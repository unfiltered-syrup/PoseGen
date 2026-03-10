import argparse
import json
import os
import sys
import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator
from tqdm import tqdm


THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(THIS_DIR, "data", "lpc_raw")
OUTPUT_DIR = os.path.join(THIS_DIR, "outputs")

LPC_REPO_DIR = os.path.join(THIS_DIR, "Universal-LPC-spritesheet")

DEFAULT_SHEET = os.path.join(LPC_REPO_DIR, "body", "male", "light.png")


def _p(data):
    """Convert flat list of (x,y) pairs."""
    pts = [(data[i], data[i+1]) for i in range(0, len(data), 2)]
    return np.array(pts, dtype=np.float32)

# Canonical rest 
CANONICAL_REST = _p([
    .50,.10,              # nose
    .40,.26,  .60,.26,   # l_shoulder, r_shoulder
    .38,.40,  .62,.40,   # l_elbow,    r_elbow
    .37,.52,  .63,.52,   # l_wrist,    r_wrist
    .44,.54,  .56,.54,   # l_hip,      r_hip
    .44,.70,  .56,.70,   # l_knee,     r_knee
    .44,.88,  .56,.88,   # l_ankle,    r_ankle
])


BONE_TOPOLOGY = [
    (0,  1),   # nose  <- l_shoulder
    (1,  7),   # l_shoulder <- l_hip  (torso L)
    (2,  8),   # r_shoulder <- r_hip  (torso R)
    (3,  1),   # l_elbow <- l_shoulder (upper arm L)
    (4,  2),   # r_elbow <- r_shoulder (upper arm R)
    (5,  3),   # l_wrist <- l_elbow  (forearm L)
    (6,  4),   # r_wrist <- r_elbow  (forearm R)
    (7,  8),   # l_hip <- r_hip      (pelvis)
    (9,  7),   # l_knee <- l_hip     (upper leg L)
    (10, 8),   # r_knee <- r_hip     (upper leg R)
    (11, 9),   # l_ankle <- l_knee   (lower leg L)
    (12, 10),  # r_ankle <- r_knee   (lower leg R)
]

def _bone_length(pts, child, parent):
    """Euclidean distance between two joints in pts array."""
    d = pts[child] - pts[parent]
    return float(np.linalg.norm(d)) + 1e-8


def retarget_pose(src_ctrl: np.ndarray, canonical_frame: np.ndarray) -> np.ndarray:

    n = len(src_ctrl)
    retargeted = src_ctrl.copy()

    # Per-joint delta in canonical space
    delta = canonical_frame - CANONICAL_REST   # shape (13, 2)

    # Joints not in topology get scale 1
    body_h_canon = float(CANONICAL_REST[12, 1] - CANONICAL_REST[0, 1]) + 1e-8
    body_h_src   = float(src_ctrl[12, 1] - src_ctrl[0, 1]) + 1e-8
    global_scale  = body_h_src / body_h_canon

    joint_scale = np.full(n, global_scale, dtype=np.float32)
    for (child, parent) in BONE_TOPOLOGY:
        if child < n and parent < n:
            canon_len = _bone_length(CANONICAL_REST, child, parent)
            src_len   = _bone_length(src_ctrl,       child, parent)
            joint_scale[child] = src_len / canon_len

    # retargeted[i] = src[i] + delta[i] * per_joint_scale
    for i in range(n):
        retargeted[i] = src_ctrl[i] + delta[i] * joint_scale[i]

    # Clamp to [0,1]
    retargeted = np.clip(retargeted, 0.0, 1.0)
    return retargeted.astype(np.float32)


MIRROR_SECOND_HALF = set()

MOTION_LIBRARY = {
    "run": [
        _p([.50,.11,  .40,.27,  .60,.27,  .34,.40,  .68,.37,  .32,.51,  .70,.49,
             .44,.52,  .56,.52,  .40,.67,  .64,.59,  .38,.85,  .66,.72]),
        _p([.50,.11,  .40,.27,  .60,.27,  .36,.40,  .66,.38,  .34,.50,  .68,.48,
             .44,.51,  .56,.51,  .43,.66,  .60,.60,  .42,.83,  .62,.73]),
        _p([.50,.11,  .40,.27,  .60,.27,  .30,.38,  .72,.34,  .26,.47,  .76,.44,
             .44,.51,  .56,.51,  .34,.64,  .68,.57,  .30,.79,  .70,.69]),
        _p([.50,.10,  .40,.26,  .60,.26,  .32,.36,  .70,.35,  .28,.45,  .72,.45,
             .44,.50,  .56,.50,  .36,.62,  .66,.59,  .34,.74,  .64,.72]),
        _p([.50,.11,  .40,.27,  .60,.27,  .32,.37,  .66,.40,  .30,.49,  .68,.51,
             .44,.52,  .56,.52,  .36,.59,  .60,.67,  .34,.72,  .62,.85]),
        _p([.50,.11,  .40,.27,  .60,.27,  .38,.39,  .64,.38,  .36,.49,  .66,.48,
             .44,.51,  .56,.51,  .42,.64,  .60,.61,  .40,.78,  .62,.74]),
    ],
}


def list_available_sheets():
    if not os.path.exists(LPC_REPO_DIR):
        return
    found = 0
    for root, dirs, files in os.walk(LPC_REPO_DIR):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '_build']
        for fn in sorted(files):
            if fn.lower().endswith('.png'):
                full = os.path.join(root, fn)
                rel  = os.path.relpath(full, LPC_REPO_DIR)
                found += 1


FRAME_W, FRAME_H = 64, 64

LPC_SHEET_ANIMS = {
    "spellcast": (0,  7),
    "thrust":    (4,  8),
    "walk":      (8,  9),
    "slash":     (12, 6),
    "shoot":     (16, 13),
    "hurt":      (20, 6),
    "run":       (22, 8),
}
DIRECTIONS = ["down", "left", "right", "up"]


def extract_lpc_frames(sheet_path: str, animation: str = "walk", direction: str = "down"):
    row_offset, n_frames = LPC_SHEET_ANIMS[animation]
    dir_idx = DIRECTIONS.index(direction)
    sheet = Image.open(sheet_path).convert("RGBA")
    sheet_w, sheet_h = sheet.size
    row = row_offset + dir_idx
    frames = []
    for i in range(n_frames):
        x = i * FRAME_W
        y = row * FRAME_H
        if x + FRAME_W > sheet_w or y + FRAME_H > sheet_h:
            print(f"  [WARN] Frame {i} out of bounds at row {row}, skipping")
            break
        frame = sheet.crop((x, y, x + FRAME_W, y + FRAME_H))
        frames.append(frame)
    return frames


def get_idle_frame(sheet_path: str) -> Image.Image:
    frames = extract_lpc_frames(sheet_path, "walk", "down")
    return frames[0] if frames else None



def fallback_control_points(image_rgba: Image.Image):
    arr = np.array(image_rgba)
    alpha = arr[:, :, 3]
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h_img, w_img = arr.shape[:2]

    cx = (cmin + cmax) / 2.0 / w_img
    top  = rmin / h_img
    bot  = rmax / h_img
    span = bot - top

    # Rough skeleton heuristic based on body proportions
    pts = np.array([
        [cx,        top + span * 0.07],   # nose
        [cx - 0.10, top + span * 0.22],   # l_shoulder
        [cx + 0.10, top + span * 0.22],   # r_shoulder
        [cx - 0.13, top + span * 0.38],   # l_elbow
        [cx + 0.13, top + span * 0.36],   # r_elbow
        [cx - 0.14, top + span * 0.52],   # l_wrist
        [cx + 0.14, top + span * 0.50],   # r_wrist
        [cx - 0.07, top + span * 0.54],   # l_hip
        [cx + 0.07, top + span * 0.54],   # r_hip
        [cx - 0.07, top + span * 0.70],   # l_knee
        [cx + 0.07, top + span * 0.70],   # r_knee
        [cx - 0.07, top + span * 0.88],   # l_ankle
        [cx + 0.07, top + span * 0.88],   # r_ankle
    ], dtype=np.float32)
    return pts


def get_control_points(image_rgba: Image.Image, rig_pts: np.ndarray = None):
    if rig_pts is not None and rig_pts.shape == (13, 2):
        return rig_pts.astype(np.float32)
    return fallback_control_points(image_rgba)


_BOUNDARY_ANCHORS = np.array([
    [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
    [0.0, 0.5],             [1.0, 0.5],
    [0.0, 1.0], [0.5, 1.0], [1.0, 1.0],
], dtype=np.float32)


def build_warp(src_pts: np.ndarray, dst_pts: np.ndarray, img_shape):
    h, w = img_shape
    scale = np.array([w, h], dtype=np.float32)

    # Append identity-mapped boundary anchors
    src_aug = np.vstack([src_pts, _BOUNDARY_ANCHORS])
    dst_aug = np.vstack([dst_pts, _BOUNDARY_ANCHORS])

    src_px = src_aug * scale
    dst_px = dst_aug * scale

    rbf = RBFInterpolator(dst_px, src_px, kernel="thin_plate_spline", smoothing=1e-3)
    return rbf


def warp_image(src_rgba: np.ndarray, rbf_inv):
    h, w = src_rgba.shape[:2]

    # Build grid of destination pixel coords
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    dst_coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)

    # Map each dest pixel to source pixel
    src_coords = rbf_inv(dst_coords)
    src_x = src_coords[:, 0].reshape(h, w)
    src_y = src_coords[:, 1].reshape(h, w)

    # Bilinear sample source image at mapped coords
    out = np.zeros_like(src_rgba)
    map_x = src_x.astype(np.float32)
    map_y = src_y.astype(np.float32)

    for c in range(4):
        out[:, :, c] = cv2.remap(
            src_rgba[:, :, c].astype(np.float32),
            map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    # Preserve alpha, zero out pixels where original alpha was 0
    out[:, :, 3] = (out[:, :, 3] > 10).astype(np.uint8) * out[:, :, 3]
    return out


def generate_animation(
    source_path: str,
    anim_type: str = "run",
    upscale_warp: int = 4,
    verbose: bool = True,
    rig_pts: np.ndarray = None,
):

    if anim_type not in MOTION_LIBRARY:
        raise ValueError(f"Unknown anim_type '{anim_type}'. Choose from: {list(MOTION_LIBRARY)}")

    source_img = Image.open(source_path).convert("RGBA")
    orig_w, orig_h = source_img.size

    work_img = source_img.resize(
        (orig_w * upscale_warp, orig_h * upscale_warp), Image.NEAREST
    )
    work_arr = np.array(work_img)  # (H, W, 4)
    work_h, work_w = work_arr.shape[:2]

    src_ctrl = get_control_points(source_img, rig_pts)
    if verbose:
        print(f"Control points: {src_ctrl.shape[0]} joints")

    target_poses = MOTION_LIBRARY[anim_type]
    frames_out = []
    desc = f"Generating '{anim_type}' frames"

    for i, tgt_pose in enumerate(tqdm(target_poses, desc=desc, disable=not verbose)):
        tgt_retargeted = retarget_pose(src_ctrl, tgt_pose)

        rbf_inv = build_warp(src_ctrl, tgt_retargeted, (work_h, work_w))
        warped = warp_image(work_arr, rbf_inv)

        frame_pil = Image.fromarray(warped, "RGBA").resize(
            (orig_w, orig_h), Image.LANCZOS
        )
        frames_out.append(frame_pil)

    return frames_out


def save_frames(frames, output_base_dir: str, sprite_name: str, anim_type: str,
                flat: bool = False):

    if flat:
        out_dir = os.path.join(output_base_dir, f"{sprite_name}_{anim_type}")
    else:
        out_dir = os.path.join(output_base_dir, sprite_name, anim_type)
    os.makedirs(out_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        path = os.path.join(out_dir, f"frame_{i:04d}.png")
        frame.save(path)
    print(f"Saved {len(frames)} frames: {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description= "Warp sprite animation generator"
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
    )
    parser.add_argument(
        "--anim", "-a", type=str, default="run",
        choices=list(MOTION_LIBRARY.keys()),
    )
    parser.add_argument(
        "--list-sheets", "--download", action="store_true",
        dest="list_sheets",
    )
    parser.add_argument(
        "--upscale", type=int, default=4,
    )
    parser.add_argument(
        "--rig", type=str, default=None,
        metavar="JSON",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        metavar="DIR",
    )
    parser.add_argument(
        "--flat", action="store_true",
    )
    args = parser.parse_args()

    if args.list_sheets:
        list_available_sheets()
        sys.exit(0)

    if args.input:
        source_path = args.input
    else:
        idle = get_idle_frame(DEFAULT_SHEET)

        os.makedirs(DATA_DIR, exist_ok=True)
        tmp_path = os.path.join(DATA_DIR, "_idle_extracted.png")
        idle.save(tmp_path)
        source_path = tmp_path


    sprite_name = os.path.splitext(os.path.basename(source_path))[0]

    # Parse optional manual rig from JSON arg
    rig_pts = None
    if args.rig:
        try:
            raw = json.loads(args.rig)
            rig_pts = np.array(raw, dtype=np.float32)
            if rig_pts.shape != (13, 2):
                rig_pts = None
        except Exception as e:
            rig_pts = None

    frames = generate_animation(
        source_path=source_path,
        anim_type=args.anim,
        upscale_warp=args.upscale,
        verbose=True,
        rig_pts=rig_pts,
    )

    base_out = args.output_dir if args.output_dir else OUTPUT_DIR
    out_dir = save_frames(frames, base_out, sprite_name, args.anim, flat=args.flat)
    print(f"Done, {len(frames)} frames in:\n  {out_dir}")


if __name__ == "__main__":
    main()
