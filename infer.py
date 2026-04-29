import argparse
import os
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from train import FRAME_SIZE, SpriteAnimDataset, SpriteSeq2Seq


# These lengths match common LPC row animation cycles.
ROW_NUM_FRAMES = {
    **{row: 6 for row in range(0, 4)},
    **{row: 7 for row in range(4, 7)},
    **{row: 8 for row in range(7, 11)},
    **{row: 5 for row in range(11, 15)},
    **{row: 12 for row in range(15, 19)},
    19: 5,
}
DEFAULT_NUM_FRAMES = 8


# Inference defaults target the restored best old checkpoint.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./best_model_old.pt')
    parser.add_argument('--data_dir', type=str, default='./data_output/frames')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='model_output')
    parser.add_argument('--plot_dir', type=str, default='./plot')
    parser.add_argument('--rows', type=int, nargs='+', default=None)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--gif_duration', type=int, default=120)
    return parser.parse_args()


def frame_to_rgba(frame: torch.Tensor, alpha_threshold: float = 0.04) -> Image.Image:
    # Near-black generated pixels are treated as transparent background.
    rgb = frame.clamp(0, 1)
    mask = (rgb > alpha_threshold).any(dim=0)
    # GIF and PNG exports need an explicit alpha channel.
    alpha = mask.to(torch.uint8) * 255
    rgb_img = TF.to_pil_image(rgb).convert('RGBA')
    red, green, blue, _ = rgb_img.split()
    alpha_img = Image.fromarray(alpha.numpy(), mode='L')
    return Image.merge('RGBA', (red, green, blue, alpha_img))


def save_strip(frames: torch.Tensor, path: str) -> None:
    # PNG strips keep every generated frame inspectable side by side.
    num_frames, _, height, width = frames.shape
    strip = Image.new('RGBA', (width * num_frames, height), (0, 0, 0, 0))
    for i, frame in enumerate(frames):
        rgba = frame_to_rgba(frame)
        strip.paste(rgba, (i * width, 0), mask=rgba.split()[3])
    strip.save(path)


def save_gif(frames: torch.Tensor, path: str, duration: int = 120) -> None:
    pil_frames = []
    for frame in frames:
        rgba = frame_to_rgba(frame)
        width, height = rgba.size
        # Nearest-neighbor upscaling keeps pixel art crisp.
        pil_frames.append(rgba.resize((width * 4, height * 4), Image.NEAREST).convert('RGBA'))

    if not pil_frames:
        return

    # Disposal clears each pixel-art frame before drawing the next.
    pil_frames[0].save(
        path,
        format='GIF',
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
        disposal=2,
    )


def save_grid(strips, plot_dir: str, title: str, filename: str) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    # The grid compares multiple sampled rows in one static image.
    n_rows = len(strips)
    n_cols = max(frames.shape[0] for _, frames in strips)
    # Empty cells handle rows with shorter animation cycles.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2), squeeze=False)
    for row_idx, (label, frames) in enumerate(strips):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx < frames.shape[0]:
                ax.imshow(frames[col_idx].permute(1, 2, 0).clamp(0, 1).numpy())
                if col_idx == 0:
                    ax.set_title(f'row {label}', fontsize=6)
            else:
                ax.set_visible(False)
            ax.axis('off')
    plt.suptitle(title, fontsize=8)
    plt.tight_layout()
    path = os.path.join(plot_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'grid saved: {path}')


def load_model(checkpoint_path: str, device: torch.device, nhead: int) -> tuple[SpriteSeq2Seq, dict]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')

    # Loading on the target device avoids a second model transfer.
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['model_state']
    # The restored training path expects a learned BOS token.
    if 'bos_token' not in state:
        raise RuntimeError('checkpoint does not match the restored bos-token transformer')

    # Model shape is inferred from weights to avoid stale CLI defaults.
    d_model = state['patch_proj.weight'].shape[0]
    dim_feedforward = state['encoder.layers.0.linear1.weight'].shape[0]
    max_frames = state['temporal_pos.pe'].shape[0]
    # Layer counts are counted from serialized Transformer blocks.
    enc_layers = sum(1 for key in state if key.startswith('encoder.layers.') and key.endswith('.norm1.weight'))
    dec_layers = sum(1 for key in state if key.startswith('decoder.layers.') and key.endswith('.norm1.weight'))

    model = SpriteSeq2Seq(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        max_frames=max_frames,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
    ).to(device)
    model.load_state_dict(state)
    # Eval mode disables dropout for deterministic sample exports.
    model.eval()
    print(
        f'checkpoint: {checkpoint_path}  d_model={d_model}  dim_ff={dim_feedforward}  '
        f'enc={enc_layers}  dec={dec_layers}  max_frames={max_frames}  epoch={ckpt.get("epoch", "?")}'
    )
    return model, ckpt


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # The dataset provides real first frames for conditioning.
    dataset = SpriteAnimDataset(args.data_dir)
    if args.rows is not None:
        row_set = set(args.rows)
        dataset.samples = [sample for sample in dataset.samples if sample[1] in row_set]
    if len(dataset) == 0:
        raise RuntimeError(f'no samples found in {args.data_dir!r}')

    model, _ = load_model(args.checkpoint, device, args.nhead)
    num_samples = min(args.num_samples, len(dataset))
    # Random sampling gives a quick qualitative spread across rows.
    indices = random.sample(range(len(dataset)), num_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    strips = []
    for sample_num, idx in enumerate(indices):
        frame0_gt, _, row_label = dataset[idx]
        # Row-specific lengths mirror the LPC animation layout.
        num_generate = ROW_NUM_FRAMES.get(row_label, DEFAULT_NUM_FRAMES)
        frame0 = frame0_gt.unsqueeze(0).to(device)
        label = torch.tensor([row_label], dtype=torch.long, device=device)
        with torch.no_grad():
            # Generation returns only predicted future frames.
            generated = model.generate(frame0, label, num_generate)

        # Prefix the original frame for easier visual comparison.
        all_frames = torch.cat([frame0.squeeze(0).cpu().unsqueeze(0), generated.cpu()])
        base_name = f'sample_{sample_num:02d}_row{row_label:02d}'
        strip_path = os.path.join(args.output_dir, f'{base_name}.png')
        gif_path = os.path.join(args.output_dir, f'{base_name}.gif')
        # Save both debugging strips and presentation-friendly GIFs.
        save_strip(all_frames, strip_path)
        save_gif(all_frames, gif_path, duration=args.gif_duration)
        strips.append((row_label, all_frames))
        print(f'[{sample_num + 1}/{num_samples}] row={row_label} frames={num_generate}  {strip_path}  {gif_path}')

    save_grid(strips, args.plot_dir, 'GAN + perceptual generated animations', 'inference_grid.png')


if __name__ == '__main__':
    main()
