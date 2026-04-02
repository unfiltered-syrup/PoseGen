import argparse
import os

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from train import SpriteSeq2Seq, SpriteAnimDataset, FRAME_SIZE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt')
    parser.add_argument('--data_dir', type=str, default='./data_output/frames')
    parser.add_argument('--num_generate', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='model_output')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--rows', type=int, nargs='+', default=None)
    return parser.parse_args()


def save_strip(frames: torch.Tensor, path: str) -> None:
    N, C, H, W = frames.shape
    strip = Image.new('RGB', (W * N, H))
    for i, f in enumerate(frames):
        strip.paste(TF.to_pil_image(f.clamp(0, 1)), (i * W, 0))
    strip.save(path)


def save_grid(strips, plot_dir, title, filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n_rows, n_cols = len(strips), strips[0][1].shape[0]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2), squeeze=False)
    for r, (label, frames) in enumerate(strips):
        for c, frame in enumerate(frames):
            axes[r][c].imshow(frame.permute(1, 2, 0).clamp(0, 1).numpy())
            axes[r][c].axis('off')
            if c == 0:
                axes[r][c].set_title(f'row {label}', fontsize=6)
    plt.suptitle(title, fontsize=8)
    plt.tight_layout()
    path = os.path.join(plot_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'grid saved: {path}')


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = SpriteAnimDataset(args.data_dir)
    if args.rows is not None:
        row_set = set(args.rows)
        dataset.samples = [s for s in dataset.samples if s[1] in row_set]
    if len(dataset) == 0:
        raise RuntimeError(f'no samples found in {args.data_dir!r}')

    num_samples = min(args.num_samples, len(dataset))

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f'checkpoint not found: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device)
    d_model = ckpt['model_state']['patch_proj.weight'].shape[0]
    model = SpriteSeq2Seq(d_model=d_model).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'checkpoint: {args.checkpoint}  d_model={d_model}  epoch={ckpt.get("epoch","?")}')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('./plot', exist_ok=True)
    strips = []

    for idx in range(num_samples):
        frame0_gt, _, row_label = dataset[idx]
        frame0 = frame0_gt.unsqueeze(0).to(device)
        label = torch.tensor([row_label], dtype=torch.long, device=device)
        with torch.no_grad():
            generated = model.generate(frame0, label, args.num_generate)
        all_frames = torch.cat([frame0.squeeze(0).cpu().unsqueeze(0), generated.cpu()])
        out_path = os.path.join(args.output_dir, f'sample_{idx:02d}_row{row_label:02d}.png')
        save_strip(all_frames, out_path)
        strips.append((row_label, all_frames))
        print(f'[{idx+1}/{num_samples}] {out_path}')

    save_grid(strips, './plot', 'GAN+Perceptual — Generated Animations', 'inference_grid.png')


if __name__ == '__main__':
    main()
