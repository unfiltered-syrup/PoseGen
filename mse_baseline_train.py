import argparse
import math
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

try:
    from einops import rearrange
    HAS_EINOPS = True
except ImportError:
    HAS_EINOPS = False


PATCH_SIZE = 8
FRAME_SIZE = 64
NUM_PATCHES = (FRAME_SIZE // PATCH_SIZE) ** 2
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3


def frame_to_patches(frame: torch.Tensor) -> torch.Tensor:
    if HAS_EINOPS:
        if frame.dim() == 3:
            return rearrange(frame, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=PATCH_SIZE, p2=PATCH_SIZE)
        return rearrange(frame, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=PATCH_SIZE, p2=PATCH_SIZE)
    batched = frame.dim() == 4
    if not batched:
        frame = frame.unsqueeze(0)
    B, C, H, W = frame.shape
    ph, pw = H // PATCH_SIZE, W // PATCH_SIZE
    x = frame.view(B, C, ph, PATCH_SIZE, pw, PATCH_SIZE).permute(0, 2, 4, 3, 5, 1).contiguous()
    x = x.view(B, ph * pw, PATCH_SIZE * PATCH_SIZE * C)
    return x.squeeze(0) if not batched else x


def patches_to_frame(patches: torch.Tensor) -> torch.Tensor:
    if HAS_EINOPS:
        ph = pw = FRAME_SIZE // PATCH_SIZE
        if patches.dim() == 2:
            return rearrange(patches, '(h w) (p1 p2 c) -> c (h p1) (w p2)',
                             h=ph, w=pw, p1=PATCH_SIZE, p2=PATCH_SIZE, c=3)
        return rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                         h=ph, w=pw, p1=PATCH_SIZE, p2=PATCH_SIZE, c=3)
    batched = patches.dim() == 3
    if not batched:
        patches = patches.unsqueeze(0)
    B, N, D = patches.shape
    ph = pw = FRAME_SIZE // PATCH_SIZE
    x = patches.view(B, ph, pw, PATCH_SIZE, PATCH_SIZE, 3).permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, 3, FRAME_SIZE, FRAME_SIZE)
    return x.squeeze(0) if not batched else x


class SpatialPosEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        ph = pw = FRAME_SIZE // PATCH_SIZE
        self.row_embed = nn.Embedding(ph, d_model // 2)
        self.col_embed = nn.Embedding(pw, d_model // 2)

    def forward(self, device):
        ph = pw = FRAME_SIZE // PATCH_SIZE
        r = self.row_embed(torch.arange(ph, device=device)).unsqueeze(1).expand(ph, pw, -1)
        c = self.col_embed(torch.arange(pw, device=device)).unsqueeze(0).expand(ph, pw, -1)
        return torch.cat([r, c], dim=-1).view(NUM_PATCHES, -1)


class TemporalPosEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, frame_idx: int) -> torch.Tensor:
        return self.pe[frame_idx]


class SpriteSeq2Seq(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=1024, dropout=0.1, max_frames=64, num_row_labels=21):
        super().__init__()
        self.d_model = d_model
        self.patch_proj = nn.Linear(PATCH_DIM, d_model)
        self.spatial_pos = SpatialPosEncoding(d_model)
        self.temporal_pos = TemporalPosEncoding(d_model, max_len=max_frames)
        self.label_embed = nn.Embedding(num_row_labels, d_model)
        self.bos_token = nn.Parameter(torch.randn(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                         dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_encoder_layers)
        dec = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                         dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec, num_layers=num_decoder_layers)
        self.out_proj = nn.Linear(d_model, PATCH_DIM)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_frame(self, frame, frame_idx):
        tokens = self.patch_proj(frame_to_patches(frame))
        tokens = tokens + self.spatial_pos(frame.device).unsqueeze(0)
        tokens = tokens + self.temporal_pos(frame_idx).unsqueeze(0).unsqueeze(0)
        return tokens

    def forward_train(self, frame0, target_frames, row_labels):
        B, T, C, H, W = target_frames.shape
        enc = torch.cat([self.label_embed(row_labels).unsqueeze(1), self.encode_frame(frame0, 0)], dim=1)
        memory = self.encoder(enc)
        all_tok = torch.cat([self.encode_frame(target_frames[:, t], t + 1) for t in range(T)], dim=1)
        dec_in = torch.cat([self.bos_token.expand(B, -1, -1), all_tok[:, :-1, :]], dim=1)
        mask = nn.Transformer.generate_square_subsequent_mask(dec_in.size(1), device=frame0.device)
        dec_out = self.decoder(dec_in, memory, tgt_mask=mask)
        return torch.sigmoid(self.out_proj(dec_out))

    @torch.no_grad()
    def generate(self, frame0, row_label, num_frames):
        device = frame0.device
        enc = torch.cat([self.label_embed(row_label).unsqueeze(1), self.encode_frame(frame0, 0)], dim=1)
        memory = self.encoder(enc)
        dec_input = self.bos_token.clone()
        generated = []
        for frame_idx in range(1, num_frames + 1):
            for patch_idx in range(NUM_PATCHES):
                mask = nn.Transformer.generate_square_subsequent_mask(dec_input.size(1), device=device)
                dec_out = self.decoder(dec_input, memory, tgt_mask=mask)
                pred = torch.sigmoid(self.out_proj(dec_out[:, -1:, :]))
                generated.append(pred.squeeze(1))
                next_tok = self.patch_proj(pred)
                next_tok = next_tok + self.spatial_pos(device)[patch_idx].unsqueeze(0).unsqueeze(0)
                next_tok = next_tok + self.temporal_pos(frame_idx).unsqueeze(0).unsqueeze(0)
                dec_input = torch.cat([dec_input, next_tok], dim=1)
        patches = torch.stack(generated, dim=1).view(1, num_frames, NUM_PATCHES, PATCH_DIM)
        return torch.stack([patches_to_frame(patches[0, i]) for i in range(num_frames)])


class SpriteAnimDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples = []
        pattern = re.compile(r'entry_\d+_(?:male|female)_row(\d+)\.png$')
        for png in sorted(self.data_dir.glob('*.png')):
            m = pattern.match(png.name)
            if m is None:
                continue
            try:
                img = Image.open(png).convert('RGB')
            except Exception:
                continue
            w, h = img.size
            if h != FRAME_SIZE:
                continue
            n = w // FRAME_SIZE
            if n >= 2:
                self.samples.append((png, int(m.group(1)), n))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torchvision.transforms.functional as TF
        png_path, row_label, num_frames = self.samples[idx]
        t = TF.to_tensor(Image.open(png_path).convert('RGB'))
        frames = [t[:, :, i * FRAME_SIZE:(i + 1) * FRAME_SIZE] for i in range(num_frames)]
        return frames[0], torch.stack(frames[1:]), row_label


def collate_fn(batch):
    frame0_list, target_list, labels = zip(*batch)
    lengths = [t.size(0) for t in target_list]
    T_max = max(lengths)
    padded = torch.zeros(len(batch), T_max, 3, FRAME_SIZE, FRAME_SIZE)
    for i, t in enumerate(target_list):
        padded[i, :t.size(0)] = t
    return torch.stack(frame0_list), padded, torch.tensor(labels, dtype=torch.long), torch.tensor(lengths)


def compute_loss(pred, target, lengths):
    B, T, C, H, W = target.shape
    tp = frame_to_patches(target.view(B * T, C, H, W)).view(B, T * NUM_PATCHES, PATCH_DIM)
    mask = torch.zeros(B, T * NUM_PATCHES, device=pred.device)
    for i, l in enumerate(lengths):
        mask[i, :l * NUM_PATCHES] = 1.0
    diff = pred - tp
    mse = (diff ** 2 * mask.unsqueeze(-1)).sum() / (mask.sum() * PATCH_DIM + 1e-8)
    l1  = (diff.abs() * mask.unsqueeze(-1)).sum() / (mask.sum() * PATCH_DIM + 1e-8)
    return mse + 0.1 * l1


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for frame0, target, labels, lengths in loader:
        frame0, target, labels, lengths = (x.to(device) for x in (frame0, target, labels, lengths))
        loss = compute_loss(model.forward_train(frame0, target, labels), target, lengths)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total = 0.0
    for frame0, target, labels, lengths in loader:
        frame0, target, labels, lengths = (x.to(device) for x in (frame0, target, labels, lengths))
        total += compute_loss(model.forward_train(frame0, target, labels), target, lengths).item()
    return total / len(loader)


def save_checkpoint(model, optimizer, epoch, path, train_losses=None, val_losses=None):
    torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_losses': train_losses or [], 'val_losses': val_losses or []}, path)


def plot_losses(train_losses, val_losses, plot_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='train')
    plt.plot(epochs, val_losses, label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('MSE Baseline Loss')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(plot_dir, 'mse_loss_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'loss curve: {path}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_dir', type=str, default='./data_output/frames')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_mse')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--rows', type=int, nargs='+', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    dataset = SpriteAnimDataset(args.data_dir)
    if args.rows is not None:
        row_set = set(args.rows)
        dataset.samples = [s for s in dataset.samples if s[1] in row_set]
    print(f'samples: {len(dataset)}')

    val_size = max(1, int(0.2 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = SpriteSeq2Seq(d_model=args.d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    start_epoch, best_val = 0, float('inf')
    train_losses, val_losses = [], []

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        train_losses = ckpt.get('train_losses', [])
        val_losses = ckpt.get('val_losses', [])
        start_epoch = ckpt['epoch'] + 1
        print(f'resumed epoch {ckpt["epoch"]}')

    for epoch in range(start_epoch, args.epochs):
        tl = train_one_epoch(model, train_loader, optimizer, device)
        vl = eval_one_epoch(model, val_loader, device)
        scheduler.step()
        train_losses.append(tl)
        val_losses.append(vl)
        print(f'epoch {epoch+1:3d}/{args.epochs}  train={tl:.5f}  val={vl:.5f}')

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch,
                            os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pt'),
                            train_losses, val_losses)
            plot_losses(train_losses, val_losses, './plot')

        if vl < best_val:
            best_val = vl
            save_checkpoint(model, optimizer, epoch,
                            os.path.join(args.checkpoint_dir, 'best_model.pt'),
                            train_losses, val_losses)

    plot_losses(train_losses, val_losses, './plot')
    print(f'done. best val={best_val:.5f}')


if __name__ == '__main__':
    main()
