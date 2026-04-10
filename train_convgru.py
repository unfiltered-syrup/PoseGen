"""
train_convgru.py — Lightweight ConvGRU-based delta predictor for sprite animation generation.

Architecture:
  - CNN Encoder  : 4→32→64→128→256 channels, stride-2 for first two layers → (B,256,16,16)
  - Label Embed  : nn.Embedding(20, 256) broadcast-added to spatial latent
  - ConvGRUCell  : Standard reset/update/new gates on (B,256,16,16) hidden state
  - CNN Decoder  : 256→128→64→32→4 via transposed convs + tanh → delta (B,4,64,64)
  - Frame synth  : frame[t] = clamp(frame[t-1] + delta[t], -1, 1)
"""

import argparse
import os
import re
import struct
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from PIL import Image
from tqdm import tqdm

FRAME_SIZE = 64
NUM_CLASSES = 20


class ConvGRUCell(nn.Module):
    """Single ConvGRU recurrent cell operating on spatial feature maps."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        # reset gate
        self.reset_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        # update gate
        self.update_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        # new gate
        self.new_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        xh = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.reset_gate(xh))
        z = torch.sigmoid(self.update_gate(xh))
        xrh = torch.cat([x, r * h], dim=1)
        n = torch.tanh(self.new_gate(xrh))
        h_new = (1.0 - z) * h + z * n
        return h_new


class SpriteConvGRU(nn.Module):

    def __init__(self, num_classes: int = NUM_CLASSES, hidden_dim: int = 256, label_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(4,   32,  3, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32,  64,  3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64,  128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.label_embed = nn.Embedding(num_classes, label_dim)
        self.label_proj  = nn.Linear(label_dim, hidden_dim) 


        self.gru_cell = ConvGRUCell(input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=3)


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,  32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32,   4, 3, stride=1, padding=1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        return self.encoder(frame)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h)

    def forward(
        self,
        frame0: torch.Tensor,
        target_frames: torch.Tensor,
        row_labels: torch.Tensor,
    ):

        B, T, C, H, W = target_frames.shape
        device = frame0.device

        frame0_n = frame0 * 2.0 - 1.0

        context = self.encode(frame0_n)

        label_bias = self.label_proj(self.label_embed(row_labels))  
        label_bias = label_bias.view(B, self.hidden_dim, 1, 1)      
        context = context + label_bias                              

        h = torch.zeros(B, self.hidden_dim, H // 4, W // 4, device=device)

        pred_frames_list = []
        pred_deltas_list = []

        prev_frame_n = frame0_n

        for t in range(T):
            h = self.gru_cell(context, h)         

            # Decode hidden → delta
            delta = self.decode(h)                

            # Frame synthesis
            next_frame_n = torch.clamp(prev_frame_n + delta, -1.0, 1.0)

            # Convert back to [0,1] for output
            pred_frames_list.append((next_frame_n + 1.0) / 2.0)
            pred_deltas_list.append(delta)

            prev_frame_n = next_frame_n

        pred_frames = torch.stack(pred_frames_list, dim=1) 
        pred_deltas = torch.stack(pred_deltas_list, dim=1) 
        return pred_frames, pred_deltas

    @torch.no_grad()
    def generate(
        self,
        frame0: torch.Tensor,
        row_label: torch.Tensor,
        num_frames: int,
    ):
        
        
        B = frame0.size(0)
        device = frame0.device
        H, W = frame0.shape[-2], frame0.shape[-1]

        frame0_n = frame0 * 2.0 - 1.0
        context  = self.encode(frame0_n)

        label_bias = self.label_proj(self.label_embed(row_label))
        label_bias = label_bias.view(B, self.hidden_dim, 1, 1)
        context = context + label_bias

        h = torch.zeros(B, self.hidden_dim, H // 4, W // 4, device=device)
        prev_frame_n = frame0_n
        frames = []

        for _ in range(num_frames):
            h = self.gru_cell(context, h)
            delta = self.decode(h)
            next_frame_n = torch.clamp(prev_frame_n + delta, -1.0, 1.0)
            frames.append((next_frame_n + 1.0) / 2.0)
            prev_frame_n = next_frame_n

        return frames

class SpriteAnimDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples = []
        self._scan()

    @staticmethod
    def _png_size(path):
        try:
            with open(path, 'rb') as f:
                header = f.read(24)
            if len(header) < 24 or header[:8] != b'\x89PNG\r\n\x1a\n':
                return None
            w = struct.unpack('>I', header[16:20])[0]
            h = struct.unpack('>I', header[20:24])[0]
            return w, h
        except Exception:
            return None

    def _scan(self):
        from multiprocessing.pool import ThreadPool

        pattern = re.compile(r'entry_\d+_(?:male|female)_row(\d+)\.png$')
        all_pngs = sorted(self.data_dir.glob('*.png'))

        candidates = []
        for png in all_pngs:
            m = pattern.match(png.name)
            if m is not None:
                candidates.append((png, int(m.group(1))))

        n_threads = min(32, os.cpu_count() or 4)

        def _check(args):
            png, row_label = args
            dims = self._png_size(png)
            if dims is None:
                return None
            w, h = dims
            if h != FRAME_SIZE:
                return None
            num_frames = w // FRAME_SIZE
            if num_frames >= 2:
                return (png, row_label, num_frames)
            return None

        with ThreadPool(n_threads) as pool:
            results = list(tqdm(
                pool.imap(_check, candidates, chunksize=256),
                total=len(candidates),
                desc='scanning dataset',
                unit='file',
                ascii=True,
                dynamic_ncols=True,
            ))

        self.samples = [r for r in results if r is not None]
        self.samples.sort(key=lambda x: str(x[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        png_path, row_label, num_frames = self.samples[idx]
        import torchvision.transforms.functional as TF
        # Load as RGBA to get the alpha channel
        img = Image.open(png_path).convert('RGBA')
        img_tensor = TF.to_tensor(img)  # (4, 64, H*num_frames) — actually (4, 64, W)
        frames = [img_tensor[:, :, i * FRAME_SIZE:(i + 1) * FRAME_SIZE] for i in range(num_frames)]
        return frames[0], torch.stack(frames[1:], dim=0), row_label


class FrameCountBucketSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size: int,
                 drop_last: bool = True, shuffle: bool = True,
                 rank: int = 0, world_size: int = 1):
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.shuffle    = shuffle
        self.rank       = rank
        self.world_size = world_size
        self.epoch      = 0

        if hasattr(dataset, 'indices'):
            actual_dataset = dataset.dataset
            valid_indices  = set(dataset.indices)
            abs_to_local   = {abs_idx: local_idx
                              for local_idx, abs_idx in enumerate(dataset.indices)}
        else:
            actual_dataset = dataset
            valid_indices  = None
            abs_to_local   = None

        buckets: dict = defaultdict(list)
        for i, (_, _, nf) in enumerate(actual_dataset.samples):
            if valid_indices is not None and i not in valid_indices:
                continue
            local_i = abs_to_local[i] if abs_to_local is not None else i
            buckets[nf].append(local_i)
        self._buckets = dict(buckets)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        import random as _random
        rng = _random.Random(self.epoch)
        batches = []
        for nf, idxs in self._buckets.items():
            idxs = list(idxs)
            if self.shuffle:
                rng.shuffle(idxs)
            for start in range(0, len(idxs), self.batch_size):
                batch = idxs[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        batches = batches[self.rank::self.world_size]
        for batch in batches:
            yield batch

    def __len__(self):
        total = 0
        for idxs in self._buckets.values():
            total += len(idxs) // self.batch_size
        return (total // self.world_size) * self.batch_size



def collate_fn(batch, max_frames=None):
    frame0_list, target_list, labels = zip(*batch)
    if max_frames is not None:
        target_list = [t[:max_frames] for t in target_list]
    lengths = [t.size(0) for t in target_list]
    T_max = max(lengths)
    C = target_list[0].size(1)
    padded = torch.zeros(len(batch), T_max, C, FRAME_SIZE, FRAME_SIZE)
    for i, t in enumerate(target_list):
        padded[i, :t.size(0)] = t
    return (
        torch.stack(frame0_list),
        padded,
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )


def _gaussian_kernel(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Returns 1 - mean SSIM (lower is better)."""
    C = pred.size(1)
    k1d = _gaussian_kernel(window_size, 1.5, pred.device, pred.dtype)
    kernel = (k1d.unsqueeze(0) * k1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(C, 1, window_size, window_size)
    pad = window_size // 2

    def _c(x):
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu_p, mu_t = _c(pred), _c(target)
    mu_p2 = mu_p * mu_p
    mu_t2 = mu_t * mu_t
    mu_pt = mu_p * mu_t
    s_p2 = _c(pred * pred) - mu_p2
    s_t2 = _c(target * target) - mu_t2
    s_pt = _c(pred * target) - mu_pt
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_pt + C1) * (2 * s_pt + C2)) / (
        (mu_p2 + mu_t2 + C1) * (s_p2 + s_t2 + C2)
    )
    return 1.0 - ssim_map.mean()


def sparse_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    prev_target: torch.Tensor,
    threshold: float = 0.05,
    weight: float = 10.0,
) -> torch.Tensor:

    pixel_l1 = F.l1_loss(pred, target, reduction='none')
    change_mask = (torch.abs(target - prev_target) > threshold).float()
    w = 1.0 + (weight - 1.0) * change_mask
    return (pixel_l1 * w).mean()


def compute_loss(
    pred_frames: torch.Tensor,
    pred_deltas: torch.Tensor,
    target_frames: torch.Tensor,
    lengths: torch.Tensor,
    ssim_weight: float = 0.0,
) -> torch.Tensor:
    B, T, C, H, W = target_frames.shape
    device = pred_frames.device
    mask = torch.zeros(B, T, device=device)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0

    valid_flat = mask.view(B * T).bool()
    if valid_flat.sum() == 0:
        return pred_frames.sum() * 0.0

    pf = pred_frames.view(B * T, C, H, W)
    tf = target_frames.view(B * T, C, H, W)
    pd = pred_deltas.view(B * T, C, H, W)  

    prev_tf_full = torch.zeros_like(target_frames)
    prev_tf_full[:, 1:] = target_frames[:, :-1]
    prev_tf = prev_tf_full.view(B * T, C, H, W)

    pf_v      = pf[valid_flat]
    tf_v      = tf[valid_flat]
    prev_tf_v = prev_tf[valid_flat]
    pd_v      = pd[valid_flat]

    gt_delta_01 = tf_v - prev_tf_v   
    gt_delta_norm = gt_delta_01 * 2.0
    spl1 = sparse_l1_loss(pf_v, tf_v, prev_tf_v, threshold=0.05, weight=10.0)

    delta_l1 = F.l1_loss(pd_v, gt_delta_norm)

    loss = 0.6 * spl1 + 0.4 * delta_l1

    if ssim_weight > 0.0:
        loss = loss + ssim_weight * ssim_loss(pf_v, tf_v)

    return loss


def get_curriculum_max_frames(epoch: int, total_epochs: int,
                               min_frames: int = 3, max_frames: int = None) -> int | None:
    if max_frames is None:
        return None
    halfway = total_epochs // 2
    if epoch >= halfway:
        return max_frames
    progress = epoch / max(halfway - 1, 1)
    return max(min_frames, int(min_frames + progress * (max_frames - min_frames)))


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def save_checkpoint(model, optimizer, epoch: int, path: str, best_val: float = None,
                    train_losses=None, val_losses=None, scheduler=None):
    ckpt = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    if best_val is not None:
        ckpt['best_val'] = best_val
    if train_losses is not None:
        ckpt['train_losses'] = train_losses
    if val_losses is not None:
        ckpt['val_losses'] = val_losses
    if scheduler is not None:
        ckpt['scheduler_state'] = scheduler.state_dict()
    torch.save(ckpt, path)


def plot_losses(train_losses: list, val_losses: list, plot_dir: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, label='train')
    ax.plot(epochs[:len(val_losses)], val_losses, label='val')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('ConvGRU Training Loss')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(plot_dir, 'loss_curve.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    tqdm.write(f'loss curve → {path}')


def save_gif(frames_01: list, path: str, scale: int = 4, duration_ms: int = 120):

    import torchvision.transforms.functional as TF
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    pil_frames = []
    for f in frames_01:
        if f.dim() == 3 and f.size(0) in (3, 4):
            img = TF.to_pil_image(f.clamp(0, 1).cpu())
        else:
            img = Image.fromarray((f.clamp(0, 1).cpu().numpy() * 255).astype('uint8'))
        W, H = img.size
        img = img.resize((W * scale, H * scale), Image.NEAREST)
        pil_frames.append(img.convert('RGBA'))

    if pil_frames:
        pil_frames[0].save(
            path, save_all=True, append_images=pil_frames[1:],
            loop=0, duration=duration_ms
        )

def train_one_epoch(model, loader, optimizer, device, scaler: GradScaler, use_amp: bool):
    model.train()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    total_loss = 0.0

    bar = tqdm(loader, desc='  train', leave=False, unit='bat', ascii=True, dynamic_ncols=True)
    optimizer.zero_grad()
    for step, (frame0, target, row_labels, lengths) in enumerate(bar):
        frame0     = frame0.to(device, non_blocking=True)
        target     = target.to(device, non_blocking=True)
        row_labels = row_labels.to(device, non_blocking=True)
        lengths    = lengths.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            pred_frames, pred_deltas = model(frame0, target, row_labels)
            loss = compute_loss(pred_frames, pred_deltas, target, lengths)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        optimizer.zero_grad()

        total_loss += loss.item()
        bar.set_postfix(loss=f'{loss.item():.4f}')

    bar.close()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_one_epoch(model, loader, device, use_amp: bool):
    model.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    total_loss = 0.0

    bar = tqdm(loader, desc='  val  ', leave=False, unit='bat', ascii=True, dynamic_ncols=True)
    for frame0, target, row_labels, lengths in bar:
        frame0     = frame0.to(device, non_blocking=True)
        target     = target.to(device, non_blocking=True)
        row_labels = row_labels.to(device, non_blocking=True)
        lengths    = lengths.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            pred_frames, pred_deltas = model(frame0, target, row_labels)
            loss = compute_loss(pred_frames, pred_deltas, target, lengths)

        total_loss += loss.item()
        bar.set_postfix(loss=f'{loss.item():.4f}')

    bar.close()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def generate_val_gif(model, val_dataset, device, epoch: int, plot_dir: str):
    model.eval()
    sample = val_dataset[0]
    frame0_raw, target_frames_raw, row_label = sample

    frame0 = frame0_raw.unsqueeze(0).to(device)
    label  = torch.tensor([row_label], dtype=torch.long, device=device)
    num_frames = target_frames_raw.size(0)

    raw_model = model.module if isinstance(model, DDP) else model
    frames = raw_model.generate(frame0, label, num_frames)
    all_frames = [frame0_raw] + [f.squeeze(0).cpu() for f in frames]

    os.makedirs(plot_dir, exist_ok=True)
    gif_path = os.path.join(plot_dir, f'sample_epoch_{epoch+1:04d}.gif')
    save_gif(all_frames, gif_path, scale=4, duration_ms=120)
    tqdm.write(f'val gif → {gif_path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='ConvGRU delta predictor for sprite animation generation'
    )
    parser.add_argument('--data_dir',       type=str,   default='data_output/frames',
                        help='Directory with PNG sprite sheets')
    parser.add_argument('--checkpoint_dir', type=str,   default='checkpoints_convgru',
                        help='Directory to save checkpoints')
    parser.add_argument('--plot_dir',       type=str,   default='plot_convgru',
                        help='Directory to save plots and sample GIFs')
    parser.add_argument('--epochs',         type=int,   default=200)
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--hidden_dim',     type=int,   default=256,
                        help='ConvGRU hidden / encoder channel dimension')
    parser.add_argument('--num_workers',    type=int,   default=4)
    parser.add_argument('--amp',            action='store_true',
                        help='Enable automatic mixed precision')
    parser.add_argument('--resume',         type=str,   default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--min_frames',     type=int,   default=3,
                        help='Minimum frames for curriculum schedule')
    parser.add_argument('--max_frames',     type=int,   default=None,
                        help='Maximum frames cap (auto from dataset if omitted)')
    parser.add_argument('--save_every',     type=int,   default=10,
                        help='Save checkpoint every N epochs')
    return parser.parse_args()


def main():
    args = parse_args()

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    ddp = world_size > 1

    if ddp:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tqdm.write(f'device: {device}  |  ddp: {ddp}  |  world_size: {world_size}')

    use_amp = args.amp and device.type == 'cuda'
    scaler  = GradScaler(enabled=use_amp)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tqdm.write(f'AMP: {use_amp}  ({amp_dtype})')

    dataset = SpriteAnimDataset(args.data_dir)
    tqdm.write(f'total samples: {len(dataset)}')

    dataset_max_frames = max(s[2] - 1 for s in dataset.samples)
    cli_max_frames = args.max_frames if args.max_frames is not None else dataset_max_frames
    tqdm.write(f'dataset_max_frames: {dataset_max_frames}  |  cap: {cli_max_frames}')

    val_size  = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    model = SpriteConvGRU(
        num_classes=NUM_CLASSES,
        hidden_dim=args.hidden_dim,
        label_dim=args.hidden_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tqdm.write(f'SpriteConvGRU parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    raw_model = model.module if ddp else model

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    start_epoch    = 0
    best_val_loss  = float('inf')
    train_losses: list = []
    val_losses:   list = []

    if args.resume and os.path.isfile(args.resume):
        if ddp:
            dist.barrier()
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        train_losses  = ckpt.get('train_losses', [])
        val_losses    = ckpt.get('val_losses', [])
        best_val_loss = ckpt.get('best_val', float('inf'))
        start_epoch   = ckpt['epoch'] + 1
        tqdm.write(f'resumed from epoch {ckpt["epoch"]}')

    epoch_bar = tqdm(
        range(start_epoch, args.epochs),
        desc='epochs', unit='ep', ascii=True, dynamic_ncols=True
    )

    for epoch in epoch_bar:
        curr_max = get_curriculum_max_frames(
            epoch, args.epochs,
            min_frames=args.min_frames,
            max_frames=cli_max_frames,
        )

        cf = partial(collate_fn, max_frames=curr_max)

        if ddp:
            train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
            train_sampler.set_epoch(epoch)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, sampler=train_sampler,
                collate_fn=cf, num_workers=args.num_workers, pin_memory=True,
                persistent_workers=False,
            )
            val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, sampler=val_sampler,
                collate_fn=cf, num_workers=args.num_workers, pin_memory=True,
                persistent_workers=False,
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                collate_fn=cf, num_workers=args.num_workers, pin_memory=True,
                persistent_workers=False,
            )
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                collate_fn=cf, num_workers=args.num_workers, pin_memory=True,
                persistent_workers=False,
            )

        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, use_amp)

        val_loss = eval_one_epoch(model, val_loader, device, use_amp)

        scheduler.step()

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        if is_main_process():
            epoch_bar.set_postfix(
                train=f'{train_loss:.4f}',
                val=f'{val_loss:.4f}',
                lr=f'{current_lr:.2e}',
                cur=curr_max,
            )
            tqdm.write(
                f'Epoch {epoch+1:3d}/{args.epochs}'
                f' | Train Loss: {train_loss:.4f}'
                f' | Val Loss: {val_loss:.4f}'
                f' | LR: {current_lr:.2e}'
                f' | cur_max_frames: {curr_max}'
            )

            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    raw_model, optimizer, epoch,
                    os.path.join(args.checkpoint_dir, f'epoch_{epoch+1:04d}.pt'),
                    best_val=best_val_loss,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    scheduler=scheduler,
                )
                plot_losses(train_losses, val_losses, args.plot_dir)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    raw_model, optimizer, epoch,
                    os.path.join(args.checkpoint_dir, 'best_model.pt'),
                    best_val=best_val_loss,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    scheduler=scheduler,
                )
                tqdm.write(f'  ↳ new best val loss: {best_val_loss:.4f}')

            generate_val_gif(raw_model, val_ds, device, epoch, args.plot_dir)

    if is_main_process():
        plot_losses(train_losses, val_losses, args.plot_dir)
        save_checkpoint(
            raw_model, optimizer, args.epochs - 1,
            os.path.join(args.checkpoint_dir, 'final_model.pt'),
            best_val=best_val_loss,
            train_losses=train_losses,
            val_losses=val_losses,
            scheduler=scheduler,
        )
        tqdm.write(f'Done. Best val loss: {best_val_loss:.4f}')

    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()