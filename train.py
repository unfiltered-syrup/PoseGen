import argparse
import torch.utils.checkpoint
import math
import os
import re
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
from einops import rearrange
from tqdm import tqdm


try:
    import torchvision.models as tvm
    HAS_TORCHVISION_MODELS = True
except ImportError:
    HAS_TORCHVISION_MODELS = False


PATCH_SIZE = 4
FRAME_SIZE = 64
NUM_PATCHES = (FRAME_SIZE // PATCH_SIZE) ** 2
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3


def frame_to_patches(frame: torch.Tensor) -> torch.Tensor:
    if frame.dim() == 3:
        return rearrange(frame, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=PATCH_SIZE, p2=PATCH_SIZE)
    return rearrange(frame, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=PATCH_SIZE, p2=PATCH_SIZE)


def patches_to_frame(patches: torch.Tensor) -> torch.Tensor:
    ph = pw = FRAME_SIZE // PATCH_SIZE
    if patches.dim() == 2:
        return rearrange(patches, '(h w) (p1 p2 c) -> c (h p1) (w p2)',
                         h=ph, w=pw, p1=PATCH_SIZE, p2=PATCH_SIZE, c=3)
    return rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                     h=ph, w=pw, p1=PATCH_SIZE, p2=PATCH_SIZE, c=3)


class SpatialPosEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        ph = pw = FRAME_SIZE // PATCH_SIZE
        self.row_embed = nn.Embedding(ph, d_model // 2)
        self.col_embed = nn.Embedding(pw, d_model // 2)

    def forward(self, device):
        ph = pw = FRAME_SIZE // PATCH_SIZE
        rows = torch.arange(ph, device=device)
        cols = torch.arange(pw, device=device)
        r_emb = self.row_embed(rows).unsqueeze(1).expand(ph, pw, -1)
        c_emb = self.col_embed(cols).unsqueeze(0).expand(ph, pw, -1)
        return torch.cat([r_emb, c_emb], dim=-1).view(NUM_PATCHES, -1)


class TemporalPosEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, frame_idx: int) -> torch.Tensor:
        return self.pe[frame_idx]


class SpriteSeq2Seq(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 16,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 12,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
        max_frames: int = 24,
        num_row_labels: int = 21,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_proj = nn.Linear(PATCH_DIM, d_model)
        self.spatial_pos = SpatialPosEncoding(d_model)
        self.temporal_pos = TemporalPosEncoding(d_model, max_len=max_frames)
        self.label_embed = nn.Embedding(num_row_labels, d_model)
        self.bos_embed = nn.Embedding(num_row_labels, d_model)
        self.length_embed = nn.Embedding(max_frames + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.out_proj = nn.Linear(d_model, PATCH_DIM)
        self.skip_proj  = nn.Linear(d_model, d_model)
        self.skip_gamma = nn.Embedding(num_row_labels, d_model)
        self.skip_beta  = nn.Embedding(num_row_labels, d_model)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_frame(self, frame: torch.Tensor, frame_idx: int) -> torch.Tensor:
        tokens = self.patch_proj(frame_to_patches(frame))
        tokens = tokens + self.spatial_pos(frame.device).unsqueeze(0)
        tokens = tokens + self.temporal_pos(frame_idx).unsqueeze(0).unsqueeze(0)
        return tokens

    def _apply_skip(self, dec_out: torch.Tensor, memory: torch.Tensor, row_labels: torch.Tensor) -> torch.Tensor:
        B = dec_out.size(0)
        enc_proj = self.skip_proj(memory[:, 2:, :])
        tgt_len = dec_out.size(1)
        nf = tgt_len // NUM_PATCHES
        rem = tgt_len % NUM_PATCHES
        if nf > 0:
            tiled = enc_proj.unsqueeze(2).expand(B, NUM_PATCHES, nf, -1)
            tiled = tiled.permute(0, 2, 1, 3).contiguous().view(B, nf * NUM_PATCHES, -1)
            skip = torch.cat([tiled, enc_proj[:, :rem, :]], dim=1) if rem > 0 else tiled
        else:
            skip = enc_proj[:, :rem, :]
        # FiLM modulation
        gamma = self.skip_gamma(row_labels).unsqueeze(1)  # (B, 1, d_model)
        beta  = self.skip_beta(row_labels).unsqueeze(1)   # (B, 1, d_model)
        return dec_out + gamma * skip + beta

    def forward_inference(self, frame0, target_frames, row_labels):
        B, T = frame0.size(0), target_frames.size(1)
        seq_len = torch.tensor([T], dtype=torch.long, device=frame0.device).expand(B)
        enc_tokens = torch.cat([
            self.label_embed(row_labels).unsqueeze(1),
            self.length_embed(seq_len).unsqueeze(1),
            self.encode_frame(frame0, 0)
        ], dim=1)
        memory = torch.utils.checkpoint.checkpoint(self.encoder, enc_tokens, use_reentrant=False)
        bos = self.bos_embed(row_labels).unsqueeze(1)  # (B, 1, d_model)
        if T > 1:
            dec_input = torch.cat([bos] + [self.encode_frame(target_frames[:, t], t + 1)
                                           for t in range(T - 1)], dim=1)
        else:
            dec_input = bos
        tgt_len = dec_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=frame0.device).bool()
        dec_out = torch.utils.checkpoint.checkpoint(
            self.decoder, dec_input, memory,
            tgt_mask,   
            None,       
            None,       
            None,       
            use_reentrant=False
        )
        return self.out_proj(self._apply_skip(dec_out, memory, row_labels))

    def forward(self, frame0, target_frames, row_labels, lengths=None):
        B, T, C, H, W = target_frames.shape
        seq_len = torch.tensor([T], dtype=torch.long, device=frame0.device).expand(B)
        enc_tokens = torch.cat([
            self.label_embed(row_labels).unsqueeze(1),
            self.length_embed(seq_len).unsqueeze(1),
            self.encode_frame(frame0, 0)
        ], dim=1)
        memory = torch.utils.checkpoint.checkpoint(self.encoder, enc_tokens, use_reentrant=False)
        all_tokens = torch.cat([self.encode_frame(target_frames[:, t], t + 1)
                                 for t in range(T)], dim=1)
        bos = self.bos_embed(row_labels).unsqueeze(1)
        dec_input = torch.cat([bos, all_tokens[:, :-1, :]], dim=1)
        tgt_len = dec_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=frame0.device).bool()
        dec_out = torch.utils.checkpoint.checkpoint(
            self.decoder, dec_input, memory,
            tgt_mask,    
            None,        
            None,        
            None,        
            use_reentrant=False
        )
        return torch.sigmoid(self.out_proj(self._apply_skip(dec_out, memory, row_labels)))

    @torch.no_grad()
    def generate(self, frame0, row_label, num_frames):
        dummy_targets = frame0.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        pred = self.forward(frame0, dummy_targets, row_label)
        B, T_patches, D = pred.shape
        T = T_patches // NUM_PATCHES
        patches = pred.view(B, T, NUM_PATCHES, PATCH_DIM)
        return torch.stack([patches_to_frame(patches[0, i]) for i in range(T)])


class SpriteAnimDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples = []
        self._scan()

    @staticmethod
    def _png_size(path) -> tuple[int, int] | None:
        import struct
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
        import os
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
        img_tensor = TF.to_tensor(Image.open(png_path).convert('RGB'))
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

        # Handle Subset datasets
        if hasattr(dataset, 'indices'):
            actual_dataset = dataset.dataset
            valid_indices = set(dataset.indices)
            abs_to_local = {abs_idx: local_idx for local_idx, abs_idx in enumerate(dataset.indices)}
        else:
            actual_dataset = dataset
            valid_indices = None
            abs_to_local = None

        from collections import defaultdict
        buckets = defaultdict(list)
        for i, (_, _, nf) in enumerate(actual_dataset.samples):
            if valid_indices is not None and i not in valid_indices:
                continue
            if abs_to_local is not None:
                local_i = abs_to_local.get(i)
                if local_i is None:
                    continue
                buckets[nf].append(local_i)
            else:
                buckets[nf].append(i)
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
            n = len(idxs) // self.batch_size
            total += n
        return (total // self.world_size) * self.batch_size


def collate_fn(batch, max_frames=None):
    frame0_list, target_list, labels = zip(*batch)
    if max_frames is not None:
        target_list = [t[:max_frames] for t in target_list]
    lengths = [t.size(0) for t in target_list]
    T_max = max(lengths)
    padded = torch.zeros(len(batch), T_max, 3, FRAME_SIZE, FRAME_SIZE)
    for i, t in enumerate(target_list):
        padded[i, :t.size(0)] = t
    return torch.stack(frame0_list), padded, torch.tensor(labels, dtype=torch.long), torch.tensor(lengths)


def get_curriculum_max_frames(epoch, total_epochs, min_frames=2, max_frames=None):
    if max_frames is None:
        return None
    halfway = total_epochs // 2
    if epoch >= halfway:
        return max_frames
    progress = epoch / max(halfway - 1, 1)
    return max(min_frames, int(min_frames + progress * (max_frames - min_frames)))


class PerceptualLoss(nn.Module):
    def __init__(self, vgg: nn.Module):
        super().__init__()
        self.vgg = vgg
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        with torch.no_grad():
            feat_t = self.vgg(self._normalize(target))
        return F.l1_loss(self.vgg(self._normalize(pred)), feat_t)


def _gaussian_kernel(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def ssim_loss(pred, target, window_size=7):
    C = pred.size(1)
    k1d = _gaussian_kernel(window_size, 1.5, pred.device, pred.dtype)
    kernel = (k1d.unsqueeze(0) * k1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0).expand(C, 1, window_size, window_size)
    pad = window_size // 2

    def _c(x):
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu_p, mu_t = _c(pred), _c(target)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    s_p2 = _c(pred * pred) - mu_p2
    s_t2 = _c(target * target) - mu_t2
    s_pt = _c(pred * target) - mu_pt
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_pt + C1) * (2 * s_pt + C2)) / ((mu_p2 + mu_t2 + C1) * (s_p2 + s_t2 + C2))
    return 1.0 - ssim_map.mean()


def compute_loss(pred, target, lengths, perceptual_loss_fn=None, row_labels=None):
    B, T, C, H, W = target.shape
    pred_frames = patches_to_frame(pred.view(B * T, NUM_PATCHES, PATCH_DIM))
    target_flat = target.view(B * T, C, H, W)
    mask = torch.zeros(B, T, device=pred.device)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0
    valid = mask.view(B * T).bool()
    if valid.sum() == 0:
        return pred.sum() * 0.0
    pv, tv = pred_frames[valid], target_flat[valid]
    if row_labels is not None:
        frame_weights = (1.0 / lengths.float().clamp(min=1)).to(pred.device)
        frame_weights = frame_weights / frame_weights.mean()
        weight_expanded = frame_weights.unsqueeze(1).expand(B, T).reshape(B * T)
        valid_weights = weight_expanded[valid]
    else:
        valid_weights = None
    perc = perceptual_loss_fn(pv, tv) if perceptual_loss_fn is not None else torch.tensor(0.0, device=pred.device)
    if valid_weights is not None:
        l1 = (F.l1_loss(pv, tv, reduction='none').mean(dim=[1, 2, 3]) * valid_weights).mean()
        return 0.1 * l1 + 0.1 * perc + 0.5 * ssim_loss(pv, tv)
    else:
        return 0.1 * F.l1_loss(pv, tv) + 0.1 * perc + 0.5 * ssim_loss(pv, tv)


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 2, 1),
        )

    def forward(self, x):
        return self.net(x)


GAN_WARMUP_EPOCHS = 5
D_NOISE_STD = 0.05
D_REAL_LABEL = 0.9
ADV_LOSS_WEIGHT = 0.001


def train_one_epoch(model, loader, optimizer, device,
                    discriminator=None, optimizer_D=None,
                    perceptual_loss_fn=None, epoch: int = 0,
                    scaler: GradScaler = None, grad_accum: int = 1,
                    epoch_bar: tqdm = None):
    model.train()
    if discriminator is not None:
        discriminator.train()
    total_G = total_D = 0.0
    bce = nn.BCEWithLogitsLoss()
    gan_active = discriminator is not None and optimizer_D is not None and epoch >= GAN_WARMUP_EPOCHS
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_amp = scaler is not None

    optimizer.zero_grad()
    batch_bar = tqdm(loader, desc=f'  batches', leave=False, unit='bat', ascii=True, dynamic_ncols=True)
    for step, (frame0, target, row_labels, lengths) in enumerate(batch_bar):
        frame0, target = frame0.to(device, non_blocking=True), target.to(device, non_blocking=True)
        row_labels, lengths = row_labels.to(device, non_blocking=True), lengths.to(device, non_blocking=True)
        B, T, C, H, W = target.shape

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            pred = model(frame0, target, row_labels, lengths)
            pred_frames = patches_to_frame(pred.view(B * T, NUM_PATCHES, PATCH_DIM))
            target_flat = target.view(B * T, C, H, W)

        mask = torch.zeros(B, T, device=device)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1.0
        valid = mask.view(B * T).bool()

        if gan_active:
            real = target_flat[valid]
            fake = pred_frames[valid].detach().float()
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                r_pred = discriminator(real + torch.randn_like(real) * D_NOISE_STD)
                f_pred = discriminator(fake + torch.randn_like(fake) * D_NOISE_STD)
                loss_D = 0.5 * (bce(r_pred, torch.full_like(r_pred, D_REAL_LABEL)) +
                                bce(f_pred, torch.zeros_like(f_pred)))
            optimizer_D.zero_grad()
            if use_amp:
                scaler.scale(loss_D).backward()
                scaler.unscale_(optimizer_D)
                nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                scaler.step(optimizer_D)
                scaler.update()
            else:
                loss_D.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                optimizer_D.step()
            total_D += loss_D.item()

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            recon = compute_loss(pred, target, lengths, perceptual_loss_fn, row_labels=row_labels)
            if gan_active:
                adv_out = discriminator(pred_frames[valid])
                loss_G = (recon + ADV_LOSS_WEIGHT * bce(adv_out, torch.ones_like(adv_out))) / grad_accum
            else:
                loss_G = recon / grad_accum

        if use_amp:
            scaler.scale(loss_G).backward()
        else:
            loss_G.backward()

        is_last_step = (step + 1) == len(loader)
        if (step + 1) % grad_accum == 0 or is_last_step:
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_G += loss_G.item() * grad_accum
        running_G = total_G / (step + 1)
        running_D = total_D / (step + 1) if gan_active else 0.0
        batch_bar.set_postfix(G=f'{running_G:.4f}', D=f'{running_D:.4f}')

    batch_bar.close()
    n = len(loader)
    return total_G / n, total_D / n


@torch.no_grad()
def eval_one_epoch(model, loader, device, perceptual_loss_fn=None, use_amp: bool = False):
    model.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    total = 0.0
    val_bar = tqdm(loader, desc='  val', leave=False, unit='bat', ascii=True, dynamic_ncols=True)
    for frame0, target, row_labels, lengths in val_bar:
        frame0, target = frame0.to(device, non_blocking=True), target.to(device, non_blocking=True)
        row_labels, lengths = row_labels.to(device, non_blocking=True), lengths.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            pred = model(frame0, target, row_labels, lengths)
            loss = compute_loss(pred, target, lengths, perceptual_loss_fn, row_labels=row_labels).item()
        total += loss
        val_bar.set_postfix(loss=f'{loss:.4f}')
    val_bar.close()
    return total / len(loader)


def save_checkpoint(model, optimizer, epoch, path,
                    discriminator=None, optimizer_D=None, curriculum_max_frames=None,
                    train_losses_G=None, train_losses_D=None, val_losses=None):
    ckpt = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
    if discriminator is not None:
        ckpt['discriminator_state'] = discriminator.state_dict()
    if optimizer_D is not None:
        ckpt['optimizer_D_state'] = optimizer_D.state_dict()
    if curriculum_max_frames is not None:
        ckpt['curriculum_max_frames'] = curriculum_max_frames
    if train_losses_G is not None:
        ckpt['train_losses_G'] = train_losses_G
    if train_losses_D is not None:
        ckpt['train_losses_D'] = train_losses_D
    if val_losses is not None:
        ckpt['val_losses'] = val_losses
    torch.save(ckpt, path)


def plot_losses(train_losses_G, train_losses_D, val_losses, plot_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)
    epochs = range(1, len(train_losses_G) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_losses_G, label='train G')
    axes[0].plot(epochs, val_losses, label='val')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].set_title('Generator Loss')
    axes[0].legend()

    axes[1].plot(epochs, train_losses_D, label='train D', color='tab:orange')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('loss')
    axes[1].set_title('Discriminator Loss')
    axes[1].legend()

    fig.suptitle('PoseGen Training')
    fig.tight_layout()
    path = os.path.join(plot_dir, 'loss_curve.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    tqdm.write(f'loss curve → {path}')


def generate_animation(model, frame_0_path: str, row_label: int, num_frames: int, output_path: str):
    import torchvision.transforms.functional as TF
    device = next(model.parameters()).device
    model.eval()
    img = Image.open(frame_0_path).convert('RGB').crop((0, 0, FRAME_SIZE, FRAME_SIZE))
    frame0 = TF.to_tensor(img).unsqueeze(0).to(device)
    label = torch.tensor([row_label], dtype=torch.long, device=device)
    frames = model.generate(frame0, label, num_frames)
    all_frames = torch.cat([frame0.squeeze(0).cpu().unsqueeze(0), frames.cpu()])
    strip = Image.new('RGB', (FRAME_SIZE * all_frames.size(0), FRAME_SIZE))
    for i, f in enumerate(all_frames):
        strip.paste(TF.to_pil_image(f.clamp(0, 1)), (i * FRAME_SIZE, 0))
    strip.save(output_path)


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--data_dir', type=str, default='./data_output/frames')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=4096)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--rows', type=int, nargs='+', default=None)
    parser.add_argument('--num_workers', type=int, default=10,
                        help='DataLoader worker processes per loader')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps (simulate larger batch)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision (bf16/fp16)')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for scheduler')
    parser.add_argument('--min_frames', type=int, default=3,
                        help='Minimum frames for curriculum schedule and dynamic batch scaling')
    parser.add_argument('--use_bucket_sampler', action='store_true',
                        help='Use FrameCountBucketSampler to group batches by frame count (reduces padding waste)')
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
    tqdm.write(f'device: {device}')

    use_amp = not args.no_amp and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tqdm.write(f'AMP: {use_amp} ({amp_dtype}), grad_accum: {args.grad_accum}, num_workers: {args.num_workers}')

    dataset = SpriteAnimDataset(args.data_dir)
    if args.rows is not None:
        row_set = set(args.rows)
        dataset.samples = [s for s in dataset.samples if s[1] in row_set]
    tqdm.write(f'samples: {len(dataset)}')

    val_size = max(1, int(0.2 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    dataset_max_frames = max(s[2] - 1 for s in dataset.samples)

    val_loader = None

    model = SpriteSeq2Seq(d_model=args.d_model, dim_feedforward=args.dim_feedforward).to(device)
    tqdm.write(f'params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(1, args.epochs // 2), T_mult=1, eta_min=args.min_lr)

    perceptual_loss_fn = None
    if HAS_TORCHVISION_MODELS:
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features[:9]
        for p in vgg.parameters():
            p.requires_grad_(False)
        perceptual_loss_fn = PerceptualLoss(vgg.to(device).eval()).to(device)

    discriminator = PatchDiscriminator().to(device)
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        discriminator = DDP(discriminator, device_ids=[local_rank])

    raw_model = model.module if ddp else model
    raw_disc  = discriminator.module if ddp else discriminator

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    start_epoch = 0
    best_val_loss = float('inf')
    resume_curriculum_min = 2
    train_losses_G, train_losses_D, val_losses = [], [], []

    if args.resume and os.path.isfile(args.resume):
        if ddp:
            dist.barrier()
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        disc_state = ckpt.get('discriminator_state')
        opt_d_state = ckpt.get('optimizer_D_state')
        if disc_state:
            raw_disc.load_state_dict(disc_state)
        if opt_d_state:
            optimizer_D.load_state_dict(opt_d_state)
        resume_curriculum_min = ckpt.get('curriculum_max_frames', 2)
        train_losses_G = ckpt.get('train_losses_G', [])
        train_losses_D = ckpt.get('train_losses_D', [])
        val_losses = ckpt.get('val_losses', [])
        start_epoch = ckpt['epoch'] + 1
        tqdm.write(f'resumed epoch {ckpt["epoch"]}')

    train_loader = None
    val_loader = None
    prev_batch = None
    epoch_bar = tqdm(range(start_epoch, args.epochs), desc='epochs', unit='ep', ascii=True, dynamic_ncols=True)
    for epoch in epoch_bar:
        curr_max = max(resume_curriculum_min,
                       get_curriculum_max_frames(epoch, args.epochs, max_frames=dataset_max_frames))

        effective_batch = max(1, int(args.batch_size * args.min_frames / curr_max))
        grad_accum = max(args.grad_accum, args.batch_size // effective_batch)

        if train_loader is None or effective_batch != prev_batch:
            if train_loader is not None:
                del train_loader
            if args.use_bucket_sampler:
                bucket_sampler = FrameCountBucketSampler(
                    train_ds,
                    batch_size=effective_batch,
                    drop_last=True,
                    shuffle=True,
                    rank=local_rank if ddp else 0,
                    world_size=world_size if ddp else 1,
                )
                bucket_sampler.set_epoch(epoch)
                train_loader = DataLoader(
                    train_ds,
                    batch_sampler=bucket_sampler,
                    collate_fn=partial(collate_fn, max_frames=curr_max),
                    num_workers=args.num_workers,
                    pin_memory=True,
                    persistent_workers=False,
                )
            elif ddp:
                train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
                train_loader = DataLoader(train_ds, batch_size=effective_batch,
                                          sampler=train_sampler,
                                          collate_fn=partial(collate_fn, max_frames=curr_max),
                                          num_workers=args.num_workers, pin_memory=True,
                                          persistent_workers=False)
            else:
                train_loader = DataLoader(train_ds, batch_size=effective_batch, shuffle=True,
                                          collate_fn=partial(collate_fn, max_frames=curr_max),
                                          num_workers=args.num_workers, pin_memory=True,
                                          persistent_workers=False)
            prev_batch = effective_batch
        else:
            del train_loader
            if args.use_bucket_sampler:
                bucket_sampler = FrameCountBucketSampler(
                    train_ds,
                    batch_size=effective_batch,
                    drop_last=True,
                    shuffle=True,
                    rank=local_rank if ddp else 0,
                    world_size=world_size if ddp else 1,
                )
                bucket_sampler.set_epoch(epoch)
                train_loader = DataLoader(
                    train_ds,
                    batch_sampler=bucket_sampler,
                    collate_fn=partial(collate_fn, max_frames=curr_max),
                    num_workers=args.num_workers,
                    pin_memory=True,
                    persistent_workers=False,
                )
            elif ddp:
                train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
                train_loader = DataLoader(train_ds, batch_size=effective_batch,
                                          sampler=train_sampler,
                                          collate_fn=partial(collate_fn, max_frames=curr_max),
                                          num_workers=args.num_workers, pin_memory=True,
                                          persistent_workers=False)
            else:
                train_loader = DataLoader(train_ds, batch_size=effective_batch, shuffle=True,
                                          collate_fn=partial(collate_fn, max_frames=curr_max),
                                          num_workers=args.num_workers, pin_memory=True,
                                          persistent_workers=False)

        if val_loader is not None:
            del val_loader
        if ddp:
            val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                    sampler=val_sampler,
                                    collate_fn=partial(collate_fn, max_frames=curr_max),
                                    num_workers=args.num_workers, pin_memory=True,
                                    persistent_workers=False)
        else:
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                    collate_fn=partial(collate_fn, max_frames=curr_max),
                                    num_workers=args.num_workers,
                                    pin_memory=True, persistent_workers=False)

        if ddp and not args.use_bucket_sampler:
            train_sampler.set_epoch(epoch)
        elif args.use_bucket_sampler:
            bucket_sampler.set_epoch(epoch)

        loss_G, loss_D = train_one_epoch(model, train_loader, optimizer, device,
                                         discriminator=discriminator, optimizer_D=optimizer_D,
                                         perceptual_loss_fn=perceptual_loss_fn, epoch=epoch,
                                         scaler=scaler, grad_accum=grad_accum,
                                         epoch_bar=epoch_bar)
        val_loss = eval_one_epoch(model, val_loader, device, perceptual_loss_fn=perceptual_loss_fn,
                                  use_amp=use_amp)
        scheduler.step()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        train_losses_G.append(loss_G)
        train_losses_D.append(loss_D)
        val_losses.append(val_loss)

        if is_main_process():
            epoch_bar.set_postfix(G=f'{loss_G:.4f}', D=f'{loss_D:.4f}', val=f'{val_loss:.4f}', cur=curr_max)
            tqdm.write(f'epoch {epoch+1:3d}/{args.epochs}  G={loss_G:.5f}  D={loss_D:.5f}  val={val_loss:.5f}  cur={curr_max}')

        if (epoch + 1) % 10 == 0:
            if is_main_process():
                save_checkpoint(raw_model, optimizer, epoch,
                                os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pt'),
                                discriminator=raw_disc, optimizer_D=optimizer_D,
                                curriculum_max_frames=curr_max,
                                train_losses_G=train_losses_G, train_losses_D=train_losses_D,
                                val_losses=val_losses)
                plot_losses(train_losses_G, train_losses_D, val_losses, './plot')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if is_main_process():
                save_checkpoint(raw_model, optimizer, epoch,
                                os.path.join(args.checkpoint_dir, 'best_model.pt'),
                                discriminator=raw_disc, optimizer_D=optimizer_D,
                                curriculum_max_frames=curr_max,
                                train_losses_G=train_losses_G, train_losses_D=train_losses_D,
                                val_losses=val_losses)

    if is_main_process():
        plot_losses(train_losses_G, train_losses_D, val_losses, './plot')
        tqdm.write(f'done. best val={best_val_loss:.5f}')

    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

