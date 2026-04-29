import argparse
import math
import os
import re
import struct
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split
from tqdm import tqdm

try:
    import torchvision.models as tvm

    HAS_TORCHVISION_MODELS = True
except ImportError:
    # Training can still run without perceptual loss.
    HAS_TORCHVISION_MODELS = False


PATCH_SIZE = 4
FRAME_SIZE = 64
NUM_PATCHES = (FRAME_SIZE // PATCH_SIZE) ** 2
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3


# Split each 64x64 sprite frame into transformer patch tokens.
def frame_to_patches(frame: torch.Tensor) -> torch.Tensor:
    if frame.dim() == 3:
        return rearrange(frame, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=PATCH_SIZE, p2=PATCH_SIZE)
    return rearrange(frame, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=PATCH_SIZE, p2=PATCH_SIZE)


# Reassemble predicted patch tokens back into RGB frames.
def patches_to_frame(patches: torch.Tensor) -> torch.Tensor:
    ph = pw = FRAME_SIZE // PATCH_SIZE
    if patches.dim() == 2:
        return rearrange(
            patches,
            '(h w) (p1 p2 c) -> c (h p1) (w p2)',
            h=ph,
            w=pw,
            p1=PATCH_SIZE,
            p2=PATCH_SIZE,
            c=3,
        )
    return rearrange(
        patches,
        'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
        h=ph,
        w=pw,
        p1=PATCH_SIZE,
        p2=PATCH_SIZE,
        c=3,
    )


class SpatialPosEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        ph = pw = FRAME_SIZE // PATCH_SIZE
        self.row_embed = nn.Embedding(ph, d_model // 2)
        self.col_embed = nn.Embedding(pw, d_model // 2)

    def forward(self, device: torch.device) -> torch.Tensor:
        # Separate row and column embeddings preserve 2D patch layout.
        ph = pw = FRAME_SIZE // PATCH_SIZE
        rows = torch.arange(ph, device=device)
        cols = torch.arange(pw, device=device)
        row_emb = self.row_embed(rows).unsqueeze(1).expand(ph, pw, -1)
        col_emb = self.col_embed(cols).unsqueeze(0).expand(ph, pw, -1)
        return torch.cat([row_emb, col_emb], dim=-1).view(NUM_PATCHES, -1)


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
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_frames: int = 64,
        num_row_labels: int = 21,
    ):
        super().__init__()
        # This BOS-token architecture matches best_model_old.pt.
        self.patch_proj = nn.Linear(PATCH_DIM, d_model)
        self.spatial_pos = SpatialPosEncoding(d_model)
        self.temporal_pos = TemporalPosEncoding(d_model, max_len=max_frames)
        self.label_embed = nn.Embedding(num_row_labels, d_model)
        self.bos_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.out_proj = nn.Linear(d_model, PATCH_DIM)
        self.skip_proj = nn.Linear(d_model, d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        # Xavier matches the checkpoint-era transformer initialization.
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def encode_frame(self, frame: torch.Tensor, frame_idx: int) -> torch.Tensor:
        # Frame tokens combine patch color, spatial position, and time.
        tokens = self.patch_proj(frame_to_patches(frame))
        tokens = tokens + self.spatial_pos(frame.device).unsqueeze(0)
        tokens = tokens + self.temporal_pos(frame_idx).unsqueeze(0).unsqueeze(0)
        return tokens

    def _apply_skip(self, dec_out: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch_size = dec_out.size(0)
        # Skip encoded input patches to preserve sprite identity.
        encoded_frame = self.skip_proj(memory[:, 1:, :])
        target_len = dec_out.size(1)
        full_frames = target_len // NUM_PATCHES
        remainder = target_len % NUM_PATCHES
        if full_frames > 0:
            skip = encoded_frame.unsqueeze(2).expand(batch_size, NUM_PATCHES, full_frames, -1)
            skip = skip.permute(0, 2, 1, 3).contiguous().view(batch_size, full_frames * NUM_PATCHES, -1)
            if remainder > 0:
                skip = torch.cat([skip, encoded_frame[:, :remainder, :]], dim=1)
        else:
            skip = encoded_frame[:, :remainder, :]
        return dec_out + skip

    def _encode_context(self, frame0: torch.Tensor, row_labels: torch.Tensor) -> torch.Tensor:
        # Row label tells the model which animation strip to generate.
        label_token = self.label_embed(row_labels).unsqueeze(1)
        enc_tokens = torch.cat([label_token, self.encode_frame(frame0, 0)], dim=1)
        return torch.utils.checkpoint.checkpoint(self.encoder, enc_tokens, use_reentrant=False)

    def forward(self, frame0: torch.Tensor, target_frames: torch.Tensor, row_labels: torch.Tensor, lengths=None) -> torch.Tensor:
        batch_size, num_frames = frame0.size(0), target_frames.size(1)
        memory = self._encode_context(frame0, row_labels)
        # Teacher forcing feeds shifted ground-truth target patches.
        all_tokens = torch.cat([self.encode_frame(target_frames[:, t], t + 1) for t in range(num_frames)], dim=1)
        bos = self.bos_token.expand(batch_size, -1, -1)
        dec_input = torch.cat([bos, all_tokens[:, :-1, :]], dim=1)
        # Causal masking prevents target patches from seeing future patches.
        target_mask = nn.Transformer.generate_square_subsequent_mask(dec_input.size(1), device=frame0.device).bool()
        dec_out = torch.utils.checkpoint.checkpoint(
            self.decoder,
            dec_input,
            memory,
            target_mask,
            None,
            None,
            None,
            use_reentrant=False,
        )
        return torch.sigmoid(self.out_proj(self._apply_skip(dec_out, memory)))

    def forward_train(self, frame0: torch.Tensor, target_frames: torch.Tensor, row_labels: torch.Tensor, lengths=None) -> torch.Tensor:
        return self.forward(frame0, target_frames, row_labels, lengths)

    @torch.no_grad()
    def generate(self, frame0: torch.Tensor, row_label: torch.Tensor, num_frames: int) -> torch.Tensor:
        # Generation reuses frame0 as dummy input for the trained decoder path.
        dummy_targets = frame0.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        pred = self.forward(frame0, dummy_targets, row_label)
        batch_size, patch_tokens, _ = pred.shape
        frames = patch_tokens // NUM_PATCHES
        patches = pred.view(batch_size, frames, NUM_PATCHES, PATCH_DIM)
        return torch.stack([patches_to_frame(patches[0, i]) for i in range(frames)])


class SpriteAnimDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples: list[tuple[Path, int, int]] = []
        self._scan()

    @staticmethod
    def _png_size(path: Path) -> tuple[int, int] | None:
        # PNG headers are enough to validate row-strip dimensions.
        try:
            with open(path, 'rb') as f:
                header = f.read(24)
            if len(header) < 24 or header[:8] != b'\x89PNG\r\n\x1a\n':
                return None
            return struct.unpack('>I', header[16:20])[0], struct.unpack('>I', header[20:24])[0]
        except OSError:
            return None

    def _scan(self) -> None:
        from multiprocessing.pool import ThreadPool

        pattern = re.compile(r'entry_\d+_(?:male|female)_row(\d+)\.png$')
        candidates = []
        for png in sorted(self.data_dir.glob('*.png')):
            match = pattern.match(png.name)
            if match is not None:
                candidates.append((png, int(match.group(1))))

        def check_sample(args: tuple[Path, int]) -> tuple[Path, int, int] | None:
            png, row_label = args
            dims = self._png_size(png)
            if dims is None:
                return None
            width, height = dims
            num_frames = width // FRAME_SIZE
            if height == FRAME_SIZE and num_frames >= 2:
                return png, row_label, num_frames
            return None

        # Header scanning avoids opening every sprite during startup.
        n_threads = min(32, os.cpu_count() or 4)
        with ThreadPool(n_threads) as pool:
            results = list(
                tqdm(
                    pool.imap(check_sample, candidates, chunksize=256),
                    total=len(candidates),
                    desc='scanning dataset',
                    unit='file',
                    ascii=True,
                    dynamic_ncols=True,
                )
            )
        self.samples = sorted([sample for sample in results if sample is not None], key=lambda x: str(x[0]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        import torchvision.transforms.functional as TF

        png_path, row_label, num_frames = self.samples[idx]
        img_tensor = TF.to_tensor(Image.open(png_path).convert('RGB'))
        # Frame zero is the conditioning image; later frames are targets.
        frames = [img_tensor[:, :, i * FRAME_SIZE:(i + 1) * FRAME_SIZE] for i in range(num_frames)]
        return frames[0], torch.stack(frames[1:], dim=0), row_label


class FrameCountBucketSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        from collections import defaultdict

        if hasattr(dataset, 'indices'):
            # Subset indices must map back to DataLoader-local indices.
            actual_dataset = dataset.dataset
            abs_to_local = {abs_idx: local_idx for local_idx, abs_idx in enumerate(dataset.indices)}
        else:
            actual_dataset = dataset
            abs_to_local = None

        buckets = defaultdict(list)
        for abs_idx, (_, _, num_frames) in enumerate(actual_dataset.samples):
            local_idx = abs_to_local.get(abs_idx) if abs_to_local is not None else abs_idx
            if local_idx is not None:
                buckets[num_frames].append(local_idx)
        self._buckets = dict(buckets)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        import random as random_module

        # Epoch-seeded shuffling stays deterministic across DDP workers.
        rng = random_module.Random(self.epoch)
        batches = []
        for indices in self._buckets.values():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches[self.rank::self.world_size]

    def __len__(self) -> int:
        total = 0
        for indices in self._buckets.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return max(0, (total + self.world_size - 1 - self.rank) // self.world_size)


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, int]], max_frames=None):
    frame0_list, target_list, labels = zip(*batch)
    if max_frames is not None:
        # Curriculum training starts with shorter target sequences.
        target_list = [target[:max_frames] for target in target_list]
    lengths = [target.size(0) for target in target_list]
    max_len = max(lengths)
    # Padding lets variable-length animation rows share one batch tensor.
    padded = torch.zeros(len(batch), max_len, 3, FRAME_SIZE, FRAME_SIZE)
    for i, target in enumerate(target_list):
        padded[i, :target.size(0)] = target
    return torch.stack(frame0_list), padded, torch.tensor(labels, dtype=torch.long), torch.tensor(lengths)


def get_curriculum_max_frames(epoch: int, total_epochs: int, min_frames: int = 2, max_frames: int | None = None):
    # Sequence length ramps up over the first half of training.
    if max_frames is None:
        return None
    halfway = max(total_epochs // 2, 1)
    if epoch >= halfway:
        return max_frames
    progress = epoch / max(halfway - 1, 1)
    return max(min_frames, int(min_frames + progress * (max_frames - min_frames)))


class PerceptualLoss(nn.Module):
    def __init__(self, vgg: nn.Module):
        super().__init__()
        self.vgg = vgg
        # ImageNet normalization is required for VGG feature loss.
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_features = self.vgg(self._normalize(target))
        pred_features = self.vgg(self._normalize(pred))
        return F.l1_loss(pred_features, target_features)


def _gaussian_kernel(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    channels = pred.size(1)
    # Gaussian SSIM emphasizes sprite structure and edge alignment.
    kernel_1d = _gaussian_kernel(window_size, 1.5, pred.device, pred.dtype)
    kernel = (kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(channels, 1, window_size, window_size)
    padding = window_size // 2

    def blur(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=padding, groups=channels)

    mu_pred, mu_target = blur(pred), blur(target)
    mu_pred2, mu_target2, mu_cross = mu_pred * mu_pred, mu_target * mu_target, mu_pred * mu_target
    sigma_pred2 = blur(pred * pred) - mu_pred2
    sigma_target2 = blur(target * target) - mu_target2
    sigma_cross = blur(pred * target) - mu_cross
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_cross + c1) * (2 * sigma_cross + c2))
    ssim_map = ssim_map / ((mu_pred2 + mu_target2 + c1) * (sigma_pred2 + sigma_target2 + c2))
    return 1.0 - ssim_map.mean()


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    perceptual_loss_fn: PerceptualLoss | None = None,
) -> torch.Tensor:
    batch_size, max_len, channels, height, width = target.shape
    pred_frames = patches_to_frame(pred.view(batch_size * max_len, NUM_PATCHES, PATCH_DIM))
    target_flat = target.view(batch_size * max_len, channels, height, width)

    mask = torch.zeros(batch_size, max_len, device=pred.device)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1.0
    # Padding frames are ignored in reconstruction losses.
    valid = mask.view(batch_size * max_len).bool()
    if valid.sum() == 0:
        return pred.sum() * 0.0

    pred_valid = pred_frames[valid]
    target_valid = target_flat[valid]
    perceptual = torch.tensor(0.0, device=pred.device)
    if perceptual_loss_fn is not None:
        perceptual = perceptual_loss_fn(pred_valid, target_valid)
    # L1 anchors color while perceptual and SSIM sharpen structure.
    return 0.1 * F.l1_loss(pred_valid, target_valid) + 0.1 * perceptual + 0.5 * ssim_loss(pred_valid, target_valid)


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Patch logits reward local pixel-art sharpness.
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


GAN_WARMUP_EPOCHS = 5
D_NOISE_STD = 0.05
D_REAL_LABEL = 0.9
ADV_LOSS_WEIGHT = 0.001


# Prefer bf16 on supported CUDA hardware for stability.
def get_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def train_one_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    discriminator=None,
    optimizer_D=None,
    perceptual_loss_fn=None,
    epoch: int = 0,
    scaler: GradScaler | None = None,
    grad_accum: int = 1,
):
    model.train()
    if discriminator is not None:
        discriminator.train()

    total_g = 0.0
    total_d = 0.0
    bce = nn.BCEWithLogitsLoss()
    # The discriminator starts after reconstruction has stabilized.
    gan_active = discriminator is not None and optimizer_D is not None and epoch >= GAN_WARMUP_EPOCHS
    use_amp = scaler is not None and scaler.is_enabled()
    amp_dtype = get_amp_dtype(device)

    optimizer.zero_grad(set_to_none=True)
    batch_bar = tqdm(loader, desc='  batches', leave=False, unit='bat', ascii=True, dynamic_ncols=True)
    for step, (frame0, target, row_labels, lengths) in enumerate(batch_bar):
        frame0 = frame0.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        row_labels = row_labels.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        batch_size, max_len, channels, height, width = target.shape

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(frame0, target, row_labels, lengths)
            pred_frames = patches_to_frame(pred.view(batch_size * max_len, NUM_PATCHES, PATCH_DIM))
            target_flat = target.view(batch_size * max_len, channels, height, width)

        mask = torch.zeros(batch_size, max_len, device=device)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1.0
        valid = mask.view(batch_size * max_len).bool()

        if gan_active:
            real = target_flat[valid]
            fake = pred_frames[valid].detach().float()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                # Noisy labels make the small patch discriminator less brittle.
                real_pred = discriminator(real + torch.randn_like(real) * D_NOISE_STD)
                fake_pred = discriminator(fake + torch.randn_like(fake) * D_NOISE_STD)
                # Real labels are smoothed to reduce discriminator overconfidence.
                loss_d = 0.5 * (
                    bce(real_pred, torch.full_like(real_pred, D_REAL_LABEL))
                    + bce(fake_pred, torch.zeros_like(fake_pred))
                )
            optimizer_D.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss_d).backward()
                scaler.unscale_(optimizer_D)
                nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                scaler.step(optimizer_D)
                scaler.update()
            else:
                loss_d.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                optimizer_D.step()
            total_d += loss_d.item()

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            recon = compute_loss(pred, target, lengths, perceptual_loss_fn)
            if gan_active:
                adv_out = discriminator(pred_frames[valid])
                # The adversarial term is intentionally tiny.
                loss_g = recon + ADV_LOSS_WEIGHT * bce(adv_out, torch.ones_like(adv_out))
            else:
                loss_g = recon
            loss_g = loss_g / grad_accum

        if use_amp:
            scaler.scale(loss_g).backward()
        else:
            loss_g.backward()

        is_last_step = (step + 1) == len(loader)
        if (step + 1) % grad_accum == 0 or is_last_step:
            if use_amp:
                scaler.unscale_(optimizer)
                # Gradient clipping protects long sequence steps from spikes.
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_g += loss_g.item() * grad_accum
        running_g = total_g / (step + 1)
        running_d = total_d / (step + 1) if gan_active else 0.0
        batch_bar.set_postfix(G=f'{running_g:.4f}', D=f'{running_d:.4f}')

    batch_bar.close()
    return total_g / len(loader), total_d / len(loader)


@torch.no_grad()
def eval_one_epoch(model, loader, device: torch.device, perceptual_loss_fn=None, use_amp: bool = False) -> float:
    model.eval()
    amp_dtype = get_amp_dtype(device)
    total = 0.0
    val_bar = tqdm(loader, desc='  val', leave=False, unit='bat', ascii=True, dynamic_ncols=True)
    for frame0, target, row_labels, lengths in val_bar:
        frame0 = frame0.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        row_labels = row_labels.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(frame0, target, row_labels, lengths)
            loss = compute_loss(pred, target, lengths, perceptual_loss_fn).item()
        total += loss
        val_bar.set_postfix(loss=f'{loss:.4f}')
    val_bar.close()
    return total / len(loader)


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    path: str,
    discriminator=None,
    optimizer_D=None,
    curriculum_max_frames=None,
    train_losses_G=None,
    train_losses_D=None,
    val_losses=None,
    train_config=None,
) -> None:
    ckpt = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    # Optional fields keep older checkpoints loadable.
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
    if train_config is not None:
        ckpt['train_config'] = train_config
    torch.save(ckpt, path)


def plot_losses(train_losses_G, train_losses_D, val_losses, plot_dir: str) -> None:
    import matplotlib

    # Agg backend writes plots on headless training machines.
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
    tqdm.write(f'loss curve -> {path}')


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def build_perceptual_loss(device: torch.device, enabled: bool):
    # VGG features reward crisp structure beyond raw pixel loss.
    if not enabled or not HAS_TORCHVISION_MODELS:
        return None
    try:
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features[:9]
    except Exception as exc:
        tqdm.write(f'perceptual loss disabled: {exc}')
        return None
    for param in vgg.parameters():
        param.requires_grad_(False)
    return PerceptualLoss(vgg.to(device).eval()).to(device)


def build_train_loader(train_ds, args, curr_max: int, effective_batch: int, ddp: bool, local_rank: int, world_size: int, epoch: int):
    if args.use_bucket_sampler:
        # Bucketing reduces wasted padding for uneven animation lengths.
        sampler = FrameCountBucketSampler(
            train_ds,
            batch_size=effective_batch,
            drop_last=True,
            shuffle=True,
            rank=local_rank if ddp else 0,
            world_size=world_size if ddp else 1,
        )
        sampler.set_epoch(epoch)
        return DataLoader(
            train_ds,
            batch_sampler=sampler,
            collate_fn=partial(collate_fn, max_frames=curr_max),
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=False,
        ), sampler
    if ddp:
        sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        sampler.set_epoch(epoch)
        return DataLoader(
            train_ds,
            batch_size=effective_batch,
            sampler=sampler,
            collate_fn=partial(collate_fn, max_frames=curr_max),
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=False,
        ), sampler
    return DataLoader(
        train_ds,
        batch_size=effective_batch,
        shuffle=True,
        collate_fn=partial(collate_fn, max_frames=curr_max),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    ), None


def build_val_loader(val_ds, args, curr_max: int, ddp: bool):
    if ddp:
        sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
        return DataLoader(
            val_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=partial(collate_fn, max_frames=curr_max),
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
    return DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, max_frames=curr_max),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    # Defaults mirror the best_model_old.pt training run.
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--data_dir', type=str, default='./data_output/frames')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--plot_dir', type=str, default='./plot')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--rows', type=int, nargs='+', default=None)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--max_frames', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--min_frames', type=int, default=2)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--no_perceptual', action='store_true')
    parser.add_argument('--use_bucket_sampler', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    ddp = world_size > 1

    if ddp:
        # Distributed runs assign one CUDA device per local process.
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tqdm.write(f'device: {device}')

    use_amp = not args.no_amp and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    tqdm.write(f'AMP: {use_amp} ({get_amp_dtype(device)}), grad_accum: {args.grad_accum}, num_workers: {args.num_workers}')

    dataset = SpriteAnimDataset(args.data_dir)
    if args.rows is not None:
        row_set = set(args.rows)
        dataset.samples = [sample for sample in dataset.samples if sample[1] in row_set]
    if len(dataset) == 0:
        raise RuntimeError(f'no training samples found in {args.data_dir!r}')
    tqdm.write(f'samples: {len(dataset)}')

    val_size = max(1, int(0.2 * len(dataset)))
    train_ds, val_ds = random_split(
        dataset,
        [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    dataset_max_frames = min(args.max_frames, max(sample[2] - 1 for sample in dataset.samples))

    model = SpriteSeq2Seq(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        max_frames=args.max_frames,
    ).to(device)
    tqdm.write(f'params: {sum(param.numel() for param in model.parameters() if param.requires_grad):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Warm restarts produced the best old checkpoint schedule.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(1, args.epochs // 2),
        T_mult=1,
        eta_min=args.min_lr,
    )
    perceptual_loss_fn = build_perceptual_loss(device, enabled=not args.no_perceptual)
    discriminator = PatchDiscriminator().to(device)
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        # The discriminator is also wrapped so GAN updates stay synchronized.
        discriminator = DDP(discriminator, device_ids=[local_rank])

    raw_model = model.module if ddp else model
    raw_disc = discriminator.module if ddp else discriminator
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    start_epoch = 0
    resume_curriculum_min = args.min_frames
    train_losses_G, train_losses_D, val_losses = [], [], []

    if args.resume and os.path.isfile(args.resume):
        if ddp:
            dist.barrier()
        ckpt = torch.load(args.resume, map_location=device)
        # Checkpoints include model, optimizers, curriculum, and loss curves.
        raw_model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        if ckpt.get('discriminator_state') is not None:
            raw_disc.load_state_dict(ckpt['discriminator_state'])
        if ckpt.get('optimizer_D_state') is not None:
            optimizer_D.load_state_dict(ckpt['optimizer_D_state'])
        resume_curriculum_min = ckpt.get('curriculum_max_frames', args.min_frames)
        train_losses_G = ckpt.get('train_losses_G', [])
        train_losses_D = ckpt.get('train_losses_D', [])
        val_losses = ckpt.get('val_losses', [])
        start_epoch = ckpt['epoch'] + 1
        tqdm.write(f'resumed epoch {ckpt["epoch"]}')

    best_val_loss = min(val_losses) if val_losses else float('inf')
    # Resumed runs continue best-model tracking from saved validation history.
    epoch_bar = tqdm(range(start_epoch, args.epochs), desc='epochs', unit='ep', ascii=True, dynamic_ncols=True)

    for epoch in epoch_bar:
        curr_max = max(
            resume_curriculum_min,
            get_curriculum_max_frames(epoch, args.epochs, min_frames=args.min_frames, max_frames=dataset_max_frames),
        )
        # Batch size shrinks as longer frame sequences enter training.
        effective_batch = max(1, int(args.batch_size * args.min_frames / curr_max))
        grad_accum = max(args.grad_accum, math.ceil(args.batch_size / effective_batch))

        train_loader, _ = build_train_loader(train_ds, args, curr_max, effective_batch, ddp, local_rank, world_size, epoch)
        val_loader = build_val_loader(val_ds, args, curr_max, ddp)

        loss_G, loss_D = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            discriminator=discriminator,
            optimizer_D=optimizer_D,
            perceptual_loss_fn=perceptual_loss_fn,
            epoch=epoch,
            scaler=scaler,
            grad_accum=grad_accum,
        )
        val_loss = eval_one_epoch(model, val_loader, device, perceptual_loss_fn=perceptual_loss_fn, use_amp=use_amp)
        scheduler.step()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        train_losses_G.append(loss_G)
        train_losses_D.append(loss_D)
        val_losses.append(val_loss)

        if is_main_process():
            epoch_bar.set_postfix(G=f'{loss_G:.4f}', D=f'{loss_D:.4f}', val=f'{val_loss:.4f}', cur=curr_max)
            tqdm.write(
                f'epoch {epoch + 1:3d}/{args.epochs}  '
                f'G={loss_G:.5f}  D={loss_D:.5f}  val={val_loss:.5f}  cur={curr_max}'
            )

        if is_main_process() and (epoch + 1) % 10 == 0:
            # Periodic checkpoints make long training runs recoverable.
            save_checkpoint(
                raw_model,
                optimizer,
                epoch,
                os.path.join(args.checkpoint_dir, f'epoch_{epoch + 1}.pt'),
                discriminator=raw_disc,
                optimizer_D=optimizer_D,
                curriculum_max_frames=curr_max,
                train_losses_G=train_losses_G,
                train_losses_D=train_losses_D,
                val_losses=val_losses,
                train_config=vars(args),
            )
            plot_losses(train_losses_G, train_losses_D, val_losses, args.plot_dir)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if is_main_process():
                # best_model.pt is always the lowest validation checkpoint.
                save_checkpoint(
                    raw_model,
                    optimizer,
                    epoch,
                    os.path.join(args.checkpoint_dir, 'best_model.pt'),
                    discriminator=raw_disc,
                    optimizer_D=optimizer_D,
                    curriculum_max_frames=curr_max,
                    train_losses_G=train_losses_G,
                    train_losses_D=train_losses_D,
                    val_losses=val_losses,
                    train_config=vars(args),
                )

    if is_main_process():
        plot_losses(train_losses_G, train_losses_D, val_losses, args.plot_dir)
        tqdm.write(f'done. best val={best_val_loss:.5f}')

    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
