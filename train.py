import argparse
import math
import os
import re
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from einops import rearrange

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
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_frames: int = 64,
        num_row_labels: int = 21,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_proj = nn.Linear(PATCH_DIM, d_model)
        self.spatial_pos = SpatialPosEncoding(d_model)
        self.temporal_pos = TemporalPosEncoding(d_model, max_len=max_frames)
        self.label_embed = nn.Embedding(num_row_labels, d_model)
        self.bos_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.out_proj = nn.Linear(d_model, PATCH_DIM)
        self.skip_proj = nn.Linear(d_model, d_model)
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

    def _apply_skip(self, dec_out: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        B = dec_out.size(0)
        enc_proj = self.skip_proj(memory[:, 1:, :])
        tgt_len = dec_out.size(1)
        nf = tgt_len // NUM_PATCHES
        rem = tgt_len % NUM_PATCHES
        if nf > 0:
            tiled = enc_proj.unsqueeze(2).expand(B, NUM_PATCHES, nf, -1)
            tiled = tiled.permute(0, 2, 1, 3).contiguous().view(B, nf * NUM_PATCHES, -1)
            skip = torch.cat([tiled, enc_proj[:, :rem, :]], dim=1) if rem > 0 else tiled
        else:
            skip = enc_proj[:, :rem, :]
        return dec_out + skip

    def forward(self, frame0, target_frames, row_labels):
        B, T = frame0.size(0), target_frames.size(1)
        enc_tokens = torch.cat([self.label_embed(row_labels).unsqueeze(1),
                                 self.encode_frame(frame0, 0)], dim=1)
        memory = self.encoder(enc_tokens)
        bos = self.bos_token.expand(B, -1, -1)
        if T > 1:
            dec_input = torch.cat([bos] + [self.encode_frame(target_frames[:, t], t + 1)
                                           for t in range(T - 1)], dim=1)
        else:
            dec_input = bos
        tgt_len = dec_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=frame0.device)
        dec_out = self.decoder(dec_input, memory, tgt_mask=tgt_mask)
        return self.out_proj(self._apply_skip(dec_out, memory))

    def forward_train(self, frame0, target_frames, row_labels):
        B, T, C, H, W = target_frames.shape
        enc_tokens = torch.cat([self.label_embed(row_labels).unsqueeze(1),
                                 self.encode_frame(frame0, 0)], dim=1)
        memory = self.encoder(enc_tokens)
        all_tokens = torch.cat([self.encode_frame(target_frames[:, t], t + 1)
                                 for t in range(T)], dim=1)
        dec_input = torch.cat([self.bos_token.expand(B, -1, -1), all_tokens[:, :-1, :]], dim=1)
        tgt_len = dec_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=frame0.device)
        dec_out = self.decoder(dec_input, memory, tgt_mask=tgt_mask)
        return torch.sigmoid(self.out_proj(self._apply_skip(dec_out, memory)))

    @torch.no_grad()
    def generate(self, frame0, row_label, num_frames):
        device = frame0.device
        enc_tokens = torch.cat([self.label_embed(row_label).unsqueeze(1),
                                 self.encode_frame(frame0, 0)], dim=1)
        memory = self.encoder(enc_tokens)
        enc_proj = self.skip_proj(memory[:, 1:, :])
        dec_input = self.bos_token.clone()
        generated = []

        for frame_idx in range(1, num_frames + 1):
            for patch_idx in range(NUM_PATCHES):
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    dec_input.size(1), device=device)
                dec_out = self.decoder(dec_input, memory, tgt_mask=tgt_mask)
                skip_tok = enc_proj[:, patch_idx % NUM_PATCHES, :].unsqueeze(1)
                pred_patch = torch.sigmoid(self.out_proj(dec_out[:, -1:, :] + skip_tok))
                generated.append(pred_patch.squeeze(1))
                next_tok = self.patch_proj(pred_patch)
                next_tok = next_tok + self.spatial_pos(device)[patch_idx].unsqueeze(0).unsqueeze(0)
                next_tok = next_tok + self.temporal_pos(frame_idx).unsqueeze(0).unsqueeze(0)
                dec_input = torch.cat([dec_input, next_tok], dim=1)

        patches = torch.stack(generated, dim=1).view(1, num_frames, NUM_PATCHES, PATCH_DIM)
        return torch.stack([patches_to_frame(patches[0, i]) for i in range(num_frames)])


class SpriteAnimDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples = []
        self._scan()

    def _scan(self):
        pattern = re.compile(r'entry_\d+_(?:male|female)_row(\d+)\.png$')
        for png in sorted(self.data_dir.glob('*.png')):
            m = pattern.match(png.name)
            if m is None:
                continue
            row_label = int(m.group(1))
            try:
                img = Image.open(png).convert('RGB')
            except Exception:
                continue
            w, h = img.size
            if h != FRAME_SIZE:
                continue
            num_frames = w // FRAME_SIZE
            if num_frames >= 2:
                self.samples.append((png, row_label, num_frames))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        png_path, row_label, num_frames = self.samples[idx]
        import torchvision.transforms.functional as TF
        img_tensor = TF.to_tensor(Image.open(png_path).convert('RGB'))
        frames = [img_tensor[:, :, i * FRAME_SIZE:(i + 1) * FRAME_SIZE] for i in range(num_frames)]
        return frames[0], torch.stack(frames[1:], dim=0), row_label


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


def compute_loss(pred, target, lengths, perceptual_loss_fn=None):
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
    perc = perceptual_loss_fn(pv, tv) if perceptual_loss_fn is not None else torch.tensor(0.0, device=pred.device)
    return 0.1 * F.l1_loss(pv, tv) + 1.0 * perc + 0.5 * ssim_loss(pv, tv)


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
                    perceptual_loss_fn=None, epoch: int = 0):
    model.train()
    if discriminator is not None:
        discriminator.train()
    total_G = total_D = 0.0
    bce = nn.BCEWithLogitsLoss()
    gan_active = discriminator is not None and optimizer_D is not None and epoch >= GAN_WARMUP_EPOCHS

    for frame0, target, row_labels, lengths in loader:
        frame0, target = frame0.to(device), target.to(device)
        row_labels, lengths = row_labels.to(device), lengths.to(device)
        B, T, C, H, W = target.shape

        pred = model.forward_train(frame0, target, row_labels)
        pred_frames = patches_to_frame(pred.view(B * T, NUM_PATCHES, PATCH_DIM))
        target_flat = target.view(B * T, C, H, W)

        mask = torch.zeros(B, T, device=device)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1.0
        valid = mask.view(B * T).bool()

        if gan_active:
            real = target_flat[valid]
            fake = pred_frames[valid].detach()
            r_pred = discriminator(real + torch.randn_like(real) * D_NOISE_STD)
            f_pred = discriminator(fake + torch.randn_like(fake) * D_NOISE_STD)
            loss_D = 0.5 * (bce(r_pred, torch.full_like(r_pred, D_REAL_LABEL)) +
                            bce(f_pred, torch.zeros_like(f_pred)))
            optimizer_D.zero_grad()
            loss_D.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            optimizer_D.step()
            total_D += loss_D.item()

        recon = compute_loss(pred, target, lengths, perceptual_loss_fn)
        if gan_active:
            adv_out = discriminator(pred_frames[valid])
            loss_G = recon + ADV_LOSS_WEIGHT * bce(adv_out, torch.ones_like(adv_out))
        else:
            loss_G = recon

        optimizer.zero_grad()
        loss_G.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_G += loss_G.item()

    n = len(loader)
    return total_G / n, total_D / n


@torch.no_grad()
def eval_one_epoch(model, loader, device, perceptual_loss_fn=None):
    model.eval()
    total = 0.0
    for frame0, target, row_labels, lengths in loader:
        frame0, target = frame0.to(device), target.to(device)
        row_labels, lengths = row_labels.to(device), lengths.to(device)
        pred = model.forward_train(frame0, target, row_labels)
        total += compute_loss(pred, target, lengths, perceptual_loss_fn).item()
    return total / len(loader)


def save_checkpoint(model, optimizer, epoch, path,
                    discriminator=None, optimizer_D=None, curriculum_max_frames=None):
    ckpt = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
    if discriminator is not None:
        ckpt['discriminator_state'] = discriminator.state_dict()
    if optimizer_D is not None:
        ckpt['optimizer_D_state'] = optimizer_D.state_dict()
    if curriculum_max_frames is not None:
        ckpt['curriculum_max_frames'] = curriculum_max_frames
    torch.save(ckpt, path)


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_dir', type=str, default='./data_output/frames')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--d_model', type=int, default=512)
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
    dataset_max_frames = max(s[2] - 1 for s in dataset.samples)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    model = SpriteSeq2Seq(d_model=args.d_model).to(device)
    print(f'params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    perceptual_loss_fn = None
    if HAS_TORCHVISION_MODELS:
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features[:9]
        for p in vgg.parameters():
            p.requires_grad_(False)
        perceptual_loss_fn = PerceptualLoss(vgg.to(device).eval()).to(device)

    discriminator = PatchDiscriminator().to(device)
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    start_epoch = 0
    best_val_loss = float('inf')
    resume_curriculum_min = 2

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        disc_state = ckpt.get('discriminator_state')
        opt_d_state = ckpt.get('optimizer_D_state')
        if disc_state:
            discriminator.load_state_dict(disc_state)
        if opt_d_state:
            optimizer_D.load_state_dict(opt_d_state)
        resume_curriculum_min = ckpt.get('curriculum_max_frames', 2)
        start_epoch = ckpt['epoch'] + 1
        print(f'resumed epoch {ckpt["epoch"]}')

    for epoch in range(start_epoch, args.epochs):
        curr_max = max(resume_curriculum_min,
                       get_curriculum_max_frames(epoch, args.epochs, max_frames=dataset_max_frames))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=partial(collate_fn, max_frames=curr_max), num_workers=0)

        loss_G, loss_D = train_one_epoch(model, train_loader, optimizer, device,
                                         discriminator=discriminator, optimizer_D=optimizer_D,
                                         perceptual_loss_fn=perceptual_loss_fn, epoch=epoch)
        val_loss = eval_one_epoch(model, val_loader, device, perceptual_loss_fn=perceptual_loss_fn)
        scheduler.step()

        print(f'epoch {epoch+1:3d}/{args.epochs}  G={loss_G:.5f}  D={loss_D:.5f}  val={val_loss:.5f}  cur={curr_max}')

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch,
                            os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pt'),
                            discriminator=discriminator, optimizer_D=optimizer_D,
                            curriculum_max_frames=curr_max)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch,
                            os.path.join(args.checkpoint_dir, 'best_model.pt'),
                            discriminator=discriminator, optimizer_D=optimizer_D,
                            curriculum_max_frames=curr_max)

    print(f'done. best val={best_val_loss:.5f}')


if __name__ == '__main__':
    main()
