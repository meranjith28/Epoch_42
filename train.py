"""
train.py - standalone, reproducible training entry point.

Reproduces the submitted checkpoint from scratch: same seed, same dataset split logic,
same architecture (imported from model.py, the same file inference.py uses - so training
and inference can never see different architectures), same loss, same optimizer/schedule.

Usage:
    python train.py --data_root /path/to/dataset --output_dir /path/to/save/checkpoint

--data_root should contain (at any depth) a "NoisyLR" folder and a "GT" folder of paired
.npy files with matching filenames. If omitted, falls back to auto-detecting them under
/kaggle/input or /kaggle/working (matching the original Kaggle notebook's behaviour), so
this also runs unmodified inside the Kaggle notebook environment.
"""
import os
import sys
import glob
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# model.py may sit either next to this script, or inside a models/ subfolder
# (the required submission layout is team_name/models/) - support both, matching run.py.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_models_dir = os.path.join(_script_dir, "models")
if os.path.isdir(_models_dir):
    sys.path.insert(0, _models_dir)
else:
    sys.path.insert(0, _script_dir)

from model import HighResSemiconductorNet


class SemiconductorAugmentedDataset(Dataset):
    def __init__(self, noisy_files, gt_files, is_train=True):
        self.noisy_files = noisy_files
        self.gt_files = gt_files
        self.is_train = is_train

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_arr = np.load(self.noisy_files[idx]).astype(np.float32)
        gt_arr = np.load(self.gt_files[idx]).astype(np.float32)

        if gt_arr.max() > 1.0:
            gt_arr /= 255.0
            noisy_arr /= 255.0

        if noisy_arr.ndim == 2:
            noisy_arr = np.expand_dims(noisy_arr, axis=0)
        if gt_arr.ndim == 2:
            gt_arr = np.expand_dims(gt_arr, axis=0)

        if self.is_train:
            if np.random.rand() > 0.5:
                noisy_arr = np.flip(noisy_arr, axis=2).copy()
                gt_arr = np.flip(gt_arr, axis=2).copy()
            if np.random.rand() > 0.5:
                noisy_arr = np.flip(noisy_arr, axis=1).copy()
                gt_arr = np.flip(gt_arr, axis=1).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                noisy_arr = np.rot90(noisy_arr, k, axes=(1, 2)).copy()
                gt_arr = np.rot90(gt_arr, k, axes=(1, 2)).copy()

        return torch.from_numpy(noisy_arr), torch.from_numpy(gt_arr), os.path.basename(self.noisy_files[idx])


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + (self.eps ** 2)))


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, img1, img2):
        mu1 = F.avg_pool2d(img1, self.window_size, 1, self.window_size // 2)
        mu2 = F.avg_pool2d(img2, self.window_size, 1, self.window_size // 2)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.avg_pool2d(img1 * img1, self.window_size, 1, self.window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, self.window_size, 1, self.window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, self.window_size, 1, self.window_size // 2) - mu1_mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, pred, target):
        gx_p = F.conv2d(pred, self.kx, padding=1)
        gy_p = F.conv2d(pred, self.ky, padding=1)
        gx_t = F.conv2d(target, self.kx, padding=1)
        gy_t = F.conv2d(target, self.ky, padding=1)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


class HighFidelityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.gradient = GradientLoss()

    def forward(self, pred, target):
        return (self.charbonnier(pred, target)
                + 1.0 * self.ssim(pred, target)
                + 0.1 * self.gradient(pred, target))


def locate_dataset(data_root):
    if data_root:
        noisy_dirs = glob.glob(os.path.join(data_root, '**', 'NoisyLR'), recursive=True)
        gt_dirs = glob.glob(os.path.join(data_root, '**', 'GT'), recursive=True)
        if noisy_dirs and gt_dirs:
            return noisy_dirs[0], gt_dirs[0]

    noisy_dirs = glob.glob('/kaggle/input/**/NoisyLR', recursive=True)
    gt_dirs = glob.glob('/kaggle/input/**/GT', recursive=True)
    if not noisy_dirs:
        noisy_dirs = glob.glob('/kaggle/working/**/NoisyLR', recursive=True)
        gt_dirs = glob.glob('/kaggle/working/**/GT', recursive=True)
    assert noisy_dirs and gt_dirs, (
        "Could not locate 'NoisyLR' or 'GT' folders. Pass --data_root pointing at a "
        "directory that contains them (at any depth)."
    )
    return noisy_dirs[0], gt_dirs[0]


def main():
    p = argparse.ArgumentParser(description="Train the KLA restoration model from scratch.")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)

    noisy_dir, gt_dir = locate_dataset(args.data_root)
    print(f"NoisyLR: {noisy_dir}")
    print(f"GT:      {gt_dir}")

    all_noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.npy")))
    all_gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
    assert len(all_noisy_files) == len(all_gt_files), \
        f"Mismatch: {len(all_noisy_files)} noisy files vs {len(all_gt_files)} GT files."

    sample_noisy = np.load(all_noisy_files[0])
    sample_gt = np.load(all_gt_files[0])
    noisy_hw = sample_noisy.shape[-2:] if sample_noisy.ndim >= 2 else sample_noisy.shape
    gt_hw = sample_gt.shape[-2:] if sample_gt.ndim >= 2 else sample_gt.shape
    raw_scale = gt_hw[0] / noisy_hw[0]
    scale_factor = max(1, round(raw_scale))
    print(f"Detected scale factor: {scale_factor}x (raw ratio {raw_scale:.3f})")

    total_count = len(all_noisy_files)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(total_count)
    train_count = int(0.9 * total_count)
    train_idx, val_idx = perm[:train_count], perm[train_count:]

    train_noisy = [all_noisy_files[i] for i in train_idx]
    train_gt = [all_gt_files[i] for i in train_idx]
    val_noisy = [all_noisy_files[i] for i in val_idx]
    val_gt = [all_gt_files[i] for i in val_idx]

    train_dataset = SemiconductorAugmentedDataset(train_noisy, train_gt, is_train=True)
    val_dataset = SemiconductorAugmentedDataset(val_noisy, val_gt, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    if device.type != 'cuda':
        print("WARNING: no GPU detected - training will be slow.")

    model = HighResSemiconductorNet(scale_factor=scale_factor).to(device)
    criterion = HighFidelityLoss().to(device)
    ssim_metric = SSIMLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_balanced_score = -1e9
    history = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for noisy, gt, _ in train_loader:
            noisy, gt = noisy.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            target_hw = gt.shape[-2:]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                output = model(noisy, target_hw=target_hw)
            loss = criterion(output.float(), gt.float())

            if not torch.isfinite(loss):
                print(f" Skipping batch at epoch {epoch+1}: non-finite loss detected.")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        val_psnr, val_ssim = 0.0, 0.0
        with torch.no_grad():
            for noisy, gt, _ in val_loader:
                noisy, gt = noisy.to(device), gt.to(device)
                output = torch.clamp(model(noisy, target_hw=gt.shape[-2:]), 0.0, 1.0)
                mse = torch.mean((output - gt) ** 2)
                psnr_val = 20 * torch.log10(1.0 / torch.sqrt(mse)).item() if mse > 0 else 100
                val_psnr += psnr_val
                val_ssim += (1.0 - ssim_metric(output, gt)).item()

        avg_psnr = val_psnr / len(val_loader)
        avg_ssim = val_ssim / len(val_loader)
        balanced_score = avg_psnr + (100 * avg_ssim)
        history.append({"epoch": epoch + 1, "train_loss": train_loss / len(train_loader),
                         "val_psnr": avg_psnr, "val_ssim": avg_ssim, "balanced_score": balanced_score})

        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val PSNR: {avg_psnr:.2f} dB | Val SSIM: {avg_ssim:.4f} | Balanced: {balanced_score:.2f}")

        if balanced_score > best_balanced_score:
            best_balanced_score = balanced_score
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f" --> New best model saved (PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f})")

    config = {
        "scale_factor": scale_factor,
        "gt_height": int(gt_hw[0]),
        "gt_width": int(gt_hw[1]),
        "seed": args.seed,
        "model_selection_metric": "balanced_score = val_PSNR + 100 * val_SSIM",
        "best_balanced_score": best_balanced_score,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. best_model.pth, config.json and training_history.json written to {args.output_dir}")


if __name__ == "__main__":
    main()
