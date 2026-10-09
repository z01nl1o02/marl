"""
Heatmap prior for MARL landmark detection (hybrid scheme: heatmap prior + RL refinement).

A lightweight fully-convolutional network predicts a gaussian heatmap per landmark
from the normalized face crop (same normalization as the RL observation: (px-128)/128).
Decoded heatmap peaks are used to initialize RL agents close to their targets, so the
marl agents only need to search in a small neighborhood instead of the whole face.

Entry points (via main.py Fire):
    python -m marl.src.main train_heatmap --dataset_root D:/dataset/WFLW
"""
import os
import random
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import Dataset, DataLoader

try:  # package-style import (python -m marl.src.main)
    from .main import DatasetWFLW
except ImportError:  # script-style import
    from main import DatasetWFLW


# ---------------------------------------------------------------- model
class HeatmapNet(nn.Module):
    """
    Lightweight encoder-decoder for landmark heatmaps.
    Input : (B, 3, S, S)  normalized face crop
    Output: (B, num_points, S/2, S/2) heatmaps (output stride = 2)
    """
    def __init__(self, num_points: int, width: int = 32):
        super().__init__()
        w = width
        def down(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 4, stride=2, padding=1),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        def up(cin, cout):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.enc1 = down(3, w)          # S   -> S/2
        self.enc2 = down(w, 2 * w)      # S/2 -> S/4
        self.enc3 = down(2 * w, 4 * w)  # S/4 -> S/8
        self.enc4 = down(4 * w, 8 * w)  # S/8 -> S/16
        self.dec1 = up(8 * w, 4 * w)    # S/16 -> S/8 (+ skip enc3)
        self.dec2 = up(4 * w, 2 * w)    # S/8  -> S/4 (+ skip enc2)
        self.dec3 = up(2 * w, w)        # S/4  -> S/2 (+ skip enc1)
        self.head = nn.Conv2d(w, num_points, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        d = self.dec1(x4) + x3
        d = self.dec2(d) + x2
        d = self.dec3(d) + x1
        return self.head(d)


# ---------------------------------------------------------------- target / decode
def make_heatmap_target(xy_norm: np.ndarray, hm_size: int, sigma_px: float, face_size: int) -> np.ndarray:
    """
    xy_norm: (n, 2) landmark coords normalized to [0, 1] (x, y).
    returns: (n, hm_size, hm_size) float32 gaussian heatmaps.
    sigma_px is in input-image pixels; converted to heatmap cells via stride.
    """
    n = xy_norm.shape[0]
    stride = face_size / hm_size
    sigma = max(sigma_px / stride, 0.5)
    targets = np.zeros((n, hm_size, hm_size), dtype=np.float32)
    yy, xx = np.mgrid[0:hm_size, 0:hm_size].astype(np.float32)
    for i, (x, y) in enumerate(xy_norm):
        cx, cy = x * hm_size, y * hm_size
        targets[i] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    return targets


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, peak_weight: float = 50.0) -> torch.Tensor:
    """MSE up-weighted near the gaussian peaks, counteracting dilution by the zero background."""
    w = 1.0 + peak_weight * target
    return ((pred - target) ** 2 * w).mean()


def decode_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """
    heatmaps: (n, H, W) float32.
    returns:  (n, 2) normalized xy, argmax + 3-point quadratic sub-pixel refinement.
    """
    n, H, W = heatmaps.shape
    xy = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        hm = heatmaps[i]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)

        def _subpixel(m1, m0, p1):
            denom = m1 - 2.0 * m0 + p1
            if abs(denom) < 1e-12:
                return 0.0
            d = 0.5 * (m1 - p1) / denom
            return float(np.clip(d, -0.5, 0.5))

        dx = _subpixel(hm[y, max(x - 1, 0)], hm[y, x], hm[y, min(x + 1, W - 1)])
        dy = _subpixel(hm[max(y - 1, 0), x], hm[y, x], hm[min(y + 1, H - 1), x])
        xy[i, 0] = (x + dx + 0.5) / W
        xy[i, 1] = (y + dy + 0.5) / H
    return xy


# ---------------------------------------------------------------- dataset
class HeatmapDataset(Dataset):
    """Wraps DatasetWFLW to produce (image_tensor, xy_norm, heatmap_target)."""
    def __init__(self, ds: DatasetWFLW, hm_size: int, sigma_px: float):
        self.ds = ds
        self.hm_size = hm_size
        self.sigma_px = sigma_px
        self.face_size = ds.face_size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        image, xy, _ = self.ds.get(index)  # image: (H, W, 3) uint8 BGR, xy: (n, 2) normalized
        img = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 128.0 - 1.0
        target = torch.from_numpy(make_heatmap_target(xy, self.hm_size, self.sigma_px, self.face_size))
        return img, torch.from_numpy(xy.astype(np.float32)), target


# ---------------------------------------------------------------- train
def train_heatmap(dataset_root: str, device: str = "cuda:0", face_size: int = 512,
                  epochs: int = 30, batch_size: int = 8, lr: float = 1e-4,
                  width: int = 32, sigma_px: float = 10.0, peak_weight: float = 50.0,
                  num_workers: int = 2,
                  dataset_size_max: int | None = None, eval_subset: int = 500,
                  weights_file: str | None = None, out_dir: str | None = None):
    """Supervised training of the heatmap prior network on WFLW (clean subset)."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if out_dir is None:
        out_dir = os.path.join("runs", f"heatmap_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    logger.add(os.path.join(out_dir, "heatmap.log"))
    logger.info(f"heatmap training | device={device} face_size={face_size} epochs={epochs} bs={batch_size}")

    ds_train = DatasetWFLW(dataset_root, split="train", face_size=face_size, dataset_size_max=dataset_size_max)
    ds_val = DatasetWFLW(dataset_root, split="test", face_size=face_size, dataset_size_max=None)
    num_points = ds_train.get_point_num()
    hm_size = face_size // 2
    logger.info(f"train {len(ds_train)} | val {len(ds_val)} | points {num_points} | hm_size {hm_size}")

    dl_train = DataLoader(HeatmapDataset(ds_train, hm_size, sigma_px), batch_size=batch_size,
                          shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    dl_val = DataLoader(HeatmapDataset(ds_val, hm_size, sigma_px), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    net = HeatmapNet(num_points, width=width).to(torch.device(device))
    if weights_file is not None:
        net.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))
        logger.warning(f"resumed from {weights_file}")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = lambda p, t: weighted_mse_loss(p, t, peak_weight=peak_weight)

    best_err = np.finfo("float").max
    weights_path = os.path.join(out_dir, "heatmap_net_best.pth")

    @torch.no_grad()
    def evaluate(loader, max_batches: int | None = None) -> tuple[float, float]:
        net.eval()
        errs = []
        for bi, (img, xy, _) in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            pred = net(img.to(device)).cpu().numpy()
            for b in range(pred.shape[0]):
                pxy = decode_heatmaps(pred[b])
                errs.append(np.linalg.norm(pxy - xy[b].numpy(), axis=-1))  # normalized units
        errs = np.concatenate(errs)
        return float(np.mean(errs) * face_size), float(np.mean(errs < 0.01))

    for epoch in range(epochs):
        net.train()
        loss_sum, nb = 0.0, 0
        for img, _, target in dl_train:
            img, target = img.to(device, non_blocking=True), target.to(device, non_blocking=True)
            pred = net(img)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            nb += 1
        scheduler.step()

        val_err_px, val_acc = evaluate(dl_val, max_batches=max(1, eval_subset // batch_size))
        logger.info(f"epoch {epoch:0>3d} | train loss {loss_sum / max(nb, 1):.5f} | "
                    f"val err {val_err_px:.2f}px | val acc(<5px) {val_acc:.3f}")
        if val_err_px < best_err:
            best_err = val_err_px
            torch.save(net.state_dict(), weights_path)
            logger.info(f"NEW BEST val err {best_err:.2f}px -> {weights_path}")

    logger.info(f"training done, best val err {best_err:.2f}px")
    return weights_path


# ---------------------------------------------------------------- inference provider
def load_heatmap_provider(weights_file: str, num_points: int, face_size: int = 512,
                          device: str = "cuda:0", width: int = 32):
    """
    Returns a callable: image_bgr_uint8 (H, W, 3) -> xy (num_points, 2) normalized.
    Used by MARLEnv to initialize agents at heatmap peaks (RL then refines locally).
    """
    dev = torch.device(device)
    net = HeatmapNet(num_points, width=width).to(dev)
    net.load_state_dict(torch.load(weights_file, map_location=dev, weights_only=True))
    net.eval()
    hm_size = face_size // 2

    @torch.no_grad()
    def provider(image_bgr: np.ndarray) -> np.ndarray:
        image_bgr = cv2.resize(image_bgr, (face_size, face_size), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(np.ascontiguousarray(image_bgr.transpose(2, 0, 1))).float() / 128.0 - 1.0
        img = img.unsqueeze(0).to(dev)
        heatmaps = net(img)[0].cpu().numpy()
        return decode_heatmaps(heatmaps)

    return provider
