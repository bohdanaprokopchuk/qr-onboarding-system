from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import qrcode
import torch

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .binarization import proposed_integral_threshold
from .preprocessing import dynamic_illumination_equalization, suppress_screen_artifacts, unsharp_masking


def _conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class TinyUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=16):
        super().__init__()
        self.enc1 = _conv(in_channels, base)
        self.enc2 = _conv(base, base * 2)
        self.enc3 = _conv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = _conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = _conv(base * 2, base)
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


class QRResearchEnhancer(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.backbone = TinyUNet(1, base, base)
        self.shared = nn.Sequential(nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True))
        self.seg_head = nn.Conv2d(base, 1, 1)
        self.deblur_head = nn.Conv2d(base, 1, 1)
        self.sr_head = nn.Sequential(nn.Conv2d(base, base * 4, 3, padding=1), nn.ReLU(inplace=True), nn.PixelShuffle(2), nn.Conv2d(base, 1, 3, padding=1))

    def forward(self, x):
        feat = self.shared(self.backbone(x))
        return {
            'segmentation': torch.sigmoid(self.seg_head(feat)),
            'deblurred': torch.sigmoid(self.deblur_head(feat)),
            'super_res': torch.sigmoid(self.sr_head(feat)),
        }


@dataclass
class EnhancementOutputs:
    segmentation: np.ndarray
    deblurred: np.ndarray
    super_res: np.ndarray
    masked_super_res: np.ndarray


class QRSyntheticDataset(Dataset):
    def __init__(self, count=256, image_size=128):
        self.count = count
        self.image_size = image_size

    def __len__(self):
        return self.count

    def _make_clean(self, seed):
        payload = f'{{"ssid":"demo-{seed}","registration-id":"rid-{seed}"}}'
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=4, border=4)
        qr.add_data(payload)
        qr.make(fit=True)
        img = qr.make_image(fill_color='black', back_color='white').convert('L').resize((self.image_size, self.image_size), Image.NEAREST)
        return np.array(img, dtype=np.uint8)

    def _add_watermark(self, image: np.ndarray) -> np.ndarray:
        out = image.copy()
        overlay = np.full_like(out, 255)
        text = random.choice(['CONFIDENTIAL', 'DEMO', 'LAB', 'WATERMARK'])
        cv2.putText(overlay, text, (8, out.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 180, 2, cv2.LINE_AA)
        alpha = random.uniform(0.12, 0.28)
        return cv2.addWeighted(out, 1.0 - alpha, overlay, alpha, 0)

    def _add_screen_like_noise(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        xx = np.linspace(0, 2 * np.pi * random.uniform(8, 16), w, dtype=np.float32)
        yy = np.linspace(0, 2 * np.pi * random.uniform(8, 14), h, dtype=np.float32)
        pattern = (8.0 * np.sin(xx)[None, :] + 6.0 * np.sin(yy)[:, None]).astype(np.float32)
        rgb_strip = np.tile(np.array([[-3, 2, 1]], dtype=np.float32), (h, max(1, w // 3 + 1)))[:, :w]
        return np.clip(image.astype(np.float32) + pattern + rgb_strip, 0, 255).astype(np.uint8)

    def _corrupt(self, image):
        out = image.copy()
        if random.random() < 0.8:
            out = cv2.GaussianBlur(out, (0, 0), random.uniform(0.6, 2.0))
        if random.random() < 0.7:
            out = np.clip(out.astype(np.float32) + np.random.normal(0, random.uniform(4, 18), out.shape), 0, 255).astype(np.uint8)
        if random.random() < 0.6:
            scale = random.uniform(0.5, 0.9)
            h, w = out.shape
            small = cv2.resize(out, (max(32, int(w * scale)), max(32, int(h * scale))), interpolation=cv2.INTER_AREA)
            out = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        if random.random() < 0.55:
            out = self._add_watermark(out)
        if random.random() < 0.45:
            out = self._add_screen_like_noise(out)
        return out

    def __getitem__(self, idx):
        clean = self._make_clean(idx)
        noisy = self._corrupt(clean)
        seg = (clean < 128).astype(np.float32)
        return {
            'input': torch.from_numpy(noisy[None] / 255.0).float(),
            'target_clean': torch.from_numpy(clean[None] / 255.0).float(),
            'target_seg': torch.from_numpy(seg[None]).float(),
            'target_sr': torch.from_numpy(cv2.resize(clean, (self.image_size * 2, self.image_size * 2), interpolation=cv2.INTER_NEAREST)[None] / 255.0).float(),
        }


def compute_training_loss(outputs, batch):
    return F.binary_cross_entropy(outputs['segmentation'], batch['target_seg']) + F.l1_loss(outputs['deblurred'], batch['target_clean']) + F.l1_loss(outputs['super_res'], batch['target_sr'])


class MLEnhancer:
    def __init__(self, model=None, checkpoint=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model or QRResearchEnhancer()
        self.model.to(self.device).eval()
        self.checkpoint_path = Path(checkpoint) if checkpoint else None
        self.has_trained_weights = bool(self.checkpoint_path and self.checkpoint_path.exists())
        if self.has_trained_weights:
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))

    @staticmethod
    def _heuristic_enhance(gray: np.ndarray) -> EnhancementOutputs:
        eq = dynamic_illumination_equalization(gray)
        screen = suppress_screen_artifacts(eq)
        deblur = unsharp_masking(screen, amount=1.2, threshold=2.0)
        sr = cv2.resize(deblur, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        seg = proposed_integral_threshold(screen).binary
        seg_up = cv2.resize(seg, (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_NEAREST)
        masked = np.where(seg_up == 0, sr, 255).astype(np.uint8)
        return EnhancementOutputs(seg, deblur, sr, masked)

    def enhance(self, image: np.ndarray) -> EnhancementOutputs:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not self.has_trained_weights:
            return self._heuristic_enhance(gray)
        tensor = torch.from_numpy(gray[None, None] / 255.0).float().to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        seg = (out['segmentation'][0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        deblur = (out['deblurred'][0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        sr = (out['super_res'][0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        seg_up = cv2.resize(seg, (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_NEAREST)
        masked = np.where(seg_up > 80, sr, 255).astype(np.uint8)
        return EnhancementOutputs(seg, deblur, sr, masked)
