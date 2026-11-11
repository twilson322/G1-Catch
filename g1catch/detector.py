"""Learned ball detector: a small fully convolutional network that outputs a centre heatmap.

The network is trained on 96x96 crops and, being fully convolutional, runs on any image size.
At run time it sees the whole frame until a track exists and a window around the predicted
pixel afterwards.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision import Detection

CROP_PX = 96
HEATMAP_SIGMA_PX = 1.5
DETECTION_THRESHOLD = 0.35        # peak heatmap probability below which there is no ball
CENTROID_HALF_WINDOW_PX = 4
MAX_CANDIDATES = 3
PEAK_SEPARATION_PX = 8
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "results" / "detector.pt"


class HeatmapNet(nn.Module):
    """Two-level U-Net with 54,809 weights. Input RGB in [0, 1], output one logit per pixel."""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Conv2d(24, 24, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(32 + 24, 24, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(24 + 16, 16, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(16, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        return self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))


def gaussian_target(centres: np.ndarray, visible: np.ndarray, size: int = CROP_PX) -> np.ndarray:
    """Heatmap targets (N, size, size). Rows with visible == False are all zero."""
    grid = np.arange(size, dtype=np.float32)
    du = grid[None, None, :] - centres[:, 0, None, None]
    dv = grid[None, :, None] - centres[:, 1, None, None]
    target = np.exp(-(du**2 + dv**2) / (2 * HEATMAP_SIGMA_PX**2)).astype(np.float32)
    return target * visible[:, None, None]


def peak_centroid(prob: np.ndarray) -> tuple[np.ndarray, float]:
    """Sub-pixel peak of a probability map: the weighted centroid of a window around the argmax."""
    v, u = np.unravel_index(int(np.argmax(prob)), prob.shape)
    w = CENTROID_HALF_WINDOW_PX
    v0, v1 = max(v - w, 0), min(v + w + 1, prob.shape[0])
    u0, u1 = max(u - w, 0), min(u + w + 1, prob.shape[1])
    win = prob[v0:v1, u0:u1]
    total = float(win.sum())
    us = np.arange(u0, u1)
    vs = np.arange(v0, v1)
    centre = np.array([(win.sum(0) * us).sum() / total, (win.sum(1) * vs).sum() / total])
    return centre, float(prob[v, u])


class CnnDetector:
    def __init__(self, weights: Path | str = DEFAULT_WEIGHTS, threshold: float = DETECTION_THRESHOLD):
        # One thread: the network is small, and the experiments run several processes side by side.
        torch.set_num_threads(1)
        self.net = HeatmapNet()
        self.net.load_state_dict(torch.load(weights, map_location="cpu"))
        self.net.eval()
        self.threshold = threshold

    @torch.no_grad()
    def heatmap(self, image: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        return torch.sigmoid(self.net(x))[0, 0].numpy()

    def detect(self, image: np.ndarray, roi=None) -> Detection | None:
        """The strongest detection. roi = (u0, v0, u1, v1) restricts the search to a window,
        which is 8x cheaper."""
        found = self.detect_all(image, roi, max_peaks=1)
        return found[0] if found else None

    def detect_all(self, image: np.ndarray, roi=None, max_peaks: int = MAX_CANDIDATES) -> list[Detection]:
        """Up to max_peaks detections, strongest first, at least PEAK_SEPARATION_PX apart.

        One peak per image is not enough to start a track: the detector fires on any small
        ball-coloured disc, and a brighter one in the background would hide the real ball.
        The tracker picks the pair that triangulates to a plausible place.
        """
        if roi is None:
            u0 = v0 = 0
            prob = self.heatmap(image)
        else:
            u0, v0, u1, v1 = roi
            prob = self.heatmap(np.ascontiguousarray(image[v0:v1, u0:u1]))
        prob = prob.copy()
        found = []
        for _ in range(max_peaks):
            centre, peak = peak_centroid(prob)
            if peak < self.threshold:
                break
            found.append(Detection(centre + np.array([u0, v0]), peak))
            cu, cv = int(round(centre[0])), int(round(centre[1]))
            r = PEAK_SEPARATION_PX
            prob[max(cv - r, 0):cv + r + 1, max(cu - r, 0):cu + r + 1] = 0.0     # suppress and look again
        return found


def roi_around(pixel: np.ndarray, width: int, height: int, size: int = CROP_PX):
    """A size x size window centred on `pixel`, shifted to stay inside the image."""
    u0 = int(np.clip(round(pixel[0]) - size // 2, 0, width - size))
    v0 = int(np.clip(round(pixel[1]) - size // 2, 0, height - size))
    return u0, v0, u0 + size, v0 + size
