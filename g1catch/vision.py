"""Stereo camera geometry: projection, triangulation and a classical colour detector.

Pixel coordinates put integer values at pixel centres, u to the right and v down, which matches
array indexing image[v, u]. MuJoCo cameras look along their -z axis with +x right and +y up.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

IMAGE_WIDTH_PX = 400
IMAGE_HEIGHT_PX = 300
CAMERA_NAMES = ("cam_left", "cam_right")


@dataclass(frozen=True)
class Intrinsics:
    focal_px: float
    cx: float
    cy: float
    width: int
    height: int

    @staticmethod
    def from_fovy(fovy_deg: float, width: int = IMAGE_WIDTH_PX, height: int = IMAGE_HEIGHT_PX) -> "Intrinsics":
        focal = 0.5 * height / np.tan(0.5 * np.radians(fovy_deg))
        # Pixel i covers [i, i+1) in OpenGL window coordinates, so its centre is at i + 0.5 and
        # the optical axis lands at (W/2 - 0.5, H/2 - 0.5) in centre-indexed pixels.
        return Intrinsics(focal, 0.5 * width - 0.5, 0.5 * height - 0.5, width, height)


@dataclass(frozen=True)
class CameraPose:
    """World pose of one camera: position and the 3x3 matrix whose columns are its axes."""
    position: np.ndarray
    rotation: np.ndarray


def camera_poses(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[CameraPose, CameraPose]:
    """Poses of the two head cameras for the configuration currently in `data`."""
    return tuple(CameraPose(data.cam(n).xpos.copy(), data.cam(n).xmat.reshape(3, 3).copy())
                 for n in CAMERA_NAMES)


def project(intr: Intrinsics, pose: CameraPose, points_world: np.ndarray):
    """Pixels (..., 2) and depths (...,) of world points (..., 3). Depth <= 0 is behind the camera."""
    pc = (np.asarray(points_world) - pose.position) @ pose.rotation
    depth = -pc[..., 2]
    safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    u = intr.cx + intr.focal_px * pc[..., 0] / safe
    v = intr.cy - intr.focal_px * pc[..., 1] / safe
    return np.stack([u, v], axis=-1), depth


def in_image(intr: Intrinsics, pixel: np.ndarray, margin_px: float = 0.0) -> bool:
    return bool(margin_px <= pixel[0] <= intr.width - 1 - margin_px
                and margin_px <= pixel[1] <= intr.height - 1 - margin_px)


def projection_matrix(intr: Intrinsics, pose: CameraPose) -> np.ndarray:
    """3x4 matrix P with [u w, v w, w] = P [X, 1], w = depth."""
    k = np.array([[intr.focal_px, 0.0, -intr.cx],
                  [0.0, -intr.focal_px, -intr.cy],
                  [0.0, 0.0, -1.0]])
    rt = np.hstack([pose.rotation.T, (-pose.rotation.T @ pose.position)[:, None]])
    return k @ rt


def triangulate(intr: Intrinsics, poses, pixels) -> np.ndarray:
    """Direct linear transform: the world point that best explains one pixel per camera."""
    rows = []
    for pose, px in zip(poses, pixels):
        p = projection_matrix(intr, pose)
        rows.append(px[0] * p[2] - p[0])
        rows.append(px[1] * p[2] - p[1])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    x = vt[-1]
    return x[:3] / x[3]


def triangulation_covariance(intr: Intrinsics, poses, point: np.ndarray, pixel_sigma: float) -> np.ndarray:
    """First-order covariance of a triangulated point for isotropic pixel noise."""
    eps = 1e-5
    jac = np.empty((4, 3))
    for axis in range(3):
        dp = np.zeros(3)
        dp[axis] = eps
        hi = np.concatenate([project(intr, pose, point + dp)[0] for pose in poses])
        lo = np.concatenate([project(intr, pose, point - dp)[0] for pose in poses])
        jac[:, axis] = (hi - lo) / (2 * eps)
    return pixel_sigma**2 * np.linalg.inv(jac.T @ jac)


@dataclass
class Detection:
    pixel: np.ndarray       # (u, v), sub-pixel
    confidence: float       # detector-specific, higher is better


class BlobDetector:
    """Colour threshold and centroid. The classical baseline for the learned detector.

    Works in chromaticity (r, g fractions of the brightness) so a global brightness change does
    not move the threshold. It is tuned for the nominal yellow-green ball and has no notion of
    shape, so a colour cast or a similar-coloured object defeats it.
    """
    NOMINAL_CHROMA = np.array([0.425, 0.445])   # (r, g) fractions of the ball under white light
    CHROMA_TOLERANCE = 0.07
    MIN_BRIGHTNESS = 60.0                     # sum of 8-bit channels
    MIN_PIXELS = 2

    def detect(self, image: np.ndarray, roi=None) -> Detection | None:
        img = image.astype(np.float32)
        total = img.sum(axis=2)
        chroma = img[..., :2] / np.maximum(total[..., None], 1.0)
        mask = (np.abs(chroma - self.NOMINAL_CHROMA).max(axis=2) < self.CHROMA_TOLERANCE) \
            & (total > self.MIN_BRIGHTNESS)
        if roi is not None:
            u0, v0, u1, v1 = roi
            keep = np.zeros_like(mask)
            keep[v0:v1, u0:u1] = True
            mask &= keep
        count = int(mask.sum())
        if count < self.MIN_PIXELS:
            return None
        vs, us = np.nonzero(mask)
        return Detection(np.array([us.mean(), vs.mean()]), float(count))
