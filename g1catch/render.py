"""Offscreen stereo rendering with appearance randomization.

Rendering is software (OSMesa) in this project, so the head cameras draw only geom groups that
can enter their view (see scripts/build_model.py) and skip shadows.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .vision import CAMERA_NAMES, IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX

HIDDEN_GEOM_GROUP = 1
EXPOSURE_S = 0.004


@dataclass
class Appearance:
    """Everything about a frame's look that is not geometry."""
    ball_rgb: tuple = (0.85, 0.9, 0.15)
    light_scale: float = 1.0
    light_tint: tuple = (1.0, 1.0, 1.0)
    floor_rgb: tuple = (1.0, 1.0, 1.0)
    wall_rgb: tuple = (1.0, 1.0, 1.0)
    gain: float = 1.0
    noise_sigma: float = 2.0          # 8-bit counts
    distractor: bool = False          # a ball-coloured object fixed in the background

    @staticmethod
    def nominal() -> "Appearance":
        return Appearance()

    @staticmethod
    def random(rng: np.random.Generator, strength: float = 1.0) -> "Appearance":
        """Sample around nominal. strength 1 is the training distribution."""
        s = strength
        base = np.array([0.85, 0.9, 0.15])
        return Appearance(
            ball_rgb=tuple(np.clip(base + s * rng.uniform(-0.15, 0.15, 3), 0.0, 1.0)),
            light_scale=float(np.exp(s * rng.uniform(-0.7, 0.4))),
            light_tint=tuple(1.0 + s * rng.uniform(-0.25, 0.25, 3)),
            floor_rgb=tuple(np.clip(1.0 + s * rng.uniform(-0.5, 0.5, 3), 0.1, 1.6)),
            wall_rgb=tuple(np.clip(1.0 + s * rng.uniform(-0.5, 0.5, 3), 0.1, 1.6)),
            gain=float(np.exp(s * rng.uniform(-0.3, 0.3))),
            noise_sigma=float(2.0 + s * rng.uniform(0.0, 6.0)),
            distractor=bool(rng.random() < 0.3 * s),
        )


class StereoRenderer:
    def __init__(self, model: mujoco.MjModel, width: int = IMAGE_WIDTH_PX, height: int = IMAGE_HEIGHT_PX):
        self.model = model
        self.renderer = mujoco.Renderer(model, height, width)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self.option = mujoco.MjvOption()
        self.option.geomgroup[HIDDEN_GEOM_GROUP] = 0
        self._ball_mat = model.material("ball_mat").id
        self._floor_mat = model.material("floor_mat").id
        self._wall_mat = model.material("wall_mat").id
        self._distractor = model.geom("distractor").id
        self._light_diffuse0 = model.light_diffuse.copy()
        self._light_ambient0 = model.light_ambient.copy()

    def close(self) -> None:
        self.renderer.close()

    def _apply(self, look: Appearance) -> None:
        m = self.model
        m.mat_rgba[self._ball_mat, :3] = look.ball_rgb
        m.mat_rgba[self._floor_mat, :3] = look.floor_rgb
        m.mat_rgba[self._wall_mat, :3] = look.wall_rgb
        tint = np.asarray(look.light_tint) * look.light_scale
        m.light_diffuse[:] = np.clip(self._light_diffuse0 * tint, 0.0, 1.0)
        m.light_ambient[:] = np.clip(self._light_ambient0 * tint, 0.0, 1.0)
        m.geom_rgba[self._distractor, 3] = 1.0 if look.distractor else 0.0

    def render(self, data: mujoco.MjData, look: Appearance, rng: np.random.Generator,
               blur_px=None) -> tuple[np.ndarray, np.ndarray]:
        """Left and right uint8 RGB images.

        blur_px: per-camera (centre pixel, image-plane displacement during the exposure) or None.
        Motion blur is approximated by averaging shifted copies of a window around the ball,
        which also smears the background inside that window.
        """
        self._apply(look)
        images = []
        for i, name in enumerate(CAMERA_NAMES):
            self.renderer.update_scene(data, name, scene_option=self.option)
            img = self.renderer.render().astype(np.float32)
            if blur_px is not None and blur_px[i] is not None:
                img = _motion_blur(img, *blur_px[i])
            img = img * look.gain + rng.normal(0.0, look.noise_sigma, img.shape)
            images.append(np.clip(img, 0, 255).astype(np.uint8))
        return images[0], images[1]


def _motion_blur(img: np.ndarray, centre: np.ndarray, shift: np.ndarray, half_window: int = 20,
                 taps: int = 5) -> np.ndarray:
    length = float(np.linalg.norm(shift))
    if length < 0.5:
        return img
    h, w = img.shape[:2]
    cu, cv = int(round(centre[0])), int(round(centre[1]))
    pad = half_window + int(np.ceil(length))
    u0, u1 = max(cu - pad, 0), min(cu + pad + 1, w)
    v0, v1 = max(cv - pad, 0), min(cv + pad + 1, h)
    if u1 - u0 < 3 or v1 - v0 < 3:
        return img
    patch = img[v0:v1, u0:u1]
    acc = np.zeros_like(patch)
    for s in np.linspace(-0.5, 0.5, taps):
        du, dv = int(round(s * shift[0])), int(round(s * shift[1]))
        acc += np.roll(np.roll(patch, dv, axis=0), du, axis=1)
    out = img.copy()
    out[v0:v1, u0:u1] = acc / taps
    return out
