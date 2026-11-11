"""Sources of stereo pixel measurements: rendered images through a detector, or a noise model.

The noise model exists because rendering costs about 40 ms per stereo pair in software. Its
noise and miss rate are set from the detector's measured errors (results/detector_eval.csv), and
the experiments state which sensor produced each number.
"""
from __future__ import annotations

import numpy as np

from .detector import roi_around
from .render import EXPOSURE_S, Appearance, StereoRenderer
from .ukf import Measurement
from .vision import Intrinsics, camera_poses, in_image, project


class SyntheticSensor:
    def __init__(self, intr: Intrinsics, rng: np.random.Generator, pixel_sigma: float = 0.5,
                 miss_rate: float = 0.02, outlier_rate: float = 0.0):
        self.intr, self.rng = intr, rng
        self.pixel_sigma, self.miss_rate, self.outlier_rate = pixel_sigma, miss_rate, outlier_rate

    def capture(self, sim, predicted_pixels=None) -> Measurement:
        poses = camera_poses(sim.model, sim.data)
        pixels = []
        for pose in poses:
            px, depth = project(self.intr, pose, sim.ball_position)
            if depth < 0.15 or not in_image(self.intr, px, 1.0) or self.rng.random() < self.miss_rate:
                pixels.append(None)
            elif self.rng.random() < self.outlier_rate:
                pixels.append(self.rng.uniform([0, 0], [self.intr.width, self.intr.height]))
            else:
                pixels.append(px + self.rng.normal(0.0, self.pixel_sigma, 2))
        return Measurement(tuple(pixels), poses)


class RenderedSensor:
    """Renders both cameras and runs a detector on each image."""

    def __init__(self, intr: Intrinsics, renderer: StereoRenderer, detector, rng: np.random.Generator,
                 look: Appearance | None = None, motion_blur: bool = True):
        self.intr, self.renderer, self.detector, self.rng = intr, renderer, detector, rng
        self.look = look or Appearance.nominal()
        self.motion_blur = motion_blur
        self.last_images = None

    def capture(self, sim, predicted_pixels=None) -> Measurement:
        """predicted_pixels: per-camera pixel where the tracker expects the ball, or None.
        With a prediction the detector searches a window, otherwise the whole frame."""
        poses = camera_poses(sim.model, sim.data)
        blur = None
        if self.motion_blur:
            blur = []
            for pose in poses:
                p0, depth = project(self.intr, pose, sim.ball_position)
                p1, _ = project(self.intr, pose, sim.ball_position + EXPOSURE_S * sim.ball_velocity)
                blur.append((p0, p1 - p0) if depth > 0.15 and in_image(self.intr, p0) else None)
        images = self.renderer.render(sim.data, self.look, self.rng, blur)
        self.last_images = images
        pixels, candidates = [], []
        for i, image in enumerate(images):
            roi = None
            if predicted_pixels is not None and predicted_pixels[i] is not None:
                roi = roi_around(predicted_pixels[i], self.intr.width, self.intr.height)
            if hasattr(self.detector, "detect_all"):
                found = [d.pixel for d in self.detector.detect_all(image, roi)]
            else:
                det = self.detector.detect(image, roi)
                found = [] if det is None else [det.pixel]
            pixels.append(found[0] if found else None)
            candidates.append(found)
        # Candidates are only for starting a track. With a prediction the tracker's gate decides.
        return Measurement(tuple(pixels), poses, tuple(candidates) if predicted_pixels is None else None)
