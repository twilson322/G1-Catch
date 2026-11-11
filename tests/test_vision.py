import numpy as np
import pytest

from g1catch import vision as vis
from g1catch.sim import CatchSim, Throw


@pytest.fixture(scope="module")
def rig():
    sim = CatchSim()
    sim.reset(Throw(np.array([2.0, 0.1, 1.3]), np.zeros(3)))
    intr = vis.Intrinsics.from_fovy(sim.model.cam_fovy[0])
    return sim, intr, vis.camera_poses(sim.model, sim.data)


def test_point_on_the_optical_axis_projects_to_the_principal_point(rig):
    _, intr, poses = rig
    pose = poses[0]
    point = pose.position - 2.0 * pose.rotation[:, 2]      # cameras look along -z
    px, depth = vis.project(intr, pose, point)
    assert depth == pytest.approx(2.0)
    assert px == pytest.approx([intr.cx, intr.cy])


def test_disparity_matches_focal_times_baseline_over_depth(rig):
    _, intr, poses = rig
    baseline = np.linalg.norm(poses[0].position - poses[1].position)
    point = 0.5 * (poses[0].position + poses[1].position) - 1.7 * poses[0].rotation[:, 2]
    (ul, _), depth = vis.project(intr, poses[0], point)
    (ur, _), _ = vis.project(intr, poses[1], point)
    assert ul - ur == pytest.approx(intr.focal_px * baseline / depth, rel=1e-9)


def test_triangulation_inverts_projection(rig):
    _, intr, poses = rig
    rng = np.random.default_rng(0)
    for _ in range(20):
        point = np.array([rng.uniform(0.5, 4.0), rng.uniform(-0.8, 0.8), rng.uniform(0.3, 2.0)])
        pixels = [vis.project(intr, pose, point)[0] for pose in poses]
        assert vis.triangulate(intr, poses, pixels) == pytest.approx(point, abs=1e-8)


def test_depth_uncertainty_grows_with_the_square_of_range(rig):
    _, intr, poses = rig
    forward = -poses[0].rotation[:, 2]
    mid = 0.5 * (poses[0].position + poses[1].position)
    sigma = []
    for depth in (1.0, 2.0, 4.0):
        cov = vis.triangulation_covariance(intr, poses, mid + depth * forward, pixel_sigma=0.3)
        sigma.append(np.sqrt(forward @ cov @ forward))
    assert sigma[1] / sigma[0] == pytest.approx(4.0, rel=0.02)
    assert sigma[2] / sigma[1] == pytest.approx(4.0, rel=0.02)
    # sigma_Z = Z^2 sigma_d / (f b), and the disparity of two independent pixels has sqrt(2) the noise.
    baseline = np.linalg.norm(poses[0].position - poses[1].position)
    assert sigma[1] == pytest.approx(4.0 * np.sqrt(2) * 0.3 / (intr.focal_px * baseline), rel=0.02)


def test_projection_matrix_agrees_with_project(rig):
    _, intr, poses = rig
    point = np.array([1.5, -0.3, 1.1])
    homogeneous = vis.projection_matrix(intr, poses[1]) @ np.append(point, 1.0)
    assert homogeneous[:2] / homogeneous[2] == pytest.approx(vis.project(intr, poses[1], point)[0])


@pytest.mark.render
def test_rendered_ball_appears_where_projection_says(rig):
    """The geometry code and the renderer must agree, or every label and measurement is biased."""
    render = pytest.importorskip("g1catch.render")
    sim, intr, _ = rig
    try:
        ren = render.StereoRenderer(sim.model)
    except Exception as exc:                      # no GL context on this machine
        pytest.skip(f"offscreen rendering unavailable: {exc}")
    look = render.Appearance.nominal()
    look.noise_sigma = 0.0
    rng = np.random.default_rng(0)
    errors = []
    for _ in range(8):
        ball = np.array([rng.uniform(0.9, 1.6), rng.uniform(-0.3, 0.3), rng.uniform(1.1, 1.6)])
        sim.reset(Throw(ball, np.zeros(3)))
        poses = vis.camera_poses(sim.model, sim.data)
        images = ren.render(sim.data, look, rng)
        for image, pose in zip(images, poses):
            det = vis.BlobDetector().detect(image)
            if det is not None:
                errors.append(det.pixel - vis.project(intr, pose, ball)[0])
    ren.close()
    errors = np.array(errors)
    assert len(errors) >= 8
    assert np.abs(errors.mean(axis=0)).max() < 0.5      # shading shifts the centroid a little
    assert np.abs(errors).max() < 1.5
