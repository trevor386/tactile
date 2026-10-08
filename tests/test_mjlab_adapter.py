"""mjlab (MuJoCo-Warp) backend tests. Skipped unless mjlab is installed and a CUDA GPU is available."""

import subprocess
import sys

import pytest
import torch

from somato.sim.mjlab import MjlabSnakeConfig, is_available

pytestmark = [
    pytest.mark.mjlab,
    pytest.mark.skipif(not is_available(), reason="mjlab not installed"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="mjlab backend tests need a CUDA GPU"),
]


@pytest.fixture(scope="module")
def backend(catalog):
    from somato.robots.factory import build_robot
    from somato.sim.mjlab.backend import MjlabBackend

    desc, layout = build_robot("configs/robots/snake_3d.yaml")
    return MjlabBackend(desc, layout, catalog, MjlabSnakeConfig(num_envs=4), torch.Generator().manual_seed(0))


def test_state_shapes_and_materials(backend):
    E, Nb, Nj = 4, len(backend.desc.body_names), len(backend.desc.joint_names)
    assert set(backend.body_names) == set(backend.desc.body_names)
    gids = backend.robot.indexing.geom_ids.long()
    assert torch.allclose(backend.sim.model.geom_friction[:, gids, 0], backend.terrain.friction[:, None].expand(E, len(gids)))
    for _ in range(300):  # settle
        backend.step(torch.zeros(E, Nj, device=backend.device))
    s = backend.get_state()
    assert s.body_pos.shape == (E, Nb, 3) and s.joint_torque.shape == (E, Nj)
    assert s.contact_friction_force.shape == (E, Nb, 3) and set(s.imu) == {"imu"}
    weight = backend.desc.total_mass() * 9.81
    assert torch.allclose(s.contact_normal_force.sum(-1), torch.full((E,), weight, device=backend.device), rtol=0.05)
    # Positions are reported relative to each env's origin.
    assert s.body_pos[..., :2].abs().max() < 2.0


def test_reset_assigns_terrain(backend):
    backend.reset(torch.tensor([1, 2]), torch.tensor([4, 0]))
    assert backend.terrain.class_id[1:3].tolist() == [4, 0]
    gids = backend.robot.indexing.geom_ids.long()
    assert torch.allclose(backend.sim.model.geom_friction[1:3, gids[0], 0], backend.terrain.friction[1:3])


def test_mjlab_validation():
    result = subprocess.run([sys.executable, "scripts/mjlab/validate_mjlab.py", "--num_envs", "10"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
