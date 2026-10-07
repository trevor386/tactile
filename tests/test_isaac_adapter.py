"""Isaac Lab adapter tests.

* With the fake ``isaaclab`` (always runs): checks scene construction, body-name mapping, per-env
  terrain materials and state packing against the Isaac Lab 2.3 API surface.
* With real Isaac Lab (``-m isaac``, skipped when not installed): runs the integration checks.
"""

import importlib

import pytest
import torch

import fake_isaaclab
from somato.robots.factory import build_robot
from somato.sim import RateConfig, SimRunner, StimulusPipeline
from somato.sim.isaaclab import IsaacSnakeConfig, is_available


@pytest.fixture
def fake_isaac():
    real = is_available()
    if real:
        pytest.skip("real Isaac Lab installed; fake-module tests skipped")
    fake_isaaclab.install()
    yield
    fake_isaaclab.uninstall()


@pytest.mark.parametrize("contact_mode", ["per_body", "net"])
def test_backend_with_fake_isaaclab(fake_isaac, tmp_path, catalog, contact_mode):
    backend_mod = importlib.import_module("somato.sim.isaaclab.backend")
    desc, layout = build_robot("configs/robots/snake_3d.yaml")
    cfg = IsaacSnakeConfig(num_envs=4, device="cpu", urdf_dir=str(tmp_path), contact_mode=contact_mode)
    backend = backend_mod.IsaacLabBackend(desc, layout, catalog, cfg, torch.Generator().manual_seed(0))

    assert (tmp_path / f"{desc.name}.urdf").exists()
    assert set(backend.body_names) == set(desc.body_names)
    # per-env friction was written to the PhysX material buffer
    mats = backend.robot.root_physx_view.materials
    assert torch.allclose(mats[:, 0, 1], backend.terrain.friction.cpu())
    assert torch.allclose(mats[:, 0, 0], backend.terrain.friction.cpu() * cfg.static_friction_ratio)

    state = backend.get_state()
    E, Nb, Nj = 4, len(desc.body_names), len(desc.joint_names)
    assert state.body_pos.shape == (E, Nb, 3) and state.joint_torque.shape == (E, Nj)
    assert state.contact_normal_force.shape == (E, Nb) and (state.contact_normal_force >= 0).all()
    if contact_mode == "per_body":
        assert state.contact_friction_force.shape == (E, Nb, 3)
        assert torch.isfinite(state.contact_point).all()  # NaN (no contact) replaced
    else:
        assert state.contact_friction_force is None
    assert set(state.imu) == {"imu"}

    rates = RateConfig(physics_hz=1000, latent_hz=50, control_hz=100, sensors={"tactile": 500, "joint": 250, "imu": 250})
    runner = SimRunner(backend, StimulusPipeline(layout), lambda t: torch.zeros(E, Nj), rates)
    frame = runner.step_latent()
    assert frame.stimuli["tactile"].shape == (E, 10, len(layout.groups["tactile"]), 3)
    assert frame.stimuli["imu"].shape == (E, 5, 1, 6)
    assert torch.isfinite(frame.stimuli["tactile"]).all()
    # Layout body order is restored even though the simulator reports bodies in another order.
    order = [backend.body_names.index(b) for b in layout.body_names]
    assert torch.allclose(frame.body_pos, (backend.robot.data.body_link_pos_w - backend.scene.env_origins[:, None])[:, order])

    backend.reset(torch.tensor([1, 2]), torch.tensor([4, 0]))
    assert backend.terrain.class_id[1:3].tolist() == [4, 0]
    assert backend.robot.root_physx_view.set_calls[-1].tolist() == [1, 2]


@pytest.mark.isaac
@pytest.mark.skipif(not is_available(), reason="Isaac Lab not installed")
def test_real_isaac_validation():  # pragma: no cover - needs Isaac Sim + GPU
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "scripts/isaac/validate_isaac.py", "--headless", "--num_envs", "5"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
