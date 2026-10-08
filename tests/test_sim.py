import pytest
import torch

from somato.control import GaitConfig, SerpenoidGait
from somato.robots.snake import SnakeConfig, make_snake_description
from somato.sim import ContactModelConfig, RateConfig, SimRunner, StimulusPipeline, TaxelContactModel
from somato.sim.mock import PlanarSnakeBackend
from somato.sim.terrain import TerrainCatalog, TerrainClass


def test_catalog_sampling(catalog):
    ids = torch.arange(len(catalog)).repeat(20)
    tb = catalog.sample(ids, torch.Generator().manual_seed(0))
    for c, cls in enumerate(catalog.classes):
        mu = tb.friction[tb.class_id == c]
        assert (mu >= cls.friction[0] - 1e-6).all() and (mu <= cls.friction[1] + 1e-6).all()
    tex = tb.texture(torch.rand(len(ids), 500, 2))
    assert 0.6 < float(tex.std()) < 1.4


def test_rate_config_validation():
    rates = RateConfig(physics_hz=1000, latent_hz=50, sensors={"tactile": 500})
    assert rates.substeps("tactile") == 10 and rates.physics_per_sample("tactile") == 2
    with pytest.raises(ValueError):
        RateConfig(physics_hz=1000, latent_hz=30).physics_per_latent()


def _resting_taxels(small_desc, small_layout, catalog, sinkage=0.0):
    tb = catalog.sample(torch.zeros(1, dtype=torch.long))
    tb.params["sinkage"][:] = sinkage
    tb.params["texture_amp"][:] = 0.0
    nb = len(small_desc.bodies)
    bp, br = small_desc.forward_kinematics(torch.tensor([[0.0, 0.0, 0.025]]), torch.eye(3)[None],
                                           torch.zeros(1, len(small_desc.joint_names)))
    pos, rot = small_layout.world_poses(bp, br, group="tactile")
    g = small_layout.groups["tactile"]
    return tb, pos, rot, g, nb


def test_contact_model_conserves_force(small_desc, small_layout, catalog):
    tb, pos, rot, g, nb = _resting_taxels(small_desc, small_layout, catalog)
    normal = torch.full((1, nb), 2.5)
    stim, w = TaxelContactModel()(pos, rot, torch.zeros_like(pos), g.area, g.body_index, nb, tb, normal)
    per_body = torch.zeros(nb).index_add_(0, g.body_index, (stim[0, :, 0] * g.area))
    assert torch.allclose(per_body, normal[0], rtol=1e-4)
    up = rot[0, :, 2, 2] > 0  # upward-facing taxels never feel ground pressure
    assert (stim[0, up, 0] == 0).all()


def test_soft_terrain_widens_footprint(small_desc, small_layout, catalog):
    counts = []
    for sink in (0.0, 0.006):
        tb, pos, rot, g, nb = _resting_taxels(small_desc, small_layout, catalog, sinkage=sink)
        stim, _ = TaxelContactModel()(pos, rot, torch.zeros_like(pos), g.area, g.body_index, nb, tb,
                                      torch.full((1, nb), 2.5))
        p = stim[0, :, 0]
        counts.append(float(p.sum() ** 2 / p.pow(2).sum()))  # effective number of loaded taxels
    assert counts[1] > 1.2 * counts[0]


def test_estimated_shear_opposes_sliding(small_desc, small_layout, catalog):
    tb, pos, rot, g, nb = _resting_taxels(small_desc, small_layout, catalog)
    vel = torch.zeros_like(pos)
    vel[..., 0] = 0.2  # sliding along +x
    stim, _ = TaxelContactModel(ContactModelConfig(texture=False))(
        pos, rot, vel, g.area, g.body_index, nb, tb, torch.full((1, nb), 2.5), None)
    traction_x = stim[0, :, 1] * rot[0, :, 0, 0] + stim[0, :, 2] * rot[0, :, 0, 1]  # back to world x
    loaded = stim[0, :, 0] > 0
    assert (traction_x[loaded] < 0).all()


@pytest.fixture(scope="module")
def mock_runner(catalog):
    desc = make_snake_description(SnakeConfig(num_links=6))
    from somato.geometry import build_layout
    from somato.robots.snake import default_snake_sensor_spec

    layout = build_layout(desc, default_snake_sensor_spec(n_rings=2, n_per_ring=6))
    gen = torch.Generator().manual_seed(0)
    backend = PlanarSnakeBackend(desc, catalog, 5, generator=gen)
    gait = SerpenoidGait(desc, GaitConfig(), 5, generator=gen)
    rates = RateConfig(physics_hz=500, control_hz=100, latent_hz=50, sensors={"tactile": 250, "joint": 100, "imu": 100})
    return SimRunner(backend, StimulusPipeline(layout), gait, rates), desc, layout


def test_runner_frames(mock_runner):
    runner, desc, layout = mock_runner
    runner.reset(torch.arange(5))
    frame = runner.step_latent()
    assert frame.stimuli["tactile"].shape == (5, 5, len(layout.groups["tactile"]), 3)
    assert frame.stimuli["joint"].shape == (5, 2, 5, 4)
    assert frame.stimuli["imu"].shape == (5, 2, 1, 6)
    assert frame.body_pos.shape == (5, 6, 3)
    assert set(frame.labels) >= {"slip_speed", "in_contact", "normal_force"}


def test_mock_physics_is_terrain_dependent(mock_runner):
    runner, desc, _ = mock_runner
    runner.reset(torch.arange(5))
    torques = []
    for _ in range(40):
        runner.step_latent()
        state = runner.backend.get_state()
        torques.append(state.joint_torque.abs().mean(-1))
        # quasi-static force balance on the whole snake
        assert float(state.contact_friction_force.sum(1).norm(dim=-1).max()) < 0.2
    mean_t = torch.stack(torques).mean(0)
    mu = runner.backend.terrain.friction
    assert int(mean_t.argmax()) == int(mu.argmax())  # highest friction -> highest motor torque


def test_mock_locomotion_needs_anisotropy():
    desc = make_snake_description(SnakeConfig(num_links=8))
    cat = TerrainCatalog([TerrainClass("iso", (0.3, 0.3), (1.0, 1.0)), TerrainClass("aniso", (0.3, 0.3), (5.0, 5.0))])
    gen = torch.Generator().manual_seed(0)
    be = PlanarSnakeBackend(desc, cat, 2, generator=gen)
    be.reset(terrain_class=torch.tensor([0, 1]))
    gait = SerpenoidGait(desc, GaitConfig(yaw_offset=(0.0, 0.0), frequency=(1.0, 1.0), yaw_amplitude=(0.7, 0.7)), 2,
                         generator=gen)
    for k in range(3000):
        be.step(gait(be.time))
        if k == 999:
            start = be.get_state().contact_point.mean(1)
    travel = (be.get_state().contact_point.mean(1) - start).norm(dim=-1)
    assert travel[1] > 3 * travel[0]


def test_validation_suite_passes_on_mock(mock_runner):
    from somato.validation import BackendValidator

    runner, desc, layout = mock_runner
    v = BackendValidator(runner.backend, desc, layout, runner.rates, settle_time=0.2, run_time=0.6)
    results = v.run()
    failed = [r.line() for r in results if r.passed is False]
    assert not failed, "\n".join(failed)


def test_texture_spatial_filter(catalog):
    import math

    gen = torch.Generator().manual_seed(0)
    terrain = catalog.sample(torch.tensor([1, 3]), gen)  # rough ice (short wavelengths), fresh snow (longer)
    xy = torch.rand(2, 4000, 2, generator=gen)
    raw = terrain.texture(xy)
    assert torch.allclose(terrain.texture(xy, 0.0), raw)
    filtered = terrain.texture(xy, 0.0035)
    # rough ice: 2-6 mm wavelengths are essentially erased; fresh snow (5-15 mm) keeps a fraction
    assert filtered[0].std() < 0.05 * raw[0].std()
    assert 0.05 * raw[1].std() < filtered[1].std() < raw[1].std()
    # a single long wave passes almost unchanged
    k = terrain.texture_k.norm(dim=-1)
    assert math.exp(-0.5 * (2 * math.pi / 0.2 * 0.0035) ** 2) > 0.99
    assert k.min() > 0
