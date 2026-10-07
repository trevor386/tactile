import pytest
import torch

from somato.geometry import LayoutInfo, build_layout
from somato.geometry.rotations import random_rotation
from somato.models import (
    GroupSpec, SomatoBatch, TEMPORAL_ENCODERS, build_model, count_parameters, match_parameter_count,
)
from somato.models.graph import EDGE_FEATURE_DIMS, edge_features, knn_graph
from somato.models.spatial import SPATIAL_LAYERS, GraphConfig, SpatialConfig, SpatialStack
from somato.robots.snake import SnakeConfig, default_snake_sensor_spec, make_snake_description

from conftest import random_poses

GROUPS = {"tactile": GroupSpec("tactile", 1, 4), "joint": GroupSpec("joint", 4, 2), "imu": GroupSpec("imu", 6, 2)}
SMALL_MODEL = {
    "dim": 16,
    "temporal": {"latent_dim": 16, "params": {"hidden": 16, "conv_channels": 8}},
    "spatial": {"num_layers": 2, "graph": {"k": 6, "k_per_group": {"tactile": 5, "joint": 2, "imu": 1}},
                "params": {"num_basis": 4, "kernel_hidden": 16}},
    "heads": {"terrain": {"out_dim": 3, "hidden": 16, "cluster_layers": 1},
              "slip": {"type": "per_body", "out_dim": 1, "hidden": 16}},
}


def make_batch(desc, layout, info, B=2, L=3, seed=0, generic=True):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, L, len(desc.joint_names), generator=gen) * 0.4
    bp, br = desc.forward_kinematics(torch.zeros(B, L, 3), torch.eye(3).expand(B, L, 3, 3), q)
    pos, rot = layout.world_poses(bp, br)
    if generic:  # break the exact symmetries of the ideal layout (no KNN ties)
        pos = pos + 1e-3 * torch.randn(pos.shape, generator=gen)
    sl = layout.slices()
    readings = {g: torch.randn(B, L, s.substeps, sl[g].stop - sl[g].start, s.channels, generator=gen)
                for g, s in GROUPS.items()}
    return SomatoBatch(readings, pos, rot, info)


def transformed(batch, R, t):
    return SomatoBatch(batch.readings, batch.pos @ R.T + t, R @ batch.rot, batch.info, batch.node_mask)


# ----------------------------------------------------------------------------- graph


def test_knn_contains_true_neighbors():
    pos = torch.randn(3, 60, 3)
    g = knn_graph(pos, 7)
    ref = torch.cdist(pos, pos).topk(7, largest=False).indices
    for b in range(3):
        for n in range(60):
            assert set(ref[b, n].tolist()) <= set(g.index[b, n][g.mask[b, n]].tolist())


def test_knn_per_group_and_mask(small_info):
    pos = torch.randn(2, small_info.num_nodes, 3)
    gid = small_info.group_id
    g = knn_graph(pos, 0, group_id=gid, k_per_group={0: 4, 1: 2, 2: 1}, tie_extra=0)
    assert g.index.shape[-1] == 7
    assert (gid[g.index[..., :4]] == 0).all() and (gid[g.index[..., 4:6]] == 1).all() and (gid[g.index[..., 6]] == 2).all()
    mask = torch.ones(2, small_info.num_nodes, dtype=torch.bool)
    mask[:, 3] = False
    g = knn_graph(pos, 5, node_mask=mask)
    assert not ((g.index == 3) & g.mask).any()
    assert not g.mask[:, 3].any()


def test_knn_includes_exact_ties():
    # A node at the center of a ring of 8 equidistant nodes: k=3 must return the whole tie group.
    ang = torch.arange(8) * (2 * torch.pi / 8)
    ring = torch.stack([ang.cos(), ang.sin(), torch.zeros(8)], -1)
    pos = torch.cat([torch.zeros(1, 3), ring])[None]
    g = knn_graph(pos, 3, include_self=False, tie_extra=8)
    assert int(g.mask[0, 0].sum()) == 8


@pytest.mark.parametrize("kind", list(EDGE_FEATURE_DIMS))
def test_edge_feature_invariance(kind, small_desc, small_layout):
    pos, rot = random_poses(small_desc, small_layout, (2,))
    g = knn_graph(pos, 6)
    R, t = random_rotation(1)[0], torch.tensor([0.3, -1.0, 2.0])
    e1 = edge_features(kind, pos, rot, g, 0.05)
    e2 = edge_features(kind, pos @ R.T + t, R @ rot, g, 0.05)
    assert e1.shape[-1] == EDGE_FEATURE_DIMS[kind]
    if kind == "rel_pos_world":
        assert not torch.allclose(e1, e2, atol=1e-3)
    else:
        assert torch.allclose(e1, e2, atol=1e-4)


# ----------------------------------------------------------------------------- stage 1


@pytest.mark.parametrize("name", TEMPORAL_ENCODERS.names())
def test_temporal_streaming_equals_batch(name):
    enc = TEMPORAL_ENCODERS.build(name, in_channels=3, substeps=4, latent_dim=8)
    x = torch.randn(2, 5, 4, 6, 3)
    z, _ = enc(x)
    state, steps = None, []
    for t in range(5):
        zt, state = enc(x[:, t : t + 1], state)
        steps.append(zt)
    assert z.shape == (2, 5, 6, 8)
    assert torch.allclose(z, torch.cat(steps, 1), atol=1e-5)


@pytest.mark.parametrize("name", TEMPORAL_ENCODERS.names())
def test_temporal_encoder_is_per_sensor(name):
    """Stage 1 must not mix sensors: changing sensor 0's history leaves other latents unchanged."""
    enc = TEMPORAL_ENCODERS.build(name, in_channels=2, substeps=3, latent_dim=8)
    x = torch.randn(1, 4, 3, 5, 2)
    x2 = x.clone()
    x2[..., 0, :] += 1.0
    z1, _ = enc(x)
    z2, _ = enc(x2)
    assert torch.allclose(z1[:, :, 1:], z2[:, :, 1:], atol=1e-6)
    assert not torch.allclose(z1[:, :, 0], z2[:, :, 0])


# ----------------------------------------------------------------------------- stage 2


@pytest.mark.parametrize("layer", SPATIAL_LAYERS.names())
def test_spatial_layers_equivariance(layer, small_desc, small_layout, small_info):
    torch.manual_seed(0)
    pos, rot = random_poses(small_desc, small_layout, (2,))
    pos = pos + 1e-3 * torch.randn_like(pos)
    edge = "rel_pos_local" if layer not in ("full_attention", "none") else "none"
    stack = SpatialStack(16, SpatialConfig(type=layer, edge_features=edge, graph=GraphConfig(k=6)))
    h = torch.randn(2, small_info.num_nodes, 16)
    y = stack(h, pos, rot, small_info)
    # SE(3) invariance of features (relative, local-frame geometry only)
    R, t = random_rotation(1)[0], torch.tensor([1.0, 2.0, -0.5])
    assert torch.allclose(y, stack(h, pos @ R.T + t, R @ rot, small_info), atol=1e-4)
    # Permutation equivariance within the tactile group (does not depend on sensor indices)
    sl = small_info.slices["tactile"]
    perm = torch.arange(small_info.num_nodes)
    perm[sl] = perm[sl][torch.randperm(sl.stop - sl.start)]
    info_p = LayoutInfo(small_info.group_names, small_info.group_kinds, small_info.slices, small_info.group_id,
                        small_info.body_index[perm], small_info.area[perm], small_info.rest_pos[perm],
                        small_info.rest_rot[perm], small_info.cluster_id[perm])
    y_p = stack(h[:, perm], pos[:, perm], rot[:, perm], info_p)
    assert torch.allclose(y[:, perm], y_p, atol=1e-4)


def test_world_frame_features_are_not_invariant(small_desc, small_layout, small_info):
    pos, rot = random_poses(small_desc, small_layout, (2,))
    stack = SpatialStack(16, SpatialConfig(edge_features="rel_pos_world", graph=GraphConfig(k=6)))
    h = torch.randn(2, small_info.num_nodes, 16)
    R = random_rotation(1)[0]
    assert not torch.allclose(stack(h, pos, rot, small_info), stack(h, pos @ R.T, R @ rot, small_info), atol=1e-3)


# ----------------------------------------------------------------------------- full model


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_model(SMALL_MODEL, GROUPS).eval()


def test_model_outputs(model, small_desc, small_layout, small_info):
    out, _ = model(make_batch(small_desc, small_layout, small_info))
    assert out["terrain"].shape == (2, 3, 3)
    assert out["slip"].shape == (2, 3, len(small_desc.bodies), 1)


def test_model_streaming_equals_batch(model, small_desc, small_layout, small_info):
    batch = make_batch(small_desc, small_layout, small_info, L=4)
    full, _ = model(batch)
    state, steps = None, []
    for t in range(4):
        o, state = model(batch.step(t), state)
        steps.append(o["terrain"])
    assert torch.allclose(full["terrain"], torch.cat(steps, 1), atol=1e-5)
    last, _ = model(batch, output_steps=1)
    assert torch.allclose(last["terrain"][:, 0], full["terrain"][:, -1], atol=1e-5)


def test_model_invariant_to_rigid_motion(model, small_desc, small_layout, small_info):
    batch = make_batch(small_desc, small_layout, small_info)
    out, _ = model(batch)
    out2, _ = model(transformed(batch, random_rotation(1)[0], torch.tensor([3.0, -1.0, 0.2])))
    assert torch.allclose(out["terrain"], out2["terrain"], atol=1e-4)


def test_model_transfers_to_other_layout(model):
    """Same weights on a robot with more links and a different taxel arrangement."""
    desc = make_snake_description(SnakeConfig(num_links=8, radius=0.03))
    layout = build_layout(desc, default_snake_sensor_spec(n_rings=3, n_per_ring=4))
    info = LayoutInfo.from_layout(layout, desc)
    out, _ = model(make_batch(desc, layout, info))
    assert out["terrain"].shape == (2, 3, 3) and out["slip"].shape[2] == 8


def test_masked_sensors_are_ignored(model, small_desc, small_layout, small_info):
    batch = make_batch(small_desc, small_layout, small_info)
    mask = torch.ones(2, small_info.num_nodes, dtype=torch.bool)
    mask[:, 2] = False
    batch.node_mask = mask
    out1, _ = model(batch)
    batch.readings["tactile"][..., 2, :] += 100.0  # dead taxel goes haywire
    out2, _ = model(batch)
    assert torch.allclose(out1["terrain"], out2["terrain"], atol=1e-5)


def test_flat_baseline_and_param_matching(model, small_desc, small_layout, small_info):
    flat = build_model({"architecture": "flat_recurrent", "hidden": 16, "heads": {"terrain": {"out_dim": 3}}},
                       GROUPS, small_info)
    out, _ = flat(make_batch(small_desc, small_layout, small_info), output_steps=2)
    assert out["terrain"].shape == (2, 2, 3)
    target = count_parameters(model)
    width, matched = match_parameter_count(
        lambda w: build_model({"architecture": "flat_recurrent", "hidden": w, "heads": {"terrain": {"out_dim": 3}}},
                              GROUPS, small_info), target)
    assert abs(count_parameters(matched) - target) / target < 0.1


def test_receptor_and_other_variants_build(small_desc, small_layout, small_info):
    cfg = dict(SMALL_MODEL, temporal={"type": "receptor", "latent_dim": 16, "params": {"hidden": 16, "kernel_size": 8}},
               temporal_overrides={"imu": {"type": "gru", "params": {"hidden": 8}}},
               spatial={"type": "geo_attention", "edge_features": "ppf", "graph": {"k": 6}, "params": {"heads": 2}})
    m = build_model(cfg, GROUPS)
    out, _ = m(make_batch(small_desc, small_layout, small_info))
    assert torch.isfinite(out["terrain"]).all()


def test_normalizer_fit(model, small_desc, small_layout, small_info):
    batch = make_batch(small_desc, small_layout, small_info)
    scaled = {g: 50 * x + 3 for g, x in batch.readings.items()}
    model.fit_normalizers(scaled)
    norm = model.normalizers["tactile"](scaled["tactile"])
    assert abs(float(norm.mean())) < 0.1 and abs(float(norm.std()) - 1) < 0.1
