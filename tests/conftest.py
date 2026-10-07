from __future__ import annotations

import pytest
import torch

from somato.geometry import LayoutInfo, build_layout
from somato.robots.snake import SnakeConfig, default_snake_sensor_spec, make_snake_description
from somato.sim.terrain import TerrainCatalog, default_catalog_path

torch.set_num_threads(2)


@pytest.fixture(scope="session")
def small_desc():
    return make_snake_description(SnakeConfig(num_links=5, link_length=0.1, radius=0.025))


@pytest.fixture(scope="session")
def small_layout(small_desc):
    return build_layout(small_desc, default_snake_sensor_spec(n_rings=2, n_per_ring=6))


@pytest.fixture(scope="session")
def small_info(small_desc, small_layout):
    return LayoutInfo.from_layout(small_layout, small_desc)


@pytest.fixture(scope="session")
def catalog():
    return TerrainCatalog.from_yaml(default_catalog_path())


def random_poses(desc, layout, batch_shape, scale=0.4, seed=0):
    """Sensor poses for random joint configurations: ``(pos [*bs, N, 3], rot [*bs, N, 3, 3])``."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(*batch_shape, len(desc.joint_names), generator=gen) * scale
    bp, br = desc.forward_kinematics(torch.zeros(*batch_shape, 3), torch.eye(3).expand(*batch_shape, 3, 3), q)
    return layout.world_poses(bp, br)
