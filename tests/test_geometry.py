import math

import pytest
import torch

from somato.geometry import LayoutInfo, SensorLayout, build_layout, compute_clusters
from somato.geometry.rotations import (
    frame_from_normal, matrix_to_quat, quat_to_matrix, random_rotation, rot_z, rotation_6d, rpy_to_matrix,
)
from somato.robots.description import RobotDescription
from somato.robots.factory import build_robot


def test_quaternion_roundtrip():
    R = random_rotation(200, torch.Generator().manual_seed(0))
    assert torch.allclose(quat_to_matrix(matrix_to_quat(R)), R, atol=1e-5)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand(200, 3, 3), atol=1e-5)


def test_rpy_matches_urdf_convention():
    # URDF: fixed-axis roll, pitch, yaw -> R = Rz(yaw) Ry(pitch) Rx(roll)
    R = rpy_to_matrix(torch.tensor([0.0, 0.0, math.pi / 2]))
    assert torch.allclose(R @ torch.tensor([1.0, 0, 0]), torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    assert torch.allclose(rot_z(torch.tensor(math.pi / 2)), R, atol=1e-6)


def test_frame_from_normal_is_right_handed():
    n = torch.randn(50, 3)
    F = frame_from_normal(n)
    assert torch.allclose(torch.det(F), torch.ones(50), atol=1e-5)
    assert torch.allclose(F[..., :, 2], n / n.norm(dim=-1, keepdim=True), atol=1e-5)
    assert rotation_6d(F).shape == (50, 6)


def test_urdf_roundtrip(small_desc):
    parsed = RobotDescription.from_urdf(small_desc.to_urdf())
    assert parsed.body_names == small_desc.body_names
    assert parsed.joint_names == small_desc.joint_names
    for a, b in zip(parsed.joints, small_desc.joints):
        assert a.axis == pytest.approx(b.axis) and a.pos == pytest.approx(b.pos)
    q = torch.randn(len(small_desc.joint_names))
    p1, _ = small_desc.forward_kinematics(torch.zeros(3), torch.eye(3), q)
    p2, _ = parsed.forward_kinematics(torch.zeros(3), torch.eye(3), q)
    assert torch.allclose(p1, p2, atol=1e-5)


def test_forward_kinematics_joint_frames(small_desc):
    q = torch.tensor([0.3, -0.2, 0.1, 0.4])
    pos, rot = small_desc.forward_kinematics(torch.zeros(3), torch.eye(3), q)
    # Child origin sits at the parent's joint location; child yaw = parent yaw + q.
    for j in small_desc.joints:
        p, c = small_desc.body_names.index(j.parent), small_desc.body_names.index(j.child)
        assert torch.allclose(pos[c], pos[p] + rot[p] @ torch.tensor(j.pos), atol=1e-6)
    yaw = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])
    assert torch.allclose(yaw[1:] - yaw[:-1], q, atol=1e-5)


def test_cylinder_taxels_on_surface(small_desc, small_layout):
    g = small_layout.groups["tactile"]
    shape = small_desc.bodies[0].shapes[0]
    # In the body frame the capsule axis is the x-axis at y = z = 0.
    radial = g.local_pos[:, 1:].norm(dim=-1)
    assert torch.allclose(radial, torch.full_like(radial, shape.size["radius"] + 0.002), atol=1e-6)
    normals = g.local_rot[..., :, 2]
    assert torch.allclose(normals[:, 0], torch.zeros(len(g)), atol=1e-6)  # outward = radial
    assert ((normals[:, 1:] * g.local_pos[:, 1:]).sum(-1) > 0).all()
    assert len(g) == 5 * 2 * 6


def test_world_poses_follow_bodies(small_desc, small_layout):
    pos0, rot0 = small_layout.rest_poses(small_desc)
    R = random_rotation(1)[0]
    t = torch.tensor([1.0, -2.0, 0.5])
    bp, br = small_desc.forward_kinematics(t, R, torch.zeros(4))
    pos, rot = small_layout.world_poses(bp, br)
    assert torch.allclose(pos, pos0 @ R.T + t, atol=1e-5)
    assert torch.allclose(rot, R @ rot0, atol=1e-5)


def test_layout_save_load(tmp_path, small_layout):
    small_layout.save(tmp_path / "layout.pt")
    loaded = SensorLayout.load(tmp_path / "layout.pt")
    assert loaded.group_names == small_layout.group_names
    assert loaded.groups["joint"].joint_names == small_layout.groups["joint"].joint_names
    assert torch.equal(loaded.body_index, small_layout.body_index)


def test_joint_and_imu_placements(small_desc, small_layout):
    pos, rot = small_layout.rest_poses(small_desc)
    js = small_layout.slices()["joint"]
    assert torch.allclose(pos[js][:, 0], torch.tensor([0.1, 0.2, 0.3, 0.4]), atol=1e-6)
    assert torch.allclose(rot[js][:, :, 2], torch.tensor([0.0, 0.0, 1.0]).expand(4, 3))
    assert small_layout.groups["imu"].names == ["imu"]


def test_clusters(small_desc, small_layout):
    info = LayoutInfo.from_layout(small_layout, small_desc, "body")
    assert info.num_clusters == len(small_desc.bodies)
    rest, _ = small_layout.rest_poses(small_desc)
    fps = compute_clusters(small_layout, rest, "fps", 3)
    assert fps.unique().numel() == 3


@pytest.mark.parametrize("path", ["configs/robots/snake_planar.yaml", "configs/robots/snake_3d.yaml"])
def test_robot_configs_build(path):
    desc, layout = build_robot(path)
    assert set(layout.group_names) == {"tactile", "joint", "imu"}
    assert len(layout.groups["joint"]) == len(desc.joint_names)


def test_build_layout_rejects_unknown_bodies(small_desc):
    with pytest.raises(ValueError):
        build_layout(small_desc, {"tactile": {"placements": [{"generator": "cylinder", "bodies": "arm_.*"}]}})
