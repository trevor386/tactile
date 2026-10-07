"""Rotation and rigid-transform utilities (pure torch, batched over leading dims).

Conventions
-----------
* Quaternions are ``(w, x, y, z)`` (scalar first), matching Isaac Lab.
* Rotation matrices map *local* to *world* coordinates: ``p_world = R @ p_local + t``.
"""

from __future__ import annotations

import math

import torch


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions ``[..., 4]`` (wxyz) to rotation matrices ``[..., 3, 3]``."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        dim=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quat(m: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices ``[..., 3, 3]`` to unit quaternions ``[..., 4]`` (wxyz, w >= 0)."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    # Four candidate solutions; pick the numerically best-conditioned one per element.
    qw = torch.sqrt((1 + m00 + m11 + m22).clamp_min(0)) / 2
    qx = torch.sqrt((1 + m00 - m11 - m22).clamp_min(0)) / 2
    qy = torch.sqrt((1 - m00 + m11 - m22).clamp_min(0)) / 2
    qz = torch.sqrt((1 - m00 - m11 + m22).clamp_min(0)) / 2
    cands = torch.stack(
        [
            torch.stack([qw, (m21 - m12) / (4 * qw.clamp_min(1e-8)), (m02 - m20) / (4 * qw.clamp_min(1e-8)),
                         (m10 - m01) / (4 * qw.clamp_min(1e-8))], -1),
            torch.stack([(m21 - m12) / (4 * qx.clamp_min(1e-8)), qx, (m01 + m10) / (4 * qx.clamp_min(1e-8)),
                         (m02 + m20) / (4 * qx.clamp_min(1e-8))], -1),
            torch.stack([(m02 - m20) / (4 * qy.clamp_min(1e-8)), (m01 + m10) / (4 * qy.clamp_min(1e-8)), qy,
                         (m12 + m21) / (4 * qy.clamp_min(1e-8))], -1),
            torch.stack([(m10 - m01) / (4 * qz.clamp_min(1e-8)), (m02 + m20) / (4 * qz.clamp_min(1e-8)),
                         (m12 + m21) / (4 * qz.clamp_min(1e-8)), qz], -1),
        ],
        dim=-2,
    )
    best = torch.stack([qw, qx, qy, qz], -1).argmax(-1)
    q = torch.gather(cands, -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    return q / q.norm(dim=-1, keepdim=True)


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula. ``axis [..., 3]`` (need not be normalized), ``angle [...]``."""
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z = axis.unbind(-1)
    c, s = torch.cos(angle), torch.sin(angle)
    C = 1 - c
    m = torch.stack(
        [
            c + x * x * C, x * y * C - z * s, x * z * C + y * s,
            y * x * C + z * s, c + y * y * C, y * z * C - x * s,
            z * x * C - y * s, z * y * C + x * s, c + z * z * C,
        ],
        dim=-1,
    )
    return m.reshape(angle.shape + (3, 3))


def rot_z(angle: torch.Tensor) -> torch.Tensor:
    """Rotation about +z by ``angle [...]`` -> ``[..., 3, 3]``."""
    c, s = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack([c, -s, zero, s, c, zero, zero, zero, one], dim=-1).reshape(angle.shape + (3, 3))


def rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """URDF roll-pitch-yaw (fixed axes X, then Y, then Z): ``R = Rz(y) Ry(p) Rx(r)``."""
    r, p, y = rpy.unbind(-1)
    ex = torch.tensor([1.0, 0.0, 0.0], dtype=rpy.dtype, device=rpy.device).expand(rpy.shape)
    ey = torch.tensor([0.0, 1.0, 0.0], dtype=rpy.dtype, device=rpy.device).expand(rpy.shape)
    ez = torch.tensor([0.0, 0.0, 1.0], dtype=rpy.dtype, device=rpy.device).expand(rpy.shape)
    return axis_angle_to_matrix(ez, y) @ axis_angle_to_matrix(ey, p) @ axis_angle_to_matrix(ex, r)


def random_rotation(n: int, generator: torch.Generator | None = None, dtype=torch.float32) -> torch.Tensor:
    """Uniformly distributed random rotations ``[n, 3, 3]``."""
    q = torch.randn(n, 4, generator=generator, dtype=dtype)
    return quat_to_matrix(q)


def rotation_6d(m: torch.Tensor) -> torch.Tensor:
    """Continuous 6D rotation representation (first two columns), ``[..., 3, 3] -> [..., 6]``."""
    return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)


def frame_from_normal(normal: torch.Tensor, tangent_hint: torch.Tensor | None = None) -> torch.Tensor:
    """Build right-handed frames whose z-axis is ``normal [..., 3]``.

    The x-axis is the projection of ``tangent_hint`` onto the tangent plane (falls back to a
    deterministic perpendicular vector when the hint is parallel to the normal).
    """
    z = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if tangent_hint is None:
        tangent_hint = torch.tensor([1.0, 0.0, 0.0], dtype=z.dtype, device=z.device).expand_as(z)
    x = tangent_hint - (tangent_hint * z).sum(-1, keepdim=True) * z
    bad = x.norm(dim=-1, keepdim=True) < 1e-6
    alt = torch.tensor([0.0, 1.0, 0.0], dtype=z.dtype, device=z.device).expand_as(z)
    alt = alt - (alt * z).sum(-1, keepdim=True) * z
    x = torch.where(bad, alt, x)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def transform_points(rot: torch.Tensor, trans: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply ``R p + t`` with broadcasting: ``rot [..., 3, 3]``, ``trans/points [..., 3]``."""
    return (rot @ points.unsqueeze(-1)).squeeze(-1) + trans


def compose(rot_a: torch.Tensor, pos_a: torch.Tensor, rot_b: torch.Tensor, pos_b: torch.Tensor):
    """Compose transforms ``T_a * T_b`` -> ``(R_a R_b, R_a p_b + p_a)``."""
    return rot_a @ rot_b, transform_points(rot_a, pos_a, pos_b)


def deg2rad(x: float) -> float:
    return x * math.pi / 180.0
