"""Taxel-level contact stimulus from rigid-body contact information.

Rigid-body simulators (PhysX, the mock) resolve contact per body with a handful of contact points,
far coarser than a dense skin. This model distributes each body's ground-contact force over its
taxels:

1. **Footprint weights from geometry.** For taxel ``k`` with outward normal ``n_k`` at height ``z_k``
   above a ground plane at ``h``, the indentation is ``delta_k = (h + sinkage) - z_k`` (taxel points
   sit on the skin surface, ``skin`` outside the collision shape). Weights are
   ``w_k = facing_k * sigma * softplus(delta_k / sigma)`` with ``facing_k = max(0, -n_k . z)^p``; the
   softplus spreads load to taxels just above the ground like an elastomer skin would, and soft
   terrain (``sinkage > 0``) engages more taxels.
2. **Magnitude from physics** (``mode: hybrid``). ``p_k = F_b w_k / (a_k sum_{k in b} w_k)`` so the
   integrated taxel pressure equals the body's normal contact force. ``mode: penetration`` uses
   ``p_k = skin_stiffness * delta_k`` instead (no physics forces needed).
3. **Micro-texture** modulates pressure by ``1 + A * texture(x_k, y_k)``, a spatial random field of
   the terrain; sliding over it produces vibration at frequency ``v / wavelength`` (absent when stuck).
4. **Shear** is the body's friction force distributed like pressure, or, when the backend cannot
   report friction, a regularized Coulomb estimate ``-mu p_k v_t / sqrt(|v_t|^2 + eps^2)`` from the
   taxel's sliding velocity. It is expressed in the taxel frame (``shear_x``, ``shear_y``).

Limitations: only contact with the ground plane is localized. Contact with other objects keeps the
correct per-body force magnitude but is placed on the ground-facing taxels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from somato.sim.terrain import TerrainBatch


@dataclass
class ContactModelConfig:
    mode: str = "hybrid"  # "hybrid" | "penetration"
    spread: float = 0.001  # m, softplus width (elastomer load spreading)
    facing_power: float = 2.0
    skin_stiffness: float = 3.0e7  # Pa per m of indentation (penetration mode)
    slip_eps: float = 0.01  # m/s, Coulomb regularization for estimated shear
    texture: bool = True


class TaxelContactModel:
    def __init__(self, cfg: ContactModelConfig | None = None):
        self.cfg = cfg or ContactModelConfig()

    def weights(self, pos: torch.Tensor, rot: torch.Tensor, terrain: TerrainBatch) -> torch.Tensor:
        """Footprint weights ``[E, N]`` (units of length) for taxels at ``pos [E, N, 3]``."""
        c = self.cfg
        surface = (terrain.ground_height + terrain.sinkage).unsqueeze(-1)  # [E, 1]
        delta = surface - pos[..., 2]
        facing = (-rot[..., 2, 2]).clamp_min(0.0).pow(c.facing_power)  # normal z-component, pointing down
        return facing * c.spread * torch.nn.functional.softplus(delta / c.spread)

    def __call__(
        self,
        pos: torch.Tensor,
        rot: torch.Tensor,
        vel: torch.Tensor,
        area: torch.Tensor,
        body_index: torch.Tensor,
        num_bodies: int,
        terrain: TerrainBatch,
        normal_force: torch.Tensor | None = None,
        friction_force: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute taxel stimuli.

        Args:
            pos, rot, vel: taxel world positions ``[E, N, 3]``, frames ``[E, N, 3, 3]``, velocities ``[E, N, 3]``.
            area: ``[N]`` taxel areas; body_index: ``[N]`` index of each taxel's body.
            normal_force: ``[E, Nb]`` ground normal force per body (required in hybrid mode).
            friction_force: ``[E, Nb, 3]`` friction force per body (optional).

        Returns:
            ``(stimulus [E, N, 3], weights [E, N])`` with stimulus channels (normal, shear_x, shear_y) in Pa.
        """
        c = self.cfg
        E, N, _ = pos.shape
        w = self.weights(pos, rot, terrain)
        idx = body_index.to(pos.device).expand(E, N)
        if c.mode == "hybrid":
            if normal_force is None:
                raise ValueError("hybrid contact model needs per-body normal forces")
            w_sum = torch.zeros(E, num_bodies, device=pos.device, dtype=pos.dtype).scatter_add_(1, idx, w)
            share = w / w_sum.gather(1, idx).clamp_min(1e-12)  # fraction of the body's load on each taxel
            pressure = normal_force.clamp_min(0.0).gather(1, idx) * share / area
        elif c.mode == "penetration":
            share = None
            pressure = c.skin_stiffness * w  # w ~ facing * max(delta, 0) away from the softplus knee
        else:
            raise ValueError(f"Unknown contact mode {c.mode}")

        if friction_force is not None and share is not None:
            traction = friction_force.gather(1, idx.unsqueeze(-1).expand(E, N, 3)) * (share / area).unsqueeze(-1)
        else:
            v_t = vel.clone()
            v_t[..., 2] = 0.0
            mu = terrain.friction.unsqueeze(-1)
            traction = -(mu * pressure).unsqueeze(-1) * v_t / torch.sqrt(v_t.pow(2).sum(-1, keepdim=True) + c.slip_eps**2)

        if c.texture:
            mod = 1.0 + terrain.texture_amp.unsqueeze(-1) * terrain.texture(pos[..., :2])
            pressure = pressure * mod.clamp_min(0.0)

        shear_x = (traction * rot[..., :, 0]).sum(-1)
        shear_y = (traction * rot[..., :, 1]).sum(-1)
        return torch.stack([pressure, shear_x, shear_y], dim=-1), w
