"""mjlab (MuJoCo-Warp) backend configuration (pure python; importable without mjlab)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MjlabSnakeConfig:
    num_envs: int = 64
    env_spacing: float = 4.0
    physics_dt: float = 1.0 / 1000.0
    device: str = "cuda:0"
    # Explicit PD actuator, same model and gains as the Isaac backend (see IsaacSnakeConfig): "dc_motor" clips the PD
    # torque to the linear torque-speed curve (stall `saturation_effort`, no-load speed `velocity_limit`, continuous
    # `effort_limit`; mjlab DcMotorActuatorCfg), "ideal_pd" clips at `effort_limit` only. MuJoCo has no joint
    # velocity limit, so Isaac's former hard PhysX velocity clamp has no counterpart.
    actuator: str = "dc_motor"
    stiffness: float = 20.0
    damping: float = 0.5
    effort_limit: float = 6.0
    saturation_effort: float = 12.0
    velocity_limit: float = 8.0
    armature: float = 0.002
    # MuJoCo solver. Contact buffers are per world: a resting 16-link snake has 32 ground contacts, i.e. 128
    # pyramidal constraint rows plus joint limits (mujoco-warp's defaults, 48/64, overflow silently).
    solver: str = "newton"
    iterations: int = 20
    ls_iterations: int = 10
    # Elliptic friction cones: with mujoco-warp 3.11 the Newton solver returns NaN contact forces for pyramidal
    # cones at mu <= ~0.07 (glare ice) within ~20 ms of gait onset; elliptic cones (or the CG solver) do not.
    cone: str = "elliptic"
    impratio: float = 1.0
    nconmax: int = 64
    njmax: int = 256
    # Soft-contact parameters of the robot geoms (time constant [s], damping ratio). Robot geoms get priority 1,
    # so their friction and solref override the ground's (MuJoCo otherwise takes the max of the two frictions).
    # With `terrain_compliance` this is only the fallback for terrain classes without `contact_timeconst`.
    contact_solref: tuple[float, float] = (0.02, 1.0)
    # Terrain compliance: robot-ground contacts become explicit contact pairs (one per robot geom) whose *normal*
    # softness is set per env from the sampled terrain (`contact_timeconst`, `contact_dampratio`), so soft snow is
    # physically penetrated, while the *friction* rows keep the stiff `friction_solref`. With plain geom contacts
    # MuJoCo applies one solref to both, and a soft contact then also softens friction (links creep, locomotion
    # stops depending on friction; see docs/mjlab.md). False: geom-geom contacts with `contact_solref` (v0).
    terrain_compliance: bool = True
    friction_solref: tuple[float, float] = (0.02, 1.0)
    # Contact detection margin / gap of the robot geoms [m] (MuJoCo: contacts are detected below `margin` and active
    # below `margin - gap`). With margin 0, lightly loaded links hover at ~0 penetration and the contact set flickers.
    contact_margin: float = 0.0
    contact_gap: float = 0.0
    # Recompute kinematics, contacts and sensors after each step so the reported state is the post-step state
    # (as in Isaac Lab). Without it, body poses and sensors lag joint positions by one physics step.
    forward_after_step: bool = True
    use_native_imu: bool = True
    self_collision: bool = False
    spawn_clearance: float = 0.003  # m above resting height
