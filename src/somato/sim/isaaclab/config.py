"""Isaac Lab backend configuration (pure python; importable without Isaac Sim)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IsaacSnakeConfig:
    num_envs: int = 64
    env_spacing: float = 4.0
    physics_dt: float = 1.0 / 1000.0
    device: str = "cuda:0"
    render_interval: int = 20  # physics steps per rendered frame when not headless
    urdf_dir: str = "outputs/urdf"  # where the generated URDF (and converted USD) are written
    # Explicit PD actuator (so Isaac Lab computes the applied torque that motor sensing reports). The same actuator
    # model is used by the mjlab backend:
    #   "dc_motor": PD torque clipped to a linear DC-motor torque-speed curve: stall torque `saturation_effort` at
    #               rest falling to zero at the no-load speed `velocity_limit`, capped at the continuous
    #               `effort_limit` (Isaac Lab DCMotorCfg == mjlab DcMotorActuatorCfg). PhysX's joint velocity
    #               limit is set to `velocity_limit_sim` (effectively unlimited, as in MuJoCo).
    #   "ideal_pd": PD torque clipped at `effort_limit`; the URDF joint velocity becomes a hard PhysX velocity clamp,
    #               which MuJoCo cannot reproduce (the behaviour of datasets collected before the switch).
    actuator: str = "dc_motor"
    stiffness: float = 20.0
    damping: float = 0.5
    effort_limit: float = 6.0
    saturation_effort: float = 12.0
    velocity_limit: float = 8.0
    velocity_limit_sim: float = 1000.0
    armature: float = 0.002
    # Contact reporting:
    #   "per_body": one filtered contact sensor per link against the ground -> normal force, friction
    #               force and contact point (Isaac Lab >= 2.3).
    #   "net":      one unfiltered sensor for all links -> net normal force only (friction is estimated
    #               downstream from sliding velocity).
    contact_mode: str = "per_body"
    max_contact_data_count_per_prim: int = 16
    ground_prim: str = "/World/ground"
    # Sign applied to reported friction forces so that they act on the robot link. Verify with
    # scripts/isaac/validate_isaac.py (check "friction_opposes_sliding") and flip if needed.
    friction_sign: float = 1.0
    use_native_imu: bool = True
    self_collision: bool = False
    # Measured on Isaac Sim 5.1 at 1 kHz: with 8/1 iterations resting contacts chatter step to step (normal
    # force +-70 %, IMU acc +-20 m/s^2); 16/4 brings the normal force within 4 % of the weight for ~5 % cost.
    solver_position_iterations: int = 16
    solver_velocity_iterations: int = 4
    spawn_clearance: float = 0.003  # m above resting height
    static_friction_ratio: float = 1.1  # static = ratio * dynamic friction coefficient

    @property
    def ground_collision_prim(self) -> str:
        return f"{self.ground_prim}/terrain/GroundPlane/CollisionPlane"
