from dataclasses import dataclass

@dataclass
class JumpParams:
    x_start: float = -1.5
    x_trigger: float = -1.6
    x_land: float = -2.2
    x_end: float = -2.8
    height: float = 1.0
    thrust_vel: float = 3.15
    tuck_height: float = 0.35
    max_force: float = 350.0
    pre_land_dist: float = 0.3
    pre_land_dy: float = 0.5
    land_h_thresh: float = 0.5
    crouch_h: float = 0.38


@dataclass
class FallParams:
    zone_len: float = 0.6
    land_zone_len: float = 0.9
    crouch_h: float = 0.4
    thrust_vel: float = 0.5
    dx_scale: float = 1.2
    land_dy: float = -3.0
    tuck_x: float = 0.05
    tuck_y: float = 0.35
    extend_time: float = 0.23
    extend_h: float = 0.08
    lookahead: float = 0.1
    edge_thresh: float = 0.3
    land_dy_thresh: float = -0.5


@dataclass
class HoleParams:
    x_start: float = -4.8
    x_trigger: float = -4.93
    x_end: float = -5.5
    width: float = 0.4
    crouch_h: float = 0.4
    thrust_vel: float = 1.6
    dx_scale: float = 1.0
    tuck_y: float = 0.35
    land_dy: float = -3.0
    pre_land_dist: float = 0.35
    pre_land_dy: float = 0.7
    extend_time: float = 0.15
    extend_h: float = 0.1


@dataclass
class PreLandParams:
    lookahead: float = 0.1
    foot_offset_x: float = 0.15
    foot_offset_y: float = 0.0
    descent_vel: float = -0.5


@dataclass
class GainScaleParams:
    flight_Kp: float = 1.5
    flight_Kd: float = 1.2
    stepoff_Kp: float = 1.3
    stepoff_Kd: float = 1.4
    landing_Kp: float = 1.3
    landing_Kd: float = 1.5
    gap_Kp: float = 1.2
    gap_Kd: float = 1.3


@dataclass
class MPCParams:
    jump_y: float = 1.5
    jump_dy: float = 2.0
    landing_theta: float = 2.0
    landing_dtheta: float = 1.5