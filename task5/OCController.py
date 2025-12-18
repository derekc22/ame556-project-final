import numpy as np
from typing import Tuple
from good.fowardKin import fK
from good.params import JumpParams, FallParams, HoleParams, PreLandParams, GainScaleParams, MPCParams


class OCParams:
    def __init__(self):
        # Robot parameters
        self.forward_sign = -1.0

        self.l = 0.22
        self.a = 0.25
        self.b = 0.15
        self.m = 8.0
        self.I = (1/12) * (self.a**2 + self.b**2) * self.m
        self.mu = 0.5
        self.g = 9.81
        
        # Course geometry [x_positions; y_heights]
        self.course_geom = np.array([
            [0, -1.99, -2, -3, -3.01, -4.99, -5, -5.39, -5.4, -7.4],
            [0,  0,     0.4, 0.4,  0,     0,     0.01, 0.01,  0,   0]
        ])
        
        # Obstacle parameters
        self.jump = JumpParams()
        self.fall = FallParams()
        self.hole = HoleParams()
        self.preland = PreLandParams()
        self.gains = GainScaleParams()
        self.mpc = MPCParams()
        
        # MPC configuration
        self.Fy_lim = np.array([-10, 350])
        self.N = 30
        self.dt = 0.01
        
        # Increased X position weight to drive forward motion
        self.Q = np.diag([300, 600, 400, 120, 100, 25, 0])  
        self.R = 1.5e-4 * np.diag([1, 1, 1.2, 1.2])
        
        # Controller targets
        self.theta_des = 0.0
        self.ypos_des = 0.45
        self.airborne_tol = 0.02
        
        # Swing leg PD gains - reduced for smoother motion
        self.Kp = 0.6 * np.diag([180, 120]) 
        self.Kd = 5.0 * np.diag([1, 1])
        
        # Gain schedules
        self.schedule = {
            'dx_des': {
                'time': np.array([0, 1, 2, 3, 4]),
                'value': np.array([0.3, 0.5, 0.6, 0.7, 0.8]) 
            },
            'Kv': {
                'time': np.array([0, 3]),
                'value': np.array([0.15, 0.18])
            },
            'swing_height': {
                'time': np.array([0, 0.5, 2]),
                'value': np.array([0.01, 0.015, 0.025]) 
            },
            'stance_time': {
                'time': np.array([0, 1, 2]),
                'value': np.array([0.08, 0.12, 0.15])
            }
        }

# Helpers
class ScheduledParams:
    """Time-varying parameters"""
    def __init__(self, params: OCParams, t: float):
        self.dx_des = self._interpolate(params.schedule['dx_des'], t)
        self.Kv = self._interpolate(params.schedule['Kv'], t)
        self.swing_height = self._interpolate(params.schedule['swing_height'], t)
        self.stance_time = self._interpolate(params.schedule['stance_time'], t)
        self.ypos_des = params.ypos_des
        self.theta_des = params.theta_des
    
    @staticmethod
    def _interpolate(schedule: dict, t: float) -> float:
        t_clamp = np.clip(t, schedule['time'][0], schedule['time'][-1])
        return float(np.interp(t_clamp, schedule['time'], schedule['value']))


def hTerr(x: float, params: OCParams) -> float:
    """Interpolate terrain height at horizontal position x"""
    course = params.course_geom
    
    for i in range(course.shape[1] - 1):
        if course[0, i] <= x < course[0, i+1]:
            t_frac = (x - course[0, i]) / (course[0, i+1] - course[0, i])
            h = course[1, i] + t_frac * (course[1, i+1] - course[1, i])
            return h
    
    return course[1, -1] if x >= course[0, -1] else 0.0


def gaitCheck(p: np.ndarray, dp: np.ndarray, P1: np.ndarray, 
                  P2: np.ndarray, terrain_h: float, 
                  params: OCParams) -> str:
    """Determine gait phase based on position and foot contact"""
    
    # Check if both feet are airborne
    airborne = (P1[1] > terrain_h + params.airborne_tol and 
                P2[1] > terrain_h + params.airborne_tol)
    x = p[0]
    
    # Zone-based gait selection
    if x <= params.jump.x_start:
        return "walk"
    
    elif x <= params.jump.x_trigger:
        return "flight" if airborne else "jump"
    
    elif x <= params.jump.x_end:
        return gaitJump(p, dp, airborne, terrain_h, params)
    
    elif x <= params.jump.x_end + params.fall.zone_len:
        return gaitFall(p, airborne, terrain_h, params)
    
    elif x <= params.jump.x_end + params.fall.zone_len + params.fall.land_zone_len:
        return gaitLanding(dp, airborne, terrain_h, params)
    
    elif x <= params.hole.x_start:
        return "walk"
    
    elif x <= params.hole.x_trigger:
        return "holeFlight" if airborne else "hole_prep"
    
    elif x <= params.hole.x_end:
        return gaitHole(p, dp, airborne, params)
    
    else:
        return "walk"


def gaitJump(p: np.ndarray, dp: np.ndarray, airborne: bool, 
                   terrain_h: float, params: OCParams) -> str:
    """Select gait during box jump"""
    if airborne:
        dist_to_land = params.jump.x_land - p[0]
        if dist_to_land < params.jump.pre_land_dist and dp[1] < params.jump.pre_land_dy:
            return "pre_landing"
        else:
            return "flight"
    elif p[1] > terrain_h + params.jump.land_h_thresh:
        return "landing"
    else:
        return "walk_elevated"


def gaitFall(p: np.ndarray, airborne: bool, terrain_h: float,
                      params: OCParams) -> str:
    """Select gait during step-off from elevated surface"""
    terrain_ahead = hTerr(p[0] + params.fall.lookahead, params)
    at_edge = (terrain_h > params.fall.edge_thresh and 
               terrain_ahead < terrain_h - params.fall.edge_thresh)
    
    if at_edge:
        return "step_off_flight" if airborne else "step_off_prep"
    elif airborne:
        return "step_off_flight"
    elif terrain_h > params.fall.edge_thresh:
        return "walk_elevated"
    else:
        return "walk"


def gaitLanding(dp: np.ndarray, airborne: bool, terrain_h: float,
                         params: OCParams) -> str:
    """Select gait during step-off landing zone"""
    if airborne:
        if dp[1] < params.fall.land_dy_thresh and terrain_h < params.fall.edge_thresh:
            return "step_off_landing"
        else:
            return "step_off_flight"
    else:
        return "walk"


def gaitHole(p: np.ndarray, dp: np.ndarray, airborne: bool,
                  params: OCParams) -> str:
    """Select gait during hole jump"""
    if airborne:
        dist_to_land = (params.hole.x_trigger + params.hole.width) - p[0]
        if dist_to_land < params.hole.pre_land_dist and dp[1] < params.hole.pre_land_dy:
            return "gap_landing"
        else:
            return "holeFlight"
    else:
        return "walk"


def gs(N: int, dt: float, t: float, params_t: ScheduledParams,
                  gait: str, params: OCParams) -> np.ndarray:
    """Generate contact schedule [2 x N] over MPC prediction horizon"""
    
    if gait in ["walk", "walk_elevated"]:
        return ws(N, dt, t, params_t)
    
    elif gait in ["jump", "landing", "step_off_prep", "step_off_landing", 
                  "hole_prep", "gap_landing"]:
        return np.ones((2, N))
    
    elif gait in ["flight", "pre_landing", "step_off_flight", "holeFlight"]:
        return np.zeros((2, N))
    
    else:
        return ws(N, dt, t, params_t)


def ws(N: int, dt: float, t: float, 
                     params_t: ScheduledParams) -> np.ndarray:
    """Generate alternating left/right contact pattern"""
    period = 2 * params_t.stance_time
    sigma = np.zeros((2, N))
    
    for i in range(N):
        phase = (t + i * dt) % period
        if phase < params_t.stance_time:
            sigma[0, i] = 1  # Left foot stance
        else:
            sigma[1, i] = 1  # Right foot stance
    
    return sigma


def footVel(state: np.ndarray, params: OCParams) -> np.ndarray:
    """Compute foot velocities using leg Jacobian"""
    l = params.l
    
    theta = state[2]
    q = state[3:7]
    dtheta = state[9]
    dq = state[10:14]
    
    # Precompute trig terms
    s1 = np.sin(q[0] + theta)
    c1 = np.cos(q[0] + theta)
    s12 = np.sin(q[0] + q[1] + theta)
    c12 = np.cos(q[0] + q[1] + theta)
    s3 = np.sin(q[2] + theta)
    c3 = np.cos(q[2] + theta)
    s34 = np.sin(q[2] + q[3] + theta)
    c34 = np.cos(q[2] + q[3] + theta)
    
    # Jacobian
    J = np.array([
        [-l*(s1+s12), -l*(s1+s12), -l*s12, 0, 0],
        [l*(c1+c12), l*(c1+c12), l*c12, 0, 0],
        [-l*(s3+s34), 0, 0, -l*(s3+s34), -l*s34],
        [l*(c3+c34), 0, 0, l*(c3+c34), l*c34]
    ])
    
    dP_rel = J @ np.concatenate([[dtheta], dq])
    dP = dP_rel + np.array([state[7], state[8], state[7], state[8]])
    
    return dP


def lTime(y: float, dy: float, y_ground: float, g: float) -> float:
    """Solve ballistic equation for time to ground contact"""
    a = -0.5 * g
    b = dy
    c = y - y_ground
    disc = b**2 - 4*a*c
    
    if disc < 0:
        return 1.0
    
    t1 = (-b + np.sqrt(disc)) / (2*a)
    t2 = (-b - np.sqrt(disc)) / (2*a)
    
    # Take smallest positive root
    if t1 > 0 and t2 > 0:
        t_land = min(t1, t2)
    elif t1 > 0:
        t_land = t1
    elif t2 > 0:
        t_land = t2
    else:
        t_land = 1.0
    
    return np.clip(t_land, 0.01, 2.0)


def tuck(p: np.ndarray, theta: float, dx: float, dy: float) -> Tuple[float, float]:
    """Transform tuck offset from body frame to world frame"""
    Px = p[0] + dx * np.cos(theta) - dy * np.sin(theta)
    Py = p[1] + dx * np.sin(theta) + dy * np.cos(theta)
    return Px, Py


def flightLCon(state: np.ndarray, P_foot: np.ndarray, dP_foot: np.ndarray,
                       gait: str, terrain_h: float, params: OCParams) -> np.ndarray:
    """PD control for leg positioning during flight phases"""
    
    p = state[0:2]
    dp = state[7:9]
    theta = state[2]
    
    if gait == "flight":
        # Box jump: tuck feet under body
        Px_d, Py_d = tuck(p, theta, 0.0, -params.jump.tuck_height)
        Px_dot_d = dp[0]
        Py_dot_d = dp[1]
    
    elif gait == "step_off_flight":
        Px_d, Py_d, Px_dot_d, Py_dot_d = trajFF(p, dp, theta, params)
    
    elif gait == "pre_landing":
        landing_h = hTerr(p[0] + params.preland.lookahead, params)
        Px_d = p[0] + params.preland.foot_offset_x
        Py_d = landing_h + params.preland.foot_offset_y
        Px_dot_d = dp[0]
        Py_dot_d = params.preland.descent_vel
    
    elif gait == "holeFlight":
        Px_d, Py_d, Px_dot_d, Py_dot_d = trajHF(p, dp, theta, params)
    
    else:
        Px_d = p[0]
        Py_d = terrain_h + 0.3
        Px_dot_d = 0
        Py_dot_d = 0
    
    # Apply gait-specific gain scaling
    Kp, Kd = fGains(gait, params)
    F = Kp @ (np.array([Px_d, Py_d]) - P_foot) + Kd @ (np.array([Px_dot_d, Py_dot_d]) - dP_foot)
    
    return F


def trajFF(p: np.ndarray, dp: np.ndarray, theta: float,
                        params: OCParams) -> Tuple[float, float, float, float]:
    """Generate foot trajectory during step-off flight"""
    
    # Default: tuck position
    Px_d, Py_d = tuck(p, theta, params.fall.tuck_x, -params.fall.tuck_y)
    
    # Estimate time to landing
    landing_h = hTerr(p[0] + dp[0] * 0.2, params)
    t_land = lTime(p[1], dp[1], landing_h, params.g)
    
    # Extend feet when close to landing
    if t_land < params.fall.extend_time:
        Py_d = landing_h + params.fall.extend_h
        Px_d = p[0] + dp[0] * t_land
    
    return Px_d, Py_d, dp[0], dp[1]


def trajHF(p: np.ndarray, dp: np.ndarray, theta: float,
                    params: OCParams) -> Tuple[float, float, float, float]:
    """Generate foot trajectory during gap flight"""
    
    # Default: tuck position
    Px_d, Py_d = tuck(p, theta, 0.0, -params.hole.tuck_y)
    
    # Estimate time to landing
    landing_h = hTerr(p[0] + dp[0] * 0.2, params)
    t_land = lTime(p[1], dp[1], landing_h, params.g)
    
    # Extend feet when close to landing
    if t_land < params.hole.extend_time:
        Py_d = landing_h + params.hole.extend_h
        Px_d = p[0] + dp[0] * t_land
    
    return Px_d, Py_d, dp[0], dp[1]


def fGains(gait: str, params: OCParams) -> Tuple[np.ndarray, np.ndarray]:
    """Get scaled PD gains for flight phase control"""
    
    scale_map = {
        "flight": (params.gains.flight_Kp, params.gains.flight_Kd),
        "step_off_flight": (params.gains.fall_Kp, params.gains.fall_Kd),
        "holeFlight": (params.gains.hole_Kp, params.gains.hole_Kd),
        "pre_landing": (params.gains.landing_Kp, params.gains.landing_Kd)
    }
    
    scale_Kp, scale_Kd = scale_map.get(gait, (1.0, 1.0))
    
    return params.Kp * scale_Kp, params.Kd * scale_Kd


def mpcTarget(state: np.ndarray, t: float, params_t: ScheduledParams,
                gait: str, terrain_h: float, params: OCParams) -> np.ndarray:
    """Compute desired body [x; y] position for current gait phase"""
    
    p = state[0:2]
    dp = state[7:9]
    
    # Use simple velocity-based position tracking (like MATLAB)
    # x_des = current_x + velocity * lookahead_time
    lookahead = 0.3  # seconds - how far ahead to track
    x_des = p[0] + params_t.dx_des * lookahead
    
    gait_y_map = {
        "walk": params_t.ypos_des + terrain_h,
        "jump": params.jump.crouch_h,
        "flight": p[1],  # Maintain current height during flight
        "step_off_flight": p[1],
        "holeFlight": p[1],
        "pre_landing": hTerr(p[0] + params.preland.lookahead, params) + params_t.ypos_des,
        "landing": params_t.ypos_des + terrain_h,
        "walk_elevated": params_t.ypos_des + terrain_h,
        "step_off_prep": terrain_h + params.fall.crouch_h,
        "step_off_landing": hTerr(p[0] + params.preland.lookahead, params) + params_t.ypos_des,
        "hole_prep": terrain_h + params.hole.crouch_h,
        "gap_landing": hTerr(p[0] + params.fall.lookahead, params) + params_t.ypos_des
    }
    
    y_des = gait_y_map.get(gait, params_t.ypos_des + terrain_h)
    
    return np.array([x_des, y_des])


def swing(state: np.ndarray, P_sw: np.ndarray, dP_sw: np.ndarray,
                  P_st: np.ndarray, t: float, params_t: ScheduledParams,
                  gait: str, terrain_h: float, 
                  params: OCParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PD control for swing leg trajectory tracking"""
    
    p = state[0:2]
    dp = state[7:9]
    
    # Compute swing phase timing
    period = 2 * params_t.stance_time
    t_cycle = t % period
    ts = min(t_cycle % params_t.stance_time, params_t.stance_time)
    
    if gait in ["jump", "step_off_prep", "hole_prep"]:
        # Double stance: no swing leg motion
        F = np.zeros(2)
        pCG_des = mpcTarget(state, t, params_t, gait, terrain_h, params)
        foot_des = np.zeros(4)
        return F, pCG_des, foot_des
    
    if gait in ["walk", "walk_elevated"]:
        Px_d, Py_d, Py_dot_d = trajNS(p, dp, ts, params_t, terrain_h, params)
    
    elif gait == "landing":
        Px_d, Py_d, Py_dot_d = trajLS(p, ts, params_t, terrain_h, 1.5, params)
    
    elif gait == "step_off_landing":
        landing_h = hTerr(p[0] + params.fall.lookahead, params)
        Px_d, Py_d, Py_dot_d = trajLS(p, ts, params_t, landing_h, 2.0, params)
    
    elif gait == "gap_landing":
        landing_h = hTerr(p[0] + params.fall.lookahead, params)
        Px_d, Py_d, Py_dot_d = trajLS(p, ts, params_t, landing_h, 1.5, params)
    
    else:
        Px_d, Py_d, Py_dot_d = trajNS(p, dp, ts, params_t, terrain_h, params)
    
    Px_dot_d = 0.0
    pCG_des = np.array([(P_sw[0] + P_st[0])/2, params_t.ypos_des + terrain_h])
    
    # PD control
    F = params.Kp @ (np.array([Px_d, Py_d]) - P_sw) + params.Kd @ (np.array([Px_dot_d, Py_dot_d]) - dP_sw)
    foot_des = np.array([Px_d, Py_d, Px_dot_d, Py_dot_d])
    
    return F, pCG_des, foot_des


def trajNS(p: np.ndarray, dp: np.ndarray, ts: float,
                      params_t: ScheduledParams, terrain_h: float,
                      params: OCParams) -> Tuple[float, float, float]:
    """Generate sinusoidal swing foot trajectory for walking"""
    
    T = params_t.stance_time
    h = params_t.swing_height
    
    Px_d = p[0] + 0.5*T*params_t.dx_des - params_t.Kv*(params_t.dx_des - dp[0])
    Py_d = terrain_h + h * np.sin(np.pi*ts/T)
    Py_dot_d = (h * np.pi/T) * np.cos(np.pi*ts/T)
    
    return Px_d, Py_d, Py_dot_d


def trajLS(p: np.ndarray, ts: float, params_t: ScheduledParams,
                       terrain_h: float, h_scale: float,
                       params: OCParams) -> Tuple[float, float, float]:
    """Generate swing trajectory with scaled height for landing phases"""
    
    T = params_t.stance_time
    h = params_t.swing_height * h_scale
    
    Px_d = p[0] + 0.5*T*params_t.dx_des
    Py_d = terrain_h + h * np.sin(np.pi*ts/T)
    Py_dot_d = (h * np.pi/T) * np.cos(np.pi*ts/T)
    
    return Px_d, Py_d, Py_dot_d


def f2t(state: np.ndarray, F: np.ndarray, gait: str,
                    params: OCParams) -> np.ndarray:
    """Map foot forces to joint torques via Jacobian transpose"""
    
    l = params.l
    theta = state[2]
    q = state[3:7]
    dq = state[10:14]
    
    # Leg 1 Jacobian
    J1 = np.array([
        [l*np.cos(q[0]+theta) + l*np.cos(q[0]+q[1]+theta), l*np.cos(q[0]+q[1]+theta)],
        [l*np.sin(q[0]+theta) + l*np.sin(q[0]+q[1]+theta), l*np.sin(q[0]+q[1]+theta)]
    ])
    
    # Leg 2 Jacobian
    J2 = np.array([
        [l*np.cos(q[2]+theta) + l*np.cos(q[2]+q[3]+theta), l*np.cos(q[2]+q[3]+theta)],
        [l*np.sin(q[2]+theta) + l*np.sin(q[2]+q[3]+theta), l*np.sin(q[2]+q[3]+theta)]
    ])
    
    # Block diagonal Jacobian
    J = np.block([[J1, np.zeros((2,2))], [np.zeros((2,2)), J2]])
    tau = -J.T @ F
    
    # Velocity limiting during flight phases
    flight_gaits = ["flight", "pre_landing", "step_off_flight", "holeFlight"]
    if gait not in flight_gaits:
        return tau


def contDyn(state: np.ndarray, P_foot: np.ndarray,
                        params: OCParams) -> Tuple[np.ndarray, np.ndarray]:
    """Build continuous-time SRBD dynamics matrices"""
    
    A = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])
    
    Pcg = state[0:2]
    r1 = P_foot[0:2] - Pcg  # Vector from CoM to foot 1
    r2 = P_foot[2:4] - Pcg  # Vector from CoM to foot 2
    
    B = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1/params.m, 0, 1/params.m, 0],
        [0, 1/params.m, 0, 1/params.m],
        [-r1[1]/params.I, r1[0]/params.I, -r2[1]/params.I, r2[0]/params.I],
        [0, 0, 0, 0]
    ])
    
    return A, B


def discretize(A: np.ndarray, B: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Euler discretization of continuous dynamics"""
    n = A.shape[0]
    Ad = np.eye(n) + A * dt
    Bd = B * dt
    return Ad, Bd


def cm(A: np.ndarray, B: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build condensed prediction matrices for QP formulation"""
    
    n = A.shape[0]
    m = B.shape[1]
    
    Aqp = np.zeros((N*n, n))
    Bqp = np.zeros((N*n, N*m))
    
    A_pow = np.eye(n)
    for k in range(N):
        A_pow = A_pow @ A
        Aqp[k*n:(k+1)*n, :] = A_pow
    
    for i in range(N):
        elem = np.linalg.matrix_power(A, i) @ B
        for j in range(N - i):
            row_start = (i+j)*n
            row_end = (i+j+1)*n
            col_start = j*m
            col_end = (j+1)*m
            Bqp[row_start:row_end, col_start:col_end] = elem
    
    return Aqp, Bqp


def srbdState(state: np.ndarray, params: OCParams) -> np.ndarray:
    """Extract single rigid body dynamics state [x, y, θ, dx, dy, dθ, g]"""
    return np.concatenate([state[0:3], state[7:10], [params.g]])


def ref(X: np.ndarray, pCG_des: np.ndarray, params_t: ScheduledParams,
                    gait: str, state: np.ndarray, params: OCParams) -> np.ndarray:
    """Build reference trajectory for MPC horizon"""
    
    N = params.N
    dt = params.dt
    dp = state[7:9]
    p = state[0:2]
    
    # Set velocity targets based on gait
    if gait == "jump":
        dx_des = max(params_t.dx_des, dp[0])
        dy_des = params.jump.thrust_vel
    
    elif gait == "step_off_prep":
        dx_des = params_t.dx_des * params.fall.dx_scale
        dy_des = params.fall.thrust_vel
    
    elif gait == "hole_prep":
        dx_des = params_t.dx_des * params.hole.dx_scale
        dy_des = params.hole.thrust_vel
    
    elif gait == "landing":
        dx_des = params_t.dx_des
        dy_des = params.preland.descent_vel
    
    elif gait == "step_off_landing":
        dx_des = params_t.dx_des
        dy_des = params.fall.land_dy
    
    elif gait == "gap_landing":
        dx_des = params_t.dx_des
        dy_des = params.hole.land_dy
    
    elif gait in ["flight", "pre_landing", "step_off_flight", "holeFlight"]:
        # Ballistic trajectory
        return ball(X, dp, N, dt, params)
    
    else:
        dx_des = params_t.dx_des
        dy_des = 0.0
    
    # Build reference trajectory over horizon
    X_ref = np.zeros((7, N))
    
    for k in range(N):
        t_pred = k * dt
        
        # Position: integrate velocity from desired target
        X_ref[0, k] = pCG_des[0] + dx_des * t_pred  # x
        X_ref[1, k] = pCG_des[1] + dy_des * t_pred  # y (usually near constant)
        X_ref[2, k] = params.theta_des  # theta
        
        # Velocity: constant desired velocities
        X_ref[3, k] = dx_des  # dx
        X_ref[4, k] = dy_des  # dy
        X_ref[5, k] = 0.0     # dtheta
        X_ref[6, k] = params.g  # g
    
    return X_ref


def ball(X: np.ndarray, dp: np.ndarray, N: int, dt: float,
                        params: OCParams) -> np.ndarray:
    """Generate ballistic (gravity-only) trajectory reference"""
    
    X_ref = np.tile(X[:, np.newaxis], (1, N))
    
    for k in range(N):
        t_pred = k * dt
        X_ref[0, k] = X[0] + dp[0] * t_pred  # x
        X_ref[1, k] = X[1] + dp[1] * t_pred - 0.5 * params.g * t_pred**2  # y
        X_ref[3, k] = dp[0]  # dx
        X_ref[4, k] = dp[1] - params.g * t_pred  # dy
    
    return X_ref


def changeWeights(Q_base: np.ndarray, gait: str, params: OCParams) -> np.ndarray:
    """Scale MPC cost weights based on gait phase"""
    
    Q = Q_base.copy()
    
    if gait == "jump":
        Q[1, 1] *= params.mpc.jump_y
        Q[4, 4] *= params.mpc.jump_dy
    
    elif gait == "landing":
        Q[2, 2] *= params.mpc.landing_theta
        Q[5, 5] *= params.mpc.landing_dtheta
    
    return Q


def fLim(gait: str, params: OCParams) -> float:
    """Get maximum vertical force based on gait phase"""
    
    F_max = params.jump.max_force
    
    force_map = {
        "jump": F_max,
        "landing": F_max * 0.8,
        "step_off_prep": F_max * 0.6,
        "step_off_landing": F_max * 0.7,
        "hole_prep": F_max * 0.7,
        "gap_landing": F_max * 0.7
    }
    
    return force_map.get(gait, params.Fy_lim[1])


def constraints(N: int, contact: np.ndarray, gait: str,
                          params: OCParams) -> Tuple[np.ndarray, np.ndarray]:
    """Build friction cone and force limit constraints"""
    
    m = 4  # num inputs
    mu = params.mu
    Fy_max = fLim(gait, params)
    Fy_min = params.Fy_lim[0]
    
    n_rows = 8  # 4 friction + 4 force limits per timestep
    A_ineq = np.zeros((N * n_rows, N * m))
    B_ineq = np.zeros(N * n_rows)
    
    for k in range(N):
        u_idx_start = k*m
        u_idx_end = (k+1)*m
        row_base = k * n_rows
        
        # Friction cone: |Fx| <= mu * Fy
        A_ineq[row_base + 0, u_idx_start:u_idx_end] = [1, -mu, 0, 0]
        A_ineq[row_base + 1, u_idx_start:u_idx_end] = [-1, -mu, 0, 0]
        A_ineq[row_base + 2, u_idx_start:u_idx_end] = [0, 0, 1, -mu]
        A_ineq[row_base + 3, u_idx_start:u_idx_end] = [0, 0, -1, -mu]
        
        # Force limits: Fy_min <= Fy <= Fy_max
        A_ineq[row_base + 4, u_idx_start:u_idx_end] = [0, 1, 0, 0]
        A_ineq[row_base + 5, u_idx_start:u_idx_end] = [0, -1, 0, 0]
        A_ineq[row_base + 6, u_idx_start:u_idx_end] = [0, 0, 0, 1]
        A_ineq[row_base + 7, u_idx_start:u_idx_end] = [0, 0, 0, -1]
        
        con_k = contact[:, min(k, contact.shape[1]-1)]
        B_ineq[row_base + 4:row_base + 8] = [
            Fy_max * con_k[0],
            -Fy_min * con_k[0],
            Fy_max * con_k[1],
            -Fy_min * con_k[1]
        ]
    
    return A_ineq, B_ineq


def solveMPC(state: np.ndarray, P_foot: np.ndarray, contact: np.ndarray,
              pCG_des: np.ndarray, params_t: ScheduledParams, gait: str,
              params: OCParams) -> np.ndarray:
    """Solve QP for optimal ground reaction forces using OSQP"""
    
    if not np.any(contact[:, 0]):
        return np.zeros(4)
    
    try:
        import osqp
        import scipy.sparse as sp
        
        # Build dynamics matrices
        A, B = contDyn(state, P_foot, params)
        Ad, Bd = discretize(A, B, params.dt)
        
        # Get current state and reference trajectory
        X = srbdState(state, params)
        X_ref = ref(X, pCG_des, params_t, gait, state, params)
        y = X_ref.T.flatten()  # Column-major flatten
        
        # Build condensed QP matrices
        Aqp, Bqp = cm(Ad, Bd, params.N)
        
        # Apply gait-specific weight scaling
        Q = changeWeights(params.Q, gait, params)
        L = np.kron(np.eye(params.N), Q)
        K = np.kron(np.eye(params.N), params.R)
        
        # Formulate QP: min 0.5*u'*H*u + f'*u
        H = 2.0 * (Bqp.T @ L @ Bqp + K)
        H = 0.5 * (H + H.T)  # Ensure symmetry
        f = 2.0 * Bqp.T @ L @ (Aqp @ X - y)
        
        # Build inequality constraints: A_ineq * u <= b_ineq
        A_ineq, b_ineq = constraints(params.N, contact, gait, params)
        
        # Convert to OSQP format: l <= A * u <= u
        P_csc = sp.csc_matrix(H)
        q = f
        A_csc = sp.csc_matrix(A_ineq)
        l = -np.inf * np.ones_like(b_ineq)
        u = b_ineq
        
        # Setup and solve
        solver = osqp.OSQP()
        solver.setup(
            P=P_csc,
            q=q,
            A=A_csc,
            l=l,
            u=u,
            verbose=False,
            eps_abs=1e-3,
            eps_rel=1e-3,
            max_iter=2000,
            warm_start=True,
            polish=False,
            adaptive_rho=True
        )
        
        result = solver.solve()
        
        if result.info.status_val == osqp.constant("OSQP_SOLVED"):
            return result.x[0:4]
        else:
            # Fallback: return zero forces
            return np.zeros(4)
            
    except Exception as e:
        print(f"MPC solve error: {e}")
        return np.zeros(4)


def occontroller(state: np.ndarray, t: float,
                               params: OCParams):
    """
    Wrapper that lets the same controller run for a course in +x or -x
    without touching theta (pitch) or joint sign conventions.

    params.forward_sign = +1  -> course in +x
    params.forward_sign = -1  -> course in -x
    """
    s = float(getattr(params, "forward_sign", +1.0))

    # ---------- map WORLD -> CANONICAL controller frame ----------
    # Only flip x and dx. Do NOT touch theta (pitch) or z/dz.
    st = state.copy()
    st[0] *= s   # x
    st[7] *= s   # dx

    # FK and foot velocities in canonical frame
    FK = fK(st[0:3], st[3:7], params.l, params.a)
    P_foot = np.concatenate([FK["p2"], FK["p4"]])  # [x1,z1,x2,z2]
    dP_foot = footVel(st, params)         # [dx1,dz1,dx2,dz2]

    P1, P2 = P_foot[0:2], P_foot[2:4]
    dP1, dP2 = dP_foot[0:2], dP_foot[2:4]

    p = st[0:2]
    dp = st[7:9]

    params_t = ScheduledParams(params, t)

    # Terrain and gait logic stay canonical (they see +x-forward)
    terrain_h = hTerr(p[0], params)
    gait = gaitCheck(p, dp, P1, P2, terrain_h, params)
    contact = gs(params.N, params.dt, t, params_t, gait, params)

    foot_des = np.full(4, np.nan)

    # ---------- same control logic as before ----------
    if np.all(contact[:, 0]):
        pCG_des = mpcTarget(st, t, params_t, gait, terrain_h, params)
        F = solveMPC(st, P_foot, contact, pCG_des, params_t, gait, params)
        ctrl = f2t(st, F, gait, params)

    elif contact[0, 0] and not contact[1, 0]:
        F_sw, pCG_des, foot_des = swing(st, P2, dP2, P1, t, params_t, gait, terrain_h, params)
        F = solveMPC(st, P_foot, contact, pCG_des, params_t, gait, params)
        ctrl = f2t(st, np.concatenate([F[0:2], -F_sw]), gait, params)

    elif contact[1, 0] and not contact[0, 0]:
        F_sw, pCG_des, foot_des = swing(st, P1, dP1, P2, t, params_t, gait, terrain_h, params)
        F = solveMPC(st, P_foot, contact, pCG_des, params_t, gait, params)
        ctrl = f2t(st, np.concatenate([-F_sw, F[2:4]]), gait, params)

    else:
        F1 = flightLCon(st, P1, dP1, gait, terrain_h, params)
        F2 = flightLCon(st, P2, dP2, gait, terrain_h, params)
        ctrl = f2t(st, np.concatenate([-F1, -F2]), gait, params)
        pCG_des = np.full(2, np.nan)

    # ---------- map outputs back to WORLD frame (optional but nice) ----------
    foot_out = np.concatenate([P_foot, dP_foot, foot_des]).copy()

    # flip x components back: P_foot x's are indices 0 and 2
    foot_out[0] *= s
    foot_out[2] *= s
    # dP_foot x's are indices 4 and 6 (since foot_out = [P_foot(4), dP_foot(4), foot_des(4)])
    foot_out[4] *= s
    foot_out[6] *= s
    # foot_des x is foot_out[8] if not nan
    if not np.isnan(foot_out[8]):
        foot_out[8] *= s

    traj_des = pCG_des.copy()
    if traj_des.shape == (2,) and not np.isnan(traj_des[0]):
        traj_des[0] *= s

    return ctrl, foot_out, traj_des
