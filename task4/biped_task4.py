import mujoco
import numpy as np
# np.set_printoptions(precision=3)
np.set_printoptions(
    precision=3,
    suppress=True,
    formatter={
        'float_kind': lambda x: "0" if abs(x) < 1e-12 else f"{x:.3f}"
    }
)
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
solvers.options['show_progress'] = False
from utils.utils import *

class Biped():
    def __init__(self, xml, ctrl):
        
        self.m, self.d = load_model(xml)
        self.dt = self.m.opt.timestep
        
        # self.M = get_body_mass(m, "torso") + get_body_mass(m, "l_thigh") + get_body_mass(m, "l_calf") + get_body_mass(m, "r_thigh") + get_body_mass(m, "r_calf") 
        self.M = get_M(self.m)
        self.Izz = get_body_inertia(self.m, "torso")[1]
        self.g = np.abs(get_gravity(self.m))
        
        self.height = 0.69 # m
        
        self.thigh_qpos_min = -120.0*(np.pi/180) # rad
        self.thigh_qpos_max = 30.0*(np.pi/180) # rad
        self.thigh_qvel_limit = 30.0 # rad/s
        self.thigh_tau_limit = 30.0 # Nm
        
        self.calf_qpos_min = 0.0 # rad
        self.calf_qpos_max = 160.0*(np.pi/180) # rad
        self.calf_qvel_limit = 15.0 # rad/s
        self.calf_tau_limit= 60.0 # Nm
        
        self.l_foot_id = get_geom_id(self.m, "l_foot")
        self.r_foot_id = get_geom_id(self.m, "r_foot")
        self.floor_id = get_geom_id(self.m, "floor")
        
        self.foot_contact_list = [0, 0]
        self.feet_contact = False
        
        self.controllers = {
            "qp_stand": self.qp_stand,
            "pd_step": self.pd_step,
            "qp_pd_climb": self.qp_pd_climb,
        }
        self.ctrl = self.controllers[ctrl]
        self.u = np.array([0, 0, 0, 0])
        

        self.thigh_qpos_indices = np.array([7, 9])
        self.calf_qpos_indices = np.array([8, 10])
        
        self.thigh_qvel_indices = np.array([6, 8])
        self.calf_qvel_indices = np.array([7, 9])
        
        self.thigh_ctrl_indices = np.array([0, 2])
        self.calf_ctrl_indices = np.array([1, 3])
        

        self.l_leg_qpos_indices = np.array([7, 8])
        self.r_leg_qpos_indices = np.array([9, 10])
        self.leg_qpos_indices = np.concatenate([self.l_leg_qpos_indices, 
                                                self.r_leg_qpos_indices])
        
        self.l_leg_qvel_indices = np.array([6, 7])
        self.r_leg_qvel_indices = np.array([8, 9])
        self.leg_qvel_indices = np.concatenate([self.l_leg_qvel_indices, 
                                                self.r_leg_qvel_indices])
        
        self.l_leg_ctrl_indices = np.array([0, 1])
        self.r_leg_ctrl_indices = np.array([2, 3])
        
        self.gait_state = 2
        self.last_gait_state = 1
        self.swing_progress = 0
        self.gait_cycle_length = int(20 / self.dt)
        
        self.l_swing_p0 = None  # left foot pose at swing start
        self.r_swing_p0 = None  # right foot pose at swing start


        self.stride_length = 0.02      # step length
        self.stride_height = 0.02      # swing height
        
        self.Fy_min = 0
        self.Fy_max = 250
        self.mu = 0.7
        
        self.first_contact = False
        
        self.two_step = False
        
        reset(self.m, self.d, "init")
        self.step(self.u)
        
        
        
        


    def get_foot_contact(self) -> bool:
        
        self.foot_contact_list = [0, 0]
        
        for k in range(self.d.ncon):
            c = self.d.contact[k]
            g1 = c.geom1
            g2 = c.geom2
            
            for foot in (self.l_foot_id, self.r_foot_id):
                if (foot in (g1, g2)) and (self.floor_id in (g1, g2)):
                    if foot == self.l_foot_id:
                        self.foot_contact_list[0] = foot
                    elif foot == self.r_foot_id:
                        self.foot_contact_list[1] = foot
                        
        self.feet_contact = all(self.foot_contact_list)
                        

    
    def check_limits(self):
        if any(self.d.qpos[self.thigh_qpos_indices] < self.thigh_qpos_min):
            raise ValueError(f"Thigh qpos under minimum limit: {self.d.qpos[self.thigh_qpos_indices]}")

        if any(self.d.qpos[self.thigh_qpos_indices] > self.thigh_qpos_max):
            raise ValueError(f"Thigh qpos over maximum limit: {self.d.qpos[self.thigh_qpos_indices]}")

        if any(self.d.qpos[self.calf_qpos_indices] < self.calf_qpos_min):
            raise ValueError(f"Calf qpos under minimum limit: {self.d.qpos[self.calf_qpos_indices]}")

        if any(self.d.qpos[self.calf_qpos_indices] > self.calf_qpos_max):
            raise ValueError(f"Calf qpos over maximum limit: {self.d.qpos[self.calf_qpos_indices]}")

        if any(np.abs(self.d.qvel[self.thigh_qvel_indices]) > self.thigh_qvel_limit):
            raise ValueError(f"Thigh qvel exceeds limit: {self.d.qvel[self.thigh_qvel_indices]}")

        if any(np.abs(self.d.qvel[self.calf_qvel_indices]) > self.calf_qvel_limit):
            raise ValueError(f"Calf qvel exceeds limit: {self.d.qvel[self.calf_qvel_indices]}")

        if any(np.abs(self.d.ctrl[self.thigh_ctrl_indices]) > self.thigh_tau_limit):
            raise ValueError(f"Thigh ctrl exceeds limit: {self.d.ctrl[self.thigh_ctrl_indices]}")

        if any(np.abs(self.d.ctrl[self.calf_ctrl_indices]) > self.calf_tau_limit):
            raise ValueError(f"Calf ctrl exceeds limit: {self.d.ctrl[self.calf_ctrl_indices]}")



            

    def set_tau_limits(self, u):
        
        u[[0, 2]] = np.clip(u[[0, 2]], -self.thigh_tau_limit, self.thigh_tau_limit)
        u[[1, 3]] = np.clip(u[[1, 3]], -self.calf_tau_limit, self.calf_tau_limit)
        
        return u



    def step(self, u):
        self.get_foot_contact()
        
        if not self.first_contact and self.feet_contact:
            self.first_contact = True
        
        self.u = self.set_tau_limits(u)
        
        self.d.ctrl = self.u
        self.check_limits()




                        
    
    def qp_stand(self, xf):
        
        if not self.feet_contact: return np.zeros(self.m.nu)
        
        alpha = 0.001
        Q = np.array([
            [1,    0,  0.1],
            [0,    1,    0],
            [0.1,  0,    5],
        ])
        
        xc = self.d.qpos[0]
        yc = self.d.qpos[2]
        theta_c_quat = self.d.qpos[3:7]
        theta_c = R.from_quat(theta_c_quat, scalar_first=True).as_euler('zyx')[1]
        
        xc_dot = self.d.qvel[0]
        yc_dot = self.d.qvel[2]
        theta_c_dot = self.d.qvel[4]
        
        xc_des, yc_des, theta_c_des, xc_dot_des, yc_dot_des, theta_c_dot_des = xf
               
        Kp_x = 5
        Kd_x = 10
        Kp_y = 5
        Kd_y = 10
        Kp_theta = 0.5 
        Kd_theta = 0.5 

        x_ddot_des = Kp_x * (xc_des - xc) + Kd_x * (xc_dot_des - xc_dot)
        y_ddot_des = Kp_y * (yc_des - yc) + Kd_y * (yc_dot_des - yc_dot)
        theta_ddot_des = Kp_theta * (theta_c_des - theta_c) + Kd_theta * (theta_c_dot_des - theta_c_dot)
        
        PF1, PF2 = get_feet_xpos(self.m, self.d)[:, :]
        PF1x, PF1y = PF1[[0, 2]]
        PF2x, PF2y = PF2[[0, 2]]
        
        A = np.array([
            [1,       0,       1,       0      ],
            [0,       1,       0,       1      ],
            [yc-PF1y, PF1x-xc, yc-PF2y, PF2x-xc]
        ])
        
        b = np.array([
            [self.M * x_ddot_des],
            [self.M * (y_ddot_des + self.g)],
            [self.Izz * theta_ddot_des],
        ])
        
        n = A.shape[1]
        
        P = 2 * (A.T @ Q @ A + alpha * np.eye(n))
        q = -2 * A.T @ Q @ b        
        G = np.array([
            [0,         1,   0,        0],
            [0,         0,   0,        1],
            [0,        -1,   0,        0],
            [0,         0,   0,       -1],
            [1,  -self.mu,   0,        0],
            [0,         0,   1, -self.mu],
            [-1, -self.mu,   0,        0],
            [0,         0,  -1, -self.mu]
        ])
        h = np.array([
            [self.Fy_max],
            [self.Fy_max],
            [-self.Fy_min],
            [-self.Fy_min],
            [0],
            [0],
            [0],
            [0]
        ])
        
        P_cvx = matrix(P.astype(float))
        q_cvx = matrix(q.astype(float))
        G_cvx = matrix(G.astype(float))
        h_cvx = matrix(h.astype(float))
        
        sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
        F_GRF_star = np.array([sol['x']])[0]
        
        F_feet = -F_GRF_star
        
        F_foot_l = np.vstack([ F_feet[0], 0, F_feet[1] ])
        F_foot_r = np.vstack([ F_feet[2], 0, F_feet[3] ])

        jacp_l = np.zeros((3, self.m.nv))
        mujoco.mj_jacGeom(self.m, self.d, jacp_l, None, self.l_foot_id)

        jacp_r = np.zeros((3, self.m.nv))
        mujoco.mj_jacGeom(self.m, self.d, jacp_r, None, self.r_foot_id)
        
        tau_l_full = jacp_l.T @ F_foot_l
        tau_r_full = jacp_r.T @ F_foot_r
        
        tau_l = tau_l_full[-4:]
        tau_r = tau_r_full[-4:]
        
        tau = (tau_l + tau_r).flatten()
        
        return tau
  

    def qp_step(self, xf, stance):
        """gpt"""
        # if not self.feet_contact: return np.zeros(self.m.nu)

        alpha = 0.0001
        Q = np.array([
            [10,    0,  0.1],
            [0,    1,    0],
            [0.1,  0,    10],
        ])

        # current centroidal state
        xc = self.d.qpos[0]
        yc = self.d.qpos[2]
        theta_c_quat = self.d.qpos[3:7]
        theta_c = R.from_quat(theta_c_quat, scalar_first=True).as_euler('zyx')[1]

        xc_dot = self.d.qvel[0]
        yc_dot = self.d.qvel[2]
        theta_c_dot = self.d.qvel[4]

        # desired centroidal state (passed in)
        xc_des, yc_des, theta_c_des, xc_dot_des, yc_dot_des, theta_c_dot_des = xf

        # moderate gains, not insane stiff
        Kp_x = 5.0
        Kd_x = 3.0
        Kp_y = 5.0
        Kd_y = 3.0
        Kp_theta = 0.5
        Kd_theta = 0.5

        x_ddot_des = Kp_x * (xc_des - xc) + Kd_x * (xc_dot_des - xc_dot)
        y_ddot_des = Kp_y * (yc_des - yc) + Kd_y * (yc_dot_des - yc_dot)
        theta_ddot_des = Kp_theta * (theta_c_des - theta_c) + Kd_theta * (theta_c_dot_des - theta_c_dot)

        # foot positions
        PF1, PF2 = get_feet_xpos(self.m, self.d)[:, :]
        PF1x, PF1y = PF1[[0, 2]]
        PF2x, PF2y = PF2[[0, 2]]

        # centroidal dynamics matrix
        A = np.array([
            [1,       0,       1,       0      ],
            [0,       1,       0,       1      ],
            [yc-PF1y, PF1x-xc, yc-PF2y, PF2x-xc]
        ])

        b = np.array([
            [self.M * x_ddot_des],
            [self.M * (y_ddot_des + self.g)],
            [self.Izz * theta_ddot_des],
        ])

        n = A.shape[1]

        P = 2 * (A.T @ Q @ A + alpha * np.eye(n))
        q = -2 * (A.T @ Q @ b)

        # normal force bounds per foot
        Fy_max_L = self.Fy_max
        Fy_max_R = self.Fy_max
        Fy_min_L = self.Fy_min
        Fy_min_R = self.Fy_min

        # encode single support by killing the swing foot inside the QP
        if stance == "left":
            # right foot must have zero normal force
            Fy_max_R = 0.0
            Fy_min_R = 0.0
        elif stance == "right":
            # left foot must have zero normal force
            Fy_max_L = 0.0
            Fy_min_L = 0.0
        # stance == "both" keeps both feet active

        # inequalities G F <= h
        # 1: Fy_L <= Fy_max_L
        # 2: Fy_R <= Fy_max_R
        # 3: -Fy_L <= -Fy_min_L  -> Fy_L >= Fy_min_L
        # 4: -Fy_R <= -Fy_min_R  -> Fy_R >= Fy_min_R
        # 5,6,7,8: friction cones |Fx| <= mu * Fy for each foot
        G = np.array([
            [0,         1,   0,        0],
            [0,         0,   0,        1],
            [0,        -1,   0,        0],
            [0,         0,   0,       -1],
            [1,  -self.mu,   0,        0],
            [0,         0,   1, -self.mu],
            [-1, -self.mu,   0,        0],
            [0,         0,  -1, -self.mu]
        ])

        h = np.array([
            [Fy_max_L],
            [Fy_max_R],
            [-Fy_min_L],
            [-Fy_min_R],
            [0],
            [0],
            [0],
            [0]
        ])

        P_cvx = matrix(P.astype(float))
        q_cvx = matrix(q.astype(float))
        G_cvx = matrix(G.astype(float))
        h_cvx = matrix(h.astype(float))

        sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
        F_GRF_star = np.array([sol['x']])[0]

        # sign convention, match your original code
        F_feet = -F_GRF_star

        # map planar forces back to 3D for each foot
        F_foot_l = np.vstack([F_feet[0], 0, F_feet[1]])
        F_foot_r = np.vstack([F_feet[2], 0, F_feet[3]])

        # foot Jacobians
        jacp_l = np.zeros((3, self.m.nv))
        mujoco.mj_jacGeom(self.m, self.d, jacp_l, None, self.l_foot_id)

        jacp_r = np.zeros((3, self.m.nv))
        mujoco.mj_jacGeom(self.m, self.d, jacp_r, None, self.r_foot_id)

        tau_l_full = jacp_l.T @ F_foot_l
        tau_r_full = jacp_r.T @ F_foot_r

        # last 4 entries are the actuated joints on each leg
        tau_l = tau_l_full[-4:]
        tau_r = tau_r_full[-4:]

        tau = (tau_l + tau_r).flatten()

        return tau
    
    def pd_step(self, leg, phi):
        """
        Swing-leg operational-space PD using Jacobian transpose.

        Parameters
        ----------
        leg : str
            "left" or "right".
        phi : float
            Phase in [0, 1] within the current swing cycle.
        """

        
        xpos_feet = get_feet_xpos(self.m, self.d)
        theta_c = R.from_quat(self.d.qpos[3:7], scalar_first=True).as_euler('zyx')[1]
        
        opt_sep = 0.3
        curr_sep = xpos_feet[1][0] - xpos_feet[0][0]

        # Cartesian stiffness and damping (N/m and N·s/m)
        # gain_left = 13 * (1-theta_c) * (curr_sep-opt_sep)
        # gain_right = 14 * (1+theta_c) * (1-(curr_sep-opt_sep))
        gain_left = 1 * (1+(curr_sep-opt_sep))
        gain_right = 10000 * (1-(curr_sep-opt_sep))
        
        kp_val_left = 5.0
        kd_val_left = 2.0
        Kp_left = np.diag([kp_val_left, kp_val_left, kp_val_left])
        Kd_left = np.diag([kd_val_left, kd_val_left, kd_val_left])
        
        kp_val_right = 10.0
        kd_val_right = 2.0
        Kp_right = np.diag([kp_val_right, kp_val_right, kp_val_right])
        Kd_right = np.diag([kd_val_right, kd_val_right, kd_val_right])

        # Desired swing-foot trajectory in world coordinates
        if leg == "left":
            p0 = self.l_swing_p0 if self.l_swing_p0 is not None else xpos_feet[0]
            p1 = p0 + gain_left*np.array([self.stride_length, 0.0, 0.0])

            x_des = 80 * ((1.0 - phi) * p0[0] + phi * p1[0])
            y_des = p0[1]
            z_des = (p0[2] + self.stride_height * 5 * np.sin(np.pi * phi) + 500)
            xpos_des = np.array([x_des, y_des, z_des])
            xpos_curr = xpos_feet[0]

            jacp_full = np.zeros((3, self.m.nv))
            mujoco.mj_jacGeom(self.m, self.d, jacp_full, None, self.l_foot_id)
            jac = jacp_full[:, self.l_leg_qvel_indices]
            qvel = self.d.qvel[self.l_leg_qvel_indices]

        elif leg == "right":
            p0 = self.r_swing_p0 if self.r_swing_p0 is not None else xpos_feet[1]
            p1 = p0 + gain_right*np.array([self.stride_length, 0.0, 0.0])

            x_des = ((1.0 - phi) * p0[0] + phi * p1[0])
            y_des = p0[1]
            z_des = p0[2] + self.stride_height * 10 * np.sin(np.pi * phi) + 5
            xpos_des = np.array([x_des, y_des, z_des])
            xpos_curr = xpos_feet[1]

            jacp_full = np.zeros((3, self.m.nv))
            mujoco.mj_jacGeom(self.m, self.d, jacp_full, None, self.r_foot_id)
            jac = jacp_full[:, self.r_leg_qvel_indices]
            qvel = self.d.qvel[self.r_leg_qvel_indices]

        # if leg == "right":
            # print(f"gait_state: {self.gait_state}, curr_sep: {curr_sep :.3f}, theta_c: {theta_c :.3f}, phi: {phi :.3f}, foot: {leg}, x_curr: {xpos_curr}, x_des: {xpos_des}, p0: {p0}, p1: {p1}")
        # print(f"gl: {gain_left :.3f}, gr: {gain_right :.3f}, sep: {curr_sep :.3f}")
    
        # Cartesian spring-damper: F = Kp * x_err - Kd * v_curr
        x_err = xpos_des - xpos_curr
        v_curr = jac @ qvel
        
        Kp = Kp_left if leg == "left" else Kp_right
        Kd = Kd_left if leg == "left" else Kd_right
        
        F_cart = Kp @ x_err - Kd @ v_curr

        # Map Cartesian force to joint torques: tau = J^T * F
        tau_leg = jac.T @ F_cart

        # Construct full actuator vector [l_thigh, l_calf, r_thigh, r_calf]
        if leg == "left":
            u_l = tau_leg
            u_r = np.zeros_like(tau_leg)
        else:
            u_l = np.zeros_like(tau_leg)
            u_r = tau_leg

        return np.concatenate([u_l, u_r])
    


    def qp_pd_climb(self):
        """
        QP + PD climbing controller with torso-angle safety.

        gait_state:
        0: right swing, left stance
        1: left swing, right stance
        2: double-support safety / recovery
        """
        
        self.d.qpos[[7, 9]] = np.clip(self.d.qpos[[7, 9]], self.thigh_qpos_min, self.thigh_qpos_max)
        self.d.qpos[[8, 10]] = np.clip(self.d.qpos[[8, 10]], self.calf_qpos_min, self.calf_qpos_max)

        self.d.qvel[[6, 8]] = np.clip(self.d.qvel[[6, 8]], -self.thigh_qvel_limit, self.thigh_qvel_limit)
        self.d.qvel[[7, 9]] = np.clip(self.d.qvel[[7, 9]], -self.calf_qvel_limit, self.calf_qvel_limit)

        if not self.first_contact: return np.zeros(self.m.nu)
        
        # 1. Torso pitch from base quaternion
        theta_c_quat = self.d.qpos[3:7]
        theta_c = R.from_quat(theta_c_quat, scalar_first=True).as_euler('zyx')[1]
        theta_c_vel = self.d.qvel[4]

        
        xcm = get_x_com(self.m, self.d)[0]
        # print(f"gait: {self.gait_state}, last_gait: {self.last_gait_state}, theta_c: {theta_c :.3f}, q4: {self.d.qpos[-1:][0]:.3f}, x_cm: {xcm :.3f}, t: {self.d.time :.3f}")
        # print(f"theta_c: {theta_c :.3f},  q4: {self.d.qpos[-1] :.3f}")
        # if xcm >= 1:
        #     print(f"finished, {self.d.time}")
        #     exit()
        # print(self.two_step)
        
        # planted = self.d.qpos[-1] >= 0.2 #0.4  # 0.468
        planted = self.d.qpos[-1] >= 0.2 if not self.two_step else self.d.qpos[-1] >= -0.1
        # print(planted)

        # 2. Safety hysteresis thresholds [rad]
        theta_enter = 0.1   # enter safety if |theta| > 0.20
        theta_exit  = 0.05   # leave safety only if |theta| < 0.10
        # theta_enter = 0.1   # enter safety if |theta| > 0.20
        # theta_exit  = 0.05   # leave safety only if |theta| < 0.10
        theta_vel_exit  = 0.05   # leave safety only if |theta| < 0.10

        # 3. Update safety mode
        if self.gait_state == 2:
            # Already in safety, leave only if back in tighter band
            # if np.abs(theta_c) < theta_exit and theta_c_vel < theta_vel_exit:
            #     self.gait_state = 0
            #     self.swing_progress = 0

            # if np.abs(theta_c) < theta_exit and planted: #and theta_c_vel < theta_vel_exit:
            if np.abs(theta_c) < theta_exit and ( (planted and self.last_gait_state == 1) or self.last_gait_state == 0 ):
                if self.last_gait_state == 1:
                    self.gait_state = 0
                elif self.last_gait_state == 0:
                    self.gait_state = 0 if not self.two_step or not planted else 1
                self.swing_progress = 0
        else:
            # Not in safety, enter if exceeded large limit
            if np.abs(theta_c) > theta_enter: #and theta_c_vel > theta_vel_exit:
                if self.swing_progress < 0.5 * self.gait_cycle_length:
                    self.last_gait_state = self.gait_state
                    self.gait_state = 2
                    self.swing_progress = 0

        # 4. Safety mode: stand in double support and keep torso upright
        if self.gait_state == 2:
            xpos_com = get_x_com(self.m, self.d)
            xc_des = xpos_com[0]
            yc_des = self.height - 0.2
            theta_c_des = 0
            xcdot_des = 0.01
            ycdot_des = 0
            theta_c_dot_des = 0
            xf_stand = np.array([xc_des, yc_des, theta_c_des, xcdot_des, ycdot_des, theta_c_dot_des])
            tau = self.qp_step(xf_stand, stance="both")
            # Debug if you want:
            # print(f"SAFETY theta_c: {theta_c:.3f}, gait_state: {self.gait_state:d}")
            return tau

        # 5. Normal gait state transitions
        if self.swing_progress >= self.gait_cycle_length:
            self.swing_progress = 0
            # if self.gait_state == 0:
            #     self.gait_state = 1   # switch to left swing
            # else:
            #     self.gait_state = 0   # switch to right swing
            if self.gait_state == 0:
                self.gait_state = 0 if not self.two_step or not planted else 1  # switch to left swing
            else:
                self.gait_state = 0   # switch to right swing

        # 6. Compute stance and swing torques
        xpos_feet = get_feet_xpos(self.m, self.d)
        
        xpos_com = get_x_com(self.m, self.d)
        xc_des = xpos_com[0] #+ self.stride_length
        yc_des = self.height - 0.5
        theta_c_des = 50
        xcdot_des = -0.5
        ycdot_des = -0.85
        theta_c_dot_des = 5000
        xf_step = np.array([xc_des, yc_des, theta_c_des, xcdot_des, ycdot_des, theta_c_dot_des])

        if self.gait_state == 0:
            # Right swing, left stance
            if self.swing_progress == 0:
                # Store right swing start pose
                self.r_swing_p0 = xpos_feet[1].copy()

            phase = self.swing_progress / self.gait_cycle_length
            self.swing_progress += 1

            # Stance leg via QP
            tau = self.qp_step(xf_step, stance="left")
            # Swing leg via PD foot tracking
            tau_swing = self.pd_step("right", phase)
            tau[self.r_leg_ctrl_indices] = tau_swing[self.r_leg_ctrl_indices]
            
            self.two_step = get_feet_xpos(self.m, self.d)[1][2] > 0.2

        elif self.gait_state == 1:
            # Left swing, right stance
            if self.swing_progress == 0:
                # Store left swing start pose
                self.l_swing_p0 = xpos_feet[0].copy()

            phase = self.swing_progress / self.gait_cycle_length
            self.swing_progress += 1

            # Stance leg via QP
            tau = self.qp_step(xf_step, stance="right")
            # Swing leg via PD foot tracking
            tau_swing = self.pd_step("left", phase)
            tau[self.l_leg_ctrl_indices] = tau_swing[self.l_leg_ctrl_indices]

        return tau