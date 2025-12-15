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
    def __init__(self, xml):
        
        self.m, self.d = load_model(xml)
        self.dt = self.m.opt.timestep
        
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
        
           
        reset(self.m, self.d, "init")
        self.step(np.array([0, 0, 0, 0]))
        

    
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
        
        u_original = u.copy()
        
        u[[0, 2]] = np.clip(u[[0, 2]], -self.thigh_tau_limit, self.thigh_tau_limit)
        u[[1, 3]] = np.clip(u[[1, 3]], -self.calf_tau_limit, self.calf_tau_limit)
        
        print(f"Torque: clipped from {u_original} to {u}")
        
        return u



    def step(self, u):
        
        self.d.ctrl = self.set_tau_limits(u)
        self.check_limits()