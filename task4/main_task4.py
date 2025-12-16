import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cvxopt import matrix, solvers
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
solvers.options['show_progress'] = False
from utils.utils import *
from biped_task4 import Biped


def plot(t, data_arr, ctrl_mode):
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(11, 9))
    
    plt.subplot(4,3,1)
    plt.plot(t, data_arr[:, 0], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('x(t) [m]')
    plt.grid()

    plt.subplot(4,3,2)
    plt.plot(t, data_arr[:, 1], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('y(t) [m]')    
    plt.grid()

    plt.subplot(4,3,3)
    plt.plot(t,  data_arr[:, 2], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('θ(t) [rad]')
    plt.grid()
    
    plt.subplot(4,3,4)
    plt.plot(t,  data_arr[:, 3], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('q1(t) [rad]')
    plt.grid()
    
    plt.subplot(4,3,5)
    plt.plot(t,  data_arr[:, 4], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('q2(t) [rad]')
    plt.grid()
    
    plt.subplot(4,3,6)
    plt.plot(t,  data_arr[:, 5], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('q3(t) [rad]')
    plt.grid()
    
    plt.subplot(4,3,7)
    plt.plot(t,  data_arr[:, 6], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('q4(t) [rad]')
    plt.grid()
    
    plt.subplot(4,3,8)
    plt.plot(t,  data_arr[:, 7], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('τ1(t) [Nm]')
    plt.grid()
    
    plt.subplot(4,3,9)
    plt.plot(t,  data_arr[:, 8], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('τ2(t) [Nm]')
    plt.grid()
    
    plt.subplot(4,3,10)
    plt.plot(t,  data_arr[:, 9], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('τ3(t) [Nm]')
    plt.grid()
    
    plt.subplot(4,3,11)
    plt.plot(t,  data_arr[:, 10], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('τ4(t) [Nm]')
    plt.grid()
    
    plt.suptitle(f"x(t), θ(t), q(t), τ(t) for biped under {ctrl_mode} control")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{ctrl_mode}.png")
    plt.close()
    

def main():
    
    xml = "./task4/biped_task4.xml"
    ctrl_mode = "qp_pd_climb"
    biped = Biped(xml, ctrl_mode)
    
    viewer = mujoco.viewer.launch_passive(biped.m, biped.d)
    camera_presets = {
        "lookat": [0.0, 0.0, 0.55], 
        "distance": 2, 
        "azimuth": 90, 
        "elevation": -10
    }  
    set_cam(viewer, 
            track=True, 
            presets=camera_presets, 
            show_world_csys=True, 
            show_body_csys=False
            )

    tmax = 3
    dt = biped.dt
    
    t_steps = round(tmax/biped.dt)
    data_arr = np.zeros((t_steps, 11))
    time_arr = np.arange(0, t_steps*dt, dt)

    for t in range(t_steps):
        
        q_pos = biped.d.qpos
        xc = q_pos[0]
        yc = q_pos[2]
        theta_c_quat = q_pos[3:7]
        theta_c = R.from_quat(theta_c_quat, scalar_first=True).as_euler('zyx')[1]
        q_leg = q_pos[biped.leg_qpos_indices]

        u = biped.ctrl()
        biped.step(u)   
         
        data_arr[t] = np.concatenate([[xc], [yc], [theta_c], q_leg, u], axis=0)
        
        mujoco.mj_step(biped.m, biped.d)
        viewer.sync()
        
    viewer.close()
    plot(time_arr, data_arr, ctrl_mode)


if __name__ == "__main__":
    main()