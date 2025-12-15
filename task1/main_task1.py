import time
import mujoco
import numpy as np
import os, sys
from cvxopt import matrix, solvers

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
solvers.options['show_progress'] = False

from utils.utils import *
from biped_task1 import Biped


def main():

    xml = "biped_task1.xml"
    biped = Biped(xml)

    default_qpos = np.array(
        [0, 0, 0.52, 1, 0, 0, 0,
         -0.0471975512, 0.5707963268,
         -0.5235987756, 0.5707963268],
        dtype=float
    )

    default_qvel = np.zeros(biped.m.nv, dtype=float)

    def reset_state():
        biped.d.qpos[:] = default_qpos
        biped.d.qvel[:] = default_qvel
        biped.d.ctrl[:] = 0.0

    def run_case(label, fn, u=(0, 0, 0, 0)):
        print("Case: ", label)
        reset_state()
        try:
            fn()
            biped.step(np.array(u, dtype=float))
        except ValueError as e:
            print(f"Error: {e}", flush=True)
        time.sleep(1)
        print("------------------------------------------------------\n")

    # --------------------
    # Joint position limits
    # --------------------
    run_case(
        "thigh qpos under min (left)",
        lambda: biped.d.qpos.__setitem__(7, biped.thigh_qpos_min - 1e-3)
    )

    run_case(
        "thigh qpos over max (left)",
        lambda: biped.d.qpos.__setitem__(7, biped.thigh_qpos_max + 1e-3)
    )

    run_case(
        "thigh qpos under min (right)",
        lambda: biped.d.qpos.__setitem__(9, biped.thigh_qpos_min - 1e-3)
    )

    run_case(
        "thigh qpos over max (right)",
        lambda: biped.d.qpos.__setitem__(9, biped.thigh_qpos_max + 1e-3)
    )

    run_case(
        "calf qpos under min (left)",
        lambda: biped.d.qpos.__setitem__(8, biped.calf_qpos_min - 1e-3)
    )

    run_case(
        "calf qpos over max (left)",
        lambda: biped.d.qpos.__setitem__(8, biped.calf_qpos_max + 1e-3)
    )

    run_case(
        "calf qpos under min (right)",
        lambda: biped.d.qpos.__setitem__(10, biped.calf_qpos_min - 1e-3)
    )

    run_case(
        "calf qpos over max (right)",
        lambda: biped.d.qpos.__setitem__(10, biped.calf_qpos_max + 1e-3)
    )

    # --------------------
    # Joint velocity limits
    # --------------------
    run_case(
        "thigh qvel exceeds (left)",
        lambda: biped.d.qvel.__setitem__(6, biped.thigh_qvel_limit + 1e-3)
    )

    run_case(
        "calf qvel exceeds (left)",
        lambda: biped.d.qvel.__setitem__(7, biped.calf_qvel_limit + 1e-3)
    )

    run_case(
        "thigh qvel exceeds (right)",
        lambda: biped.d.qvel.__setitem__(8, biped.thigh_qvel_limit + 1e-3)
    )

    run_case(
        "calf qvel exceeds (right)",
        lambda: biped.d.qvel.__setitem__(9, biped.calf_qvel_limit + 1e-3)
    )

    run_case(
        "thigh qvel exceeds negative (left)",
        lambda: biped.d.qvel.__setitem__(6, -(biped.thigh_qvel_limit + 1e-3))
    )

    run_case(
        "calf qvel exceeds negative (left)",
        lambda: biped.d.qvel.__setitem__(7, -(biped.calf_qvel_limit + 1e-3))
    )

    run_case(
        "thigh qvel exceeds negative (right)",
        lambda: biped.d.qvel.__setitem__(8, -(biped.thigh_qvel_limit + 1e-3))
    )

    run_case(
        "calf qvel exceeds negative (right)",
        lambda: biped.d.qvel.__setitem__(9, -(biped.calf_qvel_limit + 1e-3))
    )

    # --------------------
    # Torque limits
    # --------------------
    run_case(
        "thigh torque exceeds (left)",
        lambda: None,
        u=(biped.thigh_tau_limit + 1e-3, 0, 0, 0)
    )

    run_case(
        "calf torque exceeds (left)",
        lambda: None,
        u=(0, biped.calf_tau_limit + 1e-3, 0, 0)
    )

    run_case(
        "thigh torque exceeds (right)",
        lambda: None,
        u=(0, 0, biped.thigh_tau_limit + 1e-3, 0)
    )

    run_case(
        "calf torque exceeds (right)",
        lambda: None,
        u=(0, 0, 0, biped.calf_tau_limit + 1e-3)
    )

    run_case(
        "thigh torque exceeds negative (left)",
        lambda: None,
        u=(-(biped.thigh_tau_limit + 1e-3), 0, 0, 0)
    )

    run_case(
        "calf torque exceeds negative (left)",
        lambda: None,
        u=(0, -(biped.calf_tau_limit + 1e-3), 0, 0)
    )

    run_case(
        "thigh torque exceeds negative (right)",
        lambda: None,
        u=(0, 0, -(biped.thigh_tau_limit + 1e-3), 0)
    )

    run_case(
        "calf torque exceeds negative (right)",
        lambda: None,
        u=(0, 0, 0, -(biped.calf_tau_limit + 1e-3))
    )


if __name__ == "__main__":
    main()
