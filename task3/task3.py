#!/usr/bin/env python3
"""
==============================================================================
AME 556 - Robot Dynamics and Control - Final Project
TASK 3: RUNNING 10 METERS WITH FLIGHT PHASE
==============================================================================

REQUIREMENTS (from Final_Project__Fall_25.pdf):
- Control the robot to run on flat ground for 10 meters
- Score = 200 / travel_time (seconds)
- MUST have a flight phase in the gait schedule (both feet off ground)

PHYSICAL CONSTRAINTS (from Task 1):
- Hip angles: -120 deg <= q <= 30 deg
- Knee angles: 0 deg <= q <= 160 deg
- Hip velocities: |dq| <= 30 rad/s
- Knee velocities: |dq| <= 15 rad/s
- Hip torques: |tau| <= 30 Nm
- Knee torques: |tau| <= 60 Nm

CONTROL APPROACH:
This controller uses a high-frequency sinusoidal gait (2.5 Hz) with:
1. Gravity compensation via MuJoCo inverse dynamics
2. PD joint tracking for the desired gait trajectory
3. Pitch stabilization to prevent forward/backward falling
4. Forward push torques for propulsion
5. Speed feedback control to maintain target velocity
6. Soft joint limit avoidance

Place this file and biped_robot.xml in the same folder.
Run: python task3.py

Outputs:
- outputs/task3_running.mp4 (video)
- outputs/task3_constraint_verification.png (constraint plots for report)
==============================================================================
"""

# =============================================================================
# PLATFORM SETUP - Must be before mujoco import
# =============================================================================
import os
import sys

# For Linux headless rendering (servers without display)
if sys.platform.startswith('linux'):
    os.environ['MUJOCO_GL'] = 'egl'

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import mujoco as mj
import imageio
from pathlib import Path

# Try to import matplotlib for plotting
try:
    import matplotlib

    matplotlib.use('Agg')  # Non-interactive backend for saving figures
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not found. Plots will not be generated.")

# =============================================================================
# PHYSICAL CONSTANTS (from Project PDF)
# =============================================================================

# Link dimensions
L = 0.22  # Link length (m)
HIP_OFFSET = 0.125  # Distance from trunk center to hip joint (m)
FOOT_R = 0.02  # Foot sphere radius (m)
TRUNK_Z0 = 0.65  # Initial trunk z position defined in XML (m)

# Torque limits (Nm) - from project PDF, Task 1 constraints
HIP_TAU_MAX = 30.0  # Maximum hip torque: |tau_hip| <= 30 Nm
KNEE_TAU_MAX = 60.0  # Maximum knee torque: |tau_knee| <= 60 Nm

# Joint angle limits (radians) - from project PDF
HIP_ANGLE_MIN = np.deg2rad(-120)  # Hip minimum: -120 degrees
HIP_ANGLE_MAX = np.deg2rad(30)  # Hip maximum: +30 degrees
KNEE_ANGLE_MIN = np.deg2rad(0)  # Knee minimum: 0 degrees
KNEE_ANGLE_MAX = np.deg2rad(160)  # Knee maximum: 160 degrees

# Joint velocity limits (rad/s) - from project PDF
HIP_VEL_MAX = 30.0  # Hip velocity limit: +/- 30 rad/s
KNEE_VEL_MAX = 15.0  # Knee velocity limit: +/- 15 rad/s

# =============================================================================
# OUTPUT DIRECTORY SETUP
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def saturate_torques(tau):
    """
    Apply torque saturation limits from project PDF (Task 1 requirement).
    - Hip joints (indices 0, 2): +/- 30 Nm
    - Knee joints (indices 1, 3): +/- 60 Nm
    """
    tau = tau.copy()
    tau[0] = np.clip(tau[0], -HIP_TAU_MAX, HIP_TAU_MAX)
    tau[1] = np.clip(tau[1], -KNEE_TAU_MAX, KNEE_TAU_MAX)
    tau[2] = np.clip(tau[2], -HIP_TAU_MAX, HIP_TAU_MAX)
    tau[3] = np.clip(tau[3], -KNEE_TAU_MAX, KNEE_TAU_MAX)
    return tau


def gravity_compensation(model, data):
    """
    Compute gravity compensation torques using MuJoCo inverse dynamics.
    """
    qacc_save = data.qacc.copy()
    data.qacc[:] = 0
    mj.mj_inverse(model, data)
    tau_g = data.qfrc_inverse[3:7].copy()
    data.qacc[:] = qacc_save
    return tau_g


def foot_positions_fk(q1, q2, q3, q4):
    """
    Forward kinematics to compute foot positions relative to hip joints.
    """
    lf_x = L * np.sin(q1) + L * np.sin(q1 + q2)
    lf_z = -L * np.cos(q1) - L * np.cos(q1 + q2)
    rf_x = L * np.sin(q3) + L * np.sin(q3 + q4)
    rf_z = -L * np.cos(q3) - L * np.cos(q3 + q4)
    return lf_x, lf_z, rf_x, rf_z


def check_constraints(q, dq):
    """
    Check if joint angle and velocity constraints are satisfied.
    Returns (violated, message) tuple.
    """
    # Check hip angle limits (-120 deg to +30 deg)
    if q[0] < HIP_ANGLE_MIN:
        return True, f"q1={np.rad2deg(q[0]):.1f} deg < -120 deg"
    if q[0] > HIP_ANGLE_MAX:
        return True, f"q1={np.rad2deg(q[0]):.1f} deg > 30 deg"
    if q[2] < HIP_ANGLE_MIN:
        return True, f"q3={np.rad2deg(q[2]):.1f} deg < -120 deg"
    if q[2] > HIP_ANGLE_MAX:
        return True, f"q3={np.rad2deg(q[2]):.1f} deg > 30 deg"

    # Check knee angle limits (0 deg to +160 deg)
    if q[1] < KNEE_ANGLE_MIN:
        return True, f"q2={np.rad2deg(q[1]):.1f} deg < 0 deg"
    if q[1] > KNEE_ANGLE_MAX:
        return True, f"q2={np.rad2deg(q[1]):.1f} deg > 160 deg"
    if q[3] < KNEE_ANGLE_MIN:
        return True, f"q4={np.rad2deg(q[3]):.1f} deg < 0 deg"
    if q[3] > KNEE_ANGLE_MAX:
        return True, f"q4={np.rad2deg(q[3]):.1f} deg > 160 deg"

    # Check hip velocity limits (+/- 30 rad/s)
    if abs(dq[0]) > HIP_VEL_MAX:
        return True, f"|dq1|={abs(dq[0]):.1f} > 30 rad/s"
    if abs(dq[2]) > HIP_VEL_MAX:
        return True, f"|dq3|={abs(dq[2]):.1f} > 30 rad/s"

    # Check knee velocity limits (+/- 15 rad/s)
    if abs(dq[1]) > KNEE_VEL_MAX:
        return True, f"|dq2|={abs(dq[1]):.1f} > 15 rad/s"
    if abs(dq[3]) > KNEE_VEL_MAX:
        return True, f"|dq4|={abs(dq[3]):.1f} > 15 rad/s"

    return False, ""


# =============================================================================
# CONSTRAINT VERIFICATION PLOTTING
# =============================================================================

def generate_constraint_plots(log, final_dist, final_time, score, flight_count):
    """
    Generate comprehensive constraint verification plots for the final report.
    Shows that ALL physical constraints are satisfied during running.
    """
    if not HAS_MATPLOTLIB:
        print("  Matplotlib not available - skipping plots")
        return

    print("\n  Generating constraint verification plots...")

    t = np.array(log['t'])

    # Create figure with 6 subplots (3 rows x 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'Task 3: Running 10m - Constraint Verification\n'
                 f'Distance: {final_dist:.2f}m | Time: {final_time:.2f}s | '
                 f'Score: {score:.2f} | Flight Phases: {flight_count}',
                 fontsize=14, fontweight='bold')

    # -------------------------------------------------------------------------
    # Plot 1: Distance vs Time (top-left) - THE MAIN TASK
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(t, log['dist'], 'b-', linewidth=2, label='Distance traveled')
    ax.axhline(10.0, color='r', linestyle='--', linewidth=2, label='Goal: 10m')
    ax.fill_between(t, 0, 10, alpha=0.1, color='green')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Distance vs Time (10m Running Task)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(12, final_dist + 1)])

    # -------------------------------------------------------------------------
    # Plot 2: Speed vs Time (top-right)
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(t, log['vx'], 'b-', linewidth=1, label='Instantaneous velocity')
    avg_speed = final_dist / final_time if final_time > 0 else 0
    ax.axhline(avg_speed, color='g', linestyle='--', linewidth=2,
               label=f'Average: {avg_speed:.3f} m/s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Forward Velocity vs Time')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 3: Joint Angles with Constraint Limits (middle-left)
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(t, log['q1'], 'b-', label='q1 (left hip)', linewidth=1)
    ax.plot(t, log['q2'], 'c-', label='q2 (left knee)', linewidth=1)
    ax.plot(t, log['q3'], 'r-', label='q3 (right hip)', linewidth=1)
    ax.plot(t, log['q4'], 'm-', label='q4 (right knee)', linewidth=1)
    # Draw constraint limits
    ax.axhline(-120, color='blue', linestyle='--', alpha=0.5, linewidth=2, label='Hip min: -120°')
    ax.axhline(30, color='blue', linestyle='--', alpha=0.5, linewidth=2, label='Hip max: +30°')
    ax.axhline(0, color='red', linestyle=':', alpha=0.5, linewidth=2, label='Knee min: 0°')
    ax.axhline(160, color='red', linestyle=':', alpha=0.5, linewidth=2, label='Knee max: +160°')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Joint Angles vs Limits - ALL WITHIN BOUNDS')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 4: Joint Velocities with Constraint Limits (middle-right)
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.plot(t, log['dq1'], 'b-', label='dq1 (left hip)', linewidth=1)
    ax.plot(t, log['dq2'], 'c-', label='dq2 (left knee)', linewidth=1)
    ax.plot(t, log['dq3'], 'r-', label='dq3 (right hip)', linewidth=1)
    ax.plot(t, log['dq4'], 'm-', label='dq4 (right knee)', linewidth=1)
    # Draw constraint limits
    ax.axhline(30, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Hip limit: ±30 rad/s')
    ax.axhline(-30, color='blue', linestyle='--', alpha=0.7, linewidth=2)
    ax.axhline(15, color='red', linestyle=':', alpha=0.7, linewidth=2, label='Knee limit: ±15 rad/s')
    ax.axhline(-15, color='red', linestyle=':', alpha=0.7, linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (rad/s)')
    ax.set_title('Joint Velocities vs Limits - ALL WITHIN BOUNDS')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 5: Joint Torques with Saturation Limits (bottom-left)
    # -------------------------------------------------------------------------
    ax = axes[2, 0]
    ax.plot(t, log['tau1'], 'b-', label='tau1 (left hip)', linewidth=1)
    ax.plot(t, log['tau2'], 'c-', label='tau2 (left knee)', linewidth=1)
    ax.plot(t, log['tau3'], 'r-', label='tau3 (right hip)', linewidth=1)
    ax.plot(t, log['tau4'], 'm-', label='tau4 (right knee)', linewidth=1)
    # Draw saturation limits
    ax.axhline(30, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Hip limit: ±30 Nm')
    ax.axhline(-30, color='blue', linestyle='--', alpha=0.7, linewidth=2)
    ax.axhline(60, color='red', linestyle=':', alpha=0.7, linewidth=2, label='Knee limit: ±60 Nm')
    ax.axhline(-60, color='red', linestyle=':', alpha=0.7, linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Joint Torques vs Limits - SATURATION APPLIED')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 6: Trunk Pitch Angle (bottom-right)
    # -------------------------------------------------------------------------
    ax = axes[2, 1]
    ax.plot(t, log['pitch'], 'b-', linewidth=1.5)
    ax.axhline(0, color='g', linestyle='--', label='Desired: 0°')
    ax.axhline(60, color='r', linestyle='--', alpha=0.7, label='Fall limit: ±60°')
    ax.axhline(-60, color='r', linestyle='--', alpha=0.7)
    ax.fill_between(t, -60, 60, alpha=0.1, color='green')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch (deg)')
    ax.set_title('Trunk Pitch Angle (STABLE - No Fall)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-70, 70])

    plt.tight_layout()

    # Save plot
    plot_path = OUTPUT_DIR / "task3_constraint_verification.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Plots saved: {plot_path}")


# =============================================================================
# MAIN RUNNING FUNCTION
# =============================================================================

def run_task3(xml_path, save_video=True):
    """
    Run Task 3: 10m running with flight phase.
    """

    # =========================================================================
    # LOAD MODEL AND CREATE DATA STRUCTURES
    # =========================================================================

    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    dt = model.opt.timestep

    # =========================================================================
    # INITIALIZATION - Split Stance Configuration
    # =========================================================================

    mj.mj_resetData(model, data)

    # Set initial joint angles for split stance
    q_init = [np.deg2rad(-50), np.deg2rad(40),  # Left hip, left knee
              np.deg2rad(-5), np.deg2rad(55)]  # Right hip, right knee
    data.qpos[3:7] = q_init

    # Compute foot positions to set correct trunk height
    lf_x, lf_z, rf_x, rf_z = foot_positions_fk(*q_init)
    min_foot_z = min(lf_z, rf_z)

    # Set trunk position so feet touch ground
    data.qpos[0] = -(lf_x + rf_x) / 2
    data.qpos[1] = HIP_OFFSET - min_foot_z + FOOT_R + 0.002 - TRUNK_Z0
    data.qpos[2] = 0.0
    data.qvel[:] = 0

    mj.mj_forward(model, data)

    # Store initial joint configuration
    q0 = data.qpos[3:7].copy()

    # =========================================================================
    # SETTLING PHASE
    # =========================================================================

    Kp = np.array([200.0, 150.0, 200.0, 150.0])
    Kd = np.array([20.0, 15.0, 20.0, 15.0])

    for _ in range(300):
        q = data.qpos[3:7]
        dq = data.qvel[3:7]
        pitch = data.qpos[2]
        dpitch = data.qvel[2]

        tau = gravity_compensation(model, data) + Kp * (q0 - q) - Kd * dq
        tau[0] += 50 * (0 - pitch) - 12 * dpitch
        tau[2] += 50 * (0 - pitch) - 12 * dpitch

        data.ctrl[:] = saturate_torques(tau)
        mj.mj_step(model, data)

    data.qvel[:] = 0
    mj.mj_forward(model, data)

    x0 = data.qpos[0]

    # =========================================================================
    # VIDEO RECORDING SETUP
    # =========================================================================

    if save_video:
        renderer = mj.Renderer(model, width=640, height=480)
        video_path = str(OUTPUT_DIR / "task3_running.mp4")
        writer = imageio.get_writer(video_path, fps=60)
        steps_per_frame = max(1, int(1.0 / (60 * dt)))

        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "trunk")
        cam.distance = 1.5
        cam.azimuth = 90
        cam.elevation = -10
        cam.lookat[2] = 0.4

    # =========================================================================
    # OPTIMIZED RUNNING PARAMETERS
    # =========================================================================

    freq = 2.5  # Hz - higher frequency for running
    hip_amp_rad = np.deg2rad(10)
    knee_amp_rad = np.deg2rad(8)
    lean_rad = np.deg2rad(-10)
    ramp_rad = np.deg2rad(4)

    Kp_pitch = 88
    Kd_pitch = 19

    push = 15
    extra_push = 10
    speed_gain = 6.0
    target_speed = 0.6

    margin = np.deg2rad(5)
    K_limit = 100.0

    # =========================================================================
    # DATA LOGGING FOR CONSTRAINT VERIFICATION
    # =========================================================================

    log = {
        't': [], 'dist': [], 'vx': [], 'pitch': [],
        'q1': [], 'q2': [], 'q3': [], 'q4': [],
        'dq1': [], 'dq2': [], 'dq3': [], 'dq4': [],
        'tau1': [], 'tau2': [], 'tau3': [], 'tau4': [],
        'flight': []
    }

    # =========================================================================
    # PRINT CONFIGURATION
    # =========================================================================

    print("=" * 60)
    print("TASK 3: RUNNING 10 METERS WITH FLIGHT PHASE")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Gait frequency: {freq} Hz")
    print(f"  Hip amplitude: {np.rad2deg(hip_amp_rad):.0f} degrees")
    print(f"  Knee amplitude: {np.rad2deg(knee_amp_rad):.0f} degrees")
    print(f"  Forward lean: {np.rad2deg(lean_rad):.0f} degrees")
    print(f"  Push torque: {push} Nm (extra: {extra_push} Nm)")
    print(f"  Speed feedback: gain={speed_gain}, target={target_speed} m/s")
    print(f"Start position: x={x0:.3f}m")
    print()

    # =========================================================================
    # MAIN SIMULATION LOOP
    # =========================================================================

    flight_count = 0
    prev_flight = False
    step = 0
    termination_reason = None

    while data.time < 60.0:
        t = data.time

        # Extract current state
        pitch = data.qpos[2]
        dpitch = data.qvel[2]
        dx = data.qvel[0]
        q = data.qpos[3:7].copy()
        dq = data.qvel[3:7].copy()
        x = data.qpos[0]
        dist = x - x0

        # Flight phase detection
        in_flight = (data.ncon == 0)
        if in_flight and not prev_flight:
            flight_count += 1
        prev_flight = in_flight

        # Check physical constraints (Task 1 requirement)
        violated, msg = check_constraints(q, dq)
        if violated:
            termination_reason = f"CONSTRAINT VIOLATION: {msg}"
            print(f"  t={t:.2f}s: {termination_reason}")
            break

        # Fall detection
        if abs(pitch) > np.deg2rad(60):
            termination_reason = f"FALL (pitch={np.rad2deg(pitch):.1f} deg)"
            print(f"  t={t:.2f}s: {termination_reason}")
            break

        # Goal reached
        if dist >= 10.0:
            print(f"GOAL at t={t:.2f}s! Distance={dist:.2f}m, flight phases={flight_count}")
            break

        # Gait trajectory generation
        phase = 2 * np.pi * freq * t
        cycle_phase = (phase % (2 * np.pi)) / (2 * np.pi)
        cycle_ramp = ramp_rad * cycle_phase

        q_des = q0.copy()
        hip_swing = hip_amp_rad * np.sin(phase)
        q_des[0] = q0[0] + lean_rad + hip_swing - cycle_ramp
        q_des[2] = q0[2] + lean_rad - hip_swing - cycle_ramp
        q_des[1] = q0[1] + knee_amp_rad * max(0, np.sin(phase))
        q_des[3] = q0[3] + knee_amp_rad * max(0, -np.sin(phase))

        # Clamp desired angles
        q_des[0] = np.clip(q_des[0], HIP_ANGLE_MIN + margin, HIP_ANGLE_MAX - margin)
        q_des[1] = np.clip(q_des[1], KNEE_ANGLE_MIN + margin, KNEE_ANGLE_MAX - margin)
        q_des[2] = np.clip(q_des[2], HIP_ANGLE_MIN + margin, HIP_ANGLE_MAX - margin)
        q_des[3] = np.clip(q_des[3], KNEE_ANGLE_MIN + margin, KNEE_ANGLE_MAX - margin)

        # Control torque computation
        tau_g = gravity_compensation(model, data)
        tau = tau_g + Kp * (q_des - q) - Kd * dq

        # Pitch stabilization
        tau_pitch = Kp_pitch * (0 - pitch) - Kd_pitch * dpitch
        tau[0] += tau_pitch
        tau[2] += tau_pitch

        # Forward push
        tau[0] += push
        tau[2] += push

        # Extra push during stance
        if np.sin(phase) < 0:
            tau[0] += extra_push
        else:
            tau[2] += extra_push

        # Speed feedback
        speed_error = target_speed - dx
        tau[0] += speed_gain * speed_error
        tau[2] += speed_gain * speed_error

        # Soft joint limit avoidance
        for i in [0, 2]:
            if q[i] > HIP_ANGLE_MAX - margin:
                tau[i] -= K_limit * (q[i] - HIP_ANGLE_MAX + margin)
            if q[i] < HIP_ANGLE_MIN + margin:
                tau[i] += K_limit * (HIP_ANGLE_MIN + margin - q[i])

        for i in [1, 3]:
            if q[i] > KNEE_ANGLE_MAX - margin:
                tau[i] -= K_limit * (q[i] - KNEE_ANGLE_MAX + margin)
            if q[i] < KNEE_ANGLE_MIN + margin:
                tau[i] += K_limit * (KNEE_ANGLE_MIN + margin - q[i])

        # Apply saturated torques
        tau_sat = saturate_torques(tau)
        data.ctrl[:] = tau_sat

        # Log data for plots
        log['t'].append(t)
        log['dist'].append(dist)
        log['vx'].append(dx)
        log['pitch'].append(np.rad2deg(pitch))
        log['q1'].append(np.rad2deg(q[0]))
        log['q2'].append(np.rad2deg(q[1]))
        log['q3'].append(np.rad2deg(q[2]))
        log['q4'].append(np.rad2deg(q[3]))
        log['dq1'].append(dq[0])
        log['dq2'].append(dq[1])
        log['dq3'].append(dq[2])
        log['dq4'].append(dq[3])
        log['tau1'].append(tau_sat[0])
        log['tau2'].append(tau_sat[1])
        log['tau3'].append(tau_sat[2])
        log['tau4'].append(tau_sat[3])
        log['flight'].append(1 if in_flight else 0)

        # Step simulation
        mj.mj_step(model, data)

        # Record video frame
        if save_video and step % steps_per_frame == 0:
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())

        # Progress output
        if step % 5000 == 0:
            print(f"  t={t:.1f}s: dist={dist:.2f}m, speed={dx:.2f}m/s, flights={flight_count}")

        step += 1

    # =========================================================================
    # CLEANUP AND RESULTS
    # =========================================================================

    if save_video:
        writer.close()
        renderer.close()

    final_dist = data.qpos[0] - x0
    final_time = data.time

    if final_dist >= 10.0:
        score = 200.0 / final_time
    else:
        score = 0.0

    # =========================================================================
    # PRINT RESULTS
    # =========================================================================

    print()
    print("=" * 60)
    print("TASK 3 RESULTS")
    print("=" * 60)
    print(f"Distance traveled: {final_dist:.2f}m {'(GOAL!)' if final_dist >= 10 else ''}")
    print(f"Time: {final_time:.2f}s")
    print(f"Flight phases: {flight_count}")
    print(f"Average speed: {final_dist / final_time:.3f} m/s")
    print(f"SCORE = 200 / {final_time:.2f} = {score:.2f}")
    if termination_reason:
        print(f"Terminated: {termination_reason}")
    if save_video:
        print(f"Video saved: {OUTPUT_DIR / 'task3_running.mp4'}")

    # =========================================================================
    # GENERATE CONSTRAINT VERIFICATION PLOTS
    # =========================================================================

    generate_constraint_plots(log, final_dist, final_time, score, flight_count)

    return {
        'distance': final_dist,
        'time': final_time,
        'score': score,
        'flights': flight_count,
        'log': log
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    xml_path = str(SCRIPT_DIR / "biped_robot.xml")

    if not Path(xml_path).exists():
        print(f"ERROR: {xml_path} not found!")
        print("Make sure biped_robot.xml is in the same folder as this script.")
        exit(1)

    result = run_task3(xml_path, save_video=True)

    print(f"\nOutput files in: {OUTPUT_DIR}")
    print(f"  - task3_running.mp4 (video)")
    print(f"  - task3_constraint_verification.png (plots for report)")