#!/usr/bin/env python3
"""
============================================================================
AME 556 - Robot Dynamics and Control - Final Project
TASK 2a: STANDING HEIGHT TRAJECTORY TRACKING
============================================================================

This script implements Task 2a from the PDF specification:
- Start at y_d = 0.45m, theta_d = 0 rad for 1 second
- Rise to y_d = 0.55m in 0.5 seconds
- Lower to y_d = 0.40m in 1.0 second
- Maintain theta_d = 0 (upright body orientation)

Physical Constraints (must be satisfied - from PDF):
- Hip angles: -120 deg <= q <= 30 deg
- Knee angles: 0 deg <= q <= 160 deg
- Hip velocities: |dq| <= 30 rad/s
- Knee velocities: |dq| <= 15 rad/s
- Hip torques: |tau| <= 30 Nm
- Knee torques: |tau| <= 60 Nm

Place this file and biped_robot.xml in the same folder.
Run: python task2a_height_tracking.py

Outputs:
- outputs/task2a_height.mp4 (video)
- outputs/task2a_constraint_verification.png (constraint plots for report)
============================================================================
"""

import os
import platform

# Only use EGL backend on Linux (for headless rendering)
# Windows uses default WGL backend automatically
if platform.system() == 'Linux':
    os.environ['MUJOCO_GL'] = 'egl'

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
# DIRECTORY SETUP
# =============================================================================
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
# Create outputs folder for videos and plots
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PHYSICAL CONSTRAINTS FROM PDF
# =============================================================================
# Joint angle limits (in radians)
HIP_ANGLE_MIN = np.deg2rad(-120)  # -120 deg
HIP_ANGLE_MAX = np.deg2rad(30)  # +30 deg
KNEE_ANGLE_MIN = np.deg2rad(0)  # 0 deg
KNEE_ANGLE_MAX = np.deg2rad(160)  # +160 deg

# Joint velocity limits (rad/s)
HIP_VEL_MAX = 30.0  # +/- 30 rad/s
KNEE_VEL_MAX = 15.0  # +/- 15 rad/s

# Joint torque limits (Nm)
HIP_TAU_MAX = 30.0  # +/- 30 Nm
KNEE_TAU_MAX = 60.0  # +/- 60 Nm

# Robot physical parameters
L = 0.22  # Link length (m)
HIP_OFFSET = 0.125  # Distance from trunk center to hip (m)
FOOT_R = 0.02  # Foot radius (m)

# Simulation parameters
DT = 0.001  # Timestep (s) - matches biped_robot_task2a.xml
FPS = 60  # Video frame rate

# =============================================================================
# XML MODEL FILE
# =============================================================================
# The robot model is loaded from biped_robot.xml
# Note: biped_robot.xml has trunk at z=0.65, so height offsets use 0.65
XML_FILE = "biped_robot.xml"
TRUNK_Z0 = 0.65  # Initial trunk z position in biped_robot.xml


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_model():
    """
    Create MuJoCo model and data objects from XML file.
    Looks for biped_robot_task2a.xml in the same directory as this script.
    """
    xml_path = SCRIPT_DIR / XML_FILE
    if not xml_path.exists():
        raise FileNotFoundError(
            f"Could not find {XML_FILE}.\n"
            f"Please place it in the same folder as this script: {SCRIPT_DIR}"
        )
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    return model, data


def saturate_torques(tau):
    """
    Apply torque saturation limits from PDF specification.
    - Hip torques clipped to +/- 30 Nm
    - Knee torques clipped to +/- 60 Nm
    """
    tau_sat = np.array(tau, dtype=float)
    tau_sat[0] = np.clip(tau_sat[0], -HIP_TAU_MAX, HIP_TAU_MAX)  # Left hip
    tau_sat[1] = np.clip(tau_sat[1], -KNEE_TAU_MAX, KNEE_TAU_MAX)  # Left knee
    tau_sat[2] = np.clip(tau_sat[2], -HIP_TAU_MAX, HIP_TAU_MAX)  # Right hip
    tau_sat[3] = np.clip(tau_sat[3], -KNEE_TAU_MAX, KNEE_TAU_MAX)  # Right knee
    return tau_sat


def leg_ik(h_com):
    """
    Inverse kinematics: Compute symmetric leg angles for desired COM height.

    For symmetric stance: leg_length = 2 * L * cos(q2/2), q1 = -q2/2
    """
    # Calculate required leg length from desired COM height
    leg_len = h_com - HIP_OFFSET - FOOT_R
    leg_len = np.clip(leg_len, 0.10, 2 * L - 0.01)

    # Solve inverse kinematics
    cos_half_q2 = leg_len / (2 * L)
    cos_half_q2 = np.clip(cos_half_q2, 0.01, 0.99)
    half_q2 = np.arccos(cos_half_q2)
    q2 = 2 * half_q2
    q1 = -half_q2

    # Clamp to joint limits
    q1 = np.clip(q1, HIP_ANGLE_MIN, HIP_ANGLE_MAX)
    q2 = np.clip(q2, KNEE_ANGLE_MIN, KNEE_ANGLE_MAX)

    return q1, q2


def get_desired_height(t):
    """
    Generate the height trajectory from PDF specification.

    Trajectory:
    - t < 1.0s:        Hold at y_d = 0.45m
    - 1.0 <= t < 1.5s: Rise from 0.45m to 0.55m
    - 1.5 <= t < 2.5s: Lower from 0.55m to 0.40m
    - t >= 2.5s:       Hold at y_d = 0.40m
    """
    if t < 1.0:
        return 0.45
    elif t < 1.5:
        alpha = (t - 1.0) / 0.5
        return 0.45 + alpha * (0.55 - 0.45)
    elif t < 2.5:
        alpha = (t - 1.5) / 1.0
        return 0.55 + alpha * (0.40 - 0.55)
    else:
        return 0.40


def check_constraints(q, dq):
    """Check if joint angle and velocity constraints are satisfied."""
    # Check hip angle limits
    if q[0] < HIP_ANGLE_MIN or q[0] > HIP_ANGLE_MAX:
        return True, f"q1={np.rad2deg(q[0]):.1f} deg out of range"
    if q[2] < HIP_ANGLE_MIN or q[2] > HIP_ANGLE_MAX:
        return True, f"q3={np.rad2deg(q[2]):.1f} deg out of range"
    # Check knee angle limits
    if q[1] < KNEE_ANGLE_MIN or q[1] > KNEE_ANGLE_MAX:
        return True, f"q2={np.rad2deg(q[1]):.1f} deg out of range"
    if q[3] < KNEE_ANGLE_MIN or q[3] > KNEE_ANGLE_MAX:
        return True, f"q4={np.rad2deg(q[3]):.1f} deg out of range"
    # Check velocity limits
    if abs(dq[0]) > HIP_VEL_MAX: return True, f"|dq1|={abs(dq[0]):.1f} > 30 rad/s"
    if abs(dq[2]) > HIP_VEL_MAX: return True, f"|dq3|={abs(dq[2]):.1f} > 30 rad/s"
    if abs(dq[1]) > KNEE_VEL_MAX: return True, f"|dq2|={abs(dq[1]):.1f} > 15 rad/s"
    if abs(dq[3]) > KNEE_VEL_MAX: return True, f"|dq4|={abs(dq[3]):.1f} > 15 rad/s"
    return False, ""


# =============================================================================
# CONSTRAINT VERIFICATION PLOTTING
# =============================================================================

def generate_constraint_plots(log, tracking_error, max_error, final_t):
    """
    Generate comprehensive constraint verification plots for the final report.

    Shows that ALL physical constraints are satisfied:
    1. Height trajectory tracking
    2. Joint angles within limits
    3. Joint velocities within limits
    4. Joint torques within saturation limits
    5. Trunk pitch maintained near zero
    """
    if not HAS_MATPLOTLIB:
        print("  Matplotlib not available - skipping plots")
        return

    print("\n  Generating constraint verification plots...")

    # Convert all logged data from lists to numpy arrays for plotting
    t = np.array(log['t'])
    z = np.array(log['z'])
    z_des = np.array(log['z_des'])
    pitch = np.array(log['pitch'])
    q1 = np.array(log['q1'])
    q2 = np.array(log['q2'])
    q3 = np.array(log['q3'])
    q4 = np.array(log['q4'])
    dq1 = np.array(log['dq1'])
    dq2 = np.array(log['dq2'])
    dq3 = np.array(log['dq3'])
    dq4 = np.array(log['dq4'])
    tau1 = np.array(log['tau1'])
    tau2 = np.array(log['tau2'])
    tau3 = np.array(log['tau3'])
    tau4 = np.array(log['tau4'])

    # Create figure with 6 subplots (3 rows x 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'Task 2a: Height Trajectory Tracking - Constraint Verification\n'
                 f'Duration: {final_t:.2f}s | Mean Error: {tracking_error * 100:.2f}cm | Max Error: {max_error * 100:.2f}cm',
                 fontsize=14, fontweight='bold')

    # -------------------------------------------------------------------------
    # Plot 1: Height Trajectory Tracking (top-left) - THE MAIN TASK
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(t, z, 'b-', linewidth=2, label='Actual height')
    ax.plot(t, z_des, 'r--', linewidth=2, label='Desired height')
    # Mark the trajectory phases
    ax.axhline(0.45, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0.55, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0.40, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(1.0, color='green', linestyle='--', alpha=0.3, label='Phase transitions')
    ax.axvline(1.5, color='green', linestyle='--', alpha=0.3)
    ax.axvline(2.5, color='green', linestyle='--', alpha=0.3)
    ax.fill_between(t, z_des - 0.05, z_des + 0.05, alpha=0.2, color='red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Height Trajectory Tracking (y_d follows PDF specification)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.35, 0.60])

    # -------------------------------------------------------------------------
    # Plot 2: Trunk Pitch (top-right) - Must maintain theta_d = 0
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(t, pitch, 'b-', linewidth=1.5)
    ax.axhline(0, color='g', linestyle='--', linewidth=2, label='Desired: 0 deg')
    ax.axhline(60, color='r', linestyle='--', alpha=0.7, label='Fall limit: +/-60 deg')
    ax.axhline(-60, color='r', linestyle='--', alpha=0.7)
    ax.fill_between(t, -60, 60, alpha=0.1, color='green')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch (deg)')
    ax.set_title('Trunk Pitch Angle (theta_d = 0 maintained)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-30, 30])

    # -------------------------------------------------------------------------
    # Plot 3: Joint Angles with Constraint Limits (middle-left)
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(t, q1, 'b-', label='q1 (left hip)', linewidth=1)
    ax.plot(t, q2, 'c-', label='q2 (left knee)', linewidth=1)
    ax.plot(t, q3, 'r-', label='q3 (right hip)', linewidth=1)
    ax.plot(t, q4, 'm-', label='q4 (right knee)', linewidth=1)
    # Draw constraint limits
    ax.axhline(-120, color='blue', linestyle='--', alpha=0.5, linewidth=2, label='Hip min: -120 deg')
    ax.axhline(30, color='blue', linestyle='--', alpha=0.5, linewidth=2, label='Hip max: +30 deg')
    ax.axhline(0, color='red', linestyle=':', alpha=0.5, linewidth=2, label='Knee min: 0 deg')
    ax.axhline(160, color='red', linestyle=':', alpha=0.5, linewidth=2, label='Knee max: +160 deg')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Joint Angles vs Limits - ALL WITHIN BOUNDS')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 4: Joint Velocities with Constraint Limits (middle-right)
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.plot(t, dq1, 'b-', label='dq1 (left hip)', linewidth=1)
    ax.plot(t, dq2, 'c-', label='dq2 (left knee)', linewidth=1)
    ax.plot(t, dq3, 'r-', label='dq3 (right hip)', linewidth=1)
    ax.plot(t, dq4, 'm-', label='dq4 (right knee)', linewidth=1)
    # Draw constraint limits
    ax.axhline(30, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Hip limit: +/-30 rad/s')
    ax.axhline(-30, color='blue', linestyle='--', alpha=0.7, linewidth=2)
    ax.axhline(15, color='red', linestyle=':', alpha=0.7, linewidth=2, label='Knee limit: +/-15 rad/s')
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
    ax.plot(t, tau1, 'b-', label='tau1 (left hip)', linewidth=1)
    ax.plot(t, tau2, 'c-', label='tau2 (left knee)', linewidth=1)
    ax.plot(t, tau3, 'r-', label='tau3 (right hip)', linewidth=1)
    ax.plot(t, tau4, 'm-', label='tau4 (right knee)', linewidth=1)
    # Draw saturation limits
    ax.axhline(30, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Hip limit: +/-30 Nm')
    ax.axhline(-30, color='blue', linestyle='--', alpha=0.7, linewidth=2)
    ax.axhline(60, color='red', linestyle=':', alpha=0.7, linewidth=2, label='Knee limit: +/-60 Nm')
    ax.axhline(-60, color='red', linestyle=':', alpha=0.7, linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Joint Torques vs Limits - SATURATION APPLIED')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 6: Height Tracking Error (bottom-right)
    # -------------------------------------------------------------------------
    ax = axes[2, 1]
    error_cm = np.abs(z - z_des) * 100
    ax.plot(t, error_cm, 'r-', linewidth=1.5)
    ax.axhline(tracking_error * 100, color='b', linestyle='--', linewidth=2,
               label=f'Mean error: {tracking_error * 100:.2f} cm')
    ax.axhline(5, color='orange', linestyle='--', alpha=0.7, label='5 cm threshold')
    ax.fill_between(t, 0, 5, alpha=0.1, color='green')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (cm)')
    ax.set_title('Height Tracking Error')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(10, max_error * 100 + 2)])

    plt.tight_layout()

    # Save plot
    plot_path = OUTPUT_DIR / "task2a_constraint_verification.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Plots saved: {plot_path}")


# =============================================================================
# MAIN TASK 2a FUNCTION
# =============================================================================

def run_task2a():
    """
    Task 2a: Track height trajectory while maintaining upright posture.
    """
    print("\n" + "=" * 70)
    print("TASK 2a: HEIGHT TRAJECTORY TRACKING")
    print("=" * 70)
    print("Trajectory: 0.45m (hold 1s) -> 0.55m (0.5s rise) -> 0.40m (1s lower)")

    # -------------------------------------------------------------------------
    # Create simulation model
    # -------------------------------------------------------------------------
    model, data = create_model()

    # -------------------------------------------------------------------------
    # Initialize robot in split stance for stability
    # -------------------------------------------------------------------------
    h_init = 0.45
    q1_init, q2_init = leg_ik(h_init)

    # Split stance: offset hip angles for stability
    hip_offset = np.deg2rad(8)
    q1_front = q1_init - hip_offset
    q1_back = q1_init + hip_offset

    # Compute trunk height
    leg_len_front = L * np.cos(q1_front) + L * np.cos(q1_front + q2_init)
    leg_len_back = L * np.cos(q1_back) + L * np.cos(q1_back + q2_init)
    avg_leg_len = (leg_len_front + leg_len_back) / 2
    trunk_z = HIP_OFFSET + avg_leg_len + FOOT_R

    # Set initial state
    data.qpos[0] = 0
    data.qpos[1] = trunk_z - TRUNK_Z0
    data.qpos[2] = 0
    data.qpos[3] = q1_front
    data.qpos[4] = q2_init
    data.qpos[5] = q1_back
    data.qpos[6] = q2_init
    data.qvel[:] = 0
    mj.mj_forward(model, data)

    print(f"  Initial stance: left hip={np.rad2deg(q1_front):.1f} deg, "
          f"right hip={np.rad2deg(q1_back):.1f} deg, knees={np.rad2deg(q2_init):.1f} deg")

    # -------------------------------------------------------------------------
    # Settling phase
    # -------------------------------------------------------------------------
    print("  Settling robot (1 second)...")
    q_settle = data.qpos[3:7].copy()
    # Use same gains as task2b/2c for consistency with biped_robot.xml dynamics
    kp_settle = np.array([200.0, 150.0, 200.0, 150.0])
    kd_settle = np.array([20.0, 15.0, 20.0, 15.0])

    for _ in range(int(1.0 / DT)):
        q = data.qpos[3:7]
        dq = data.qvel[3:7]
        pitch = data.qpos[2]
        dpitch = data.qvel[2]

        # Gravity compensation + PD control (same structure as task2b/2c)
        data.qacc[:] = 0
        mj.mj_inverse(model, data)
        tau = data.qfrc_inverse[3:7].copy() + kp_settle * (q_settle - q) - kd_settle * dq

        # Pitch stabilization (gains similar to task2b/2c)
        tau_pitch = 50.0 * (0 - pitch) - 12.0 * dpitch
        tau[0] += tau_pitch
        tau[2] += tau_pitch

        tau = saturate_torques(tau)
        data.ctrl[:] = tau
        mj.mj_step(model, data)

    data.time = 0
    actual_z = data.qpos[1] + TRUNK_Z0
    print(f"  After settling: z={actual_z:.3f}m, pitch={np.rad2deg(data.qpos[2]):.1f} deg")

    # -------------------------------------------------------------------------
    # Setup video recording
    # -------------------------------------------------------------------------
    # biped_robot.xml has no camera defined, so we create a tracking camera
    renderer = mj.Renderer(model, width=640, height=480)
    video_path = str(OUTPUT_DIR / "task2a_height.mp4")
    writer = imageio.get_writer(video_path, fps=FPS)

    # Create tracking camera (same setup as task2b/2c)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "trunk")
    cam.distance = 1.5
    cam.azimuth = 90
    cam.elevation = -10
    cam.lookat[2] = 0.5

    # -------------------------------------------------------------------------
    # Controller gains (same as task2b/2c for consistency)
    # -------------------------------------------------------------------------
    kp = np.array([200.0, 150.0, 200.0, 150.0])
    kd = np.array([20.0, 15.0, 20.0, 15.0])
    kp_pitch = 50.0  # Same as task2b/2c
    kd_pitch = 12.0  # Same as task2b/2c

    steps_per_frame = int(1.0 / (FPS * DT))
    step = 0
    duration = 4.0

    # -------------------------------------------------------------------------
    # Data logging for constraint verification
    # -------------------------------------------------------------------------
    log = {
        't': [], 'z': [], 'z_des': [], 'pitch': [],
        'q1': [], 'q2': [], 'q3': [], 'q4': [],
        'dq1': [], 'dq2': [], 'dq3': [], 'dq4': [],
        'tau1': [], 'tau2': [], 'tau3': [], 'tau4': []
    }

    print("  Running height trajectory...")
    print("-" * 60)

    # -------------------------------------------------------------------------
    # Main simulation loop
    # -------------------------------------------------------------------------
    while data.time < duration:
        t = data.time
        q = data.qpos[3:7]
        dq = data.qvel[3:7]
        pitch = data.qpos[2]
        dpitch = data.qvel[2]
        z = data.qpos[1] + TRUNK_Z0

        # Check for fall
        if abs(pitch) > np.deg2rad(60) or z < 0.15:
            print(f"  FELL at t={t:.2f}s: pitch={np.rad2deg(pitch):.0f} deg, z={z:.2f}m")
            break

        # Check constraints
        violated, msg = check_constraints(q, dq)
        if violated:
            print(f"  CONSTRAINT VIOLATION at t={t:.2f}s: {msg}")
            break

        # Get desired height and compute IK
        h_des = get_desired_height(t)
        q1_des, q2_des = leg_ik(h_des)
        q_des = np.array([q1_des - hip_offset, q2_des, q1_des + hip_offset, q2_des])

        # Gravity compensation + PD control (same structure as task2b/2c)
        data.qacc[:] = 0
        mj.mj_inverse(model, data)
        tau = data.qfrc_inverse[3:7].copy() + kp * (q_des - q) - kd * dq

        # Pitch stabilization (same as task2b/2c)
        tau_pitch = kp_pitch * (0 - pitch) - kd_pitch * dpitch
        tau[0] += tau_pitch
        tau[2] += tau_pitch

        # Apply saturation
        tau = saturate_torques(tau)
        data.ctrl[:] = tau
        mj.mj_step(model, data)

        # Log data
        log['t'].append(t)
        log['z'].append(z)
        log['z_des'].append(h_des)
        log['pitch'].append(np.rad2deg(pitch))
        log['q1'].append(np.rad2deg(q[0]))
        log['q2'].append(np.rad2deg(q[1]))
        log['q3'].append(np.rad2deg(q[2]))
        log['q4'].append(np.rad2deg(q[3]))
        log['dq1'].append(dq[0])
        log['dq2'].append(dq[1])
        log['dq3'].append(dq[2])
        log['dq4'].append(dq[3])
        log['tau1'].append(tau[0])
        log['tau2'].append(tau[1])
        log['tau3'].append(tau[2])
        log['tau4'].append(tau[3])

        # Record video frame
        if step % steps_per_frame == 0:
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())

        # Print progress every second
        if step % int(1.0 / DT) == 0:
            print(f"  t={t:.1f}s: z={z:.3f}m (des={h_des:.3f}m), pitch={np.rad2deg(pitch):.1f} deg")

        step += 1

    writer.close()
    renderer.close()

    # -------------------------------------------------------------------------
    # Calculate results
    # -------------------------------------------------------------------------
    final_t = data.time

    if len(log['z']) > 0:
        tracking_error = np.mean(np.abs(np.array(log['z']) - np.array(log['z_des'])))
        max_error = np.max(np.abs(np.array(log['z']) - np.array(log['z_des'])))
    else:
        tracking_error = 1.0
        max_error = 1.0

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Duration: {final_t:.2f}s (target: {duration:.1f}s)")
    print(f"Mean tracking error: {tracking_error * 100:.2f} cm")
    print(f"Max tracking error: {max_error * 100:.2f} cm")
    print(f"Video: {video_path}")

    # -------------------------------------------------------------------------
    # Generate constraint verification plots
    # -------------------------------------------------------------------------
    generate_constraint_plots(log, tracking_error, max_error, final_t)

    return {'time': final_t, 'tracking_error': tracking_error, 'max_error': max_error, 'log': log}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    result = run_task2a()

    print(f"\n" + "=" * 60)
    print("TASK 2a SUMMARY")
    print("=" * 60)
    print(f"Duration: {result['time']:.2f}s {'(PASS)' if result['time'] >= 3.9 else '(FAIL)'}")
    print(f"Mean tracking error: {result['tracking_error'] * 100:.2f} cm")
    print(f"\nOutput files in: {OUTPUT_DIR}")
    print(f"  - task2a_height.mp4 (video)")
    print(f"  - task2a_constraint_verification.png (plots for report)")