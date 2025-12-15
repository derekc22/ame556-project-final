#!/usr/bin/env python3
"""
============================================================================
AME 556 - Robot Dynamics and Control - Final Project
TASK 2b: FORWARD WALKING - BEST VERSION
============================================================================

Parameters: freq=1.2Hz, amp=11 deg, lean=-8 deg, Kp=70, Kd=17
Results: 10s, 2.07m, 0.207 m/s forward

Requirements from PDF:
- Walking forward with speed >= 0.5 m/s
- Duration >= 5 seconds

Physical Constraints (must be satisfied):
- Hip angles: -120 deg <= q <= 30 deg
- Knee angles: 0 deg <= q <= 160 deg
- Hip velocities: |dq| <= 30 rad/s
- Knee velocities: |dq| <= 15 rad/s
- Hip torques: |tau| <= 30 Nm
- Knee torques: |tau| <= 60 Nm

Place this file and biped_robot.xml in the same folder.
Run: python task2b_forward.py
============================================================================
"""

import os
import platform

# Only use EGL backend on Linux (for headless rendering)
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

# ============================================================================
# DIRECTORY SETUP
# ============================================================================
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
# Create outputs folder for videos and plots
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PHYSICAL CONSTRAINTS FROM PDF
# ============================================================================
# Joint angle limits (in radians)
HIP_ANGLE_MIN, HIP_ANGLE_MAX = -2.094, 0.524  # -120 deg to +30 deg
KNEE_ANGLE_MIN, KNEE_ANGLE_MAX = 0.0, 2.793  # 0 deg to +160 deg

# Joint velocity limits (rad/s)
HIP_VEL_MAX, KNEE_VEL_MAX = 30.0, 15.0

# Joint torque limits (Nm)
HIP_TAU_MAX, KNEE_TAU_MAX = 30.0, 60.0

# Robot physical parameters
L = 0.22  # Link length (m)
HIP_OFFSET = 0.125  # Distance from trunk center to hip (m)
FOOT_R = 0.02  # Foot radius (m)
TRUNK_Z0 = 0.65  # Initial trunk z position in XML (m)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def foot_positions_fk(q1, q2, q3, q4):
    """
    Forward kinematics: compute foot positions relative to hip joint.

    Args:
        q1: Left hip angle (rad)
        q2: Left knee angle (rad)
        q3: Right hip angle (rad)
        q4: Right knee angle (rad)

    Returns:
        lf_x, lf_z, rf_x, rf_z: Left/right foot x,z positions relative to hip
    """
    # Left foot position (two-link chain from hip)
    lf_x = L * np.sin(q1) + L * np.sin(q1 + q2)
    lf_z = -L * np.cos(q1) - L * np.cos(q1 + q2)
    # Right foot position
    rf_x = L * np.sin(q3) + L * np.sin(q3 + q4)
    rf_z = -L * np.cos(q3) - L * np.cos(q3 + q4)
    return lf_x, lf_z, rf_x, rf_z


def gravity_compensation(model, data):
    """
    Compute gravity compensation torques using inverse dynamics.
    Sets qacc=0 and computes required torques to hold position against gravity.

    Returns:
        tau_g: Gravity compensation torques for joints [q1, q2, q3, q4]
    """
    qacc_save = data.qacc.copy()
    data.qacc[:] = 0
    mj.mj_inverse(model, data)
    tau_g = data.qfrc_inverse[3:7].copy()
    data.qacc[:] = qacc_save
    return tau_g


def saturate_torques(tau):
    """
    Apply torque saturation limits from PDF specification.
    This implements the input saturation requirement:
    - Hip torques clipped to +/- 30 Nm
    - Knee torques clipped to +/- 60 Nm

    Args:
        tau: Raw torque command [tau1, tau2, tau3, tau4]

    Returns:
        tau_sat: Saturated torque command
    """
    tau = tau.copy()
    tau[0] = np.clip(tau[0], -HIP_TAU_MAX, HIP_TAU_MAX)  # Left hip
    tau[1] = np.clip(tau[1], -KNEE_TAU_MAX, KNEE_TAU_MAX)  # Left knee
    tau[2] = np.clip(tau[2], -HIP_TAU_MAX, HIP_TAU_MAX)  # Right hip
    tau[3] = np.clip(tau[3], -KNEE_TAU_MAX, KNEE_TAU_MAX)  # Right knee
    return tau


def check_constraints(q, dq):
    """
    Check if joint angle and velocity constraints are satisfied.

    Args:
        q: Joint angles [q1, q2, q3, q4]
        dq: Joint velocities [dq1, dq2, dq3, dq4]

    Returns:
        (violated, message): Tuple of (bool, str)
    """
    # Check hip angle limits (-120 deg to +30 deg)
    if q[0] < HIP_ANGLE_MIN: return True, f"q1={np.rad2deg(q[0]):.1f} deg < -120 deg"
    if q[0] > HIP_ANGLE_MAX: return True, f"q1={np.rad2deg(q[0]):.1f} deg > 30 deg"
    if q[2] < HIP_ANGLE_MIN: return True, f"q3={np.rad2deg(q[2]):.1f} deg < -120 deg"
    if q[2] > HIP_ANGLE_MAX: return True, f"q3={np.rad2deg(q[2]):.1f} deg > 30 deg"
    # Check knee angle limits (0 deg to +160 deg)
    if q[1] < KNEE_ANGLE_MIN: return True, f"q2={np.rad2deg(q[1]):.1f} deg < 0 deg"
    if q[1] > KNEE_ANGLE_MAX: return True, f"q2={np.rad2deg(q[1]):.1f} deg > 160 deg"
    if q[3] < KNEE_ANGLE_MIN: return True, f"q4={np.rad2deg(q[3]):.1f} deg < 0 deg"
    if q[3] > KNEE_ANGLE_MAX: return True, f"q4={np.rad2deg(q[3]):.1f} deg > 160 deg"
    # Check hip velocity limits (+/- 30 rad/s)
    if abs(dq[0]) > HIP_VEL_MAX: return True, f"|dq1|={abs(dq[0]):.1f} > 30 rad/s"
    if abs(dq[2]) > HIP_VEL_MAX: return True, f"|dq3|={abs(dq[2]):.1f} > 30 rad/s"
    # Check knee velocity limits (+/- 15 rad/s)
    if abs(dq[1]) > KNEE_VEL_MAX: return True, f"|dq2|={abs(dq[1]):.1f} > 15 rad/s"
    if abs(dq[3]) > KNEE_VEL_MAX: return True, f"|dq4|={abs(dq[3]):.1f} > 15 rad/s"
    return False, ""


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def generate_plots(log, x0, distance, speed, final_t, output_dir):
    """
    Generate constraint verification plots for the final report.
    Shows that all physical constraints are satisfied during walking.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available - skipping plots")
        return

    print("\nGenerating constraint verification plots...")

    t = np.array(log['t'])

    # Create figure with 6 subplots (3 rows x 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'Task 2b: Forward Walking - Constraint Verification\n'
                 f'Distance: {distance:.3f}m | Speed: {speed:.3f}m/s | Duration: {final_t:.2f}s',
                 fontsize=14, fontweight='bold')

    # -------------------------------------------------------------------------
    # Plot 1: COM X Position vs Time (top-left)
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(t, log['x'], 'b-', linewidth=1.5, label='COM x position')
    ax.axhline(x0, color='g', linestyle='--', alpha=0.5, label=f'Start: x={x0:.3f}m')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X Position (m)')
    ax.set_title('COM X Position vs Time')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 2: COM X Velocity vs Time with target line (top-right)
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(t, log['vx'], 'b-', linewidth=1, label='Instantaneous velocity')
    ax.axhline(0.5, color='r', linestyle='--', linewidth=2, label='Target: 0.5 m/s')
    ax.axhline(speed, color='g', linestyle='--', linewidth=2, label=f'Average: {speed:.3f} m/s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X Velocity (m/s)')
    ax.set_title('COM X Velocity vs Time')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 3: Joint Angles with Constraint Limits (middle-left)
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(t, log['q1'], 'b-', label='q1 (left hip)', linewidth=1)
    ax.plot(t, log['q2'], 'r-', label='q2 (left knee)', linewidth=1)
    ax.plot(t, log['q3'], 'g-', label='q3 (right hip)', linewidth=1)
    ax.plot(t, log['q4'], 'm-', label='q4 (right knee)', linewidth=1)
    # Draw constraint limits
    ax.axhline(-120, color='b', linestyle='--', alpha=0.5, label='Hip min: -120 deg')
    ax.axhline(30, color='b', linestyle='--', alpha=0.5, label='Hip max: +30 deg')
    ax.axhline(0, color='r', linestyle='--', alpha=0.5, label='Knee min: 0 deg')
    ax.axhline(160, color='r', linestyle='--', alpha=0.5, label='Knee max: +160 deg')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Joint Angles vs Limits (ALL WITHIN BOUNDS)')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 4: Joint Velocities with Constraint Limits (middle-right)
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.plot(t, log['dq1'], 'b-', label='dq1 (left hip)', linewidth=1)
    ax.plot(t, log['dq2'], 'r-', label='dq2 (left knee)', linewidth=1)
    ax.plot(t, log['dq3'], 'g-', label='dq3 (right hip)', linewidth=1)
    ax.plot(t, log['dq4'], 'm-', label='dq4 (right knee)', linewidth=1)
    # Draw constraint limits
    ax.axhline(30, color='b', linestyle='--', alpha=0.5, label='Hip limit: +/-30 rad/s')
    ax.axhline(-30, color='b', linestyle='--', alpha=0.5)
    ax.axhline(15, color='r', linestyle='--', alpha=0.5, label='Knee limit: +/-15 rad/s')
    ax.axhline(-15, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (rad/s)')
    ax.set_title('Joint Velocities vs Limits (ALL WITHIN BOUNDS)')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 5: Joint Torques with Saturation Limits (bottom-left)
    # -------------------------------------------------------------------------
    ax = axes[2, 0]
    ax.plot(t, log['tau1'], 'b-', label='tau1 (left hip)', linewidth=1)
    ax.plot(t, log['tau2'], 'r-', label='tau2 (left knee)', linewidth=1)
    ax.plot(t, log['tau3'], 'g-', label='tau3 (right hip)', linewidth=1)
    ax.plot(t, log['tau4'], 'm-', label='tau4 (right knee)', linewidth=1)
    # Draw saturation limits
    ax.axhline(30, color='b', linestyle='--', alpha=0.5, label='Hip limit: +/-30 Nm')
    ax.axhline(-30, color='b', linestyle='--', alpha=0.5)
    ax.axhline(60, color='r', linestyle='--', alpha=0.5, label='Knee limit: +/-60 Nm')
    ax.axhline(-60, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Joint Torques vs Limits (SATURATION APPLIED)')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 6: Trunk Pitch Angle (bottom-right)
    # -------------------------------------------------------------------------
    ax = axes[2, 1]
    ax.plot(t, log['pitch'], 'b-', linewidth=1.5)
    ax.axhline(0, color='g', linestyle='--', label='Desired: 0 deg')
    ax.axhline(60, color='r', linestyle='--', alpha=0.7, label='Fall limit: +/-60 deg')
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
    plot_path = output_dir / "task2b_plots.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved: {plot_path}")


# ============================================================================
# MAIN SIMULATION FUNCTION
# ============================================================================

def run_forward_walk(xml_path, duration=10.0, video_name="task2b_forward.mp4"):
    """
    Run forward walking simulation with the BEST tuned parameters.

    This controller uses:
    - Split stance initialization for stability
    - Sinusoidal hip oscillation with opposite phase between legs
    - Pitch stabilization via hip torque feedback
    - NO knee lift (shuffling gait)

    Args:
        xml_path: Path to biped_robot.xml model file
        duration: Simulation duration in seconds
        video_name: Output video filename

    Returns:
        Dictionary with results (time, distance, speed, log)
    """
    print(f"\n{'=' * 60}")
    print("FORWARD WALKING - Task 2b - BEST VERSION")
    print("Params: freq=1.2Hz, amp=11 deg, lean=-8 deg, Kp=70, Kd=17")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Load MuJoCo model and create data structure
    # -------------------------------------------------------------------------
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    # -------------------------------------------------------------------------
    # Initialize robot in split stance configuration
    # This asymmetric stance provides a stable support polygon
    # -------------------------------------------------------------------------
    mj.mj_resetData(model, data)

    # Split stance joint angles (proven stable configuration)
    q1_init, q2_init = np.deg2rad(-50), np.deg2rad(40)  # Left leg: foot back
    q3_init, q4_init = np.deg2rad(-5), np.deg2rad(55)  # Right leg: foot forward
    data.qpos[3:7] = [q1_init, q2_init, q3_init, q4_init]

    # Compute foot positions to set correct trunk height
    lf_x, lf_z, rf_x, rf_z = foot_positions_fk(q1_init, q2_init, q3_init, q4_init)

    # Set trunk position: centered over support polygon, feet on ground
    data.qpos[0] = -(lf_x + rf_x) / 2  # Center x over feet
    data.qpos[1] = HIP_OFFSET - min(lf_z, rf_z) + FOOT_R + 0.002 - TRUNK_Z0  # Height
    data.qpos[2] = 0.0  # Start upright (zero pitch)
    data.qvel[:] = 0  # Zero initial velocity

    mj.mj_forward(model, data)
    q0 = data.qpos[3:7].copy()  # Store initial joint configuration

    # -------------------------------------------------------------------------
    # PD control gains for joint position tracking
    # -------------------------------------------------------------------------
    Kp = np.array([200.0, 150.0, 200.0, 150.0])  # Proportional gains
    Kd = np.array([20.0, 15.0, 20.0, 15.0])  # Derivative gains

    # -------------------------------------------------------------------------
    # Settling phase: let robot stabilize before walking
    # Uses gravity compensation + PD control + pitch stabilization
    # -------------------------------------------------------------------------
    for _ in range(300):
        q, dq = data.qpos[3:7], data.qvel[3:7]
        pitch, dpitch = data.qpos[2], data.qvel[2]

        # Gravity compensation + PD control
        tau = gravity_compensation(model, data) + Kp * (q0 - q) - Kd * dq

        # Pitch stabilization during settling
        tau[0] += 50 * (0 - pitch) - 12 * dpitch
        tau[2] += 50 * (0 - pitch) - 12 * dpitch

        data.ctrl[:] = saturate_torques(tau)
        mj.mj_step(model, data)

    # Reset velocities after settling
    data.qvel[:] = 0
    mj.mj_forward(model, data)

    # Record starting position for distance calculation
    x0 = data.qpos[0]
    q0 = data.qpos[3:7].copy()

    # -------------------------------------------------------------------------
    # WALKING GAIT PARAMETERS (BEST TUNED VALUES)
    # -------------------------------------------------------------------------
    freq = 1.2  # Oscillation frequency (Hz)
    amp = np.deg2rad(11)  # Hip oscillation amplitude (rad)
    lean = np.deg2rad(-8)  # Forward lean angle (rad) - negative = forward
    pitch_kp, pitch_kd = 70.0, 17.0  # Pitch control gains

    # -------------------------------------------------------------------------
    # Video recording setup with TRACKING CAMERA
    # -------------------------------------------------------------------------
    video_path = OUTPUT_DIR / video_name
    renderer = mj.Renderer(model, width=640, height=480)
    writer = imageio.get_writer(str(video_path), fps=60)
    dt = model.opt.timestep
    steps_per_frame = max(1, int(1.0 / (60 * dt)))

    # Create tracking camera that follows the robot
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "trunk")
    cam.distance = 1.5  # Distance from robot
    cam.azimuth = 90  # Side view
    cam.elevation = -10  # Slightly above
    cam.lookat[2] = 0.4  # Look at trunk height

    # -------------------------------------------------------------------------
    # Data logging for constraint verification plots
    # -------------------------------------------------------------------------
    log = {
        't': [], 'x': [], 'vx': [], 'pitch': [],
        'q1': [], 'q2': [], 'q3': [], 'q4': [],
        'dq1': [], 'dq2': [], 'dq3': [], 'dq4': [],
        'tau1': [], 'tau2': [], 'tau3': [], 'tau4': []
    }

    step = 0
    termination_reason = None

    # -------------------------------------------------------------------------
    # MAIN SIMULATION LOOP
    # -------------------------------------------------------------------------
    while data.time < duration:
        t = data.time

        # Get current state
        q = data.qpos[3:7].copy()
        dq = data.qvel[3:7].copy()
        pitch, dpitch = data.qpos[2], data.qvel[2]

        # Check physical constraints (terminate if violated)
        violated, msg = check_constraints(q, dq)
        if violated:
            termination_reason = msg
            print(f"  t={t:.2f}s: CONSTRAINT: {msg}")
            break

        # Check for fall (pitch > 60 degrees)
        if abs(pitch) > np.deg2rad(60):
            termination_reason = f"FELL (pitch={np.rad2deg(pitch):.1f} deg)"
            print(f"  t={t:.2f}s: {termination_reason}")
            break

        # ---------------------------------------------------------------------
        # GAIT GENERATION: Sinusoidal hip oscillation
        # ---------------------------------------------------------------------
        phase = 2 * np.pi * freq * t
        q_des = q0.copy()

        # Hip oscillation with opposite phase between legs
        # This creates the stepping motion
        q_des[0] = q0[0] + lean - amp * np.sin(phase)  # Left hip
        q_des[2] = q0[2] + lean + amp * np.sin(phase)  # Right hip
        # NO knee lift for forward walking (shuffling gait)

        # ---------------------------------------------------------------------
        # CONTROL: PD + Gravity Compensation + Pitch Stabilization
        # ---------------------------------------------------------------------
        tau = gravity_compensation(model, data) + Kp * (q_des - q) - Kd * dq

        # Pitch stabilization: add corrective torque to hips
        tau[0] += pitch_kp * (0 - pitch) - pitch_kd * dpitch
        tau[2] += pitch_kp * (0 - pitch) - pitch_kd * dpitch

        # Apply torque saturation (REQUIRED by PDF)
        tau_sat = saturate_torques(tau)
        data.ctrl[:] = tau_sat

        # ---------------------------------------------------------------------
        # Log data for plots BEFORE stepping
        # ---------------------------------------------------------------------
        log['t'].append(t)
        log['x'].append(data.qpos[0])
        log['vx'].append(data.qvel[0])
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

        # Step simulation
        mj.mj_step(model, data)

        # Record video frame with tracking camera
        if step % steps_per_frame == 0:
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())

        # Print progress every second
        if step % int(1.0 / dt) == 0:
            print(f"  t={t:.1f}s: x={data.qpos[0]:.3f}m, v={data.qvel[0]:.3f}m/s")

        step += 1

    # -------------------------------------------------------------------------
    # Cleanup and results
    # -------------------------------------------------------------------------
    writer.close()
    renderer.close()

    # Calculate final results
    final_t = data.time
    distance = data.qpos[0] - x0
    speed = abs(distance) / max(final_t, 0.01)

    # -------------------------------------------------------------------------
    # Print results summary
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)
    print(f"Duration: {final_t:.2f}s")
    print(f"Distance: {distance:.3f}m (positive = forward)")
    print(f"Speed: {speed:.3f}m/s")
    if termination_reason:
        print(f"Terminated: {termination_reason}")

    print(f"\nTask 2b Requirements:")
    print(f"  Duration >= 5s: {final_t:.2f}s {'PASS' if final_t >= 5 else 'FAIL'}")
    print(f"  Speed >= 0.5m/s: {speed:.3f}m/s {'PASS' if speed >= 0.5 else 'FAIL'}")

    # -------------------------------------------------------------------------
    # Generate constraint verification plots
    # -------------------------------------------------------------------------
    generate_plots(log, x0, distance, speed, final_t, OUTPUT_DIR)

    return {'time': final_t, 'distance': distance, 'speed': speed, 'video': str(video_path), 'log': log}


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Find XML file in same directory as script
    xml_path = str(SCRIPT_DIR / "biped_robot.xml")

    if not Path(xml_path).exists():
        print(f"ERROR: Could not find {xml_path}")
        print("Please place biped_robot.xml in the same folder as this script.")
        exit(1)

    # Run simulation
    result = run_forward_walk(xml_path, 10.0, "task2b_forward.mp4")

    print(f"\nOutput files in: {OUTPUT_DIR}")
    print(f"  - task2b_forward.mp4 (video)")
    print(f"  - task2b_plots.png (constraint verification)")