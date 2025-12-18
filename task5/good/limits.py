import numpy as np

# ============================================================
# SAFETY LIMITS - STOP SIMULATION IF VIOLATED
# ============================================================
def limits(data, t_now, warmup_duration=0.5, verbose=True):
    """
    Check joint angle and velocity limits. Returns (safe, message).
    If unsafe, simulation should stop immediately.
    
    During warmup period, velocity limits are relaxed to allow stabilization.
    """
    # Joint angle limits [rad]
    # q1, q3 (hip): -120° to 30° = -2.094 to 0.524 rad
    # q2, q4 (knee): 0° to 160° = 0 to 2.793 rad
    q_min = np.array([-2.094, 0.0, -2.094, 0.0])
    q_max = np.array([0.524, 2.793, 0.524, 2.793])
    
    # Joint velocity limits [rad/s]
    base_limits = np.array([30.0, 15.0, 30.0, 15.0])
    
    # During warmup, use 3x higher limits to allow initial transients
    in_warmup = t_now < warmup_duration
    if in_warmup:
        warmup_factor = 3.0
        dq_max = base_limits * warmup_factor
    else:
        dq_max = base_limits
    
    q = data.qpos[3:7]
    dq = data.qvel[3:7]
    
    # Check angle limits (always enforced)
    for i in range(4):
        if q[i] < q_min[i] - 0.1 or q[i] > q_max[i] + 0.1:  # Small margin
            msg = f"SAFETY VIOLATION: Joint q{i+1} = {np.rad2deg(q[i]):.1f}° (limits: {np.rad2deg(q_min[i]):.1f}° to {np.rad2deg(q_max[i]):.1f}°)"
            if verbose:
                print(f"\n{'='*60}")
                print(f" {msg}")
                print(f"{'='*60}\n")
            return False, msg
    
    # Check velocity limits
    for i in range(4):
        if abs(dq[i]) > dq_max[i]:
            if in_warmup:
                # During warmup, just warn but don't stop
                if verbose and abs(dq[i]) > base_limits[i] * 2:  # Only warn if very high
                    print(f"[WARMUP] High velocity on q{i+1}: {dq[i]:.1f} rad/s (warmup limit: ±{dq_max[i]:.1f} rad/s)")
                # Don't stop during warmup
                continue
            else:
                # After warmup, enforce strict limits
                msg = f"SAFETY VIOLATION: Joint q{i+1} velocity = {dq[i]:.1f} rad/s (limit: ±{dq_max[i]:.1f} rad/s)"
                if verbose:
                    print(f"\n{'='*60}")
                    print(f" {msg}")
                    print(f"{'='*60}\n")
                return False, msg
    
    return True, ""