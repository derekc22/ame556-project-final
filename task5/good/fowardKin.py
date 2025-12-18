import numpy as np
# ============================================================
# FORWARD KINEMATICS
# ============================================================
def fK(pos, q, l, a):
    """
    Compute foot positions using homogeneous transforms
    Z-UP COORDINATE SYSTEM
    
    pos: [x, z, theta] - trunk COM position and orientation  
    q: [q1, q2, q3, q4] - joint angles
    l: link length (0.22m)
    a: trunk total height (0.25m)
    
    MATLAB: H = trans * rot (translate then rotate)
    """
    x, z, theta = pos
    q1, q2, q3, q4 = q
    
    def rot2D(angle):
        """2D rotation matrix"""
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
    
    def homo(d, r):
        """
        Homogeneous transform
        H = trans * rot (translate THEN rotate)
        d: [dx, dz] translation vector
        r: 2x2 rotation matrix
        """
        trans = np.array([
            [1, 0, d[0]],
            [0, 1, d[1]],
            [0, 0, 1]
        ])
        rot = np.array([
            [r[0, 0], r[0, 1], 0],
            [r[1, 0], r[1, 1], 0],
            [0, 0, 1]
        ])
        return trans @ rot
    
    # Body frame in world (Z-up: [x, z])
    H_b = homo(np.array([x, z]), rot2D(theta))
    
    # Hip joints (down by a/2 in body frame)
    H_b_q1 = homo(np.array([0, -a/2]), rot2D(q1))
    H_b_q3 = homo(np.array([0, -a/2]), rot2D(q3))
    
    # Knees
    H_q1_1 = homo(np.array([0, -l]), rot2D(q2))
    H_q3_3 = homo(np.array([0, -l]), rot2D(q4))
    
    # Feet
    H_q1_q2 = homo(np.array([0, -l]), rot2D(0))
    H_q3_q4 = homo(np.array([0, -l]), rot2D(0))
    
    # Global Frames 
    H_o_q1 = H_b @ H_b_q1
    H_o_1 = H_o_q1 @ H_q1_1
    H_o_2 = H_o_1 @ H_q1_q2
    
    H_o_q3 = H_b @ H_b_q3
    H_o_3 = H_o_q3 @ H_q3_3
    H_o_4 = H_o_3 @ H_q3_q4
    
    # Extract positions
    p1 = np.array([H_o_1[0, 2], H_o_1[1, 2]])
    p2 = np.array([H_o_2[0, 2], H_o_2[1, 2]])
    p3 = np.array([H_o_3[0, 2], H_o_3[1, 2]])
    p4 = np.array([H_o_4[0, 2], H_o_4[1, 2]])
    
    return {'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4}