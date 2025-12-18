import matplotlib.pyplot as plt
import numpy as np

def plot_results(T, X, Z, Zref, TH, Qlog, DQlog, TAU):
    """Generate plots"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Results', fontsize=14, fontweight='bold')
    
    # Height tracking
    ax = axes[0, 0]
    ax.plot(T, Z, 'b-', linewidth=2, label='Actual')
    ax.plot(T, Zref, 'r--', linewidth=2, label='Desired')
    ax.fill_between(T, Z, Zref, alpha=0.2, color='red')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Height [m]')
    ax.set_title('Trunk Height Tracking')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # X position vs time
    ax = axes[0, 1]
    ax.plot(T, X, 'k-', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('X Position [m]')
    ax.set_title('Trunk X Position')
    ax.grid(True, alpha=0.3)
    
    # Body angle
    ax = axes[1, 0]
    ax.plot(T, np.rad2deg(TH), 'g-', linewidth=2)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angle [deg]')
    ax.set_title('Body Orientation')
    ax.grid(True, alpha=0.3)
    
    # Joint angles
    ax = axes[1, 1]
    labels = ['Hip L', 'Knee L', 'Hip R', 'Knee R']
    for i in range(4):
        ax.plot(T, np.rad2deg(Qlog[:, i]), label=labels[i])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angle [deg]')
    ax.set_title('Joint Angles')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Joint velocities
    ax = axes[2, 0]
    for i in range(4):
        ax.plot(T, np.rad2deg(DQlog[:, i]), label=labels[i])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [deg/s]')
    ax.set_title('Joint Velocities')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Torques
    ax = axes[2, 1]
    for i in range(4):
        ax.plot(T, TAU[:, i], label=labels[i])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Torque [Nm]')
    ax.set_title('Joint Torques')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    print("\nPlot saved: results.png")
    plt.show()
