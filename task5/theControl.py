import numpy as np
import mujoco
import mujoco.viewer
import time
import cv2

from OCController import occontroller, OCParams
from good.fowardKin import fK
from good.limits import limits
import matplotlib.pyplot as plt
from good.plot import plot_results


def main():

    # parameters
    params = OCParams()
    # model
    model = mujoco.MjModel.from_xml_path("biped_robot.xml")
    data = mujoco.MjData(model)

    # ICs
    data.qpos[:] = 0.0
    data.qpos[0] = 0.0       # x
    data.qpos[1] = 0.45      # z (height)
    data.qpos[2] = 0.0       # theta

    # Initial joint angles from MATLAB: [q1, q2, q3, q4]
    q1_init = -np.pi/3      # -60 degrees
    q2_init = 1.38          # ~79 degrees  
    q3_init = -np.pi/6      # -30 degrees
    q4_init = np.pi/2       # 90 degrees

    data.qpos[3:7] = [q1_init, q2_init, q3_init, q4_init]
    mujoco.mj_forward(model, data)

    forwardK = fK(
        [data.qpos[0], data.qpos[1], data.qpos[2]],
        data.qpos[3:7],
        params.l,
        params.a
    )
    print("Initial foot positions (x,z):", forwardK["p2"], forwardK["p4"])
    print(f"Initial body height: {data.qpos[1]:.3f}m\n")

    # for plotting
    T, X, Z, TH = [], [], [], []
    Qlog, DQlog, TAU = [], [], []
    
    # Video
    record_video = True
    video_fps = 30
    video_dt = 1.0 / video_fps
    next_frame_time = 0.0
    frames = []
    
    # Offscreen renderer for video
    if record_video:
        width, height = 1280, 720
        video_renderer = mujoco.Renderer(model, width=width, height=height)
        video_cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(video_cam)
        video_cam.azimuth = 90.0
        video_cam.elevation = 0.0
        video_cam.distance = 3.5
        video_cam.lookat[:] = [data.qpos[0], 0.0, 0.5]
    
    # Controller state
    prev_ctrl_t = -params.dt
    tau = np.zeros(4)
    
    # Warmup period - gentler limits initially or no limits
    warmup_duration = 1.0
    
    # Debugging
    print("="*60)
    print("OBSTACLE COURSE SIMULATION - REAL-TIME MODE")
    print("="*60)
    print("Obstacles:")
    print("  - Box jump at x=2.0m (height 0.4m)")
    print("  - Gap at x=5.0m (width 0.4m)")
    print("  - Goal at x=7.4m")
    print("\nSafety limits ACTIVE:")
    print("  - Joint angles: q1,q3 [-120°,30°], q2,q4 [0°,160°]")
    print("  - Joint velocities: q1,q3 [±30 rad/s], q2,q4 [±15 rad/s]")
    print(f"  - Warmup period: {warmup_duration}s (3x relaxed velocity limits)")
    print("    During warmup: q1,q3 [±90 rad/s], q2,q4 [±45 rad/s]")
    print(f"\n{' Video recording: ENABLED' if record_video else 'Video recording: disabled'}")
    print("\nPress ESC to stop simulation")
    print("="*60 + "\n")

    last_report_time = 0.0
    simulation_active = True
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera to side view
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = 0.0
        viewer.cam.distance = 3.5
        viewer.cam.lookat[:] = [3.5, 0.0, 0.5]
        
        while viewer.is_running() and simulation_active:
            step_start = time.time()
            t_now = data.time
            
            # limits
            safe, msg = limits(data, t_now, warmup_duration, verbose=True)
            if not safe:
                print(f"\n SIMULATION STOPPED at t={t_now:.3f}s")
                print(f"Reason: {msg}\n")
                simulation_active = False
                break
            
            # controller
            if t_now >= prev_ctrl_t + params.dt - 1e-9:
                
                # state vector
                state = np.concatenate([
                    data.qpos[0:3],   # x, z, theta
                    data.qpos[3:7],   # q1, q2, q3, q4
                    data.qvel[0:3],   # dx, dz, dtheta
                    data.qvel[3:7]    # dq1, dq2, dq3, dq4
                ])
                
                # Call controller
                try:
                    tau_cmd, foot_out, traj_des = occontroller(
                        state, t_now, params
                    )
                    
                    # Per-joint torque limits [q1, q2, q3, q4]
                    tau_max = np.array([30.0, 60.0, 30.0, 60.0])
                    tau_min = -tau_max

                    # Clip absolute torque
                    tau_cmd = np.clip(tau_cmd, tau_min, tau_max)

                    # Limit torque rate of change for smooth startup
                    if t_now < warmup_duration:
                        # More conservative during warmup
                        max_tau_change = np.array([15.0, 15.0, 15.0, 15.0])  # Nm per step
                    else:
                        # Slightly looser after warmup
                        max_tau_change = np.array([25.0, 25.0, 25.0, 25.0])

                    tau_change = tau_cmd - tau
                    tau_change = np.clip(tau_change, -max_tau_change, max_tau_change)
                    tau = tau + tau_change

                    
                except Exception as e:
                    print(f"[ERROR] Controller failed at t={t_now:.3f}s: {e}")
                    import traceback
                    traceback.print_exc()
                    tau = np.zeros(4)
                
                prev_ctrl_t = t_now
            
            # next step
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            
            # log
            T.append(data.time)
            X.append(data.qpos[0])
            Z.append(data.qpos[1])
            TH.append(data.qpos[2])
            Qlog.append(data.qpos[3:7].copy())
            DQlog.append(data.qvel[3:7].copy())
            TAU.append(tau.copy())
            
            # debugging info
            if t_now < warmup_duration and t_now - last_report_time >= 0.2:
                max_dq = np.max(np.abs(data.qvel[3:7]))
                print(f"[WARMUP t={t_now:.2f}s] x={data.qpos[0]:.2f}m, z={data.qpos[1]:.2f}m, max|dq|={max_dq:.1f} rad/s")
                last_report_time = t_now
            elif t_now >= warmup_duration and t_now - last_report_time >= 1.0:
                print(f"[t={t_now:.1f}s] Position: x={data.qpos[0]:.2f}m, z={data.qpos[1]:.2f}m, θ={np.rad2deg(data.qpos[2]):.1f}°")
                last_report_time = t_now
            
            # Warmup complete announcement
            if warmup_duration - 0.01 < t_now < warmup_duration + 0.01:
                print(f"\n{'='*60}")
                print("WARMUP COMPLETE - Full safety limits now active")
                print(f"{'='*60}\n")
            
            # video
            if record_video and t_now >= next_frame_time:
                # Update camera to follow robot
                video_cam.lookat[:] = [
                    data.qpos[0],   
                    0.0,            
                    0.5
                ]
                video_renderer.update_scene(data, camera=video_cam)
                frame = video_renderer.render()
                frames.append(frame.copy())
                next_frame_time += video_dt
            
            # viewer
            viewer.sync()
            
            # Real-time pacing
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # completition report
    final_x = X[-1] if X else 0.0
    final_time = T[-1] if T else 0.0
    goal_x = 7.4
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Duration: {final_time:.2f}s")
    print(f"Final position: x={final_x:.2f}m (goal: {goal_x}m)")
    print(f"Distance covered: {final_x:.2f}m")
    if final_time > 0:
        print(f"Average speed: {final_x/final_time:.2f}m/s")
    
    if not simulation_active:
        print("\n Simulation terminated due to safety violation")
    elif final_x >= goal_x - 0.5:
        print("\n SUCCESS: Robot reached the goal!")
    elif final_x >= params.jump.x_end:
        print("\n PARTIAL: Robot cleared the box jump")
    else:
        print("\n Robot did not complete the course")
    print("="*60)
    
    # data
    if len(T) > 0:
        print("\nSaving simulation data...")
        
        data_dict = {
            'time': np.array(T),
            'x': np.array(X),
            'z': np.array(Z),
            'theta': np.array(TH),
            'q': np.array(Qlog),
            'dq': np.array(DQlog),
            'tau': np.array(TAU)
        }
        
        # save
        if record_video and len(frames) > 0:
            print(f"\n Saving video ({len(frames)} frames)...")
            
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                "task5.mp4",
                fourcc,
                video_fps,
                (width, height)
            )
            
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            out.release()
            if video_renderer is not None:
                video_renderer.close()
            
            print(f" Video saved as task5.mp4 ({len(frames)/video_fps:.1f}s @ {video_fps}fps)")
        
        # Optional: Create plots
        try:            
            Zref_arr = np.ones_like(data_dict['z']) * params.ypos_des
            plot_results(
                data_dict['time'],
                data_dict['x'],
                data_dict['z'],
                Zref_arr,
                data_dict['theta'],
                data_dict['q'],
                data_dict['dq'],
                data_dict['tau']
            )
            print("Plots generated successfully")
        except Exception as e:
            print(f"Note: Could not generate plots: {e}")
        
        # Trajectory plot with terrain
        try:            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot terrain
            course = params.course_geom
            ax.plot(course[0, :], course[1, :], 'k-', linewidth=2, label='Terrain')
            ax.fill_between(course[0, :], -0.5, course[1, :], alpha=0.3, color='brown')
            
            # Plot robot trajectory
            ax.plot(data_dict['x'], data_dict['z'], 'b-', linewidth=2, label='Robot CoM')
            
            # Mark start and end
            ax.plot(data_dict['x'][0], data_dict['z'][0], 'go', markersize=10, label='Start')
            ax.plot(data_dict['x'][-1], data_dict['z'][-1], 'ro', markersize=10, label='End')
            
            # Mark obstacles
            ax.axvline(params.jump.x_start, color='r', linestyle='--', alpha=0.5, label='Jump Zone')
            ax.axvline(params.jump.x_end, color='r', linestyle='--', alpha=0.5)
            ax.axvline(params.hole.x_start, color='orange', linestyle='--', alpha=0.5, label='Gap Zone')
            ax.axvline(params.hole.x_end, color='orange', linestyle='--', alpha=0.5)
            
            # Mark goal
            ax.plot(7.4, 0.05, 'g*', markersize=20, label='Goal')
            
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Z Height (m)')
            ax.set_title('Obstacle Course Trajectory')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.5, 1.5)
            
            plt.tight_layout()
            plt.savefig('task5.png', dpi=150)
            print("Trajectory plot saved to task5.png")
            plt.close()
            
        except Exception as e:
            print(f"Note: Could not generate trajectory plot: {e}")


if __name__ == "__main__":
    main()