import os
import time
import numpy as np
import pinocchio as pin

from duco_arm_ik import DucoArmIK

def pose_error(T_sol, T_target):
    pos_err = np.linalg.norm(T_sol[:3, 3] - T_target[:3, 3])
    R_err = T_sol[:3, :3] @ T_target[:3, :3].T
    rot_err = np.linalg.norm(pin.log3(R_err))
    return pos_err, rot_err

def main():
    urdf_path = "/home/yofo/unitree/duco_support/urdf/duco_gcr5_910.urdf"
    model_dir = os.path.dirname(urdf_path)

    ik = DucoArmIK(
        urdf_path=urdf_path,
        model_dir=model_dir,
        ee_parent_joint="arm_1_joint_6",
        ee_frame_name="ee",
        ee_offset_xyz=(0.0, 0.0, 0.0),
    )

    q0 = np.zeros(6)
    dq0 = np.zeros(6)

    print("=== Test A: FK sanity ===")
    T0 = ik.fk(q0)
    print("T0 =\n", T0)

    print("\n=== Test B: single reachable perturbation ===")
    T_target = T0.copy()
    T_target[0, 3] += 0.02
    T_target[2, 3] += 0.02

    t_start = time.time()
    q_sol, tau_ff = ik.solve_ik(T_target, current_arm_q=q0, current_arm_dq=dq0)
    dt = time.time() - t_start

    T_sol = ik.fk(q_sol)
    pos_err, rot_err = pose_error(T_sol, T_target)

    print("q_sol =", q_sol)
    print("tau_ff =", tau_ff)
    print(f"solve_time = {dt*1000:.2f} ms")
    print(f"pos_err = {pos_err:.6f} m")
    print(f"rot_err = {rot_err:.6f} rad")

    print("\n=== Test C: local grid around T0 ===")
    q_prev = q0.copy()
    results = []

    for dx in [-0.02, 0.0, 0.02]:
        for dy in [-0.02, 0.0, 0.02]:
            for dz in [0.0, 0.02]:
                T_goal = T0.copy()
                T_goal[0, 3] += dx
                T_goal[1, 3] += dy
                T_goal[2, 3] += dz

                t0 = time.time()
                q_sol, _ = ik.solve_ik(T_goal, current_arm_q=q_prev, current_arm_dq=dq0)
                solve_time = time.time() - t0

                T_check = ik.fk(q_sol)
                pos_err, rot_err = pose_error(T_check, T_goal)
                dq_step = np.linalg.norm(q_sol - q_prev)

                results.append((dx, dy, dz, pos_err, rot_err, dq_step, solve_time))
                q_prev = q_sol.copy()

                print(
                    f"goal=({dx:+.2f},{dy:+.2f},{dz:+.2f}) "
                    f"pos_err={pos_err:.5f} "
                    f"rot_err={rot_err:.5f} "
                    f"dq_step={dq_step:.5f} "
                    f"time={solve_time*1000:.1f}ms"
                )

    avg_pos_err = np.mean([r[3] for r in results])
    avg_rot_err = np.mean([r[4] for r in results])
    avg_dq_step = np.mean([r[5] for r in results])
    avg_time = np.mean([r[6] for r in results])

    print("\n=== Summary ===")
    print(f"avg_pos_err = {avg_pos_err:.6f} m")
    print(f"avg_rot_err = {avg_rot_err:.6f} rad")
    print(f"avg_dq_step = {avg_dq_step:.6f}")
    print(f"avg_solve_time = {avg_time*1000:.2f} ms")


if __name__ == "__main__":
    main()