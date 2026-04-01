import os
import time
import copy
import numpy as np
import pinocchio as pin

from duco_arm_ik import DucoArmIK


np.set_printoptions(precision=5, suppress=True, linewidth=200)


def pose_error(T_sol, T_target):
    """
    返回：
    - pos_err_norm: 位置误差范数
    - rot_err_norm: 旋转误差范数(rad)
    - dp: 3维位置误差向量 = sol - target
    - w_err: 3维旋转误差向量（SO(3) log）
    """
    dp = T_sol[:3, 3] - T_target[:3, 3]
    pos_err_norm = np.linalg.norm(dp)

    R_err = T_sol[:3, :3] @ T_target[:3, :3].T
    w_err = pin.log3(R_err)
    rot_err_norm = np.linalg.norm(w_err)

    return pos_err_norm, rot_err_norm, dp, w_err


def print_pose(name, T):
    print(f"{name} =")
    print(T)


def summarize_results(results, title):
    if len(results) == 0:
        print(f"\n=== {title} ===")
        print("No results.")
        return

    pos_norm = np.array([r["pos_err"] for r in results])
    rot_norm = np.array([r["rot_err"] for r in results])
    solve_ms = np.array([r["solve_ms"] for r in results])
    dq_step = np.array([r["dq_step"] for r in results])
    dp_all = np.array([r["dp"] for r in results])
    dw_all = np.array([r["dw"] for r in results])

    print(f"\n=== {title} ===")
    print(f"num_cases         = {len(results)}")
    print(f"avg_pos_err       = {np.mean(pos_norm):.6f} m")
    print(f"max_pos_err       = {np.max(pos_norm):.6f} m")
    print(f"avg_rot_err       = {np.mean(rot_norm):.6f} rad")
    print(f"max_rot_err       = {np.max(rot_norm):.6f} rad")
    print(f"avg_solve_time    = {np.mean(solve_ms):.3f} ms")
    print(f"max_solve_time    = {np.max(solve_ms):.3f} ms")
    print(f"avg_dq_step       = {np.mean(dq_step):.6f}")
    print(f"max_dq_step       = {np.max(dq_step):.6f}")

    print("avg_abs_pos_axis  = "
          f"[{np.mean(np.abs(dp_all[:, 0])):.6f}, "
          f"{np.mean(np.abs(dp_all[:, 1])):.6f}, "
          f"{np.mean(np.abs(dp_all[:, 2])):.6f}] m")
    print("avg_abs_rot_axis  = "
          f"[{np.mean(np.abs(dw_all[:, 0])):.6f}, "
          f"{np.mean(np.abs(dw_all[:, 1])):.6f}, "
          f"{np.mean(np.abs(dw_all[:, 2])):.6f}] rad")


def run_single_case(ik, T_goal, q_seed, dq0, tag=""):
    t0 = time.time()
    q_sol, tau_ff = ik.solve_ik(T_goal, current_arm_q=q_seed, current_arm_dq=dq0)
    solve_ms = (time.time() - t0) * 1000.0

    T_sol = ik.fk(q_sol)
    pos_err, rot_err, dp, dw = pose_error(T_sol, T_goal)
    dq_step = np.linalg.norm(q_sol - q_seed)

    print(
        f"{tag:<28s} "
        f"pos={pos_err:.5f}m "
        f"rot={rot_err:.5f}rad "
        f"dp=[{dp[0]:+.4f},{dp[1]:+.4f},{dp[2]:+.4f}] "
        f"dw=[{dw[0]:+.4f},{dw[1]:+.4f},{dw[2]:+.4f}] "
        f"dq={dq_step:.5f} "
        f"time={solve_ms:.2f}ms"
    )

    return {
        "q_sol": q_sol,
        "tau_ff": tau_ff,
        "T_sol": T_sol,
        "pos_err": pos_err,
        "rot_err": rot_err,
        "dp": dp,
        "dw": dw,
        "dq_step": dq_step,
        "solve_ms": solve_ms,
    }


def make_ik(urdf_path, ee_offset_xyz):
    model_dir = os.path.dirname(urdf_path)
    ik = DucoArmIK(
        urdf_path=urdf_path,
        model_dir=model_dir,
        ee_parent_joint="arm_1_joint_6",
        ee_frame_name="ee",
        ee_offset_xyz=ee_offset_xyz,
    )
    return ik


def test_fk_and_self_consistency(ik, q0):
    print("\n=== Test A: FK sanity & self-consistency ===")
    T0 = ik.fk(q0)
    print_pose("T0", T0)

    pos_err, rot_err, dp, dw = pose_error(T0, T0)
    print(f"self-check pos_err={pos_err:.8f}, rot_err={rot_err:.8f}")
    return T0


def test_reachable_pose_recovery(ik, q_ref, dq0):
    """
    这是最关键的离线 sanity check：
    先用某个已知关节 q_ref 做 FK 得到 T_ref，
    再拿 T_ref 去做 IK，看能不能回到 q_ref 附近。

    这种测试比“在 T0 上直接平移一点点”更能排查：
    - 模型是否自洽
    - IK 是否在已知可达点上表现正常
    """
    print("\n=== Test B: reachable pose recovery ===")
    T_ref = ik.fk(q_ref)
    print_pose("T_ref(from q_ref)", T_ref)

    res = run_single_case(ik, T_ref, q_seed=np.zeros_like(q_ref), dq0=dq0, tag="recover q_ref from T_ref")

    q_sol = res["q_sol"]
    q_err = q_sol - q_ref
    print("q_ref =", q_ref)
    print("q_sol =", q_sol)
    print("q_err =", q_err)
    print(f"||q_err|| = {np.linalg.norm(q_err):.6f}")

    return T_ref, res


def test_axis_translation_grid(ik, T_base, q0, dq0, use_warm_start=True):
    """
    只保持姿态不变，分别沿 x/y/z 做小位移。
    目的是看：
    - 哪个轴的纯平移最好/最差
    - 你之前怀疑的 dz 问题是否稳定复现
    """
    print("\n=== Test C: axis translation sweep (orientation fixed) ===")

    deltas = [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03]
    results = []
    q_prev = q0.copy()

    for axis_i, axis_name in enumerate(["x", "y", "z"]):
        print(f"\n--- axis {axis_name} ---")
        for d in deltas:
            T_goal = T_base.copy()
            T_goal[axis_i, 3] += d

            seed = q_prev if use_warm_start else q0
            tag = f"{axis_name} {d:+.3f}"
            res = run_single_case(ik, T_goal, q_seed=seed, dq0=dq0, tag=tag)
            res["axis"] = axis_name
            res["delta"] = d
            results.append(res)

            if use_warm_start:
                q_prev = res["q_sol"].copy()

    summarize_results(results, "Summary C: axis translation sweep")

    for axis_name in ["x", "y", "z"]:
        sub = [r for r in results if r["axis"] == axis_name]
        summarize_results(sub, f"Summary C-{axis_name}")

    return results


def test_local_xyz_grid(ik, T_base, q0, dq0, use_warm_start=True):
    """
    在 T_base 附近做一个小网格，但这次额外打印每次误差向量。
    """
    print("\n=== Test D: local xyz grid ===")
    results = []
    q_prev = q0.copy()

    xs = [-0.02, 0.00, 0.02]
    ys = [-0.02, 0.00, 0.02]
    zs = [0.00, 0.02]

    for dx in xs:
        for dy in ys:
            for dz in zs:
                T_goal = T_base.copy()
                T_goal[0, 3] += dx
                T_goal[1, 3] += dy
                T_goal[2, 3] += dz

                seed = q_prev if use_warm_start else q0
                tag = f"grid ({dx:+.2f},{dy:+.2f},{dz:+.2f})"
                res = run_single_case(ik, T_goal, q_seed=seed, dq0=dq0, tag=tag)
                res["delta_xyz"] = (dx, dy, dz)
                results.append(res)

                if use_warm_start:
                    q_prev = res["q_sol"].copy()

    summarize_results(results, "Summary D: local xyz grid")

    # 单独看 dz=0 和 dz=+0.02 两组
    sub_dz0 = [r for r in results if abs(r["delta_xyz"][2] - 0.00) < 1e-9]
    sub_dz1 = [r for r in results if abs(r["delta_xyz"][2] - 0.02) < 1e-9]
    summarize_results(sub_dz0, "Summary D-1: dz = 0.00")
    summarize_results(sub_dz1, "Summary D-2: dz = +0.02")

    return results


def test_orientation_relaxation_proxy(ik, T_base, q0, dq0):
    """
    由于当前 DucoArmIK 没暴露 rot weight 开关，
    我们这里做一个“姿态放松代理测试”：

    构造一组“只改位置”的目标时，再额外给目标姿态施加一个很小的扰动，
    看位置误差是否明显下降。

    逻辑：
    - 如果稍微改一点姿态，位置误差明显更小，
      说明问题很可能是“6DoF 下固定姿态太硬”。
    - 如果依然不改善，更像 TCP / frame / 几何定义问题。
    """
    print("\n=== Test E: orientation-relaxation proxy ===")

    results_fixed = []
    results_relaxed = []

    # 重点盯你之前最可疑的 z 方向上抬
    cases = [
        ("z +0.02", np.array([0.00, 0.00, 0.02])),
        ("x +0.02, z +0.02", np.array([0.02, 0.00, 0.02])),
        ("y -0.02, z +0.02", np.array([0.00, -0.02, 0.02])),
    ]

    # 一个很小的额外姿态扰动：绕工具 x / y 轴转一点
    tiny_rotvecs = [
        np.array([0.08, 0.00, 0.00]),   # ~4.6 deg
        np.array([0.00, 0.08, 0.00]),   # ~4.6 deg
        np.array([0.05, 0.05, 0.00]),
    ]

    for name, dpos in cases:
        # 固定姿态
        T_goal_fixed = T_base.copy()
        T_goal_fixed[:3, 3] += dpos
        res_fixed = run_single_case(ik, T_goal_fixed, q_seed=q0, dq0=dq0, tag=f"fixed   {name}")
        results_fixed.append(res_fixed)

        # 给姿态一个小扰动
        for i, rv in enumerate(tiny_rotvecs):
            T_goal_relaxed = T_base.copy()
            T_goal_relaxed[:3, 3] += dpos
            R_pert = pin.exp3(rv)
            T_goal_relaxed[:3, :3] = R_pert @ T_goal_relaxed[:3, :3]

            res_relaxed = run_single_case(
                ik, T_goal_relaxed, q_seed=q0, dq0=dq0, tag=f"relaxed {name} #{i+1}"
            )
            res_relaxed["case_name"] = name
            results_relaxed.append(res_relaxed)

    summarize_results(results_fixed, "Summary E-1: fixed orientation")
    summarize_results(results_relaxed, "Summary E-2: tiny relaxed orientation")

    print("\nInterpretation:")
    print("- 如果 E-2 明显优于 E-1：更像“固定姿态太硬”。")
    print("- 如果 E-2 改善不明显：更像 TCP / ee frame / 坐标定义问题。")

    return results_fixed, results_relaxed


def test_tcp_offset_compare(urdf_path, offsets_to_test, q0, dq0, q_ref):
    """
    对比不同 ee_offset_xyz 下的结果。
    这是最直接排查 TCP 定义问题的方法。
    """
    print("\n=== Test F: compare different ee_offset_xyz ===")

    all_summaries = []

    for offset in offsets_to_test:
        print("\n" + "=" * 80)
        print(f"Testing ee_offset_xyz = {offset}")
        print("=" * 80)

        ik = make_ik(urdf_path, ee_offset_xyz=offset)

        T0 = test_fk_and_self_consistency(ik, q0)
        _, _ = test_reachable_pose_recovery(ik, q_ref, dq0)
        c_results = test_axis_translation_grid(ik, T0, q0, dq0, use_warm_start=False)

        avg_pos = np.mean([r["pos_err"] for r in c_results])
        avg_rot = np.mean([r["rot_err"] for r in c_results])

        all_summaries.append({
            "offset": offset,
            "avg_pos": avg_pos,
            "avg_rot": avg_rot,
        })

    print("\n=== Summary F: offset comparison ===")
    for s in all_summaries:
        print(
            f"offset={s['offset']} "
            f"avg_pos_err={s['avg_pos']:.6f} m "
            f"avg_rot_err={s['avg_rot']:.6f} rad"
        )

    print("\nInterpretation:")
    print("- 若某个 offset 明显让 z 方向平移误差下降，说明 TCP 定义确实是关键因素。")

    return all_summaries


def main():
    urdf_path = "/home/yofo/unitree/duco_support/urdf/duco_gcr5_910.urdf"

    # 先用你当前版本的默认 TCP
    ee_offset_xyz = (0.0, 0.0, 0.0)

    ik = make_ik(urdf_path, ee_offset_xyz=ee_offset_xyz)

    # q0 = np.zeros(6)
    q0_deg = np.array([-13.58, -21.84, 127.44, 99.98, 82.59, -5.08], dtype=np.float64)
    q0 = np.deg2rad(q0_deg)

    dq0 = np.zeros(6)

    # 选一个已知参考姿态：不要太激进，尽量在中间区域
    # q_ref = np.array([0.25, -0.35, 0.30, -0.20, 0.25, 0.10], dtype=np.float64)
    q_ref = q0

    # A. FK sanity
    T0 = test_fk_and_self_consistency(ik, q0)

    # B. 可达位姿恢复测试（非常重要）
    # T_ref, _ = test_reachable_pose_recovery(ik, q_ref, dq0)

    # C. 纯单轴平移扫描：判断哪个轴最差
    test_axis_translation_grid(ik, T0, q0, dq0, use_warm_start=False)

    # D. 局部 xyz 网格：复现你原来 dz 异常模式
    test_local_xyz_grid(ik, T0, q0, dq0, use_warm_start=True)

    # E. 姿态放松代理测试：判断是否是“固定姿态过硬”
    test_orientation_relaxation_proxy(ik, T0, q0, dq0)

    # F. TCP offset 对比测试
    # 这里先只给几组小偏置，你后面可自己扩充
    # offsets_to_test = [
    #     (0.0, 0.0, 0.0),
    #     (0.0, 0.0, 0.05),
    #     (0.0, 0.0, 0.10),
    #     (0.05, 0.0, 0.0),
    #     (0.0, 0.05, 0.0),
    # ]
    # test_tcp_offset_compare(urdf_path, offsets_to_test, q0, dq0, q_ref)


if __name__ == "__main__":
    main()