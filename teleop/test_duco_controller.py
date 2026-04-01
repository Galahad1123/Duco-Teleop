import os
import sys
import time
import math
import argparse

import numpy as np
import logging_mp

logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

# ---------------------------------------------------------------------
# 路径设置：与 teleop_arm.py 风格一致
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from teleop.robot_control.duco_arm_controller import duco_ArmController


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-interface", type=str, default=None)
    parser.add_argument("--duration", type=float, default=20.0, help="test duration in seconds")
    parser.add_argument("--frequency", type=float, default=50.0, help="command update frequency")
    parser.add_argument("--joint", type=int, default=0, help="which joint to excite, 0~5")
    parser.add_argument("--amplitude", type=float, default=0.20, help="sin amplitude in rad")
    parser.add_argument("--period", type=float, default=4.0, help="sin period in seconds")
    parser.add_argument("--hold", type=float, default=2.0, help="hold initial pose before motion")
    return parser


def main():
    args = build_parser().parse_args()

    if not (0 <= args.joint <= 5):
        raise ValueError(f"--joint must be in [0, 5], got {args.joint}")

    logger_mp.info("===== Duco controller minimal DDS test =====")

    # -------------------------------------------------------------
    # 1) 初始化 DDS
    # 与 sim_main.py 匹配，仿真模式必须用 Channel 1
    # -------------------------------------------------------------
    ChannelFactoryInitialize(1, networkInterface=args.network_interface)
    logger_mp.info("DDS initialized on channel 1")

    # -------------------------------------------------------------
    # 2) 初始化单臂控制器（仿真模式）
    # -------------------------------------------------------------
    arm_ctrl = duco_ArmController(
        motion_mode=False,
        simulation_mode=True,
    )

    # 让控制器启动时速度上限平滑拉起
    arm_ctrl.speed_gradual_max()

    # -------------------------------------------------------------
    # 3) 读取当前状态，作为基准位姿
    # -------------------------------------------------------------
    q0 = arm_ctrl.get_current_arm_q()
    dq0 = arm_ctrl.get_current_arm_dq()

    logger_mp.info(f"Initial q  = {np.round(q0, 4)}")
    logger_mp.info(f"Initial dq = {np.round(dq0, 4)}")

    q_cmd = q0.copy()
    tau_ff = np.zeros(6, dtype=np.float64)

    # -------------------------------------------------------------
    # 4) 先保持当前位姿一小段时间
    # -------------------------------------------------------------
    logger_mp.info(f"Holding current pose for {args.hold:.1f}s ...")
    t_hold_start = time.time()
    while time.time() - t_hold_start < args.hold:
        arm_ctrl.ctrl_arm(q_cmd, tau_ff)
        time.sleep(1.0 / args.frequency)

    # -------------------------------------------------------------
    # 5) 对单个关节做正弦激励
    #    目标：验证 lowcmd -> sim 这一整条链是否通
    # -------------------------------------------------------------
    logger_mp.info(
        f"Start sine motion: joint={args.joint}, amp={args.amplitude:.3f} rad, "
        f"period={args.period:.2f}s, duration={args.duration:.1f}s"
    )

    t0 = time.time()
    step = 0

    try:
        while True:
            now = time.time()
            t = now - t0
            if t > args.duration:
                break

            q_cmd[:] = q0
            q_cmd[args.joint] = q0[args.joint] + args.amplitude * math.sin(2.0 * math.pi * t / args.period)

            arm_ctrl.ctrl_arm(q_cmd, tau_ff)

            if step % int(max(1, args.frequency)) == 0:
                q_cur = arm_ctrl.get_current_arm_q()
                logger_mp.info(
                    f"t={t:5.2f}s | q_cmd={np.round(q_cmd, 4)} | q_cur={np.round(q_cur, 4)}"
                )

            step += 1
            time.sleep(1.0 / args.frequency)

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt received, stopping test ...")

    # -------------------------------------------------------------
    # 6) 回到初始位姿
    # -------------------------------------------------------------
    logger_mp.info("Returning to initial pose ...")
    for _ in range(int(2.0 * args.frequency)):
        arm_ctrl.ctrl_arm(q0, tau_ff)
        time.sleep(1.0 / args.frequency)

    logger_mp.info("Test finished.")


if __name__ == "__main__":
    main()