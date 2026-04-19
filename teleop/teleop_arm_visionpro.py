import time
import argparse
import threading
import os
import sys

import numpy as np
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from teleop.robot_control.duco_arm_controller import duco_ArmController
from teleop.robot_control.duco_arm_ik import DucoArmIK
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher
from sshkeyboard import listen_keyboard, stop_listening

from televuer import VisionProTVWrapperDuco

START = False
STOP = False
READY = False

def on_press(key):
    global START, STOP
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined.")

def get_state() -> dict:
    global START, STOP, READY
    return {"START": START, "STOP": STOP, "READY": READY}

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', type=float, default=30.0)
    parser.add_argument('--network-interface', type=str, default=None)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--ipc', action='store_true')
    parser.add_argument('--affinity', action='store_true')
    parser.add_argument('--motion', action='store_true')
    parser.add_argument('--hand-side', type=str, choices=['left','right'], default='right')
    parser.add_argument('--verbose-stream', action='store_true')
    parser.add_argument('--orientation-mode', type=str, choices=['fixed', 'teleop'], default='fixed',
                        help='fixed: only track position delta; teleop: track both position and wrist orientation')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only subscribe robot state and run IK debug, do not publish arm commands')
    parser.add_argument('--debug-print-every', type=int, default=10,
                        help='Print IK debug information every N control loops')
    return parser

def setup_input_listener(args):
    if args.ipc:
        ipc_server = IPC_Server(on_press=on_press, get_state=get_state)
        ipc_server.start()
        return ipc_server, None
    else:
        keyboard_thread = threading.Thread(target=listen_keyboard,
                                           kwargs={"on_press": on_press, "until": None, "sequential": False},
                                           daemon=True)
        keyboard_thread.start()
        return None, keyboard_thread

def setup_affinity():
    import psutil
    p = psutil.Process(os.getpid())
    p.cpu_affinity([0,1,2,3])
    try:
        p.nice(-20)
        logger_mp.info("Set high priority successfully.")
    except psutil.AccessDenied:
        logger_mp.warning("Failed to set high priority. Please run as root.")
    for child in p.children(recursive=True):
        try:
            child.cpu_affinity([5,6])
            child.nice(-20)
        except psutil.AccessDenied:
            pass

def get_target_pose(tele_data, hand_side: str):
    if tele_data is None:
        return None
    return tele_data.left_wrist_pose if hand_side=='left' else tele_data.right_wrist_pose

def wait_for_valid_pose(tv_wrapper, hand_side: str, timeout_sec: float=10.0):
    start_t = time.time()
    while (time.time() - start_t) < timeout_sec and not STOP:
        tele_data = tv_wrapper.get_tele_data()
        target_pose = get_target_pose(tele_data, hand_side)
        if target_pose is not None:
            return tele_data, target_pose
        time.sleep(0.02)
    return None, None

def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret

def main():
    global READY, START, STOP

    parser = build_parser()
    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    ipc_server = None
    listen_keyboard_thread = None
    tv_wrapper = None
    motion_switcher = None
    arm_ctrl = None

    try:
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            # ChannelFactoryInitialize(0, networkInterface=args.network_interface)
            ChannelFactoryInitialize(0, networkInterface="enp0s31f6", peerAddress="10.188.223.119")

        ipc_server, listen_keyboard_thread = setup_input_listener(args)

        # 使用 Duco 版 hand_pose wrapper
        tv_wrapper = VisionProTVWrapperDuco(ip="10.188.211.82", verbose=args.verbose_stream)

        if not args.motion:
            motion_switcher = MotionSwitcher()
            status, _ = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status==0 else 'Failed'}")

        duco_urdf_path = "/home/yofo/unitree/duco_support/urdf/duco_gcr5_910.urdf"
        duco_model_dir = os.path.dirname(duco_urdf_path)
        arm_ik = DucoArmIK(urdf_path=duco_urdf_path,
                           model_dir=duco_model_dir,
                           ee_parent_joint="arm_1_joint_6",
                           ee_frame_name="ee",
                           ee_offset_xyz=(0.0,0.0,0.0))

        arm_ctrl = duco_ArmController(
            motion_mode=args.motion,
            simulation_mode=args.sim,
            enable_command_stream=not args.dry_run,
        )

        # obs_joint_deg = np.array([-5.2,-20.8,109.6,99.2,90.5,-12.1], dtype=np.float64)
        obs_joint_deg = np.array([-8.25,-23.15,121.49,82.65,81.61,4.47], dtype=np.float64)
        obs_joint_rad = np.deg2rad(obs_joint_deg)
        robot_origin_pose = arm_ik.fk(obs_joint_rad)
        logger_mp.info(f"robot_origin_pose from obs_joint:\n{robot_origin_pose}")

        if args.affinity:
            setup_affinity()

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢 Press [r] to start syncing the robot with your movements.")
        logger_mp.info("🔴 Press [q] to stop and exit the program.")
        logger_mp.info("⚠️ Keep safe!")

        READY = True
        while not START and not STOP:
            time.sleep(0.033)
        if STOP: return

        logger_mp.info("🟠 Vision Pro pose buffering for 3.0 seconds...")
        init_tele_pose = None
        buffer_start = time.time()
        while (time.time() - buffer_start) < 3.0 and not STOP:
            tele_data = tv_wrapper.get_tele_data()
            current_pose = get_target_pose(tele_data, args.hand_side)
            if current_pose is not None:
                init_tele_pose = current_pose.copy()
            time.sleep(0.02)

        if init_tele_pose is None:
            logger_mp.warning("No valid Vision Pro pose during buffering, retry once...")
            _, init_tele_pose = wait_for_valid_pose(tv_wrapper, args.hand_side, timeout_sec=10.0)
        if init_tele_pose is None:
            raise RuntimeError("Failed to get initial Vision Pro wrist pose.")

        logger_mp.info(f"Buffered initial tele pose:\n{init_tele_pose}")
        tele_origin_inv = fast_mat_inv(init_tele_pose)
        logger_mp.info("Relative teleop origin initialized.")

        logger_mp.info("---------------------🚀 start Tracking 🚀------------------------")
        logger_mp.info(f"Orientation mode: {args.orientation_mode}")
        arm_ctrl.speed_gradual_max()

        loop_count = 0
        while not STOP:
            loop_start = time.time()
            tele_data = tv_wrapper.get_tele_data()
            current_tele_pose = get_target_pose(tele_data, args.hand_side)
            if current_tele_pose is None:
                logger_mp.warning("Current Vision Pro pose is None, skipping.")
                time.sleep(0.01)
                continue

            current_arm_q = arm_ctrl.get_current_arm_q()
            current_arm_dq = arm_ctrl.get_current_arm_dq()

            # 相对移动
            # tele_rel_pose = tele_origin_inv @ current_tele_pose
            # target_pose = robot_origin_pose @ tele_rel_pose
            delta_world = current_tele_pose[:3, 3] - init_tele_pose[:3, 3]
            target_pose = robot_origin_pose.copy()
            # target_pose[:3, :3] = robot_origin_pose[:3, :3]   # 先冻结机器人末端朝向
            if args.orientation_mode == 'teleop':
                target_pose[:3, :3] = current_tele_pose[:3, :3]
            else:
                target_pose[:3, :3] = robot_origin_pose[:3, :3]
            target_pose[:3, 3] = robot_origin_pose[:3, 3] + delta_world

            logger_mp.debug(f"current tele xyz = {current_tele_pose[:3,3]}")
            logger_mp.debug(f"target xyz = {target_pose[:3,3]}")

            sol_q, sol_tauff = arm_ik.solve_ik(target_pose, current_arm_q, current_arm_dq)
            solved_pose = arm_ik.fk(sol_q)

            if loop_count % max(args.debug_print_every, 1) == 0:
                pose_error_xyz = solved_pose[:3, 3] - target_pose[:3, 3]
                rot_trace = np.trace(solved_pose[:3, :3] @ target_pose[:3, :3].T)
                rot_trace = np.clip((rot_trace - 1.0) / 2.0, -1.0, 1.0)
                pose_error_angle = float(np.arccos(rot_trace))

                logger_mp.debug(f"[teleop_arm_visionpro] current_arm_q: {np.array2string(current_arm_q, precision=5, suppress_small=True)}")
                logger_mp.debug(f"[teleop_arm_visionpro] current_arm_dq: {np.array2string(current_arm_dq, precision=5, suppress_small=True)}")
                logger_mp.debug(f"[teleop_arm_visionpro] sol_q: {np.array2string(sol_q, precision=5, suppress_small=True)}")
                logger_mp.debug(f"[teleop_arm_visionpro] sol_tauff: {np.array2string(sol_tauff, precision=5, suppress_small=True)}")
                logger_mp.info(f"[teleop_arm_visionpro] target_pose xyz: {np.array2string(target_pose[:3,3], precision=5, suppress_small=True)}")
                logger_mp.debug(f"[teleop_arm_visionpro] fk(sol_q) xyz: {np.array2string(solved_pose[:3,3], precision=5, suppress_small=True)}")
                logger_mp.debug(f"[teleop_arm_visionpro] fk pose error xyz: {np.array2string(pose_error_xyz, precision=6, suppress_small=True)}, angle(rad): {pose_error_angle:.6f}")

            if not args.dry_run:
                arm_ctrl.ctrl_arm(sol_q, sol_tauff)

            loop_count += 1

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, (1.0/args.frequency) - elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main loop sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            if arm_ctrl: arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to go home: {e}")
        try:
            if args.ipc and ipc_server: ipc_server.stop()
            else: 
                # stop_listening()
                try:
                    stop_listening()
                    listen_keyboard_thread.join(timeout=1.0)
                except Exception as e:
                    logger_mp.warning(f"Keyboard listener already stopped: {e}")
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener: {e}")
        try:
            if tv_wrapper: tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close VisionPro wrapper: {e}")
        logger_mp.info("✅ Exiting.")
        exit(0)

if __name__=='__main__':
    main()
