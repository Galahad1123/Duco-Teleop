import argparse
from collections import deque
import os
import sys
import threading
import time

import numpy as np

import logging_mp

logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from sshkeyboard import listen_keyboard, stop_listening
from televuer import VisionProTVWrapperDuco
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from teleop.robot_control.duco_arm_controller import duco_ArmController
from teleop.robot_control.duco_arm_ik import DucoArmIK
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher


START = False
STOP = False
READY = False
EXIT_REQUESTED = False
PENDING_RECORD_CMDS = deque()
RECORD_CMD_LOCK = threading.Lock()

RECORD_CMD_NOOP = 0
RECORD_CMD_START = 1
RECORD_CMD_KEEP = 2
RECORD_CMD_DISCARD = 3
RECORD_CMD_SKIP_RESET = 4
RECORD_CMD_ABORT = 5
RECORD_CMD_PREVIEW = 6
RECORD_STATUS_MAGIC = 0x41565052
RECORD_EVENT_PREVIEW = 1
RECORD_EVENT_STARTED = 2
RECORD_EVENT_KEPT = 3
RECORD_EVENT_DISCARDED = 4
RECORD_EVENT_RESET_SKIPPED = 5
RECORD_EVENT_ABORTED = 6
RECORD_EVENT_TARGET_REACHED = 7

LAST_RECORD_STATUS_SEQ = None

REAL_CHANNEL_INTERFACE = "enp0s31f6"
REAL_CHANNEL_PEER_ADDRESS = "10.188.223.119"
VISION_PRO_IP = "10.188.215.239"
DUCO_URDF_PATH = "/home/yofo/unitree/duco_support/urdf/duco_gcr5_910.urdf"
DUCO_HOME_Q_DEG = np.array([-8.25, -23.15, 121.49, 82.65, 81.61, 4.47], dtype=np.float64)


def queue_record_command(record_cmd: int):
    with RECORD_CMD_LOCK:
        PENDING_RECORD_CMDS.append(int(record_cmd))


def pop_record_command() -> int:
    with RECORD_CMD_LOCK:
        if PENDING_RECORD_CMDS:
            return PENDING_RECORD_CMDS.popleft()
    return RECORD_CMD_NOOP


def on_press(key):
    global START, STOP, EXIT_REQUESTED
    if key == "r":
        START = True
        queue_record_command(RECORD_CMD_PREVIEW)
    elif key == "e":
        START = True
        queue_record_command(RECORD_CMD_ABORT)
        queue_record_command(RECORD_CMD_PREVIEW)
    elif key == "s":
        queue_record_command(RECORD_CMD_START)
    elif key == "right":
        queue_record_command(RECORD_CMD_KEEP)
    elif key == "left":
        queue_record_command(RECORD_CMD_DISCARD)
    elif key == "space":
        queue_record_command(RECORD_CMD_SKIP_RESET)
    elif key == "q":
        if START:
            EXIT_REQUESTED = True
        else:
            STOP = True
    else:
        logger_mp.warning("Ignored input key: %s", key)


def get_state() -> dict:
    global START, STOP, READY
    return {"START": START, "STOP": STOP, "READY": READY}


def _decode_record_status_counts(raw_counts: int):
    command_samples = int(raw_counts) & 0xFFF
    robot_states = (int(raw_counts) >> 12) & 0x3FF
    camera_frames = (int(raw_counts) >> 22) & 0x3FF
    return command_samples, robot_states, camera_frames


def decode_record_status_payload(payload):
    if payload is None or len(payload) < 4:
        return None
    if int(payload[0]) != RECORD_STATUS_MAGIC:
        return None

    header = int(payload[1])
    if header == 0:
        return None

    progress = int(payload[2])
    command_samples, robot_states, camera_frames = _decode_record_status_counts(payload[3])
    return {
        "seq": (header >> 16) & 0xFFFF,
        "event": (header >> 8) & 0xFF,
        "state": header & 0xFF,
        "episode": (progress >> 16) & 0xFFFF,
        "saved": (progress >> 8) & 0xFF,
        "target": progress & 0xFF,
        "command_samples": command_samples,
        "robot_states": robot_states,
        "camera_frames": camera_frames,
    }


def log_record_status_if_updated(arm_ctrl):
    global LAST_RECORD_STATUS_SEQ
    if arm_ctrl is None:
        return
    status = decode_record_status_payload(arm_ctrl.get_record_status_payload())
    if status is None or status["seq"] == LAST_RECORD_STATUS_SEQ:
        return
    LAST_RECORD_STATUS_SEQ = status["seq"]

    event = status["event"]
    episode = status["episode"]
    saved = status["saved"]
    target = status["target"]
    if event == RECORD_EVENT_PREVIEW:
        logger_mp.info("[RecordManager] Preview started. Teleop enabled without recording.")
    elif event == RECORD_EVENT_STARTED:
        logger_mp.info("Recorder started episode %06d task=duco_teleop", episode)
        logger_mp.info("[RecordManager] Episode %06d started (%d/%d saved).", episode, saved, target)
    elif event == RECORD_EVENT_KEPT:
        logger_mp.info(
            "Recorder saved episode %06d with %d command samples, %d robot states and %d camera frames.",
            episode,
            status["command_samples"],
            status["robot_states"],
            status["camera_frames"],
        )
        logger_mp.info(
            "[RecordManager] Episode %06d kept (%d/%d saved).",
            episode,
            saved,
            target,
        )
    elif event == RECORD_EVENT_DISCARDED:
        logger_mp.info("Recorder discarded episode %06d before save.", episode)
        logger_mp.info(
            "[RecordManager] Episode %06d discarded (%d/%d saved).",
            episode,
            saved,
            target,
        )
    elif event == RECORD_EVENT_RESET_SKIPPED:
        logger_mp.info("[RecordManager] Reset skipped by operator.")
    elif event == RECORD_EVENT_ABORTED:
        logger_mp.info("[RecordManager] Aborted. Saved %d/%d episodes.", saved, target)
    elif event == RECORD_EVENT_TARGET_REACHED:
        logger_mp.info("[RecordManager] Target %d episodes reached, back to idle.", target)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--network-interface", type=str, default=None)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--ipc", action="store_true")
    parser.add_argument("--affinity", action="store_true")
    parser.add_argument("--motion", action="store_true")
    parser.add_argument("--hand-side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--verbose-stream", action="store_true")
    parser.add_argument(
        "--orientation-mode",
        type=str,
        choices=["fixed", "teleop"],
        default="teleop",
        help="fixed: only track position delta; teleop: track both position and wrist orientation",
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=0.8,
        help="Scale factor for teleop translation delta.",
    )
    parser.add_argument(
        "--orientation-scale",
        type=float,
        default=0.5,
        help="Scale factor for teleop rotation delta.",
    )
    parser.add_argument("--orientation-scale-rx", type=float, default=-1.0)
    parser.add_argument("--orientation-scale-ry", type=float, default=-1.0)
    parser.add_argument("--orientation-scale-rz", type=float, default=1.0)
    parser.add_argument(
        "--ik-translation-fallback-threshold",
        type=float,
        default=0.02,
        help="Fallback to position-priority IK when full-pose IK translation error is too large.",
    )
    return parser


def setup_input_listener(args):
    if args.ipc:
        ipc_server = IPC_Server(on_press=on_press, get_state=get_state)
        ipc_server.start()
        return ipc_server, None

    keyboard_thread = threading.Thread(
        target=listen_keyboard,
        kwargs={"on_press": on_press, "until": None, "sequential": False},
        daemon=True,
    )
    keyboard_thread.start()
    return None, keyboard_thread


def setup_affinity():
    import psutil

    process = psutil.Process(os.getpid())
    process.cpu_affinity([0, 1, 2, 3])
    try:
        process.nice(-20)
        logger_mp.info("Process priority elevated.")
    except psutil.AccessDenied:
        logger_mp.warning("Failed to elevate process priority. Run with higher privileges if needed.")

    for child in process.children(recursive=True):
        try:
            child.cpu_affinity([5, 6])
            child.nice(-20)
        except psutil.AccessDenied:
            continue


def initialize_channel(args):
    if args.sim:
        ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        logger_mp.info("DDS channel initialized for simulation mode.")
        return

    ChannelFactoryInitialize(
        0,
        networkInterface=REAL_CHANNEL_INTERFACE,
        peerAddress=REAL_CHANNEL_PEER_ADDRESS,
    )
    logger_mp.info(
        "DDS channel initialized for real robot mode. interface=%s peer=%s",
        REAL_CHANNEL_INTERFACE,
        REAL_CHANNEL_PEER_ADDRESS,
    )


def enter_debug_mode_if_needed(args):
    if args.motion:
        return None

    motion_switcher = MotionSwitcher()
    status, _ = motion_switcher.Enter_Debug_Mode()
    if status == 0:
        logger_mp.info("Robot switched to debug command topic.")
    else:
        logger_mp.warning("Failed to switch robot to debug command topic.")
    return motion_switcher


def build_arm_ik():
    model_dir = os.path.dirname(DUCO_URDF_PATH)
    return DucoArmIK(
        urdf_path=DUCO_URDF_PATH,
        model_dir=model_dir,
        ee_parent_joint="arm_1_joint_6",
        ee_frame_name="ee",
        ee_offset_xyz=(0.0, 0.0, 0.0),
    )


def get_target_pose(tele_data, hand_side: str):
    if tele_data is None:
        return None
    return tele_data.left_wrist_pose if hand_side == "left" else tele_data.right_wrist_pose


def get_gripper_status(tele_data, hand_side: str):
    if tele_data is None:
        return None, None, None

    if hand_side == "left":
        return (
            getattr(tele_data, "left_hand_pinch", None),
            getattr(tele_data, "left_hand_pinchValue", None),
            getattr(tele_data, "left_hand_pinchNormalized", None),
        )

    return (
        getattr(tele_data, "right_hand_pinch", None),
        getattr(tele_data, "right_hand_pinchValue", None),
        getattr(tele_data, "right_hand_pinchNormalized", None),
    )


def wait_for_valid_pose(tv_wrapper, hand_side: str, timeout_sec: float = 10.0):
    deadline = time.time() + timeout_sec
    while time.time() < deadline and not STOP:
        tele_data = tv_wrapper.get_tele_data()
        target_pose = get_target_pose(tele_data, hand_side)
        if target_pose is not None:
            return target_pose.copy()
        time.sleep(0.02)
    return None


def collect_initial_pose(tv_wrapper, hand_side: str, buffer_sec: float = 3.0):
    logger_mp.info("Buffering Vision Pro wrist pose for %.1f seconds.", buffer_sec)
    deadline = time.time() + buffer_sec
    latest_pose = None
    while time.time() < deadline and not STOP:
        tele_data = tv_wrapper.get_tele_data()
        current_pose = get_target_pose(tele_data, hand_side)
        if current_pose is not None:
            latest_pose = current_pose.copy()
        time.sleep(0.02)

    if latest_pose is not None:
        return latest_pose

    logger_mp.warning("No valid Vision Pro pose collected during buffering. Retrying once.")
    return wait_for_valid_pose(tv_wrapper, hand_side, timeout_sec=10.0)


def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


def skew(vec):
    x, y, z = np.asarray(vec, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rotation_matrix_to_rotvec(rot):
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    cos_theta = np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)

    sin_theta = np.sin(theta)
    if abs(sin_theta) < 1e-6:
        axis = np.sqrt(np.maximum((np.diag(rot) + 1.0) * 0.5, 0.0))
        if rot[2, 1] - rot[1, 2] < 0.0:
            axis[0] = -axis[0]
        if rot[0, 2] - rot[2, 0] < 0.0:
            axis[1] = -axis[1]
        if rot[1, 0] - rot[0, 1] < 0.0:
            axis[2] = -axis[2]
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            return np.zeros(3, dtype=np.float64)
        return axis / axis_norm * theta

    axis = np.array(
        [rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]],
        dtype=np.float64,
    ) / (2.0 * sin_theta)
    return axis * theta


def rotvec_to_rotation_matrix(rotvec):
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(rotvec)
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)

    axis = rotvec / theta
    k = skew(axis)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def build_target_pose(args, robot_origin_pose, init_tele_pose, tele_origin_inv, current_tele_pose):
    tele_rel_pose = tele_origin_inv @ current_tele_pose
    raw_delta_world = (current_tele_pose[:3, 3] - init_tele_pose[:3, 3]) * args.position_scale

    # Align hand translation with the robot base plane:
    # teleop lateral motion -> robot lateral motion
    # teleop forward/back motion -> robot forward/back motion
    delta_world = np.array(
        [raw_delta_world[1], -raw_delta_world[0], raw_delta_world[2]],
        dtype=np.float64,
    )

    target_pose = robot_origin_pose.copy()
    target_pose[:3, 3] = robot_origin_pose[:3, 3] + delta_world

    if args.orientation_mode == "teleop":
        rel_rotvec = rotation_matrix_to_rotvec(tele_rel_pose[:3, :3]) * args.orientation_scale
        rel_rotvec *= np.array(
            [args.orientation_scale_rx, args.orientation_scale_ry, args.orientation_scale_rz],
            dtype=np.float64,
        )
        target_pose[:3, :3] = robot_origin_pose[:3, :3] @ rotvec_to_rotation_matrix(rel_rotvec)
    else:
        target_pose[:3, :3] = robot_origin_pose[:3, :3]

    return target_pose


def solve_target_with_fallback(args, arm_ik, robot_origin_pose, current_arm_q, current_arm_dq, target_pose):
    sol_q, sol_tauff = arm_ik.solve_ik(target_pose, current_arm_q, current_arm_dq)
    solved_pose = arm_ik.fk(sol_q)

    if args.orientation_mode != "teleop":
        return sol_q, sol_tauff, solved_pose

    pos_error_norm = float(np.linalg.norm(solved_pose[:3, 3] - target_pose[:3, 3]))
    if pos_error_norm <= args.ik_translation_fallback_threshold:
        return sol_q, sol_tauff, solved_pose

    target_pose_pos_only = target_pose.copy()
    target_pose_pos_only[:3, :3] = robot_origin_pose[:3, :3]

    fallback_q, fallback_tauff = arm_ik.solve_ik(target_pose_pos_only, current_arm_q, current_arm_dq)
    fallback_pose = arm_ik.fk(fallback_q)
    fallback_error_norm = float(
        np.linalg.norm(fallback_pose[:3, 3] - target_pose_pos_only[:3, 3])
    )

    if fallback_error_norm < pos_error_norm:
        logger_mp.warning(
            "Full-pose IK translation error %.4fm exceeded threshold %.4fm. "
            "Using position-priority IK with error %.4fm.",
            pos_error_norm,
            args.ik_translation_fallback_threshold,
            fallback_error_norm,
        )
        return fallback_q, fallback_tauff, fallback_pose

    return sol_q, sol_tauff, solved_pose


def log_run_config(args, robot_origin_pose):
    logger_mp.info("----------------------------------------------------------------")
    logger_mp.info("🟢 Press [r] to start syncing the robot with your movements.")
    logger_mp.info("🔴 Press [q] to stop and exit the program.")
    logger_mp.info("[record idle] Press [s] to start a capture run, [e] to return to preview.")
    logger_mp.info("[recording] Press [right] to keep, [left] to discard.")
    logger_mp.info("[record reset] Press [space] to skip reset.")
    logger_mp.info("⚠️ Keep safe!")
    logger_mp.info(
        "Mode summary: sim=%s motion=%s hand=%s orientation=%s position_scale=%.3f orientation_scale=%.3f",
        args.sim,
        args.motion,
        args.hand_side,
        args.orientation_mode,
        args.position_scale,
        args.orientation_scale,
    )
    logger_mp.info(
        "Orientation axis scales: rx=%.3f ry=%.3f rz=%.3f",
        args.orientation_scale_rx,
        args.orientation_scale_ry,
        args.orientation_scale_rz,
    )
    logger_mp.info(
        "Robot reference pose xyz: %s",
        np.array2string(robot_origin_pose[:3, 3], precision=5, suppress_small=True),
    )


def wait_for_start_signal():
    global READY
    READY = True
    while not START and not STOP:
        time.sleep(0.033)


def drive_home_before_exit(arm_ctrl, timeout_sec: float = 8.0, tolerance: float = 0.05):
    if arm_ctrl is None:
        return

    home_q = np.zeros(6, dtype=np.float64) if arm_ctrl.simulation_mode else np.deg2rad(DUCO_HOME_Q_DEG)
    tauff = np.zeros_like(home_q)
    gripper_target = getattr(arm_ctrl, "gripper_target", 0.0)
    logger_mp.info("[duco_ArmController] Sending home target before exit.")

    try:
        arm_ctrl.speed_instant_max()
    except Exception:
        pass

    deadline = time.time() + timeout_sec
    while time.time() < deadline and not STOP:
        arm_ctrl.ctrl_arm(home_q, tauff, gripper_target=gripper_target)
        current_q = arm_ctrl.get_current_arm_q()
        if np.all(np.abs(current_q - home_q) < tolerance):
            logger_mp.info("[duco_ArmController] arm has reached the home position before exit.")
            return
        time.sleep(0.05)

    logger_mp.warning("[duco_ArmController] home-before-exit timeout.")


def cleanup(args, arm_ctrl, ipc_server, keyboard_thread, tv_wrapper):
    try:
        if arm_ctrl is not None:
            arm_ctrl.set_record_command(RECORD_CMD_ABORT)
            time.sleep(0.1)
    except Exception as exc:
        logger_mp.error("Failed to send record abort command: %s", exc)

    try:
        if args.ipc and ipc_server is not None:
            ipc_server.stop()
        else:
            stop_listening()
            if keyboard_thread is not None:
                keyboard_thread.join(timeout=1.0)
    except Exception as exc:
        logger_mp.warning("Failed to stop input listener cleanly: %s", exc)

    try:
        if tv_wrapper is not None:
            tv_wrapper.close()
    except Exception as exc:
        logger_mp.warning("Failed to close Vision Pro wrapper cleanly: %s", exc)


def main():
    global STOP, EXIT_REQUESTED

    parser = build_parser()
    args = parser.parse_args()
    logger_mp.info("Startup args: %s", args)

    ipc_server = None
    keyboard_thread = None
    tv_wrapper = None
    arm_ctrl = None

    try:
        initialize_channel(args)
        ipc_server, keyboard_thread = setup_input_listener(args)
        tv_wrapper = VisionProTVWrapperDuco(ip=VISION_PRO_IP, verbose=args.verbose_stream)
        enter_debug_mode_if_needed(args)

        arm_ik = build_arm_ik()
        arm_ctrl = duco_ArmController(
            motion_mode=args.motion,
            simulation_mode=args.sim,
            enable_command_stream=True,
        )

        robot_origin_pose = arm_ik.fk(np.deg2rad(DUCO_HOME_Q_DEG))
        if args.affinity:
            setup_affinity()

        log_run_config(args, robot_origin_pose)
        wait_for_start_signal()
        if STOP:
            return

        init_tele_pose = collect_initial_pose(tv_wrapper, args.hand_side)
        if init_tele_pose is None:
            raise RuntimeError("Failed to get initial Vision Pro wrist pose.")

        tele_origin_inv = fast_mat_inv(init_tele_pose)
        logger_mp.info("Relative teleoperation origin initialized.")
        logger_mp.info("---------------------🚀 start Tracking 🚀------------------------")
        arm_ctrl.speed_gradual_max()

        missing_pose_warn_ts = 0.0
        while not STOP:
            loop_start = time.time()
            record_cmd = pop_record_command()
            if record_cmd != RECORD_CMD_NOOP:
                arm_ctrl.set_record_command(record_cmd)
            log_record_status_if_updated(arm_ctrl)

            if EXIT_REQUESTED:
                drive_home_before_exit(arm_ctrl)
                STOP = True
                break

            tele_data = tv_wrapper.get_tele_data()
            current_tele_pose = get_target_pose(tele_data, args.hand_side)
            if current_tele_pose is None:
                now = time.time()
                if now - missing_pose_warn_ts > 1.0:
                    logger_mp.warning("Vision Pro wrist pose unavailable. Waiting for next valid frame.")
                    missing_pose_warn_ts = now
                time.sleep(0.01)
                continue

            _, _, pinch_normalized = get_gripper_status(tele_data, args.hand_side)
            current_arm_q = arm_ctrl.get_current_arm_q()
            current_arm_dq = arm_ctrl.get_current_arm_dq()
            target_pose = build_target_pose(
                args,
                robot_origin_pose,
                init_tele_pose,
                tele_origin_inv,
                current_tele_pose,
            )
            sol_q, sol_tauff, _ = solve_target_with_fallback(
                args,
                arm_ik,
                robot_origin_pose,
                current_arm_q,
                current_arm_dq,
                target_pose,
            )
            arm_ctrl.ctrl_arm(
                sol_q,
                sol_tauff,
                gripper_target=pinch_normalized,
            )

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, (1.0 / args.frequency) - elapsed))

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting...")
    except Exception:
        import traceback

        logger_mp.error(traceback.format_exc())
    finally:
        cleanup(args, arm_ctrl, ipc_server, keyboard_thread, tv_wrapper)
        logger_mp.info("Exiting teleop_arm_visionpro.")
        sys.exit(0)


if __name__ == "__main__":
    main()
