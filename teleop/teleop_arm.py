import time
import argparse
import threading
import os
import sys

import numpy as np

import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

# -----------------------------------------------------------------------------
# 路径设置
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# -----------------------------------------------------------------------------
# 外部模块导入
# -----------------------------------------------------------------------------
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

from televuer import DucoTeleVuerWrapper
from teleop.robot_control.duco_arm_controller import duco_ArmController
from teleop.robot_control.duco_arm_ik import DucoArmIK
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher
from sshkeyboard import listen_keyboard, stop_listening

# -----------------------------------------------------------------------------
# 全局状态机
# -----------------------------------------------------------------------------
START = False           # 是否开始跟随
STOP = False            # 是否退出程序
READY = False           # 是否处于可开始/可录制状态
RECORD_RUNNING = False  # 是否正在录制
RECORD_TOGGLE = False   # 录制状态切换信号


def publish_reset_category(category: int, publisher):
    """仿真场景复位信号。"""
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")


def on_press(key):
    """
    键盘/IPC 输入回调：
    - r: 开始跟随
    - s: 录制开始/保存切换（仅在 START 后有效）
    - q: 退出
    """
    global START, STOP, RECORD_TOGGLE

    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START:
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined.")


def get_state() -> dict:
    """提供给 IPC 心跳查询的当前状态。"""
    global START, STOP, READY, RECORD_RUNNING
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }


def build_parser():
    """命令行参数。"""
    parser = argparse.ArgumentParser()

    # 基本运行参数
    parser.add_argument('--frequency', type=float, default=30.0,
                        help='Control / record loop frequency')
    parser.add_argument('--display-mode', type=str,
                        choices=['immersive', 'ego', 'pass-through'],
                        default='immersive',
                        help='XR display mode')
    parser.add_argument('--img-server-ip', type=str, default='10.138.41.147',
                        help='Image server IP')
    parser.add_argument('--network-interface', type=str, default=None,
                        help='DDS network interface')

    # 运行模式
    parser.add_argument('--headless', action='store_true',
                        help='Enable headless mode')
    parser.add_argument('--sim', action='store_true',
                        help='Enable simulation mode')
    parser.add_argument('--ipc', action='store_true',
                        help='Use IPC server instead of sshkeyboard')
    parser.add_argument('--affinity', action='store_true',
                        help='Enable high priority / CPU affinity')
    parser.add_argument('--motion', action='store_true',
                        help='Enable motion mode')

    # 录制参数
    parser.add_argument('--record', action='store_true',
                        help='Enable data recording')
    parser.add_argument('--task-dir', type=str, default='./utils/data/',
                        help='Directory to save episodes')
    parser.add_argument('--task-name', type=str, default='pick cube',
                        help='Task name')
    parser.add_argument('--task-goal', type=str, default='pick up cube.',
                        help='Task goal')
    parser.add_argument('--task-desc', type=str, default='task description',
                        help='Task description')
    parser.add_argument('--task-steps', type=str,
                        default='step1: do this; step2: do that;',
                        help='Task steps')

    # 单臂控制：选择 AVP 左手或右手作为输入源
    parser.add_argument('--hand-side', type=str, choices=['left', 'right'],
                        default='right',
                        help='Use left or right AVP hand to control the robot arm')

    return parser


def setup_input_listener(args):
    """
    启动输入监听：
    - IPC 模式：启动 IPC server
    - 否则：启动 sshkeyboard 线程
    """
    if args.ipc:
        ipc_server = IPC_Server(on_press=on_press, get_state=get_state)
        ipc_server.start()
        return ipc_server, None
    else:
        keyboard_thread = threading.Thread(
            target=listen_keyboard,
            kwargs={
                "on_press": on_press,
                "until": None,
                "sequential": False,
            },
            daemon=True
        )
        keyboard_thread.start()
        return None, keyboard_thread


def setup_affinity():
    """可选：设置进程优先级和 CPU 亲和性。"""
    import psutil

    p = psutil.Process(os.getpid())
    p.cpu_affinity([0, 1, 2, 3])

    try:
        p.nice(-20)
        logger_mp.info("Set high priority successfully.")
    except psutil.AccessDenied:
        logger_mp.warning("Failed to set high priority. Please run as root.")

    for child in p.children(recursive=True):
        try:
            logger_mp.info(f"Child process {child.pid} name: {child.name()}")
            child.cpu_affinity([5, 6])
            child.nice(-20)
        except psutil.AccessDenied:
            pass


def get_target_pose(tele_data, hand_side: str):
    """
    根据 hand_side 选择用于控制单臂的目标位姿。
    单臂系统中，建议尽早在这里把“用左手还是右手控制”明确下来。
    """
    if hand_side == 'left':
        return tele_data.left_wrist_pose
    return tele_data.right_wrist_pose


def build_record_data(camera_config, head_img, left_wrist_img, right_wrist_img,
                      current_arm_q, sol_q):
    """
    组装录制数据。
    这里使用单臂命名，不再沿用双臂模板的 left/right_arm 结构。
    """
    colors = {}
    depths = {}

    # 相机图像整理
    if camera_config['head_camera']['binocular']:
        if head_img is not None:
            half_w = camera_config['head_camera']['image_shape'][1] // 2
            colors["color_0"] = head_img.bgr[:, :half_w]
            colors["color_1"] = head_img.bgr[:, half_w:]
        else:
            logger_mp.warning("Head image is None!")

        if camera_config['left_wrist_camera']['enable_zmq']:
            if left_wrist_img is not None:
                colors["color_2"] = left_wrist_img.bgr
            else:
                logger_mp.warning("Left wrist image is None!")

        if camera_config['right_wrist_camera']['enable_zmq']:
            if right_wrist_img is not None:
                colors["color_3"] = right_wrist_img.bgr
            else:
                logger_mp.warning("Right wrist image is None!")
    else:
        if head_img is not None:
            colors["color_0"] = head_img
        else:
            logger_mp.warning("Head image is None!")

        if camera_config['left_wrist_camera']['enable_zmq']:
            if left_wrist_img is not None:
                colors["color_1"] = left_wrist_img.bgr
            else:
                logger_mp.warning("Left wrist image is None!")

        if camera_config['right_wrist_camera']['enable_zmq']:
            if right_wrist_img is not None:
                colors["color_2"] = right_wrist_img.bgr
            else:
                logger_mp.warning("Right wrist image is None!")

    # 单臂状态与动作
    states = {
        "arm": {
            "qpos": current_arm_q.tolist(),
            "qvel": [],
            "torque": [],
        },
        "ee": {
            "qpos": [],
            "qvel": [],
            "torque": [],
        },
    }

    actions = {
        "arm": {
            "qpos": sol_q.tolist(),
            "qvel": [],
            "torque": [],
        },
        "ee": {
            "qpos": [],
            "qvel": [],
            "torque": [],
        },
    }

    return colors, depths, states, actions


def main():
    global READY, RECORD_RUNNING, RECORD_TOGGLE, START, STOP

    parser = build_parser()
    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    ipc_server = None
    listen_keyboard_thread = None
    img_client = None
    tv_wrapper = None
    recorder = None
    sim_state_subscriber = None
    reset_pose_publisher = None
    motion_switcher = None
    arm_ctrl = None

    try:
        # ---------------------------------------------------------------------
        # 1) 初始化通信
        # ---------------------------------------------------------------------
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ---------------------------------------------------------------------
        # 2) 启动输入监听
        # ---------------------------------------------------------------------
        ipc_server, listen_keyboard_thread = setup_input_listener(args)

        # ---------------------------------------------------------------------
        # 3) 初始化图像客户端，并读取相机配置
        # ---------------------------------------------------------------------
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)

        # debug
        head_img = img_client.get_head_frame()
        print("first head_img:", head_img)
        print("img_client request_bgr:", img_client._request_bgr)

        mgr = img_client._subscriber_manager
        print("subscriber cache keys:", list(mgr._subscriber_threads.keys()))

        sub = mgr._subscriber_threads.get((args.img_server_ip, img_client.get_cam_config()['head_camera']['zmq_port']))
        print("subscriber request_bgr internal:", getattr(sub, "_request_bgr", "N/A"))
        print("has decoder thread:", getattr(sub, "_decoder_thread", None) is not None)

        camera_config = img_client.get_cam_config()
        camera_config['head_camera']['enable_webrtc'] = False
        # logger_mp.debug(f"Camera config: {camera_config}")
        logger_mp.info(f"Camera config: {camera_config}")
        try:
            logger_mp.info(
                "[camera] head_camera: "
                f"enable_zmq={camera_config['head_camera'].get('enable_zmq')}, "
                f"enable_webrtc={camera_config['head_camera'].get('enable_webrtc')}, "
                f"binocular={camera_config['head_camera'].get('binocular')}, "
                f"image_shape={camera_config['head_camera'].get('image_shape')}, "
                f"webrtc_port={camera_config['head_camera'].get('webrtc_port')}"
            )
        except Exception as e:
            logger_mp.error(f"[camera] Failed to parse head_camera config: {e}")

        # 是否需要本地取头图再回传给 XR
        xr_need_local_img = not (
            args.display_mode == 'pass-through'
            or camera_config['head_camera']['enable_webrtc']
        )
        # debug
        logger_mp.info(
            f"[camera] display_mode={args.display_mode}, "
            f"head_enable_zmq={camera_config['head_camera']['enable_zmq']}, "
            f"head_enable_webrtc={camera_config['head_camera']['enable_webrtc']}, "
            f"xr_need_local_img={xr_need_local_img}"
        )

        # ---------------------------------------------------------------------
        # 4) 初始化 TeleVuerWrapper
        #    这里是 AVP/XR 输入数据的统一入口
        # ---------------------------------------------------------------------
        tv_wrapper = DucoTeleVuerWrapper(
            use_hand_tracking=True,
            binocular=camera_config['head_camera']['binocular'],
            img_shape=camera_config['head_camera']['image_shape'],
            display_mode=args.display_mode,
            zmq=camera_config['head_camera']['enable_zmq'],
            webrtc=camera_config['head_camera']['enable_webrtc'],
            webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
            base_offset_xyz=(0.0, 0.0, -0.15),
        )
        # tv_wrapper.set_base_offset((0.0, 0.0, -0.15))

        # ---------------------------------------------------------------------
        # 5) 进入 debug mode（如果不是 motion mode）
        # ---------------------------------------------------------------------
        if not args.motion:
            motion_switcher = MotionSwitcher()
            status, _ = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # ---------------------------------------------------------------------
        # 6) 初始化单臂 IK 和控制器
        # ---------------------------------------------------------------------
        duco_urdf_path = "/home/yofo/unitree/duco_support/urdf/duco_gcr5_910.urdf"
        duco_model_dir = os.path.dirname(duco_urdf_path)
        arm_ik = DucoArmIK(
            urdf_path=duco_urdf_path,
            model_dir=duco_model_dir,
            ee_parent_joint="arm_1_joint_6",
            ee_frame_name="ee",
            ee_offset_xyz=(0.0, 0.0, 0.0),
        )

        arm_ctrl = duco_ArmController(
            motion_mode=args.motion,
            simulation_mode=args.sim
        )

        # 第3步 relative mapping
        obs_joint_deg = np.array([-5.2, -20.8, 109.6, 99.2, 90.5, -12.1], dtype=np.float64)
        obs_joint_rad = np.deg2rad(obs_joint_deg)
        robot_origin_pose = arm_ik.fk(obs_joint_rad)
        logger_mp.info(f"robot_origin_pose from obs_joint:\n{robot_origin_pose}")

        # ---------------------------------------------------------------------
        # 7) 可选：设置 affinity
        # ---------------------------------------------------------------------
        if args.affinity:
            setup_affinity()

        # ---------------------------------------------------------------------
        # 8) 仿真模式下，初始化 reset / sim_state
        # ---------------------------------------------------------------------
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()

            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # ---------------------------------------------------------------------
        # 9) 录制器初始化
        # ---------------------------------------------------------------------
        if args.record:
            recorder = EpisodeWriter(
                task_dir=os.path.join(args.task_dir, args.task_name),
                task_goal=args.task_goal,
                task_desc=args.task_desc,
                task_steps=args.task_steps,
                frequency=args.frequency,
                rerun_log=not args.headless
            )

        # ---------------------------------------------------------------------
        # 10) 打印启动说明
        # ---------------------------------------------------------------------
        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")

        READY = True

        # ---------------------------------------------------------------------
        # 11) 等待用户按下 r 开始
        # ---------------------------------------------------------------------
        while not START and not STOP:
            time.sleep(0.033)

            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                tv_wrapper.render_to_xr(head_img)


        # 3秒缓冲期，期间继续刷新 XR 图像，避免 AVP 画面卡住
        logger_mp.info("🟠 AVP pose buffering for 3.0 seconds before tracking...")
        buffer_start = time.time()
        last_buffer_pose = None

        while (time.time() - buffer_start) < 3.0 and not STOP:
            time.sleep(0.02)

            # XR 本地图像继续刷新，避免缓冲阶段 AVP 画面卡住
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                if head_img is not None:
                    tv_wrapper.render_to_xr(head_img)

            try:
                tele_data = tv_wrapper.get_tele_data()
                last_buffer_pose = get_target_pose(tele_data, args.hand_side)
            except Exception as e:
                logger_mp.warning(f"buffering tele_data failed: {e}")

        if last_buffer_pose is not None:
            logger_mp.info(f"Buffered initial target pose:\n{last_buffer_pose}")
        else:
            logger_mp.warning("No valid AVP pose captured during buffering.")

        # 第3步 relative mapping
        tv_wrapper.reset_origin()
        logger_mp.info("Relative teleop origin has been reset.")

        logger_mp.info("---------------------🚀 start Tracking 🚀------------------------")
        arm_ctrl.speed_gradual_max()

        # ---------------------------------------------------------------------
        # 12) 主控制循环
        # ---------------------------------------------------------------------
        while not STOP:
            loop_start_time = time.time()

            # 每轮循环先将图像变量置空，避免某些分支未赋值时后面录制报错
            head_img = None
            left_wrist_img = None
            right_wrist_img = None

            # -----------------------------------------------------------------
            # 12.1 获取图像
            # -----------------------------------------------------------------
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()

                if xr_need_local_img and head_img is not None:
                    tv_wrapper.render_to_xr(head_img)

            if camera_config['left_wrist_camera']['enable_zmq'] and args.record:
                left_wrist_img = img_client.get_left_wrist_frame()

            if camera_config['right_wrist_camera']['enable_zmq'] and args.record:
                right_wrist_img = img_client.get_right_wrist_frame()

            # -----------------------------------------------------------------
            # 12.2 处理录制开关
            # -----------------------------------------------------------------
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False

                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()

                    if args.sim and reset_pose_publisher is not None:
                        publish_reset_category(1, reset_pose_publisher)

            # -----------------------------------------------------------------
            # 12.3 获取 AVP 遥操作数据
            # -----------------------------------------------------------------
            tele_data = tv_wrapper.get_tele_data()

            # -----------------------------------------------------------------
            # 12.4 获取当前机械臂状态
            # -----------------------------------------------------------------
            current_arm_q = arm_ctrl.get_current_arm_q()
            current_arm_dq = arm_ctrl.get_current_arm_dq()

            # -----------------------------------------------------------------
            # 12.5 选择控制手，并计算单臂 IK
            # -----------------------------------------------------------------
            # target_pose = get_target_pose(tele_data, args.hand_side)
            # 第3步 relative mapping
            tele_rel_pose = get_target_pose(tele_data, args.hand_side)
            target_pose = robot_origin_pose @ tele_rel_pose
            # debug: 打印目标位姿的平移部分，检查是否合理
            logger_mp.debug(f"target xyz = {target_pose[:3, 3]}")

            ik_start_time = time.time()
            sol_q, sol_tauff = arm_ik.solve_ik(
                target_pose,
                current_arm_q,
                current_arm_dq
            )
            # debug: 打印 IK 解，检查是否合理
            logger_mp.debug(f"sol_q = {np.round(sol_q, 4)}")
            ik_end_time = time.time()
            logger_mp.debug(f"ik: {round(ik_end_time - ik_start_time, 6)} s")

            # -----------------------------------------------------------------
            # 12.6 下发机械臂控制命令
            # -----------------------------------------------------------------
            arm_ctrl.ctrl_arm(sol_q, sol_tauff)

            # -----------------------------------------------------------------
            # 12.7 录制数据
            # -----------------------------------------------------------------
            if args.record:
                READY = recorder.is_ready()

                if RECORD_RUNNING:
                    colors, depths, states, actions = build_record_data(
                        camera_config=camera_config,
                        head_img=head_img,
                        left_wrist_img=left_wrist_img,
                        right_wrist_img=right_wrist_img,
                        current_arm_q=current_arm_q,
                        sol_q=sol_q,
                    )

                    if args.sim and sim_state_subscriber is not None:
                        sim_state = sim_state_subscriber.read_data()
                        recorder.add_item(
                            colors=colors,
                            depths=depths,
                            states=states,
                            actions=actions,
                            sim_state=sim_state
                        )
                    else:
                        recorder.add_item(
                            colors=colors,
                            depths=depths,
                            states=states,
                            actions=actions
                        )
            # debug
            logger_mp.debug(f"right_arm_pose: {tv_wrapper.tvuer.right_arm_pose[:3, 3]}")
            logger_mp.debug(f"right_wrist_pose: {tele_data.right_wrist_pose[:3, 3]}")
            logger_mp.debug(f"target_pose: {target_pose[:3, 3]}")


            # -----------------------------------------------------------------
            # 12.8 控制循环频率
            # -----------------------------------------------------------------
            elapsed = time.time() - loop_start_time
            sleep_time = max(0.0, (1.0 / args.frequency) - elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        # ---------------------------------------------------------------------
        # 13) 安全退出与资源清理
        # ---------------------------------------------------------------------
        try:
            if arm_ctrl is not None:
                arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")

        try:
            if args.ipc:
                if ipc_server is not None:
                    ipc_server.stop()
            else:
                stop_listening()
                if listen_keyboard_thread is not None:
                    listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")

        try:
            if img_client is not None:
                img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            if tv_wrapper is not None:
                tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if args.sim and sim_state_subscriber is not None:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")

        try:
            if args.record and recorder is not None:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")

        logger_mp.info("✅ Finally, exiting program.")
        exit(0)


if __name__ == '__main__':
    main()