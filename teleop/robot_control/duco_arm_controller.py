import time
import threading
import struct
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

import logging_mp

logger_mp = logging_mp.getLogger(__name__)

# ---------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------
# sim 侧 DucoRobotDDS 当前监听的是 rt/lowcmd，发布的是 rt/lowstate
kTopicLowCommand_Debug = "rt/lowcmd"
kTopicLowCommand_Motion = "rt/arm_sdk"
kTopicLowState = "rt/lowstate"

# ---------------------------------------------------------------------
# 基础配置
# ---------------------------------------------------------------------
ARM_DOF = 6
ALL_MOTOR_NUM = 6
CONTROL_DT = 1.0 / 250.0
DUCO_HOME_Q_DEG = np.array([-8.25, -23.15, 121.49, 82.65, 81.61, 4.47], dtype=np.float64)
DUCO_HOME_Q_RAD = np.deg2rad(DUCO_HOME_Q_DEG)
GRIPPER_RESERVE_INDEX = 0
RECORD_CMD_RESERVE_INDEX = 1
RECORD_CMD_NOOP = 0
RECORD_CMD_PREVIEW = 6
RECORD_CMD_HOLD_CYCLES = 25


@dataclass
class MotorState:
    q: float = 0.0
    dq: float = 0.0


class LowStateBuffer:
    def __init__(self, motor_num: int):
        self.motor_state = [MotorState() for _ in range(motor_num)]
        self.reserve = [0, 0, 0, 0]


class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def get_data(self):
        with self.lock:
            return self.data

    def set_data(self, data):
        with self.lock:
            self.data = data


class duco_ArmController:
    """
    Duco 单臂控制器（最小可运行版）

    设计原则：
    1. 与 teleop_arm.py 当前接口保持一致
    2. 与 sim 侧 duco_robot_dds.py / action_provider_duco_dds.py 对齐
    3. 先优先支持仿真链路：rt/lowstate <-> rt/lowcmd
    """

    def __init__(
        self,
        motion_mode: bool = False,
        simulation_mode: bool = False,
        enable_command_stream: bool = True,
    ):
        logger_mp.info("Initialize duco_ArmController...")

        self.motion_mode = motion_mode
        self.simulation_mode = simulation_mode
        self.enable_command_stream = enable_command_stream

        # 控制目标
        self.q_target = np.zeros(ARM_DOF, dtype=np.float64)
        self.tauff_target = np.zeros(ARM_DOF, dtype=np.float64)
        self.gripper_target = 0.0
        self.pending_record_cmd = RECORD_CMD_NOOP
        self.active_record_cmd = RECORD_CMD_NOOP
        self.record_cmd_hold_remaining = 0

        # 控制参数
        self.kp_arm = 80.0
        self.kd_arm = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5

        self.arm_velocity_limit = 20.0
        self.control_dt = CONTROL_DT

        # 启动阶段速度渐增
        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        # 状态缓存 / 锁
        self.lowstate_buffer = DataBuffer()
        self.ctrl_lock = threading.Lock()

        # -------------------------------------------------------------
        # DDS 通信初始化
        # -------------------------------------------------------------
        # 关键点：
        # - simulation_mode=True 时，必须发 rt/lowcmd，因为 sim 侧就订阅这个
        # - 真机时才考虑 motion_mode -> rt/arm_sdk
        if self.simulation_mode:
            topic_cmd = kTopicLowCommand_Debug
        else:
            topic_cmd = kTopicLowCommand_Motion if self.motion_mode else kTopicLowCommand_Debug

        self.lowcmd_publisher = None
        if self.enable_command_stream:
            self.lowcmd_publisher = ChannelPublisher(topic_cmd, hg_LowCmd)
            self.lowcmd_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()

        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()

        # 这两个字段在 hg_LowCmd 里常见，沿用官方写法
        self.msg.mode_pr = 0
        self.msg.mode_machine = 0
        self.msg.reserve[GRIPPER_RESERVE_INDEX] = self._encode_gripper_target(self.gripper_target)
        self.msg.reserve[RECORD_CMD_RESERVE_INDEX] = RECORD_CMD_NOOP

        # 启动状态订阅线程
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state, daemon=True)
        self.subscribe_thread.start()

        # 等待第一帧状态
        while self.lowstate_buffer.get_data() is None:
            time.sleep(0.1)
            logger_mp.warning("[duco_ArmController] Waiting for robot state from rt/lowstate...")
        logger_mp.info("[duco_ArmController] State subscription ready.")

        # 用当前状态初始化目标
        current_q = self.get_current_arm_q()
        self.q_target[:] = current_q

        # 初始化命令消息
        self._init_joint_command_from_current_state(current_q)

        # 启动控制发送线程
        self.publish_thread = None
        if self.enable_command_stream:
            self.publish_thread = threading.Thread(target=self._ctrl_motor_state, daemon=True)
            self.publish_thread.start()

        logger_mp.info(f"[duco_ArmController] DDS command topic: {topic_cmd}")
        logger_mp.info(f"[duco_ArmController] DDS command stream enabled: {self.enable_command_stream}")
        logger_mp.info("Initialize duco_ArmController OK!")

    # -----------------------------------------------------------------
    # DDS 状态订阅
    # -----------------------------------------------------------------
    def _subscribe_motor_state(self):
        while True:
            try:
                msg = self.lowstate_subscriber.Read()
                if msg is not None:
                    lowstate = LowStateBuffer(ALL_MOTOR_NUM)
                    for i in range(ALL_MOTOR_NUM):
                        lowstate.motor_state[i].q = float(msg.motor_state[i].q)
                        lowstate.motor_state[i].dq = float(msg.motor_state[i].dq)
                    try:
                        lowstate.reserve = [int(msg.reserve[i]) for i in range(4)]
                    except Exception:
                        lowstate.reserve = [0, 0, 0, 0]
                    self.lowstate_buffer.set_data(lowstate)
            except Exception as e:
                logger_mp.error(f"[duco_ArmController] _subscribe_motor_state error: {e}")

            time.sleep(0.002)

    # -----------------------------------------------------------------
    # 命令消息初始化
    # -----------------------------------------------------------------
    def _init_joint_command_from_current_state(self, current_q: np.ndarray):
        """
        初始化 6 个关节的 kp/kd/q/mode
        这里直接采用最小单臂版本：
        - 只使用前 6 个 motor_cmd 槽位
        - q 先锁定在当前状态
        """
        for i in range(ARM_DOF):
            self.msg.motor_cmd[i].mode = 1

            if i >= 4:
                self.msg.motor_cmd[i].kp = self.kp_wrist
                self.msg.motor_cmd[i].kd = self.kd_wrist
            else:
                self.msg.motor_cmd[i].kp = self.kp_arm
                self.msg.motor_cmd[i].kd = self.kd_arm

            self.msg.motor_cmd[i].q = float(current_q[i])
            self.msg.motor_cmd[i].dq = 0.0
            self.msg.motor_cmd[i].tau = 0.0

    @staticmethod
    def _encode_gripper_target(gripper_target: float) -> int:
        gripper_target = float(np.clip(gripper_target, 0.0, 1.0))
        return struct.unpack("<I", struct.pack("<f", gripper_target))[0]

    # -----------------------------------------------------------------
    # 限速逻辑
    # -----------------------------------------------------------------
    def clip_arm_q_target(self, target_q: np.ndarray, velocity_limit: float) -> np.ndarray:
        current_q = self.get_current_arm_q()
        delta = target_q - current_q

        max_step = velocity_limit * self.control_dt
        motion_scale = np.max(np.abs(delta)) / max(max_step, 1e-8)
        clipped_target_q = current_q + delta / max(motion_scale, 1.0)

        return clipped_target_q

    # -----------------------------------------------------------------
    # 控制发送线程
    # -----------------------------------------------------------------
    def _ctrl_motor_state(self):
        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target = self.q_target.copy()
                arm_tauff_target = self.tauff_target.copy()
                gripper_target = float(self.gripper_target)
                if self.pending_record_cmd != RECORD_CMD_NOOP:
                    self.active_record_cmd = int(self.pending_record_cmd)
                    self.record_cmd_hold_remaining = RECORD_CMD_HOLD_CYCLES
                    self.pending_record_cmd = RECORD_CMD_NOOP
                record_cmd = int(self.active_record_cmd)
                if self.record_cmd_hold_remaining > 0:
                    self.record_cmd_hold_remaining -= 1
                else:
                    self.active_record_cmd = RECORD_CMD_NOOP

            # 仿真里可直接发送；真机建议限速
            if self.simulation_mode:
                clipped_arm_q_target = arm_q_target
            else:
                clipped_arm_q_target = self.clip_arm_q_target(
                    arm_q_target, velocity_limit=self.arm_velocity_limit
                )

            # 写入前 6 个关节命令
            for i in range(ARM_DOF):
                self.msg.motor_cmd[i].q = float(clipped_arm_q_target[i])
                self.msg.motor_cmd[i].dq = 0.0
                self.msg.motor_cmd[i].tau = float(arm_tauff_target[i])

            self.msg.reserve[GRIPPER_RESERVE_INDEX] = self._encode_gripper_target(gripper_target)
            self.msg.reserve[RECORD_CMD_RESERVE_INDEX] = int(record_cmd)

            if self.lowcmd_publisher is not None:
                self.msg.crc = self.crc.Crc(self.msg)
                self.lowcmd_publisher.Write(self.msg)

            # 启动阶段逐渐增加速度上限
            if self._speed_gradual_max and self._gradual_start_time is not None:
                t_elapsed = start_time - self._gradual_start_time
                ratio = min(1.0, t_elapsed / max(self._gradual_time, 1e-6))
                self.arm_velocity_limit = 20.0 + 10.0 * ratio

            elapsed = time.time() - start_time
            sleep_time = max(0.0, self.control_dt - elapsed)
            time.sleep(sleep_time)

    # -----------------------------------------------------------------
    # 状态读取接口：与 teleop_arm.py 对齐
    # -----------------------------------------------------------------
    def get_current_arm_q(self) -> np.ndarray:
        data = self.lowstate_buffer.get_data()
        if data is None:
            return np.zeros(ARM_DOF, dtype=np.float64)

        return np.array([data.motor_state[i].q for i in range(ARM_DOF)], dtype=np.float64)

    def get_current_arm_dq(self) -> np.ndarray:
        data = self.lowstate_buffer.get_data()
        if data is None:
            return np.zeros(ARM_DOF, dtype=np.float64)

        return np.array([data.motor_state[i].dq for i in range(ARM_DOF)], dtype=np.float64)

    def get_current_motor_q(self) -> np.ndarray:
        return self.get_current_arm_q()

    def get_record_status_payload(self):
        data = self.lowstate_buffer.get_data()
        if data is None:
            return None
        return list(getattr(data, "reserve", [0, 0, 0, 0]))

    # -----------------------------------------------------------------
    # 控制接口：与 teleop_arm.py / duco_arm_ik.py 对齐
    # -----------------------------------------------------------------
    def ctrl_arm(
        self,
        q_target: Sequence[float],
        tauff_target: Optional[Sequence[float]] = None,
        gripper_target: Optional[float] = None,
        record_cmd: int = RECORD_CMD_NOOP,
    ):
        q_target = np.asarray(q_target, dtype=np.float64).reshape(-1)
        if q_target.shape[0] != ARM_DOF:
            raise ValueError(f"q_target shape mismatch: expect {ARM_DOF}, got {q_target.shape[0]}")

        if tauff_target is None:
            tauff_target = np.zeros(ARM_DOF, dtype=np.float64)
        else:
            tauff_target = np.asarray(tauff_target, dtype=np.float64).reshape(-1)

        if tauff_target.shape[0] != ARM_DOF:
            raise ValueError(f"tauff_target shape mismatch: expect {ARM_DOF}, got {tauff_target.shape[0]}")

        if gripper_target is None:
            gripper_target = self.gripper_target
        gripper_target = float(np.clip(gripper_target, 0.0, 1.0))

        with self.ctrl_lock:
            self.q_target = q_target.copy()
            self.tauff_target = tauff_target.copy()
            self.gripper_target = gripper_target
            if int(record_cmd) != RECORD_CMD_NOOP:
                self.pending_record_cmd = int(record_cmd)

    def set_record_command(self, record_cmd: int):
        record_cmd = int(record_cmd)
        if record_cmd == RECORD_CMD_NOOP:
            return
        with self.ctrl_lock:
            self.pending_record_cmd = record_cmd

    # -----------------------------------------------------------------
    # 启停辅助
    # -----------------------------------------------------------------
    def speed_gradual_max(self, t: float = 5.0):
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        self.arm_velocity_limit = 30.0

    def ctrl_arm_go_home(self):
        logger_mp.info("[duco_ArmController] ctrl_arm_go_home start...")
        max_attempts = 100
        tolerance = 0.05

        if self.simulation_mode:
            home_q = np.zeros(ARM_DOF, dtype=np.float64)
        else:
            home_q = DUCO_HOME_Q_RAD.copy()

        with self.ctrl_lock:
            self.q_target = home_q
            self.tauff_target = np.zeros(ARM_DOF, dtype=np.float64)

        for _ in range(max_attempts):
            current_q = self.get_current_arm_q()
            if np.all(np.abs(current_q - home_q) < tolerance):
                logger_mp.info("[duco_ArmController] arm has reached the home position.")
                return
            time.sleep(0.05)

        logger_mp.warning("[duco_ArmController] ctrl_arm_go_home timeout.")

    def ctrl_dual_arm_go_home(self):
        """
        兼容 teleop_arm.py finally 里的旧调用名。
        单臂版本直接复用 ctrl_arm_go_home。
        """
        self.ctrl_arm_go_home()
