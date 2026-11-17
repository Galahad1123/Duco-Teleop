# visionpro_tv_wrapper_duco.py
import numpy as np
from dataclasses import dataclass
from avp_stream import VisionProStreamer
from .hand_pose_utils import compute_gripper_state, compute_hand_pose

@dataclass
class TeleData:
    head_pose: np.ndarray = None
    left_wrist_pose: np.ndarray = None
    right_wrist_pose: np.ndarray = None
    left_hand_pos: np.ndarray = None
    right_hand_pos: np.ndarray = None
    left_hand_rot: np.ndarray = None
    right_hand_rot: np.ndarray = None
    left_hand_pinch: bool = False
    right_hand_pinch: bool = False
    left_hand_pinchValue: float = 0.0
    right_hand_pinchValue: float = 0.0
    left_hand_pinchNormalized: float = 0.0
    right_hand_pinchNormalized: float = 0.0

class VisionProTVWrapperDuco:
    """
    Duco-specific VisionPro wrapper.

    - Uses hand_pose_utils to compute stable tip_middle-based wrist poses.
    - Ignores raw wrist translation from streamer.
    - Rotation can be fixed or taken from hand_pose.
    """
    def __init__(self, ip: str, verbose: bool=False):
        self.streamer = VisionProStreamer(ip=ip, record=False, verbose=verbose)
        self.streamer.start_webrtc()

    def get_tele_data(self):
        latest = self.streamer.get_latest()
        if latest is None:
            return None

        # Compute stable hand_pose using thumb/index geometry
        right_pose, right_tip, right_base = compute_hand_pose(latest, side="right")
        left_pose, left_tip, left_base = compute_hand_pose(latest, side="left")
        right_gripper = compute_gripper_state(latest, side="right")
        left_gripper = compute_gripper_state(latest, side="left")

        # Construct wrist poses (tip_middle + optional rotation)
        right_wrist_pose = right_pose.copy()
        left_wrist_pose = left_pose.copy()

        # 你可以选择固定旋转:
        # right_wrist_pose[:3, :3] = np.eye(3)
        # left_wrist_pose[:3, :3] = np.eye(3)

        return TeleData(
            head_pose=None,  # 暂时不使用
            left_wrist_pose=left_wrist_pose,
            right_wrist_pose=right_wrist_pose,
            left_hand_pos=None,
            right_hand_pos=None,
            left_hand_rot=None,
            right_hand_rot=None,
            left_hand_pinch=left_gripper.is_pinching,
            right_hand_pinch=right_gripper.is_pinching,
            left_hand_pinchValue=left_gripper.pinch_value_cm,
            right_hand_pinchValue=right_gripper.pinch_value_cm,
            left_hand_pinchNormalized=left_gripper.pinch_normalized,
            right_hand_pinchNormalized=right_gripper.pinch_normalized,
        )

    def close(self):
        self.streamer.cleanup()
