import numpy as np
from dataclasses import dataclass

from televuer import TeleVuer
from .duco_relative_mapper import DucoRelativeMapper, RelativePoseState


def safe_mat_update(prev_mat, mat):
    det = np.linalg.det(mat)
    if not np.isfinite(det) or np.isclose(det, 0.0, atol=1e-6):
        return prev_mat, False
    return mat, True


def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


# 保留原项目里 OpenXR -> Robot basis 的定义
T_ROBOT_OPENXR = np.array([
    [ 0, 0,-1, 0],
    [-1, 0, 0, 0],
    [ 0, 1, 0, 0],
    [ 0, 0, 0, 1],
], dtype=np.float64)

T_OPENXR_ROBOT = fast_mat_inv(T_ROBOT_OPENXR)

# 启动时 raw pose 不可用时的兜底常量
CONST_HEAD_POSE = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 1.5],
    [0, 0, 1,-0.2],
    [0, 0, 0, 1],
], dtype=np.float64)

CONST_LEFT_ARM_POSE = np.array([
    [1, 0, 0,-0.15],
    [0, 1, 0, 1.13],
    [0, 0, 1,-0.30],
    [0, 0, 0, 1],
], dtype=np.float64)

CONST_RIGHT_ARM_POSE = np.array([
    [1, 0, 0, 0.15],
    [0, 1, 0, 1.13],
    [0, 0, 1,-0.30],
    [0, 0, 0, 1],
], dtype=np.float64)


@dataclass
class DucoTeleData:
    head_pose: np.ndarray
    left_wrist_pose: np.ndarray
    right_wrist_pose: np.ndarray
    left_hand_pinch: bool = False
    right_hand_pinch: bool = False
    left_hand_squeeze: bool = False
    right_hand_squeeze: bool = False


class DucoTeleVuerWrapper:
    """
    最小版 Duco wrapper：

    1. 复用底层 TeleVuer 拿 AVP 原始 pose
    2. 只做 OpenXR basis -> Robot basis
    3. 不做 Unitree humanoid arm convention 修正
    4. 不做 head->waist 固定偏移
    """

    def __init__(
        self,
        use_hand_tracking=True,
        binocular=False,
        img_shape=(480, 640),
        display_mode="pass-through",
        zmq=False,
        webrtc=False,
        webrtc_url=None,
        base_offset_xyz=(0.0, 0.0, 0.0),
    ):
        self.use_hand_tracking = use_hand_tracking

        self.tvuer = TeleVuer(
            use_hand_tracking=use_hand_tracking,
            binocular=binocular,
            img_shape=img_shape,
            display_mode=display_mode,
            zmq=zmq,
            webrtc=webrtc,
            webrtc_url=webrtc_url,
        )

        # 上一帧有效值缓存，避免 raw pose 突然无效时直接炸掉
        self._prev_head = CONST_HEAD_POSE.copy()
        self._prev_left = CONST_LEFT_ARM_POSE.copy()
        self._prev_right = CONST_RIGHT_ARM_POSE.copy()

        self._rel_mapper = DucoRelativeMapper()
        self.base_offset_xyz = np.array(base_offset_xyz, dtype=np.float64)

    def _to_robot_basis(self, pose_openxr: np.ndarray) -> np.ndarray:
        return T_ROBOT_OPENXR @ pose_openxr @ T_OPENXR_ROBOT
    
    def set_base_offset(self, offset_xyz):
        self.base_offset_xyz = np.array(offset_xyz, dtype=np.float64)

    def _apply_base_offset(self, pose: np.ndarray) -> np.ndarray:
        pose_out = pose.copy()
        pose_out[:3, 3] += self.base_offset_xyz
        return pose_out

    # def get_tele_data(self) -> DucoTeleData:
    #     head_pose_openxr, _ = safe_mat_update(self._prev_head, self.tvuer.head_pose)
    #     self._prev_head = head_pose_openxr.copy()

    #     left_pose_openxr, _ = safe_mat_update(self._prev_left, self.tvuer.left_arm_pose)
    #     self._prev_left = left_pose_openxr.copy()

    #     right_pose_openxr, _ = safe_mat_update(self._prev_right, self.tvuer.right_arm_pose)
    #     self._prev_right = right_pose_openxr.copy()

    #     head_pose_robot = self._to_robot_basis(head_pose_openxr)
    #     left_pose_robot = self._to_robot_basis(left_pose_openxr)
    #     right_pose_robot = self._to_robot_basis(right_pose_openxr)

    #     return DucoTeleData(
    #         head_pose=head_pose_robot,
    #         left_wrist_pose=left_pose_robot,
    #         right_wrist_pose=right_pose_robot,
    #         left_hand_pinch=bool(self.tvuer.left_hand_pinch),
    #         right_hand_pinch=bool(self.tvuer.right_hand_pinch),
    #         left_hand_squeeze=bool(self.tvuer.left_hand_squeeze),
    #         right_hand_squeeze=bool(self.tvuer.right_hand_squeeze),
    #     )

    def _get_abs_tele_data(self) -> DucoTeleData:
        head_pose_openxr, _ = safe_mat_update(self._prev_head, self.tvuer.head_pose)
        self._prev_head = head_pose_openxr.copy()

        left_pose_openxr, _ = safe_mat_update(self._prev_left, self.tvuer.left_arm_pose)
        self._prev_left = left_pose_openxr.copy()

        right_pose_openxr, _ = safe_mat_update(self._prev_right, self.tvuer.right_arm_pose)
        self._prev_right = right_pose_openxr.copy()

        head_pose_robot = self._to_robot_basis(head_pose_openxr)
        left_pose_robot = self._to_robot_basis(left_pose_openxr)
        right_pose_robot = self._to_robot_basis(right_pose_openxr)

        return DucoTeleData(
            head_pose=head_pose_robot,
            left_wrist_pose=left_pose_robot,
            right_wrist_pose=right_pose_robot,
            left_hand_pinch=bool(self.tvuer.left_hand_pinch),
            right_hand_pinch=bool(self.tvuer.right_hand_pinch),
            left_hand_squeeze=bool(self.tvuer.left_hand_squeeze),
            right_hand_squeeze=bool(self.tvuer.right_hand_squeeze),
        )
    
    def get_tele_data(self) -> DucoTeleData:
        tele_abs = self._get_abs_tele_data()

        rel_state = self._rel_mapper.apply(
            RelativePoseState(
                head_pose=tele_abs.head_pose,
                left_wrist_pose=tele_abs.left_wrist_pose,
                right_wrist_pose=tele_abs.right_wrist_pose,
            )
        )

        # return DucoTeleData(
        #     head_pose=rel_state.head_pose,
        #     left_wrist_pose=rel_state.left_wrist_pose,
        #     right_wrist_pose=rel_state.right_wrist_pose,
        #     left_hand_pinch=tele_abs.left_hand_pinch,
        #     right_hand_pinch=tele_abs.right_hand_pinch,
        #     left_hand_squeeze=tele_abs.left_hand_squeeze,
        #     right_hand_squeeze=tele_abs.right_hand_squeeze,
        # )

        left_pose = self._apply_base_offset(rel_state.left_wrist_pose)
        right_pose = self._apply_base_offset(rel_state.right_wrist_pose)

        return DucoTeleData(
            head_pose=rel_state.head_pose,
            left_wrist_pose=left_pose,
            right_wrist_pose=right_pose,
            left_hand_pinch=tele_abs.left_hand_pinch,
            right_hand_pinch=tele_abs.right_hand_pinch,
            left_hand_squeeze=tele_abs.left_hand_squeeze,
            right_hand_squeeze=tele_abs.right_hand_squeeze,
        )

    def render_to_xr(self, img):
        if img is None:
            return

        arr = None
        if isinstance(img, np.ndarray):
            arr = img
        elif hasattr(img, "bgr") and isinstance(img.bgr, np.ndarray):
            arr = img.bgr

        if arr is None:
            print("[Duco render_to_xr] skip frame: no ndarray available")
            return

        self.tvuer.render_to_xr(arr)

    def close(self):
        self.tvuer.close()

    def reset_origin(self):
        tele_abs = self._get_abs_tele_data()
        state = RelativePoseState(
            head_pose=tele_abs.head_pose,
            left_wrist_pose=tele_abs.left_wrist_pose,
            right_wrist_pose=tele_abs.right_wrist_pose,
        )
        self._rel_mapper.reset_origin(state)