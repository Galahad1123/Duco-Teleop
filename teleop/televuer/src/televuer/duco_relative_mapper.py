'''
它只负责三件事：

保存 origin
把当前 pose 变成相对 pose
提供 reset_origin() / apply() 接口
'''
import numpy as np
from dataclasses import dataclass


def fast_mat_inv(mat: np.ndarray) -> np.ndarray:
    ret = np.eye(4, dtype=np.float64)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


@dataclass
class RelativePoseState:
    head_pose: np.ndarray
    left_wrist_pose: np.ndarray
    right_wrist_pose: np.ndarray


class DucoRelativeMapper:
    """
    将“当前绝对 pose”转换为“相对 origin 的 pose”。

    用法：
    1. reset_origin(abs_state)
    2. apply(abs_state) -> rel_state
    """

    def __init__(self):
        self._origin_head = np.eye(4, dtype=np.float64)
        self._origin_left = np.eye(4, dtype=np.float64)
        self._origin_right = np.eye(4, dtype=np.float64)
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def reset_origin(self, state: RelativePoseState):
        self._origin_head = state.head_pose.copy()
        self._origin_left = state.left_wrist_pose.copy()
        self._origin_right = state.right_wrist_pose.copy()
        self._enabled = True

    def clear_origin(self):
        self._enabled = False

    def apply(self, state: RelativePoseState) -> RelativePoseState:
        if not self._enabled:
            return state

        head_rel = fast_mat_inv(self._origin_head) @ state.head_pose
        left_rel = fast_mat_inv(self._origin_left) @ state.left_wrist_pose
        right_rel = fast_mat_inv(self._origin_right) @ state.right_wrist_pose

        return RelativePoseState(
            head_pose=head_rel,
            left_wrist_pose=left_rel,
            right_wrist_pose=right_rel,
        )