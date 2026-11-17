# hand_pose_utils.py
from dataclasses import dataclass

import numpy as np


PINCH_DISTANCE_CLOSE_M = 0.02
PINCH_DISTANCE_MIN_CM = 5.0
PINCH_DISTANCE_MAX_CM = 7.0
PINCH_NORMALIZED_MIN_CM = 1.0
PINCH_NORMALIZED_MAX_CM = 5.0


@dataclass(frozen=True)
class HandGripperState:
    pinch_distance_m: float
    pinch_value_cm: float
    is_pinching: bool
    openness: float
    pinch_normalized: float


def _extract_hand_points(hand, side="right"):
    wrist = hand[f"{side}_wrist"]
    fingers = wrist @ hand[f"{side}_fingers"]

    thumb_tip = fingers[4, :3, -1]
    thumb_base = fingers[2, :3, -1]
    index_tip = fingers[9, :3, -1]
    index_base = fingers[7, :3, -1]

    return fingers, thumb_tip, thumb_base, index_tip, index_base


def compute_hand_pose(hand, side="right"):
    """
    Compute a hand pose (4x4 matrix) from wrist and fingers.

    Args:
        hand: dict from VisionProStreamer, e.g., hand['right_wrist'], hand['right_fingers']
        side: "left" or "right"

    Returns:
        hand_pose: 4x4 numpy array, world frame
        tip_middle: 3D numpy array of fingertip midpoint
        base_middle: 3D numpy array of finger base midpoint
    """
    _, thumb_tip, thumb_base, index_tip, index_base = _extract_hand_points(hand, side=side)

    base_middle = (thumb_base + index_base) * 0.5
    tip_middle = (thumb_tip + index_tip) * 0.5

    z_axis = tip_middle - base_middle
    norm_z = np.linalg.norm(z_axis)
    if norm_z < 1e-6:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        z_axis /= norm_z

    x_axis = index_base - thumb_base
    x_axis -= np.dot(x_axis, z_axis) * z_axis
    norm_x = np.linalg.norm(x_axis)
    if norm_x < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis /= norm_x

    y_axis = np.cross(z_axis, x_axis)
    norm_y = np.linalg.norm(y_axis)
    if norm_y < 1e-6:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        y_axis /= norm_y

    rot = np.column_stack((x_axis, y_axis, z_axis))
    det = np.linalg.det(rot)
    if np.isfinite(det) and abs(det) > 1e-6:
        rot /= np.cbrt(det)

    hand_pose = np.eye(4)
    hand_pose[:3, :3] = rot
    hand_pose[:3, 3] = tip_middle

    return hand_pose, tip_middle, base_middle


def compute_gripper_state(
    hand,
    side="right",
    pinch_close_m=PINCH_DISTANCE_CLOSE_M,
    pinch_min_cm=PINCH_DISTANCE_MIN_CM,
    pinch_max_cm=PINCH_DISTANCE_MAX_CM,
    pinch_normalized_min_cm=PINCH_NORMALIZED_MIN_CM,
    pinch_normalized_max_cm=PINCH_NORMALIZED_MAX_CM,
):
    """
    Compute gripper-related pinch metrics from thumb/index geometry.

    `pinch_value_cm` matches the convention used by the existing Unitree
    gripper control path in this repo.
    """
    _, thumb_tip, _, index_tip, _ = _extract_hand_points(hand, side=side)

    geom_distance_m = float(np.linalg.norm(index_tip - thumb_tip))
    api_distance = hand.get(f"{side}_pinch_distance")
    pinch_distance_m = float(api_distance) if api_distance is not None else geom_distance_m
    pinch_value_cm = pinch_distance_m * 100.0

    denom = max(pinch_max_cm - pinch_min_cm, 1e-6)
    openness = float(np.clip((pinch_value_cm - pinch_min_cm) / denom, 0.0, 1.0))
    normalized_denom = max(pinch_normalized_max_cm - pinch_normalized_min_cm, 1e-6)
    pinch_normalized = float(
        np.clip(
            (pinch_value_cm - pinch_normalized_min_cm) / normalized_denom,
            0.0,
            1.0,
        )
    )

    return HandGripperState(
        pinch_distance_m=pinch_distance_m,
        pinch_value_cm=pinch_value_cm,
        is_pinching=pinch_distance_m < pinch_close_m,
        openness=openness,
        pinch_normalized=pinch_normalized,
    )


def compute_hand_pose_and_gripper(hand, side="right"):
    hand_pose, tip_middle, base_middle = compute_hand_pose(hand, side=side)
    gripper_state = compute_gripper_state(hand, side=side)
    return hand_pose, tip_middle, base_middle, gripper_state
