"""
Standalone Duco gripper LowCmd sender.

This script runs on the teleoperation host. It reproduces the gripper part of
teleop_arm_visionpro.py without starting Vision Pro tracking or arm IK:

AVP pinch distance(cm) -> pinch_normalized -> LowCmd.reserve[0] float32 bits
record command -> ZMQ record command channel

The first 6 motor_cmd.q values are held at a fixed joint target so the control
host can reuse its normal LowCmd parser without moving the arm intentionally.
"""

from __future__ import annotations

import argparse
import struct
import time
from types import SimpleNamespace
from typing import Iterable

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd
    from unitree_sdk2py.utils.crc import CRC
except ModuleNotFoundError:
    ChannelFactoryInitialize = None
    ChannelPublisher = None
    unitree_hg_msg_dds__LowCmd_ = None
    hg_LowCmd = None
    CRC = None

from record_zmq import RecordCmdClient


ARM_DOF = 6
TOPIC_LOWCMD_DEBUG = "rt/lowcmd"
TOPIC_LOWCMD_MOTION = "rt/arm_sdk"
GRIPPER_RESERVE_INDEX = 0

RECORD_CMD_NOOP = 0
RECORD_CMD_START = 1
RECORD_CMD_KEEP = 2
RECORD_CMD_DISCARD = 3
RECORD_CMD_SKIP_RESET = 4
RECORD_CMD_ABORT = 5
RECORD_CMD_PREVIEW = 6

PINCH_NORMALIZED_MIN_CM = 1.0
PINCH_NORMALIZED_MAX_CM = 5.0


def encode_float32_to_uint32(value: float) -> int:
    value = min(1.0, max(0.0, float(value)))
    return struct.unpack("<I", struct.pack("<f", value))[0]


def decode_uint32_to_float32(raw: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(raw)))[0]


def pinch_cm_to_normalized(pinch_value_cm: float) -> float:
    denom = max(PINCH_NORMALIZED_MAX_CM - PINCH_NORMALIZED_MIN_CM, 1e-6)
    return min(1.0, max(0.0, (float(pinch_value_cm) - PINCH_NORMALIZED_MIN_CM) / denom))


def build_wave_sample(sample_index: int, hz: float, period_sec: float) -> tuple[float, float]:
    t = sample_index / max(hz, 1e-6)
    phase = (t % max(period_sec, 1e-6)) / max(period_sec, 1e-6)
    # Triangle wave: 0 -> 1 -> 0. This creates frequent changed targets.
    normalized = 1.0 - abs(2.0 * phase - 1.0)
    pinch_cm = PINCH_NORMALIZED_MIN_CM + normalized * (
        PINCH_NORMALIZED_MAX_CM - PINCH_NORMALIZED_MIN_CM
    )
    return pinch_cm, normalized


def parse_joint_q(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != ARM_DOF:
        raise argparse.ArgumentTypeError(f"expected {ARM_DOF} comma-separated joint values")
    return values


def make_lowcmd(joint_q: Iterable[float], gripper_target: float, crc: CRC):
    if unitree_hg_msg_dds__LowCmd_ is None:
        msg = SimpleNamespace(
            mode_pr=0,
            mode_machine=0,
            motor_cmd=[SimpleNamespace(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0) for _ in range(ARM_DOF)],
            reserve=[0, 0],
            crc=0,
        )
    else:
        msg = unitree_hg_msg_dds__LowCmd_()
    msg.mode_pr = 0
    msg.mode_machine = 0
    for idx, q in enumerate(list(joint_q)[:ARM_DOF]):
        msg.motor_cmd[idx].mode = 1
        msg.motor_cmd[idx].q = float(q)
        msg.motor_cmd[idx].dq = 0.0
        msg.motor_cmd[idx].tau = 0.0
        msg.motor_cmd[idx].kp = 0.0
        msg.motor_cmd[idx].kd = 0.0

    msg.reserve[GRIPPER_RESERVE_INDEX] = encode_float32_to_uint32(gripper_target)
    msg.crc = 0 if crc is None else crc.Crc(msg)
    return msg


def init_channel(args) -> None:
    if args.dry_run:
        return
    if ChannelFactoryInitialize is None:
        raise RuntimeError("unitree_sdk2py is required when --dry-run is not set.")
    if args.peer_address:
        ChannelFactoryInitialize(
            0,
            networkInterface=args.network_interface,
            peerAddress=args.peer_address,
        )
    else:
        ChannelFactoryInitialize(0, networkInterface=args.network_interface)


def send_msg(publisher, msg, dry_run: bool) -> None:
    if not dry_run:
        publisher.Write(msg)


def send_record_command(record_client: RecordCmdClient | None, record_cmd: int, dry_run: bool) -> None:
    if dry_run:
        print(f"dry-run record cmd={record_cmd}")
        return
    if record_client is not None:
        record_client.send_cmd(record_cmd)


def run_episode(args, publisher, crc: CRC, record_client: RecordCmdClient | None, episode_index: int) -> None:
    sample_count = max(1, int(round(args.episode_seconds * args.hz)))
    period = 1.0 / max(args.hz, 1e-6)
    start_time = time.time()

    for sample_index in range(sample_count):
        loop_start = time.time()
        pinch_cm, gripper_target = build_wave_sample(sample_index, args.hz, args.wave_period)
        if sample_index == 0:
            send_record_command(record_client, RECORD_CMD_START, args.dry_run)
        msg = make_lowcmd(args.joint_q, gripper_target, crc)
        send_msg(publisher, msg, args.dry_run)

        if sample_index % max(1, int(args.hz)) == 0:
            decoded = decode_uint32_to_float32(msg.reserve[GRIPPER_RESERVE_INDEX])
            print(
                f"episode={episode_index:03d} sample={sample_index:04d} "
                f"pinch_cm={pinch_cm:.3f} gripper={gripper_target:.3f} "
                f"reserve0=0x{msg.reserve[GRIPPER_RESERVE_INDEX]:08x} decoded={decoded:.3f}"
            )

        elapsed = time.time() - loop_start
        time.sleep(max(0.0, period - elapsed))

    finish_cmd = RECORD_CMD_DISCARD if args.discard else RECORD_CMD_KEEP
    send_record_command(record_client, finish_cmd, args.dry_run)

    print(
        f"episode={episode_index:03d} finished save={not args.discard} "
        f"duration={time.time() - start_time:.2f}s"
    )


def run(args) -> None:
    topic = TOPIC_LOWCMD_MOTION if args.motion else TOPIC_LOWCMD_DEBUG
    init_channel(args)
    publisher = None
    if not args.dry_run:
        if ChannelPublisher is None or hg_LowCmd is None:
            raise RuntimeError("unitree_sdk2py is required when --dry-run is not set.")
        publisher = ChannelPublisher(topic, hg_LowCmd)
        publisher.Init()

    crc = None if CRC is None else CRC()
    record_client = None
    if not args.dry_run:
        record_client = RecordCmdClient(
            ctrl_host=None if args.record_zmq_mode == "teleop-bind" else args.record_host,
            cmd_port=args.record_cmd_port,
            status_port=args.record_status_port,
            bind=args.record_zmq_mode == "teleop-bind",
            bind_host=args.record_bind_host,
        )
    print(f"topic={topic} dry_run={args.dry_run} hz={args.hz} episodes={args.episodes}")

    try:
        if args.send_preview:
            print("sending PREVIEW command before episodes")
            send_record_command(record_client, RECORD_CMD_PREVIEW, args.dry_run)

        for episode_index in range(args.episodes):
            run_episode(args, publisher, crc, record_client, episode_index)
            if episode_index < args.episodes - 1:
                send_record_command(record_client, RECORD_CMD_SKIP_RESET, args.dry_run)
                time.sleep(max(0.0, args.reset_pause_sec))

        if args.abort_at_end:
            send_record_command(record_client, RECORD_CMD_ABORT, args.dry_run)
    finally:
        if record_client is not None:
            record_client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", type=str, default="enp0s31f6")
    parser.add_argument("--peer-address", type=str, default="10.188.223.119")
    parser.add_argument("--record-host", type=str, default="10.188.223.119")
    parser.add_argument("--record-zmq-mode", choices=("control-bind", "teleop-bind"), default="control-bind")
    parser.add_argument("--record-bind-host", type=str, default="0.0.0.0")
    parser.add_argument("--record-cmd-port", type=int, default=5587)
    parser.add_argument("--record-status-port", type=int, default=5588)
    parser.add_argument("--motion", action="store_true", help="publish to rt/arm_sdk instead of rt/lowcmd")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument("--wave-period", type=float, default=2.0)
    parser.add_argument("--reset-pause-sec", type=float, default=0.5)
    parser.add_argument("--joint-q", type=parse_joint_q, default=[0.0] * ARM_DOF)
    parser.add_argument("--send-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--discard", action="store_true", help="send DISCARD instead of KEEP")
    parser.add_argument("--abort-at-end", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print encode/decode data without DDS publish")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
