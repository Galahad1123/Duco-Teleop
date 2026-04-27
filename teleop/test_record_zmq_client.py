"""
Minimal teleop-host ZMQ record transport test.

Run on the teleop/vision host:
    python test_record_zmq_client.py --host <control-host-ip>

The script first performs raw TCP probes to the command/status ports, then sends
repeated ZMQ pings and waits for PUB/SUB acknowledgements.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import time

import zmq


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def probe_tcp(host: str, port: int, timeout_sec: float) -> bool:
    started = time.time()
    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout_sec)) as sock:
            local = sock.getsockname()
            elapsed = time.time() - started
            logger.info("TCP probe OK host=%s port=%d local=%s elapsed=%.3fs", host, port, local, elapsed)
            return True
    except OSError as exc:
        elapsed = time.time() - started
        logger.warning("TCP probe FAILED host=%s port=%d elapsed=%.3fs error=%s", host, port, elapsed, exc)
        return False


def run(args: argparse.Namespace) -> None:
    logger.info("Target command endpoint: tcp://%s:%d", args.host, args.cmd_port)
    logger.info("Target status endpoint:  tcp://%s:%d", args.host, args.status_port)

    cmd_ok = probe_tcp(args.host, args.cmd_port, args.tcp_timeout)
    status_ok = probe_tcp(args.host, args.status_port, args.tcp_timeout)
    if args.probe_only:
        raise SystemExit(0 if cmd_ok and status_ok else 2)

    ctx = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.SNDHWM, 128)
    push.setsockopt(zmq.LINGER, 0)
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 128)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.SUBSCRIBE, b"")

    push.connect(f"tcp://{args.host}:{args.cmd_port}")
    sub.connect(f"tcp://{args.host}:{args.status_port}")
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    logger.info("ZMQ sockets connected; waiting %.2fs for SUB handshake.", args.warmup_sec)
    time.sleep(max(0.0, args.warmup_sec))

    ack_count = 0
    try:
        for seq in range(1, args.count + 1):
            msg = {
                "type": "ping",
                "seq": seq,
                "client_time": time.time(),
                "payload": args.payload,
            }
            push.send_json(msg)
            logger.info("sent ping seq=%d", seq)

            deadline = time.time() + args.ack_timeout
            got_ack = False
            while time.time() < deadline:
                remaining_ms = max(1, int((deadline - time.time()) * 1000))
                events = dict(poller.poll(timeout=remaining_ms))
                if sub not in events:
                    continue
                ack = sub.recv_json()
                ack_count += 1
                logger.info("recv ack #%d: %s", ack_count, json.dumps(ack, ensure_ascii=False))
                got_ack = True
                break
            if not got_ack:
                logger.warning("no ack for seq=%d within %.2fs", seq, args.ack_timeout)
            time.sleep(max(0.0, args.interval))
    finally:
        push.close(linger=0)
        sub.close(linger=0)
        ctx.term()

    if ack_count <= 0:
        logger.error("No ZMQ acknowledgements received.")
        raise SystemExit(1)
    logger.info("ZMQ test finished: sent=%d ack=%d", args.count, ack_count)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--cmd-port", type=int, default=5587)
    parser.add_argument("--status-port", type=int, default=5588)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--ack-timeout", type=float, default=2.0)
    parser.add_argument("--tcp-timeout", type=float, default=2.0)
    parser.add_argument("--warmup-sec", type=float, default=0.5)
    parser.add_argument("--payload", type=str, default="record-zmq-test")
    parser.add_argument("--probe-only", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
