"""
Reverse-direction ZMQ test for networks that block teleop -> bridge TCP.

Run on the teleop/vision host first:
    python test_record_zmq_teleop_bind.py --bind-host 0.0.0.0

Then run DucoTeleoperate/test_record_zmq_bridge_connect.py on the bridge host.
This keeps the application data direction the same:
    teleop PUSH command -> bridge PULL command
    bridge PUB status  -> teleop SUB status
but the TCP connections are initiated from the bridge host.
"""

from __future__ import annotations

import argparse
import logging
import socket
import time

import zmq


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def local_ipv4_addresses() -> list[str]:
    addresses = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(info[4][0])
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            addresses.add(sock.getsockname()[0])
    except OSError:
        pass
    return sorted(addresses)


def run(args: argparse.Namespace) -> None:
    ctx = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.SNDHWM, 128)
    push.setsockopt(zmq.LINGER, 0)
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 128)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.SUBSCRIBE, b"")

    cmd_endpoint = f"tcp://{args.bind_host}:{args.cmd_port}"
    status_endpoint = f"tcp://{args.bind_host}:{args.status_port}"
    push.bind(cmd_endpoint)
    sub.bind(status_endpoint)

    logger.info("PUSH bound: %s", cmd_endpoint)
    logger.info("SUB  bound: %s", status_endpoint)
    logger.info("Local IPv4 candidates: %s", ", ".join(local_ipv4_addresses()) or "unknown")
    logger.info("Start bridge_connect test on bridge host, then wait for acks.")
    time.sleep(max(0.0, args.warmup_sec))

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)
    ack_count = 0

    try:
        for seq in range(1, args.count + 1):
            msg = {"type": "ping", "seq": seq, "client_time": time.time()}
            push.send_json(msg)
            logger.info("sent ping seq=%d", seq)

            deadline = time.time() + args.ack_timeout
            got_ack = False
            while time.time() < deadline:
                events = dict(poller.poll(timeout=max(1, int((deadline - time.time()) * 1000))))
                if sub not in events:
                    continue
                ack = sub.recv_json()
                ack_count += 1
                logger.info("recv ack #%d: %s", ack_count, ack)
                got_ack = True
                break
            if not got_ack:
                logger.warning("no ack for seq=%d within %.2fs", seq, args.ack_timeout)
            time.sleep(max(0.0, args.interval))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
    finally:
        push.close(linger=0)
        sub.close(linger=0)
        ctx.term()

    logger.info("finished: sent=%d ack=%d", args.count, ack_count)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", type=str, default="0.0.0.0")
    parser.add_argument("--cmd-port", type=int, default=5587)
    parser.add_argument("--status-port", type=int, default=5588)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--ack-timeout", type=float, default=2.0)
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
