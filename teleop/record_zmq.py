import itertools
import logging
import time

import zmq


logger = logging.getLogger(__name__)

DEFAULT_CMD_PORT = 5587
DEFAULT_STATUS_PORT = 5588


class RecordCmdClient:
    """Teleop-side ZMQ transport for sparse recording commands."""

    def __init__(
        self,
        ctrl_host: str | None,
        cmd_port: int = DEFAULT_CMD_PORT,
        status_port: int = DEFAULT_STATUS_PORT,
        bind: bool = False,
        bind_host: str = "0.0.0.0",
    ):
        self._ctx = zmq.Context()
        self._seq = itertools.count(1)

        self._cmd_sock = self._ctx.socket(zmq.PUSH)
        self._cmd_sock.setsockopt(zmq.SNDHWM, 8)
        self._cmd_sock.setsockopt(zmq.LINGER, 0)

        self._status_sock = self._ctx.socket(zmq.SUB)
        self._status_sock.setsockopt(zmq.RCVHWM, 32)
        self._status_sock.setsockopt(zmq.LINGER, 0)
        self._status_sock.setsockopt(zmq.SUBSCRIBE, b"")

        if bind:
            self._cmd_sock.bind(f"tcp://{bind_host}:{cmd_port}")
            self._status_sock.bind(f"tcp://{bind_host}:{status_port}")
            logger.info(
                "[record-zmq] client bound cmd=tcp://%s:%d status=tcp://%s:%d",
                bind_host,
                cmd_port,
                bind_host,
                status_port,
            )
        else:
            if ctrl_host is None:
                raise ValueError("ctrl_host is required when bind=False")
            self._cmd_sock.connect(f"tcp://{ctrl_host}:{cmd_port}")
            self._status_sock.connect(f"tcp://{ctrl_host}:{status_port}")
            logger.info(
                "[record-zmq] client connected cmd=tcp://%s:%d status=tcp://%s:%d",
                ctrl_host,
                cmd_port,
                ctrl_host,
                status_port,
            )

    def send_cmd(self, cmd: int) -> int | None:
        seq = next(self._seq)
        try:
            self._cmd_sock.send_json({"seq": seq, "cmd": int(cmd), "ts": time.time()}, flags=zmq.NOBLOCK)
            return seq
        except zmq.Again:
            logger.warning("[record-zmq] command queue full, dropped cmd=%d seq=%d", int(cmd), seq)
            return None

    def recv_status(self) -> dict | None:
        try:
            return self._status_sock.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None

    def close(self):
        self._cmd_sock.close(linger=0)
        self._status_sock.close(linger=0)
        self._ctx.term()
