import time

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState

TOPIC = "rt/lowstate"
ARM_DOF = 6


def format_lowstate(msg):
    q = [round(float(msg.motor_state[i].q), 6) for i in range(ARM_DOF)]
    dq = [round(float(msg.motor_state[i].dq), 6) for i in range(ARM_DOF)]
    return f"tick={int(msg.tick)}, q={q}, dq={dq}"


if __name__ == "__main__":
    # ChannelFactoryInitialize(0, networkInterface="enp0s31f6")
    ChannelFactoryInitialize(0, networkInterface="enp0s31f6", peerAddress="10.188.223.119")
    sub = ChannelSubscriber(TOPIC, hg_LowState)
    sub.Init()

    while True:
        msg = sub.Read()
        if msg is not None:
            print(f"Subscribe success. {format_lowstate(msg)}")
        else:
            print("No data subscribed.")
            break
        time.sleep(0.01)

    sub.Close()
