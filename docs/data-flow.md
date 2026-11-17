# DUCO 数据链路与绘图依据

本文依据仓库代码静态阅读整理，不代表运行或性能测试结果。

## 主控制链路
1. `teleop/teleop_arm_visionpro.py` 的 `main` 创建 `VisionProTVWrapperDuco`，通过 `get_tele_data()` 读取手腕位姿与归一化捏合值。定制 AVP 包装器已收录在 `teleop/televuer`；其字段语义详见下方核实记录。
2. `build_target_pose` 以初始手腕位姿为参考，将平移差映射到机器人坐标方向，并应用位置缩放。姿态支持 fixed 与 teleop 两种模式，后者对相对旋转进行缩放和轴向调整。
3. `solve_target_with_fallback` 调用 `DucoArmIK.solve_ik`；全位姿解的位置误差超过阈值时，尝试固定参考姿态的位置优先解，仅在位置误差更小时采用。
4. `teleop/robot_control/duco_arm_ik.py` 使用 URDF、Pinocchio、CasADi 与 IPOPT。目标包含位置、旋转、关节正则和平滑项，施加关节限位，并使用加权移动滤波。
5. `duco_ArmController.ctrl_arm` 更新关节目标与夹爪目标。独立发送线程进行目标限速、DDS 消息编码与发送；夹爪目标编码进 reserve 字段。
6. 控制器发布 `rt/lowcmd`，motion 模式可使用 `rt/arm_sdk`；订阅 `rt/lowstate` 获得关节位置与速度，返回主循环供下一次求解使用。机器人侧桥接与硬件执行作为外部端点展示，不推断驱动内部实现。

## 录制交互链路
`on_press` 将键盘操作加入录制命令队列；主循环通过 `record_zmq.py` 中的 `RecordCmdClient.send_cmd` 发送 ZeroMQ PUSH 消息，通过 SUB 接收状态。支持预览、开始、保留、丢弃、跳过复位与中止。外部录制服务负责实际采集与保存，本客户端仅发送操作并接收事件；不把 `episode_writer.py` 画成此入口的直接调用。

## 首页图的抽象
- 左：AVP 操作者，输出手腕位姿与捏合输入。
- 中：遥操作主机，完成位姿映射、IK 与命令发送。
- 右：机器人侧桥接与 DUCO 单臂，DDS 命令向右、状态向左。
- 下：键盘录制交互，通过 ZeroMQ 与外部录制服务交换指令及状态。
- 用户提供的机器人图片只用于设备展示；图片中的移动底座不表示本项目实现底盘遥操作。
- 不展示端口、IP、频率和优化权重，避免淹没主要信息。

## AVP 原生输入：服务器源码核实

2026-11-17 从 3090 项目取得定制 televuer 提交 `8579f46dc7a91bddf9dafdad36a21b788abd7c55`，已原样纳入仓库，替代原来的子模块指针。

实际链路：VisionProTeleop / Tracking Streamer → avp_stream.VisionProStreamer → get_latest() → compute_hand_pose / compute_gripper_state → TeleData → 位姿映射 → IK → DDS。

`visionpro_tv_wrapper.py` 创建 VisionProStreamer、调用 start_webrtc()，通过 get_latest() 获取数据。不能仅凭 start_webrtc() 将位姿数据的传输协议确定为 WebRTC。

`hand_pose_utils.py` 将手指变换乘以手腕变换，以拇指和食指指尖中点为控制位置，根据指根和指尖几何关系建立朝向。因此 TeleData 的 wrist_pose 字段实际承载的是重建控制位姿，并非原始手腕位姿。夹爪使用流中的 pinch_distance，缺失时由指尖间距计算，再归一化。

外部项目 https://github.com/Improbable-AI/VisionProTeleop 文档说明其 visionOS 原生追踪基于 Apple ARKit。已确认本项目使用其 Python 接口；未取得设备端安装版本或独立 Swift 修改，不宣称作者从零实现了 ARKit 追踪器。AR 实机使用场景由作者提供的说明和演示材料支持，代码静态查阅不等于运行验证。

上游文档也含真机和透视模式入口，不使用“上游仅支持 Isaac Lab VR”的绝对比较。
