import os
import time
from collections import deque
from typing import Optional, Tuple

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin


class SimpleWeightedMovingFilter:
    """
    一个小型平滑器 把连续几次 IK 解做加权平均，减少关节跳动
    """

    def __init__(self, weights: np.ndarray, data_dim: int):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights = self.weights / np.sum(self.weights)
        self.data_dim = data_dim
        self.buffer = deque(maxlen=len(self.weights)) # 取最近几次解
        self.filtered_data = np.zeros(data_dim, dtype=np.float64)

    def add_data(self, data: np.ndarray):
        data = np.asarray(data, dtype=np.float64).reshape(-1)
        self.buffer.appendleft(data.copy())

        weighted_sum = np.zeros(self.data_dim, dtype=np.float64)
        total_weight = 0.0
        for i, x in enumerate(self.buffer):
            w = self.weights[i]
            weighted_sum += w * x
            total_weight += w

        if total_weight > 0:
            self.filtered_data = weighted_sum / total_weight # 加权平均
        else:
            self.filtered_data = data.copy()


class DucoArmIK:
    """
    Single-arm IK solver for DUCO GCR5_910-like 6DoF manipulator.

    Input:
        target_pose: 4x4 homogeneous transform in world/base frame 
        需确认：传进来的 target_pose,和 Pinocchio 模型使用的 base/world 坐标系一致。
        current_arm_q: current joint position, shape (6,)
        current_arm_dq: current joint velocity, shape (6,), optional

    Output:
        sol_q: solved joint target, shape (6,)
        sol_tauff: feedforward torque from pin.rnea, shape (6,)
    """

    def __init__(
        self,
        urdf_path: str,
        model_dir: Optional[str] = None,
        ee_parent_joint: str = "arm_1_joint_6",
        ee_frame_name: str = "ee",
        ee_offset_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        visualization: bool = False,
    ):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.urdf_path = urdf_path
        self.model_dir = model_dir if model_dir is not None else os.path.dirname(urdf_path)
        self.visualization = visualization

        # 机器人模型加载
        self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)
        self.model = self.robot.model # 静态结构
        # self.data = self.model.createData() # 运动学动力学计算时的临时数据结构

        # 在最后一个关节上手工挂了一个自定义操作点 frame
        parent_joint_id = self.model.getJointId(ee_parent_joint)
        if parent_joint_id == 0:
            raise ValueError(f"Cannot find ee_parent_joint='{ee_parent_joint}' in model.")

        self.model.addFrame(
            pin.Frame(
                ee_frame_name,
                parent_joint_id,
                pin.SE3(np.eye(3), np.array(ee_offset_xyz, dtype=np.float64)),
                pin.FrameType.OP_FRAME,
            )
        )
        self.ee_frame_name = ee_frame_name
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        self.data = self.model.createData() # 运动学动力学计算时的临时数据结构

        # CasADi 符号模型
        # 把 Pinocchio 模型“符号化”，这样就能在 CasADi 里写优化问题
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q", self.model.nq, 1) # 待求解的关节变量
        self.cTf = casadi.SX.sym("tf", 4, 4) # 目标末端位姿参数

        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Error terms
        # 位置误差：末端实际位置 - 目标位置
        # 旋转误差：末端实际旋转 * 目标旋转的
        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf],
            [
                self.cdata.oMf[self.ee_frame_id].translation - self.cTf[:3, 3]
            ],
        )

        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf],
            [
                cpin.log3(
                    self.cdata.oMf[self.ee_frame_id].rotation @ self.cTf[:3, :3].T
                )
            ],
        )

        # Optimization
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.model.nq) # 优化变量
        self.var_q_last = self.opti.parameter(self.model.nq) # 上一次求解的关节值，用于平滑
        self.param_tf = self.opti.parameter(4, 4) # 目标位姿参数

        # 平方和
        self.trans_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf)) # 位置误差
        self.rot_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf)) # 姿态误差
        self.reg_cost = casadi.sumsqr(self.var_q) # 正则项
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last) # 平滑项

        self.opti.subject_to(
            self.opti.bounded(
                self.model.lowerPositionLimit,
                self.var_q,
                self.model.upperPositionLimit,
            )
        ) # 关节限位约束

        # Similar spirit to official IK: position is dominant, add smooth + regularization
        # 总目标函数
        self.opti.minimize(
            200.0 * self.trans_cost # 位置不对
            + 1.0 * self.rot_cost # 姿态不对
            + 0.01 * self.reg_cost # 喜欢极端角度
            + 0.01 * self.smooth_cost # 解很跳
        )

        opts = {
            "expand": True,
            "detect_simple_bounds": True,
            "calc_lam_p": False,
            "print_time": False,
            "ipopt.sb": "yes",
            "ipopt.print_level": 0,
            "ipopt.max_iter": 50, # 最多 50 次迭代，通常不到 10 次就收敛了 IK较慢可减少
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 5e-4, # 见好就收
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.derivative_test": "none",
            "ipopt.jacobian_approximation": "exact",
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.model.nq, dtype=np.float64)

        # Simple smoothing
        self.smooth_filter = SimpleWeightedMovingFilter(
            np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64),
            self.model.nq,
        )


    def fk(self, q: np.ndarray) -> np.ndarray:
        """
        给你一个关节角 q,返回当前末端 EE 的 4×4 齐次变换矩阵
        """
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        pin.framesForwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        T = self.data.oMf[self.ee_frame_id]
        return T.homogeneous.copy()

    def solve_ik(
        self,
        target_pose: np.ndarray,
        current_arm_q: Optional[np.ndarray] = None,
        current_arm_dq: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve IK for one target pose.

        Args:
            target_pose: 4x4 目标末端位姿
            current_arm_q: 当前 6 维关节角
            current_arm_dq: 当前 6 维关节角速度(仅用于计算 feedforward tauff,实际 IK 求解时并未使用) 可选

        Returns:
            sol_q: 解出的 6 维关节角
            sol_tauff: 根据解出的 sol_q 和 current_arm_dq 计算的 6 维前馈力矩，使用 pin.rnea 计算
        """
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if target_pose.shape != (4, 4):
            raise ValueError(f"target_pose must be (4,4), got {target_pose.shape}")

        if current_arm_q is not None:
            current_arm_q = np.asarray(current_arm_q, dtype=np.float64).reshape(-1)
            if current_arm_q.shape[0] != self.model.nq:
                raise ValueError(
                    f"current_arm_q dim mismatch: expect {self.model.nq}, got {current_arm_q.shape[0]}"
                )
            self.init_data = current_arm_q.copy()

        # 设置优化参数和初始值，开始求解
        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.param_tf, target_pose)
        self.opti.set_value(self.var_q_last, self.init_data)

        try:
            # 求解 IK
            sol = self.opti.solve()
            sol_q = np.asarray(sol.value(self.var_q), dtype=np.float64).reshape(-1)

            # debug
            # self.smooth_filter.add_data(sol_q) # 平滑
            # sol_q = self.smooth_filter.filtered_data.copy()

            if current_arm_dq is not None:
                v = np.asarray(current_arm_dq, dtype=np.float64).reshape(-1) * 0.0
            else:
                v = np.zeros(self.model.nv, dtype=np.float64)

            self.init_data = sol_q.copy()
            # 计算前馈力矩，注意这里直接把 current_arm_dq 乘以 0 了，因为我们不想让速度对力矩有影响，你也可以根据需要调整这个逻辑
            sol_tauff = pin.rnea(
                self.model,
                self.data,
                sol_q,
                v,
                np.zeros(self.model.nv, dtype=np.float64),
            )
            return sol_q, sol_tauff

        except Exception as e:
            print(f"[DucoArmIK] IK solve failed: {e}")

            try:
                # 首先尝试取 debug 解
                sol_q = np.asarray(self.opti.debug.value(self.var_q), dtype=np.float64).reshape(-1)
            except Exception:
                # 退回初始关节 不乱动
                sol_q = self.init_data.copy()

            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data.copy()

            if current_arm_dq is not None:
                v = np.asarray(current_arm_dq, dtype=np.float64).reshape(-1) * 0.0
            else:
                v = np.zeros(self.model.nv, dtype=np.float64)

            self.init_data = sol_q.copy()
            sol_tauff = pin.rnea(
                self.model,
                self.data,
                sol_q,
                v,
                np.zeros(self.model.nv, dtype=np.float64),
            )

            if current_arm_q is not None:
                return current_arm_q.copy(), np.zeros(self.model.nv, dtype=np.float64)
            return sol_q, sol_tauff


if __name__ == "__main__":
    # Example usage:
    urdf_path = "duco_gcr5_910.urdf"   # replace with your actual path
    model_dir = os.path.dirname(os.path.abspath(urdf_path))

    ik = DucoArmIK(
        urdf_path=urdf_path,
        model_dir=model_dir,
        ee_parent_joint="arm_1_joint_6",
        ee_frame_name="ee",
        # If TCP is not exactly at joint_6 origin, tune this offset later
        # 你真正想控制的末端，往往不是第 6 关节原点，而是：
        # 法兰盘中心
        # 工具中心点
        # 夹爪中心
        # 腕部相机坐标系附近某个点
        # 所以后面你很可能需要把 ee_offset_xyz 调成一个小偏置。
        ee_offset_xyz=(0.0, 0.0, 0.0),
    )

    q0 = np.zeros(6)
    T0 = ik.fk(q0)
    print("Initial FK:")
    print(T0)

    # Make a small reachable perturbation
    T_target = T0.copy()
    T_target[0, 3] += 0.03
    T_target[2, 3] += 0.02

    q_sol, tau_ff = ik.solve_ik(T_target, current_arm_q=q0)
    print("Solved q:", q_sol)
    print("Feedforward tau:", tau_ff)

    T_sol = ik.fk(q_sol)
    print("FK after solve:")
    print(T_sol)