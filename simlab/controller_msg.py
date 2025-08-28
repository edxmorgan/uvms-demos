# controller_msg.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Optional
import numpy as np

from std_msgs.msg import Header, Float64MultiArray
from control_msgs.msg import DynamicInterfaceGroupValues, InterfaceValue

@dataclass
class SixDofState:
    """Vehicle pose, twist, and wrench in consistent frames.
    pose: [x, y, z, roll, pitch, yaw]
    twist: [u, v, w, p, q, r] in body
    wrench: [Fx, Fy, Fz, Tx, Ty, Tz] body frame
    """
    pose: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))
    twist: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))
    wrench: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))

@dataclass
class ArmState:
    """Manipulator joint vectors. Order must match your system."""
    q: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))
    dq: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))
    effort: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))

@dataclass
class FullRobotMsg:
    """Single container for vehicle and arm commands or states."""
    prefix: str
    vehicle: SixDofState = field(default_factory=SixDofState)
    arm: ArmState = field(default_factory=ArmState)

    # -------------------------
    # ROS conversion, publishing
    # -------------------------
    def to_vehicle_dynamic_group(
        self,
        now_ros_time,
        frame_id: Optional[str] = None
    ) -> DynamicInterfaceGroupValues:
        """Builds DynamicInterfaceGroupValues with wrench in the expected order."""
        msg = DynamicInterfaceGroupValues()
        msg.header = Header()
        msg.header.stamp = now_ros_time
        msg.header.frame_id = frame_id or f"{self.prefix}map"

        group = f"{self.prefix}IOs"
        msg.interface_groups = [group]

        iv = InterfaceValue()
        iv.interface_names = [
            "force.x", "force.y", "force.z",
            "torque.x", "torque.y", "torque.z",
        ]
        Fx, Fy, Fz, Tx, Ty, Tz = self.vehicle.wrench.astype(float).tolist()
        iv.values = [Fx, Fy, Fz, Tx, Ty, Tz]
        msg.interface_values = [iv]
        return msg

    def to_arm_effort_array(self) -> Float64MultiArray:
        """Manipulator effort message in your expected order [e, d, c, b, a]."""
        msg = Float64MultiArray()
        msg.data = self.arm.effort.astype(float).tolist()
        return msg

    # Convenience setters
    def set_vehicle_wrench(self, wrench_body_6: Sequence[float]) -> None:
        self.vehicle.wrench = np.asarray(wrench_body_6, dtype=float).reshape(6)

    def set_vehicle_pose(self, pose_ned_rpy_6: Sequence[float]) -> None:
        self.vehicle.pose = np.asarray(pose_ned_rpy_6, dtype=float).reshape(6)

    def set_vehicle_twist(self, twist_6: Sequence[float]) -> None:
        self.vehicle.twist = np.asarray(twist_6, dtype=float).reshape(6)

    def set_arm_effort(self, effort_5: Sequence[float]) -> None:
        self.arm.effort = np.asarray(effort_5, dtype=float).reshape(5)