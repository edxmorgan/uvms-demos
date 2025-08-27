# Copyright (C) 2025 Edward Morgan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import rclpy
from rclpy.node import Node
import numpy as np
from control_msgs.msg import DynamicJointState
from robot import Robot
from task import Task
from rclpy.qos import QoSProfile, QoSHistoryPolicy


class CoverageTask(Node):
    def __init__(self):
        super().__init__('coverage_task',
                          automatically_declare_parameters_from_overrides=True)

        # Get parameter values
        self.no_robot = self.get_parameter('no_robot').value
        self.no_efforts = self.get_parameter('no_efforts').value
        self.robots_prefix = self.get_parameter('robots_prefix').value
        self.record = self.get_parameter('record_data').value
        self.controllers = self.get_parameter('controllers').value
        self.get_logger().info(f"robots controllers : {self.controllers}")

        self.get_logger().info(f"robot prefixes found in task node: {self.robots_prefix}")
        self.total_no_efforts = self.no_robot * self.no_efforts
        self.get_logger().info(f"robots total number of commands : {self.total_no_efforts}")
        
        initial_pos = np.array([0.0, 0.0, 8.0, 0,0,0, 3.1, 0.7, 0.4, 2.1])


        self.robots_and_tasks = [(Robot(self, k, 4, prefix, initial_pos, self.record, controller), Task(initial_pos)) for k, (prefix, controller) in enumerate(list(zip(self.robots_prefix, self.controllers)))]
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # self.uvms_publisher_ = self.create_publisher(Command, '/uvms_controller/uvms/commands', qos_profile)

        frequency = 150  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)
        self.get_logger().info("CoverageTask node has been initialized with optimal control.")


    def timer_callback(self):
        pass
        # command_msg = Command()
        # command_msg.command_type = self.controllers
        # command_msg.acceleration.data = []
        # command_msg.twist.data = []
        # command_msg.pose.data = []

        # for robot_and_task in self.robots_and_tasks:
        #     robot, task = robot_and_task
        #     state = robot.get_state()
        #     if state['status']=='active':
        #         sim_t = state['sim_time']
        #         sim_dt = state['dt']

        #         ref_ned_vel, ref_ned_pos = task.square_velocity_uv_ref(sim_t, T_side=40.0, speed=0.1) #task

        #         robot.set_robot_goals(ref_ned_vel, ref_ned_pos)
        #         robot.publish_reference_path()
        #         # robot.publish_ops_reference_path()
                
        #         robot.publish_robot_path()
        #         robot.get_robot_goals('ref_pos')
        #         command_msg.acceleration.data.extend(robot.get_robot_goals('ref_acc'))
        #         command_msg.acceleration.data.extend([0.0]) #endeffector

        #         command_msg.twist.data.extend(robot.get_robot_goals('ref_vel'))
        #         command_msg.twist.data.extend([0.0]) #endeffector

        #         command_msg.pose.data.extend(robot.get_robot_goals('ref_pos'))
        #         command_msg.pose.data.extend([0.0]) #endeffector

        #         if len(robot.trajectory_twist) > 500:
        #             robot.trajectory_twist.pop(0)
        #             robot.trajectory_poses.pop(0)
                
        # # Publish the command
        # # self.get_logger().info(f'{command_msg.pose.data}')
        # self.uvms_publisher_.publish(command_msg)

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    coveragetask = CoverageTask()

    try:
        rclpy.spin(coveragetask)
    except KeyboardInterrupt:
        coveragetask.get_logger().info('CoverageTask node stopped by KeyboardInterrupt.')
    finally:
        coveragetask.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()