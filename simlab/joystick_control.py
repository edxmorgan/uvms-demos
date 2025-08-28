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

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.qos import QoSProfile, QoSHistoryPolicy
from std_msgs.msg import Float64MultiArray
from control_msgs.msg import DynamicInterfaceGroupValues
from control_msgs.msg import DynamicInterfaceGroupValues, InterfaceValue
from robot import Robot
from std_msgs.msg import Header


###############################################################################
# ROS2 Node that uses the PS4 controller for ROV teleoperation.
#
# The ROV command is built as follows:
#   - ROV Command (6 elements): [surge, sway, heave, roll, pitch, yaw]
#       surge  = - (left stick vertical)   (inverted so that pushing forward is positive)
#       sway   = left stick horizontal
#       heave  = analog value from triggers
#       roll   = 0.0 (unused)
#       pitch  = right stick vertical
#       yaw    = right stick horizontal
#
#   - Manipulator Command (5 elements): all zeros.
#
# Total command for each robot is 11 elements.
###############################################################################

class PS4TeleopNode(Node):
    def __init__(self):
        super().__init__('ps4_teleop_node',
                         automatically_declare_parameters_from_overrides=True)

        # Retrieve parameters (e.g. number of robots, efforts, and robot prefixes).
        self.no_robot = self.get_parameter('no_robot').value
        self.no_efforts = self.get_parameter('no_efforts').value
        self.robots_prefix = self.get_parameter('robots_prefix').value
        self.record = self.get_parameter('record_data').value
        self.controllers = self.get_parameter('controllers').value

        self.get_logger().info(f"Robot prefixes found: {self.robots_prefix}")
        self.total_no_efforts = self.no_robot * self.no_efforts
        self.get_logger().info(f"Total number of commands: {self.total_no_efforts}")

        # Initialize robots (make sure your Robot class is defined properly).
        initial_pos = np.array([0.0, 0.0, 0.0, 0, 0, 0, 3.1, 0.7, 0.4, 2.1])

        # Setup a publisher with a QoS profile.
        qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.robots = []
        for k, (prefix, controller) in enumerate(list(zip(self.robots_prefix, self.controllers))):
            vehicle_command_publisher_k = self.create_publisher(
                DynamicInterfaceGroupValues,
                f"vehicle_effort_controller_{prefix}/commands",
                qos_profile
            )
            manipulator_command_publisher_k = self.create_publisher(
                Float64MultiArray,
                f"manipulation_effort_controller_{prefix}/commands",
                qos_profile
            )
            robot_k = Robot(self, k, 4, prefix, initial_pos, self.record, controller, vehicle_command_publisher_k, manipulator_command_publisher_k)
            self.robots.append(robot_k)

        # Create a timer callback to publish commands at 1000 Hz.
        frequency = 1000  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)

    def build_vehicle_message(self, robot, surge, sway, heave, roll, pitch, yaw):
        """
        Build DynamicInterfaceGroupValues for the GPIO group, order must match controller config.
        """
        msg = DynamicInterfaceGroupValues()

        # Fill header
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"{robot.prefix}map"

        # The group name must match your GPIO name in URDF, <gpio name="${prefix}IOs">
        gpio_group = f"{robot.prefix}IOs"
        msg.interface_groups = [gpio_group]

        iv = InterfaceValue()
        iv.interface_names = [
            "force.x", "force.y", "force.z",
            "torque.x", "torque.y", "torque.z",
        ]
        iv.values = [
            float(surge), float(sway), float(heave),
            float(roll), float(pitch), float(yaw),
        ]
        msg.interface_values = [iv]
        return msg
    

    def timer_callback(self):
        # Create a new command message.
        # Build the full command list for all robots.
        for robot in self.robots:
            robot.publish_robot_path()
            robot.publish_gt_path()
            [surge, sway, heave, roll, pitch, yaw] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            [e_joint, d_joint, c_joint, b_joint, a_joint] = [0.0, 0.0, 0.0, 0.0, 0.0]

            if robot.has_joystick_interface:
                # Safely acquire the latest controller values.
                with robot.controller_lock:
                    surge = robot.rov_surge
                    sway = robot.rov_sway
                    heave = robot.rov_z
                    roll = robot.rov_roll
                    pitch = robot.rov_pitch
                    yaw = robot.rov_yaw

                    e_joint= robot.jointe
                    d_joint= robot.jointd
                    c_joint= robot.jointc
                    b_joint= robot.jointb
                    a_joint= robot.jointa

            # Vehicle, DynamicInterfaceGroupValues payload
            vehicle_msg = self.build_vehicle_message(
                robot, surge, sway, heave, roll, pitch, yaw
            )

            # Manipulator, Float64MultiArray to ForwardCommandController
            manipulator_msg = Float64MultiArray()
            manipulator_msg.data = [
                float(e_joint), float(d_joint), float(c_joint),
                float(b_joint), float(a_joint)
            ]

            # Publish
            robot.vehicle_command_publisher.publish(vehicle_msg)
            robot.manipulator_command_publisher.publish(manipulator_msg)

            robot.write_data_to_file()
            robot.publish_robot_path()


    def destroy_node(self):
        # Optionally, stop the PS4 controller listener here if needed.
        super().destroy_node()


###############################################################################
# Main entry point.
###############################################################################
def main(args=None):
    rclpy.init(args=args)
    teleop_node = PS4TeleopNode()
    try:
        rclpy.spin(teleop_node)
    except KeyboardInterrupt:
        teleop_node.get_logger().info('PS4 Teleop node stopped by KeyboardInterrupt.')
    finally:
        teleop_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
