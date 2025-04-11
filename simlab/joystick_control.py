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

# Import ROS2 QoS settings and message type.
from rclpy.qos import QoSProfile, QoSHistoryPolicy
from uvms_interfaces.msg import Command

# Import your robot class
from robot import Robot


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

        self.get_logger().info(f"Robot prefixes found: {self.robots_prefix}")
        self.total_no_efforts = self.no_robot * self.no_efforts
        self.get_logger().info(f"Total number of commands: {self.total_no_efforts}")

        # Initialize robots (make sure your Robot class is defined properly).
        initial_pos = np.array([0.0, 0.0, 0.0, 0, 0, 0, 3.1, 0.7, 0.4, 2.1])
        self.robots = [Robot(self, k, 4, prefix, initial_pos, self.record) for k, prefix in enumerate(self.robots_prefix)]

        # Setup a publisher with a QoS profile.
        qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.publisher_ = self.create_publisher(Command, '/uvms_controller/uvms/commands', qos_profile)

        # Create a timer callback to publish commands at 1000 Hz.
        frequency = 1000  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)


    def timer_callback(self):
        # Create a new command message.
        command_msg = Command()
        command_msg.command_type = ["force"]*self.no_robot
       
        # Build the full command list for all robots.
        data = []
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


            rov_command = [surge, sway, heave, roll, pitch, yaw]
            manipulator_command = [e_joint, d_joint, c_joint, b_joint, a_joint] #[0]*5 # Manipulator command (unused).

            robot.write_data_to_file()
            # robot.publish_robot_path()  # Assumes each Robot instance handles its own publishing.
            data.extend(rov_command + manipulator_command)

        command_msg.force.data = [float(value) for value in data]
        # Publish the command.
        self.publisher_.publish(command_msg)

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
