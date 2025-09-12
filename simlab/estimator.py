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
from robot import Robot

class ParamEstimatorNode(Node):
    def __init__(self):
        super().__init__('param_estimator_node',
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
        initial_ref_pos = np.array([0.0, 0.0, 0.0, 0, 0, 0, 3.1, 0.7, 0.4, 2.1])

        self.robots = []
        for k, (prefix, controller) in enumerate(list(zip(self.robots_prefix, self.controllers))):
            robot_k = Robot(self, k, 4, prefix, initial_ref_pos, self.record, controller)
            self.robots.append(robot_k)
            self.get_logger().info(f"\033[35mEstimator running on {robot_k.robot_name}\033[0m")

        # Create a timer callback to publish commands at 1000 Hz.
        frequency = 1000  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)    

    def timer_callback(self):
        for robot in self.robots:
            pass


    def destroy_node(self):
        for robot in self.robots:
            if robot.record:
                robot.close_csv()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    estimator_node = ParamEstimatorNode()
    try:
        rclpy.spin(estimator_node)
    except KeyboardInterrupt:
        estimator_node.get_logger().info('Param Estimator node node stopped by KeyboardInterrupt.')
    finally:
        estimator_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
