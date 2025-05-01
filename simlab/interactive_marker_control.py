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
import numpy as np
np.float = float  # Patch NumPy to satisfy tf_transformations' use of np.float

import copy
import math
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
import casadi as ca
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from rclpy.qos import QoSProfile, QoSHistoryPolicy
from uvms_interfaces.msg import Command
from robot import Robot
import tf2_ros
from geometry_msgs.msg import TransformStamped
import ament_index_python
import os
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial import ConvexHull
from blue_rov import Params as blue
from alpha_reach import Params as alpha
from tf_transformations import quaternion_matrix, quaternion_from_matrix


class BasicControlsNode(Node):
    def __init__(self):
        super().__init__('uvms_interactive_controls',
                         automatically_declare_parameters_from_overrides=True)
        
        package_share_directory = ament_index_python.get_package_share_directory('simlab')
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        uvms_FK_path = os.path.join(package_share_directory, 'fk_eval.casadi')
        self.uvms_FK = ca.Function.load(uvms_FK_path)

        # Example: get some parameters
        self.no_robot = self.get_parameter('no_robot').value
        self.no_efforts = self.get_parameter('no_efforts').value
        self.robots_prefix = self.get_parameter('robots_prefix').value
        self.record = self.get_parameter('record_data').value
        self.controllers = self.get_parameter('controllers').value
        self.total_no_efforts = self.no_robot * self.no_efforts

        qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.uvms_publisher_ = self.create_publisher(Command, '/uvms_controller/uvms/commands', qos_profile)

        self.taskspace_pc_publisher_ = self.create_publisher(PointCloud2, 'workspace_pointcloud', 10)
        self.rov_pc_publisher_ = self.create_publisher(PointCloud2, 'base_pointcloud', 10)

        workspace_pts_path = os.path.join(package_share_directory, 'workspace.npy')
        self.workspace_pts = np.load(workspace_pts_path)
        self.workspace_hull = ConvexHull(self.workspace_pts)

        self.workspace_pts_list = self.workspace_pts.tolist()
        self.rov_ellipsoid_cl_pts = self.generate_rov_ellipsoid(a=0.3, b=0.3, c=0.2, num_points=10000)
        self.vehicle_body_hull = ConvexHull(self.rov_ellipsoid_cl_pts)

        frequency = 500  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)

        initial_pos = np.array([0.0, 0.0, 0.0, 0, 0, 0, 3.1, 0.7, 0.4, 2.1])
        self.robots = [Robot(self, k, 4, prefix, initial_pos, self.record)
                       for k, prefix in enumerate(self.robots_prefix)]

        self.last_vehicle_marker_pose = Pose()
        self.last_vehicle_marker_pose.orientation.w = 1.0
        self.selected_robot_index = 0 # by default robot 0 is selected
        self.execute_plan = False

        self.arm_base_pose = Pose()
        self.arm_base_pose.position.x = 0.19
        self.arm_base_pose.position.y = 0.0
        self.arm_base_pose.position.z = -0.12
        r = R.from_euler('xyz', [3.142, 0.0, 0.0])
        q = r.as_quat()
        (self.arm_base_pose.orientation.x,
         self.arm_base_pose.orientation.y,
         self.arm_base_pose.orientation.z,
         self.arm_base_pose.orientation.w) = q

        # Create marker server, menu handler
        self.server = InteractiveMarkerServer(self, "uvms_interactive_controls")
        self.menu_handler = MenuHandler()
        self.menu_handler.insert("execute", callback=self.processFeedback)
        sub_menu_handle = self.menu_handler.insert("Robots")
        for prefix in self.robots_prefix:
            self.menu_handler.insert(f"{prefix} plan", parent=sub_menu_handle, callback=self.processFeedback)

        self.base_frame = "base_link"
        self.vehicle_marker_frame = "vehicle_marker_frame"
        self.endeffector_marker_frame = "endeffector_marker_frame"

        # Create markers
        self.uv_marker = self.make_UVMS_Dof_Marker(
            name='uv_marker',
            description='interactive marker for controlling vehicle',
            frame_id=self.base_frame,
            robot='uv',
            fixed=False,
            interaction_mode=InteractiveMarkerControl.MOVE_ROTATE_3D,
            initial_pose=self.last_vehicle_marker_pose,
            scale=1.0,
            show_6dof=True,
            ignore_dof=['roll','pitch']
        )

        self.server.insert(self.uv_marker)
        self.server.setCallback(self.uv_marker.name, self.processFeedback)

        desired_q_orientation = [
                self.last_vehicle_marker_pose.orientation.x,
                self.last_vehicle_marker_pose.orientation.y,
                self.last_vehicle_marker_pose.orientation.z,
                self.last_vehicle_marker_pose.orientation.w
            ]

        # Unpack the Euler angles from the returned array.
        roll, pitch, yaw = R.from_quat(desired_q_orientation).as_euler('xyz', degrees=False)
        [self.q0_des, self.q1_des, self.q2_des, self.q3_des, self.q4_des] = self.robots[0].arm.q_command
        self.n_int_est = ca.DM([self.last_vehicle_marker_pose.position.x,
                      self.last_vehicle_marker_pose.position.y,
                      self.last_vehicle_marker_pose.position.z,
                      roll,
                      pitch,
                      yaw,
                      self.q0_des,
                      self.q1_des,
                      self.q2_des, 
                      self.q3_des])
        temp_dm = self.uvms_FK(self.n_int_est, alpha.base_T0)
        self.last_valid_task_pose = self.dm_to_pose(temp_dm[4])

        self.task_marker = self.make_UVMS_Dof_Marker(
            name='task_marker',
            description='interactive marker for controlling endeffector',
            frame_id=self.vehicle_marker_frame,
            robot='task',
            fixed=False,
            interaction_mode=InteractiveMarkerControl.MOVE_ROTATE_3D,
            initial_pose=self.last_valid_task_pose,
            scale=0.2,
            show_6dof=True,
            ignore_dof=['yaw']
        )

        self.server.insert(self.task_marker)
        self.server.setCallback(self.task_marker.name, self.processFeedback)

        # Add menu control
        menu_control = self.make_menu_control()
        self.uv_marker.controls.append(copy.deepcopy(menu_control))
        self.menu_handler.apply(self.server, self.uv_marker.name)

        self.server.applyChanges()
        self.header = Header()
        self.header.frame_id = self.vehicle_marker_frame

    def timer_callback(self):
        self.header.stamp = self.get_clock().now().to_msg()
        cloud_msg = pc2.create_cloud_xyz32(self.header, self.workspace_pts_list)
        self.taskspace_pc_publisher_.publish(cloud_msg)

        cloud_msg = pc2.create_cloud_xyz32(self.header, self.rov_ellipsoid_cl_pts)
        self.rov_pc_publisher_.publish(cloud_msg)

        self.broadcast_pose(self.last_vehicle_marker_pose, self.base_frame, self.vehicle_marker_frame)

        command_msg = Command()
        command_msg.command_type = self.controllers
        command_msg.acceleration.data = []
        command_msg.twist.data = []
        command_msg.pose.data = []

        for k, robot in enumerate(self.robots):
            state = robot.get_state()
            if state['status'] == 'active':
                command_msg.acceleration.data.extend(robot.body_acc_command + robot.arm.ddq_command)
                command_msg.twist.data.extend(robot.body_vel_command + robot.arm.dq_command)
                robot.publish_robot_path()

                if self.execute_plan and (k == self.selected_robot_index) and (self.last_vehicle_marker_pose is not None):
                    planned = self.last_vehicle_marker_pose
                    x_nwu = planned.position.x
                    y_nwu = planned.position.y
                    z_nwu = planned.position.z
                    roll_nwu, pitch_nwu, yaw_nwu = robot.quaternion_to_euler(planned.orientation)

                    x_ned = x_nwu
                    y_ned = -y_nwu
                    z_ned = -z_nwu

                    raw_roll_ned = roll_nwu
                    raw_pitch_ned = -pitch_nwu
                    raw_yaw_ned = -yaw_nwu

                    curr_roll, curr_pitch, curr_yaw = state['pose'][3:6]

                    target_roll = robot.normalize_angle(raw_roll_ned, curr_roll)
                    target_pitch = robot.normalize_angle(raw_pitch_ned, curr_pitch)
                    target_yaw = robot.normalize_angle(raw_yaw_ned, curr_yaw)

                    robot.pose_command = [x_ned, y_ned, z_ned,target_roll, target_pitch, target_yaw]
                    robot.arm.q_command = [self.q0_des, self.q1_des, self.q2_des, self.q3_des, self.q4_des]

                    self.execute_plan = robot.set_robot_command_status()
                    command_msg.pose.data.extend(robot.pose_command + robot.arm.q_command)

                else:
                    robot.pose_command = state['pose']
                    robot.arm.q_command = state['q'] + [0.0]
                    command_msg.pose.data.extend(robot.pose_command + robot.arm.q_command)
            robot.write_data_to_file() 

        self.uvms_publisher_.publish(command_msg)

    def processFeedback(self, feedback):
        # For uv_marker
        if feedback.marker_name == "uv_marker":
            if feedback.pose:
                self.last_vehicle_marker_pose = feedback.pose
                if feedback.pose.position.z > 0.0:
                    feedback.pose.position.z = 0.0
                    self.server.setPose(feedback.marker_name, feedback.pose)
                    self.server.applyChanges()
            if feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
                if feedback.menu_entry_id == 1: 
                    if self.selected_robot_index is not None and self.last_vehicle_marker_pose is not None:
                        self.execute_plan = True
                        self.get_logger().info(
                            f"Execute clicked: plan will be applied to robot {self.robots_prefix[self.selected_robot_index]}."
                        )
                    else:
                        self.get_logger().warn("Execute clicked but no robot was selected or no planned pose available.")
                else:
                    robot_index = feedback.menu_entry_id - 3
                    if 0 <= robot_index < len(self.robots_prefix):
                        self.selected_robot_index = robot_index
                        self.get_logger().info(f"Robot {self.robots_prefix[robot_index]} selected for planning.")
            elif feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
                pass

        elif feedback.marker_name == "task_marker" and feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            task_point = np.array([feedback.pose.position.x,
                                feedback.pose.position.y,
                                feedback.pose.position.z])
            if self.is_point_valid(task_point):
                self.last_valid_task_pose = feedback.pose
                relative_pose = self.get_relative_pose(self.arm_base_pose, self.last_valid_task_pose)
                self.q0_des, self.q1_des, self.q2_des = self.robots[self.selected_robot_index].arm.ik_solver([
                    relative_pose.position.x, relative_pose.position.y, relative_pose.position.z
                ])
                self.get_logger().debug(
                    f"Task marker updated with IK: {self.q0_des, self.q1_des, self.q2_des, self.q3_des}"
                )
            else:
                # The task marker is at the boundary; compute the displacement since the last valid pose.
                dx = feedback.pose.position.x - self.last_valid_task_pose.position.x
                dy = feedback.pose.position.y - self.last_valid_task_pose.position.y
                dz = feedback.pose.position.z - self.last_valid_task_pose.position.z

                # Shift the uv_marker by this delta so that the task marker remains at the boundary.
                self.last_vehicle_marker_pose.position.x += dx
                self.last_vehicle_marker_pose.position.y += dy
                self.last_vehicle_marker_pose.position.z += dz

                # Update the uv_marker pose on the server.
                self.server.setPose("uv_marker", self.last_vehicle_marker_pose)
                self.server.applyChanges()

                # Reset the task marker back to the last valid pose (i.e. at the boundary).
                self.server.setPose("task_marker", self.last_valid_task_pose)
                self.server.applyChanges()


    def makeBox(self, fixed, scale, marker_type, initial_pose):
        marker = Marker()
        marker.type = marker_type
        marker.pose = initial_pose
        marker.scale.x = scale * 0.25
        marker.scale.y = scale * 0.25
        marker.scale.z = scale * 0.25

        if fixed:
            marker.color.r = 1.0 
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
        else:
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.color.a = 1.0
        return marker

    def makeBoxControl(self, msg, fixed, interaction_mode, marker_type,
                       scale=1.0, show_6dof=False, initial_pose=Pose(), ignore_dof=[]):
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(self.makeBox(fixed, scale, marker_type, initial_pose))
        control.interaction_mode = interaction_mode
        msg.controls.append(control)

        if show_6dof:
            if 'roll' not in ignore_dof:
                control = InteractiveMarkerControl()
                control.orientation.w = 1.0
                control.orientation.x = 1.0
                control.orientation.y = 0.0
                control.orientation.z = 0.0
                control.name = "roll"
                control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
                if fixed:
                    control.orientation_mode = InteractiveMarkerControl.FIXED
                msg.controls.append(control)

            if 'surge' not in ignore_dof:
                control = InteractiveMarkerControl()
                control.orientation.w = 1.0
                control.orientation.x = 1.0
                control.orientation.y = 0.0
                control.orientation.z = 0.0
                control.name = "surge"
                control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
                if fixed:
                    control.orientation_mode = InteractiveMarkerControl.FIXED
                msg.controls.append(control)

            if 'yaw' not in ignore_dof:
                control = InteractiveMarkerControl()
                control.orientation.w = 1.0
                control.orientation.x = 0.0
                control.orientation.y = 1.0
                control.orientation.z = 0.0
                control.name = "yaw"
                control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
                if fixed:
                    control.orientation_mode = InteractiveMarkerControl.FIXED
                msg.controls.append(control)

            if 'heave' not in ignore_dof:
                control = InteractiveMarkerControl()
                control.orientation.w = 1.0
                control.orientation.x = 0.0
                control.orientation.y = 1.0
                control.orientation.z = 0.0
                control.name = "heave"
                control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
                if fixed:
                    control.orientation_mode = InteractiveMarkerControl.FIXED
                msg.controls.append(control)

            if 'pitch' not in ignore_dof:
                control = InteractiveMarkerControl()
                control.orientation.w = 1.0
                control.orientation.x = 0.0
                control.orientation.y = 0.0
                control.orientation.z = 1.0
                control.name = "pitch"
                control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
                if fixed:
                    control.orientation_mode = InteractiveMarkerControl.FIXED
                msg.controls.append(control)

            if 'sway' not in ignore_dof:
                control = InteractiveMarkerControl()
                control.orientation.w = 1.0
                control.orientation.x = 0.0
                control.orientation.y = 0.0
                control.orientation.z = 1.0
                control.name = "sway"
                control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
                if fixed:
                    control.orientation_mode = InteractiveMarkerControl.FIXED
                msg.controls.append(control)

        return control

    def make_UVMS_Dof_Marker(self, name, description, frame_id, robot, fixed,
                            interaction_mode, initial_pose, scale,
                            show_6dof=False, ignore_dof=[]):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = frame_id
        int_marker.pose = initial_pose
        int_marker.scale = scale
        int_marker.name = name
        int_marker.description = description
        marker_type = Marker.CUBE
        if robot == 'task':
            marker_type = Marker.SPHERE
        self.makeBoxControl(int_marker, fixed, interaction_mode, marker_type,
                            int_marker.scale, show_6dof, Pose(), ignore_dof)
        if robot == 'uv':
            self.makeBoxControl(int_marker, True, InteractiveMarkerControl.NONE, Marker.CUBE, 0.2, False, self.arm_base_pose, ignore_dof)
        return int_marker

    def make_menu_control(self):
        menu_control = InteractiveMarkerControl()
        menu_control.interaction_mode = InteractiveMarkerControl.MENU
        menu_control.name = "robots_control_menu"
        menu_control.description = "target"
        menu_control.always_visible = True
        return menu_control

    def is_point_valid(self, point):
        """
        Returns True if 'point' is in the workspace hull but *not* in the vehicle hull.
        Equivalently, we want:  point ∈ (Workspace \ Vehicle) = Workspace ∩ (Vehicle)^c
        """
        inside_workspace = np.all(
            np.dot(self.workspace_hull.equations[:, :-1], point) + self.workspace_hull.equations[:, -1] <= 0
        )
        inside_vehicle = np.all(
            np.dot(self.vehicle_body_hull.equations[:, :-1], point) + self.vehicle_body_hull.equations[:, -1] <= 0
        )
        # accept the point if it is inside the workspace and *not* inside the vehicle hull.
        return inside_workspace and not inside_vehicle


    def generate_rov_ellipsoid(self, a=0.5, b=0.3, c=0.2, num_points=10000):
        points = []
        while len(points) < num_points:
            pt = np.random.uniform(-1, 1, 3)
            if (pt[0]/a)**2 + (pt[1]/b)**2 + (pt[2]/c)**2 <= 1:
                points.append(pt)
        return points

    def broadcast_pose(self, pose, parent_frame, child_frame):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        t.transform.rotation = pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def dm_to_pose(self, dm):
        pose = Pose()
        pose.position.x = float(dm[0])
        pose.position.y = float(dm[1])
        pose.position.z = float(dm[2])
        roll = float(dm[3])
        pitch = float(dm[4])
        yaw = float(dm[5])
        q = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose


    def pose_to_homogeneous(self, pose):
        """Convert a geometry_msgs/Pose into a 4x4 homogeneous transformation matrix."""
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        trans = [pose.position.x, pose.position.y, pose.position.z]
        mat = quaternion_matrix(quat)
        mat[0:3, 3] = trans
        return mat

    def homogeneous_to_pose(self, mat):
        """Convert a 4x4 homogeneous transformation matrix into a geometry_msgs/Pose."""
        pose = Pose()
        pose.position.x = mat[0, 3]
        pose.position.y = mat[1, 3]
        pose.position.z = mat[2, 3]
        quat = quaternion_from_matrix(mat)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
        return pose

    def get_relative_pose(self, marker_pose, endeffector_pose):
        """
        Compute the relative pose of the endeffector with respect to the marker.
        marker_pose and endeffector_pose should be geometry_msgs/Pose.
        Returns a Pose representing the endeffector pose in the marker's frame.
        """
        T_marker = self.pose_to_homogeneous(marker_pose)
        T_ee = self.pose_to_homogeneous(endeffector_pose)
        T_rel = np.dot(np.linalg.inv(T_marker), T_ee)
        return self.homogeneous_to_pose(T_rel)

def main(args=None):
    rclpy.init(args=args)
    node = BasicControlsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
