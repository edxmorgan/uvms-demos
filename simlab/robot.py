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

import numpy as np
from typing import Dict
from control_msgs.msg import DynamicJointState
from scipy.spatial.transform import Rotation as R
import ament_index_python
import os
import casadi as ca
from nav_msgs.msg import Path
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSHistoryPolicy
import csv
from datetime import datetime
import copy
from std_msgs.msg import Float32
from pyPS4Controller.controller import Controller
import threading
import glob
from typing import Sequence, Dict
from control_msgs.msg import DynamicInterfaceGroupValues
from std_msgs.msg import Float64MultiArray
from controller_msg import FullRobotMsg
from estimate import SerialLinksDynamicEstimator
from estimate import MarineVehicleEstimator
from estimate import EWUncertainty, EWErrorMetrics
from controllers import LowLevelController

from collections import deque

class Ring:
    def __init__(self, maxlen):
        self.buf = deque(maxlen=maxlen)

    def append(self, x):
        was_full = len(self.buf) == self.buf.maxlen
        self.buf.append(x)
        return was_full

    def is_full(self):
        return len(self.buf) == self.buf.maxlen

    def fill_ratio(self):
        return len(self.buf) / self.buf.maxlen

    def as_vstack(self):
        return np.vstack(list(self.buf))

    def __len__(self):
        return len(self.buf)
    
class PS4Controller(Controller):
    def __init__(self, ros_node, prefix, **kwargs):
        super().__init__(**kwargs)
        self.ros_node = ros_node
        
        # mode flag: False = joint control, True = light & mount control
        self.options_mode = False

        # running values
        self.light_value = 0.0
        self.mount_value = 0.0
        
        sim_gain = 5.0
        real_gain = 5.0
        self.gain = sim_gain
        self.gain = real_gain if 'real' in prefix else sim_gain

        # Gains for different DOFs
        self.max_torque = self.gain * 2.0             # for surge/sway
        self.heave_max_torque = self.gain * 5.0         # for heave (L2/R2)
        self.orient_max_torque = self.gain * 0.8        # for roll, pitch,
        self.yaw_max_torque = self.gain * 0.4 # for yaw

        # # Create a lock specifically for updating gain values.
        # self.gain_lock = threading.Lock()
        # # Start a thread to update the gain every few seconds.
        # # gain randomization for good data collection
        # self.gain_thread = threading.Thread(target=self._update_gain, daemon=True)
        # self.gain_thread.start()

    # def _update_gain(self):
    #     """Randomize the gain value every few seconds and update the torque parameters."""
    #     while True:
    #         # For example, choose a new gain between 4 and 6.
    #         new_gain = random.uniform(3, 8)
    #         with self.gain_lock:
    #             self.gain = new_gain

    #             self.max_torque = self.gain * 2.0
    #             self.heave_max_torque = self.gain * 3.0
    #             self.orient_max_torque = self.gain * 0.7
    #             self.yaw_max_torque = self.gain * 0.2
    #         # Keep this gain for 8 seconds.
    #         time.sleep(8)

   # —— Options toggles between modes ——    
    def on_options_press(self):
        self.options_mode = not self.options_mode
        # if returning to joint mode, zero out any light/mount commands
        if not self.options_mode:
            self.ros_node.light_publisher_.publish(Float32(data=0.0))
            self.ros_node.mountPitch_publisher_.publish(Float32(data=0.0))

    # —— Heave (unchanged) ——    
    def on_L2_press(self, value):
        scaled = self.heave_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = -scaled

    def on_L2_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = 0.0

    def on_R2_press(self, value):
        scaled = self.heave_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = scaled

    def on_R2_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = 0.0

    # —— Surge & Sway (unchanged) ——    
    def on_L3_up(self, value):
        scaled = self.max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_surge = -scaled

    def on_L3_down(self, value):
        scaled = self.max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_surge = -scaled

    def on_L3_right(self, value):
        scaled = self.max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_sway = scaled

    def on_L3_left(self, value):
        scaled = self.max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_sway = scaled

    def on_L3_x_at_rest(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_sway = 0.0

    def on_L3_y_at_rest(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_surge = 0.0

    # —— Roll control (unchanged) ——    
    def on_R1_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_roll =  self.orient_max_torque

    def on_L1_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_roll = -self.orient_max_torque

    def on_R1_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_roll = 0.0

    def on_L1_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_roll = 0.0

    # —— Pitch & Yaw (unchanged) ——    
    def on_R3_up(self, value):
        scaled = self.orient_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_pitch = scaled

    def on_R3_down(self, value):
        scaled = self.orient_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_pitch = scaled

    def on_R3_left(self, value):
        scaled = self.yaw_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_yaw = scaled

    def on_R3_right(self, value):
        scaled = self.yaw_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_yaw = scaled

    def on_R3_x_at_rest(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_yaw = 0.0

    def on_R3_y_at_rest(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_pitch = 0.0

    # —— D‑pad Left/Right ——    
    def on_left_arrow_press(self):
        if self.options_mode:
            self.ros_node.light_publisher_.publish(Float32(data=-10.0))
        else:
            with self.ros_node.controller_lock:
                self.ros_node.jointe = -3.0

    def on_right_arrow_press(self):
        if self.options_mode:
            self.ros_node.light_publisher_.publish(Float32(data=10.0))
        else:
            with self.ros_node.controller_lock:
                self.ros_node.jointe = 3.0

    def on_left_right_arrow_release(self):
        if self.options_mode:
            self.ros_node.light_publisher_.publish(Float32(data=0.0))
        else:
            with self.ros_node.controller_lock:
                self.ros_node.jointe = 0.0

    # —— D‑pad Up/Down ——    
    def on_up_arrow_press(self):
        if self.options_mode:
            self.ros_node.mountPitch_publisher_.publish(Float32(data=-10.0))
        else:
            with self.ros_node.controller_lock:
                self.ros_node.jointd = 2.0

    def on_down_arrow_press(self):
        if self.options_mode:
            self.ros_node.mountPitch_publisher_.publish(Float32(data=10.0))
        else:
            with self.ros_node.controller_lock:
                self.ros_node.jointd = -2.0

    def on_up_down_arrow_release(self):
        if self.options_mode:
            self.ros_node.mountPitch_publisher_.publish(Float32(data=0.0))
        else:
            with self.ros_node.controller_lock:
                self.ros_node.jointd = 0.0

    # —— Manipulator buttons (unchanged) ——    
    def on_triangle_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointc = 2.0

    def on_triangle_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointc = 0.0

    def on_x_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointc = -2.0

    def on_x_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointc = 0.0

    def on_square_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointb = 1.0

    def on_square_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointb = 0.0

    def on_circle_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointb = -1.0

    def on_circle_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointb = 0.0

    def on_R3_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointa = 1.0

    def on_R3_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointa = 0.0

    def on_L3_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointa = -1.0

    def on_L3_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointa = 0.0

class Base:
    def get_interface_value(self, msg: DynamicJointState, dof_names: list, interface_names: list):
        names = msg.joint_names
        return [
            msg.interface_values[names.index(joint_name)].values[
                msg.interface_values[names.index(joint_name)].interface_names.index(interface_name)
            ]
            for joint_name, interface_name in zip(dof_names, interface_names)
        ]

class Axis_Interface_names:
    manipulator_position = 'position'
    manipulator_filtered_position = 'filtered_position'
    manipulator_velocity = 'velocity'
    manipulator_filtered_velocity = 'filtered_velocity'
    manipulator_estimation_acceleration = "estimated_acceleration"
    manipulator_effort = 'effort'
    
    floating_base_x = 'position.x'
    floating_base_y = 'position.y'
    floating_base_z = 'position.z'

    floating_base_roll = 'roll'
    floating_base_pitch = 'pitch'
    floating_base_yaw = 'yaw'

    floating_dx = 'velocity.x'
    floating_dy = 'velocity.y'
    floating_dz = 'velocity.z'

    floating_roll_vel = 'angular_velocity.x'
    floating_pitch_vel = 'angular_velocity.y'
    floating_yaw_vel = 'angular_velocity.z'

    floating_force_x = 'force.x'
    floating_force_y = 'force.y'
    floating_force_z = 'force.z'
    floating_torque_x = 'torque.x'
    floating_torque_y = 'torque.y'
    floating_torque_z = 'torque.z'

    sim_time = 'sim_time'
    sim_period = 'sim_period'

    imu_roll = "imu_roll"
    imu_pitch = "imu_pitch"
    imu_yaw = "imu_yaw"

    imu_roll_unwrap = "imu_roll_unwrap"
    imu_pitch_unwrap = "imu_pitch_unwrap"
    imu_yaw_unwrap = "imu_yaw_unwrap"

    imu_q_w = "imu_orientation_w"
    imu_q_x = "imu_orientation_x"
    imu_q_y = "imu_orientation_y"
    imu_q_z = "imu_orientation_z"

    imu_wx = "imu_angular_vel_x"
    imu_wy = "imu_angular_vel_y"
    imu_wz = "imu_angular_vel_z"

    imu_ax = "imu_linear_acceleration_x"
    imu_ay = "imu_linear_acceleration_y"
    imu_az = "imu_linear_acceleration_z"

    depth_pressure2 = "depth_from_pressure2"

    dvl_roll = "dvl_gyro_roll"
    dvl_pitch = "dvl_gyro_pitch"
    dvl_yaw = "dvl_gyro_yaw"

    dvl_speed_x = "dvl_speed_x"
    dvl_speed_y = "dvl_speed_y"
    dvl_speed_z = "dvl_speed_z"
    
class Manipulator(Base):
    def __init__(self, node: Node, n_joint, prefix):
        self.node = node
        self.n_joint = n_joint
        self.q = [0]*n_joint
        self.dq = [0]*n_joint
        self.sim_period = [0.0]
        self.effort = [0]*n_joint
        self.alpha_axis_a = f'{prefix}_axis_a'
        self.alpha_axis_b = f'{prefix}_axis_b'
        self.alpha_axis_c = f'{prefix}_axis_c'
        self.alpha_axis_d = f'{prefix}_axis_d'
        self.alpha_axis_e = f'{prefix}_axis_e'

        self.joints = [self.alpha_axis_e, self.alpha_axis_d, self.alpha_axis_c, self.alpha_axis_b]


        self.a0 = 20e-3
        self.a1 = np.sqrt(40**2 + (154.3)**2)*(10**-3)
        self.a2 = 20e-3
        self.a3 = 0
        self.a4 = 0
        
        self.d0 = 46.2e-3
        self.d1 = 0
        self.d2 = 0
        self.d3 = -180e-3
        self.d4 = 0
        
        self.l1 = self.a1
        self.l2 = np.sqrt(self.a2**2 + self.d3**2)

        self.q_command = [3.1, 0.7, 0.4, 2.1]
        self.dq_command = [0.0, 0.0, 0.0, 0.0]
        self.ddq_command = [0.0, 0.0, 0.0, 0.0]

        theta_prev = np.array([ 1.94000000e-01,  1.89269857e-02,  1.29958327e-02,  1.92826734e-02,
                1.70332778e-02,  1.85707491e-02,  3.38742861e-03, -1.45044739e-03,
                1.07407291e-03, -4.16794673e-04,  9.60009605e-01,  8.77084974e-01,
                4.29000000e-01, -6.82922902e-03,  2.91162110e-02,  2.80386073e-02,
                3.99924053e-02,  4.98000396e-02,  2.39434153e-02, -8.72099542e-03,
                2.28957850e-02,  1.01441651e-02,  2.10688822e+00,  8.13858996e-02,
                1.15000000e-01,  6.58255578e-03,  1.14735347e-02,  9.48787928e-03,
                5.85331737e-02,  1.42889470e-01,  1.94501826e-01, -8.72523666e-02,
                1.87521719e-02,  1.10696098e-02,  8.37994156e-01,  2.92171031e-02,
                3.33000000e-01,  2.96542838e-03,  1.34841459e-02,  8.47960624e-03,
                2.80343236e-02,  2.27215753e-02,  1.58212231e-02, -4.56922163e-03,
                2.33185226e-03, -6.25990223e-03,  3.37566423e-01,  5.66586894e-02])


        fixed_blocks = {
            3: np.array([0.333, 0.00812305, 0.0197926, -0.0332988,
                        0.0371459, 0.0557153, 0.0318996,
                        -0.006127, -0.0168061, -0.00987941,
                        0.28743, 0.346754]),
            2: np.array([0.115, -0.00574854, 0.0114983, 0.0115,
                         0.279322, 0.00154219, 0.278402, 
                         0.00314941, 0.000618867, -0.00588411,
                         1.01562, 0.0332583]),
            1: np.array([0.429, -0.0418153, 0.0255531, -0.000997896,
                         0.833981, 1.45351, 0.915332, 
                         0.0362388, -0.723757, 0.0295013,
                         1.3938, 0.43046]),
        }

# Final parameters for joint 1 at t=24.5001, ros_time=1757795267.349725:
#   m1: 0.429  CI95 [0.428996, 0.429004]
#   m*rcx1: -0.0418153  CI95 [-0.0425308, -0.0410997]
#   m*rcy1: 0.0255531  CI95 [0.0228927, 0.0282135]
#   m*rcz1: -0.000997896  CI95 [-0.00241596, 0.000420166]
#   Ixx1: 0.833981  CI95 [0.809236, 0.858727]
#   Iyy1: 1.45351  CI95 [1.42534, 1.48167]
#   Izz1: 0.915332  CI95 [0.89469, 0.935974]
#   Ixy1: 0.0362388  CI95 [0.0353139, 0.0371636]
#   Ixz1: -0.723757  CI95 [-0.738847, -0.708667]
#   Iyz1: 0.0295013  CI95 [0.0284411, 0.0305615]
#   fv1: 1.3938  CI95 [1.38139, 1.4062]
#   fs1: 0.43046  CI95 [0.422955, 0.437966]

# Final parameters for joint 2 at t=29.5999, ros_time=1757793979.095591:
#   m2: 0.115  CI95 [0.114998, 0.115002]
#   m*rcx2: -0.00574854  CI95 [-0.00618164, -0.00531544]
#   m*rcy2: 0.0114983  CI95 [0.0114961, 0.0115006]
#   m*rcz2: 0.0115  CI95 [0.0114371, 0.0115629]
#   Ixx2: 0.279322  CI95 [0.274057, 0.284587]
#   Iyy2: 0.00154219  CI95 [-0.00319455, 0.00627893]
#   Izz2: 0.278402  CI95 [0.271863, 0.284941]
#   Ixy2: 0.00314941  CI95 [-0.00236994, 0.00866876]
#   Ixz2: 0.000618867  CI95 [-9.05146e-05, 0.00132825]
#   Iyz2: -0.00588411  CI95 [-0.00654028, -0.00522795]
#   fv2: 1.01562  CI95 [1.00944, 1.02179]
#   fs2: 0.0332583  CI95 [0.0316651, 0.0348515]

# Final parameters for joint 3 at t=19.3499, ros_time=1757792872.675372:
#   m3: 0.333  CI95 [0.332997, 0.333003]
#   m*rcx3: 0.00812305  CI95 [0.00755143, 0.00869468]
#   m*rcy3: 0.0197926  CI95 [0.019217, 0.0203682]
#   m*rcz3: -0.0332988  CI95 [-0.0387932, -0.0278043]
#   Ixx3: 0.0371459  CI95 [0.0366409, 0.037651]
#   Iyy3: 0.0557153  CI95 [0.0526413, 0.0587893]
#   Izz3: 0.0318996  CI95 [0.0301893, 0.0336099]
#   Ixy3: -0.006127  CI95 [-0.0064659, -0.0057881]
#   Ixz3: -0.0168061  CI95 [-0.0189188, -0.0146934]
#   Iyz3: -0.00987941  CI95 [-0.0103114, -0.00944738]
#   fv3: 0.28743  CI95 [0.283938, 0.290923]
#   fs3: 0.346754  CI95 [0.316089, 0.37742]

        # Estimator and ring buffers
        self.horizon_steps = 840 # number of time steps to stack
        self.n_params = 48
        assert self.n_params == theta_prev.shape[0], "manipulator theta_prev == manipulator n_params"
        self.manipulator_estimator = SerialLinksDynamicEstimator(dof=4,
                                                                  horizon_steps=self.horizon_steps,
                                                                  n_params=self.n_params,
                                                                    theta_prev=theta_prev, 
                                                                    fixed_blocks=fixed_blocks)

        self.manip_uncert = EWUncertainty(dim=self.n_params, alpha=0.05, eps=1e-8, jitter=1e-12)
        # Online error metrics for 4 joint torques
        self.manip_err_metrics = EWErrorMetrics(n_outputs=4, alpha=0.5)

        # Each time step we will append a 4 x p block for each active robot
        # We stack across time, not across robots, so we sum all robots into one step
        # Y_rows_buffer holds rows shaped (4, p)
        # tau_rows_buffer holds rows shaped (4, 1)
        self.Y_rows_buffer = Ring(maxlen=self.horizon_steps)
        self.tau_rows_buffer = Ring(maxlen=self.horizon_steps)
        self.has_intialize_manipulator_estimator = False


    def update_state(self, msg: DynamicJointState):
        self.q = self.get_interface_value(
            msg,
            self.joints,
            [Axis_Interface_names.manipulator_position] * 4
        )
        self.filtered_q = self.get_interface_value(
            msg,
            self.joints,
            [Axis_Interface_names.manipulator_filtered_position] * 4
        )
        self.dq = self.get_interface_value(
            msg,
            self.joints,
            [Axis_Interface_names.manipulator_velocity] * 4
        )
        self.filtered_dq = self.get_interface_value(
            msg,
            self.joints,
            [Axis_Interface_names.manipulator_filtered_velocity] * 4
        )
        self.estimated_ddq = self.get_interface_value(
            msg,
            self.joints,
            [Axis_Interface_names.manipulator_estimation_acceleration] * 4
        )
        self.effort = self.get_interface_value(
            msg,
            self.joints,
            [Axis_Interface_names.manipulator_effort] * 4
        )


        self.sim_period = self.get_interface_value(
            msg,
            [self.alpha_axis_e],
            [Axis_Interface_names.sim_period]
        )
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            'arm_effort':self.effort,
            'q':self.q,
            'dq':self.dq,
            'dt':self.sim_period[0]
        }

    def ik_solver(self, target_position, pose="overarm"):
        x = target_position[0]
        y = target_position[1]
        z = target_position[2]

        thet0, thet1, thet2 = float("nan"), float("nan"), float("nan")
        try:
            R = np.sqrt(x**2 + y**2)
            l1 = self.a1
            l2 = np.sqrt(self.a2**2 + self.d3**2)

            if pose == 'underarm':
                thet0 = np.arctan2(y, x) + np.pi
                l3 = np.sqrt((R - self.a0)**2 + (z - self.d0)**2)
                
                # Compute argument for arccos and arcsin safely by clipping
                arg1 = (l1**2 + l2**2 - l3**2) / (2 * l1 * l2)
                term1 = np.arccos(np.clip(arg1, -1, 1))
                
                term2 = np.arcsin(np.clip((2 * self.a2) / l1, -1, 1))
                term3 = np.arcsin(np.clip(self.a2 / l2, -1, 1))
                
                thet2 = term1 - term2 - term3

                arg2 = (l1**2 + l3**2 - l2**2) / (2 * l1 * l3)
                term4 = np.arccos(np.clip(arg2, -1, 1))
                
                thet1 = (np.pi / 2) + np.arctan2(z - self.d0, R - self.a0) - term4 - term2

            elif pose == 'overarm':
                thet0 = np.arctan2(y, x)
                l3 = np.sqrt((R + self.a0)**2 + (z - self.d0)**2)
                
                arg1 = (l1**2 + l2**2 - l3**2) / (2 * l1 * l2)
                term1 = np.arccos(np.clip(arg1, -1, 1))
                
                term2 = np.arcsin(np.clip((2 * self.a2) / l1, -1, 1))
                term3 = np.arcsin(np.clip(self.a2 / l2, -1, 1))
                
                thet2 = term1 - term2 - term3

                arg2 = (l1**2 + l3**2 - l2**2) / (2 * l1 * l3)
                term4 = np.arccos(np.clip(arg2, -1, 1))
                
                thet1 = ((3 * np.pi) / 2) - np.arctan2(z - self.d0, R + self.a0) - term4 - term2

        except Exception as e:
            self.node.get_logger().error(f"An error occurred: {e}")
        return thet0, thet1, thet2


class Robot(Base):
    def __init__(self, node: Node,
                  k_robot, 
                  n_joint, 
                  prefix, 
                  initial_ref_pos, 
                  record=False,  
                  controller='pid'):
        self.subscription = node.create_subscription(
                DynamicJointState,
                'dynamic_joint_states',
                self.listener_callback,
                10
            )
        self.k_robot = k_robot
        self.robot_name = f'uvms {prefix}: {k_robot}'
        self.subscription  # prevent unused variable warning
    
        package_share_directory = ament_index_python.get_package_share_directory(
                'simlab')
        fk_path = os.path.join(package_share_directory, 'manipulator/fk_eval.casadi')

        vehicle_J_path = os.path.join(package_share_directory, 'vehicle/J_uv.casadi')

        vehicle_regressor_path = os.path.join(package_share_directory, 'vehicle/vehicle_id_Y.casadi')
        manipulator_regressor_path = os.path.join(package_share_directory, 'manipulator/arm_id_Y.casadi')

        self.fk_eval = ca.Function.load(fk_path) # differential inverse kinematics
        self.vehicle_J = ca.Function.load(vehicle_J_path)
        self.vehicle_Y = ca.Function.load(vehicle_regressor_path)
        self.manipulator_Y = ca.Function.load(manipulator_regressor_path)


        m_X_du, m_Y_dv, m_Z_dw = 50, 50, 20
        I_x_K_dp, I_y_M_dq, I_z_N_dr = 0.5, 0.5, 0.5
        vehicle_initial_thet = np.array([m_X_du, m_Y_dv, m_Z_dw,
                        0.0, 0.0, 0.0, 0.0, 
                        I_x_K_dp, I_y_M_dq, I_z_N_dr, 
                        0, 0, 0, 0 , 0, 
                        -80,-80,-150,-5,-5,-5, 
                        0,0,0,0,0,0])

        self.vehicle_n_params = 27
        assert self.vehicle_n_params == vehicle_initial_thet.shape[0], "vehicle theta_prev == vehicle n_params"
        self.vehicle_n_horizon_steps = 840
        self.v_c = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # current velocity 
        self.vehicle_estimator =MarineVehicleEstimator(6, self.vehicle_n_params, self.vehicle_n_horizon_steps, vehicle_initial_thet)

        # Online error metrics for 6 dof
        self.vehicle_err_metrics = EWErrorMetrics(n_outputs=6, alpha=0.5)
        # Each time step we will append a 6 x p block for each active robot
        # We stack across time, not across robots, so we sum all robots into one step
        # Y_rows_buffer holds rows shaped (6, p)
        # tau_rows_buffer holds rows shaped (6, 1)
        self.vehicle_Y_rows_buffer = Ring(maxlen=self.vehicle_n_horizon_steps)
        self.vehicle_tau_rows_buffer = Ring(maxlen=self.vehicle_n_horizon_steps)
        self.has_intialize_vehicle_estimator = False
        self.vehicle_uncert = EWUncertainty(dim=self.vehicle_n_params, alpha=0.05, eps=1e-8, jitter=1e-12)

        self.node = node
        self.sensors = [
            Axis_Interface_names.imu_roll,
            Axis_Interface_names.imu_pitch,
            Axis_Interface_names.imu_yaw,
            Axis_Interface_names.imu_roll_unwrap,
            Axis_Interface_names.imu_pitch_unwrap,
            Axis_Interface_names.imu_yaw_unwrap,
            Axis_Interface_names.imu_q_w,
            Axis_Interface_names.imu_q_x,
            Axis_Interface_names.imu_q_y,
            Axis_Interface_names.imu_q_z,
            Axis_Interface_names.imu_wx,
            Axis_Interface_names.imu_wy,
            Axis_Interface_names.imu_wz,
            Axis_Interface_names.imu_ax,
            Axis_Interface_names.imu_ay,
            Axis_Interface_names.imu_az,
            Axis_Interface_names.depth_pressure2,
            Axis_Interface_names.dvl_roll,
            Axis_Interface_names.dvl_pitch,
            Axis_Interface_names.dvl_yaw,
            Axis_Interface_names.dvl_speed_x,
            Axis_Interface_names.dvl_speed_y,
            Axis_Interface_names.dvl_speed_z
            ]
        self.prediction_interfaces = [
            "position.x", "position.y", "position.z", "roll", "pitch", "yaw",
            "orientation.w", "orientation.x", "orientation.y", "orientation.z", 
            "velocity.x", "velocity.y", "velocity.z", 
            "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
        ]

        self.state_estimate_interfaces = [
            "position_estimate.x", "position_estimate.y", "position_estimate.z",
            "roll_estimate", "pitch_estimate", "yaw_estimate",
            "orientation_estimate.w", "orientation_estimate.x", "orientation_estimate.y", "orientation_estimate.z",
            "velocity_estimate.x", "velocity_estimate.y", "velocity_estimate.z",
            "angular_velocity_estimate.x", "angular_velocity_estimate.y", "angular_velocity_estimate.z",
            "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z",
            "angular_acceleration.x", "angular_acceleration.y", "angular_acceleration.z",
            "P_x_x", "P_y_y", "P_z_z", "P_roll_roll", "P_pitch_pitch", "P_yaw_yaw",
            "P_u_u", "P_v_v", "P_w_w", "P_p_p", "P_q_q", "P_r_r",
        ]

        self.payload_state_interfaces = ["payload.mass", "payload.Ixx", "payload.Iyy", "payload.Izz"]

        self.n_joint = n_joint
        self.floating_base_IOs = f'{prefix}IOs'
        self.arm_IOs = f'{prefix}_arm_IOs'
        self.arm = Manipulator(node, n_joint, prefix)
        self.ned_pose = [0] * 6
        self.body_vel = [0] * 6
        self.ned_vel = [0] * 6
        self.sensor_reading = [0] * len(self.sensors)
        self.prediction_readings = [0] * len(self.prediction_interfaces)
        self.state_estimate_readings = [0] * len(self.state_estimate_interfaces)
        self.payload_state_readings = [0] * len(self.payload_state_interfaces)
        self.body_forces = [0] * 6
        self.gt_measurements = [0] * 6
        self.prefix = prefix
        self.status = 'inactive'
        self.sim_time = 0.0
        self.start_time = 0.0

        # self.use_controller = controller
        self.pose_command = [0.0]*6
        self.body_vel_command = [0.0]*6
        self.body_acc_command = [0.0]*6
        self.ll_controllers = LowLevelController(self.n_joint)

 
        self.uvms_ll = [-1000, -1000, 0.0, -np.pi/6, -np.pi/6, -1000, 1, 0.01, 0.01, 0.01]
        self.uvms_ul = [ 1000, 1000, 1000, np.pi/6, np.pi/6, 1000, 5.50, 3.40, 3.40, 5.70]
        self.k0 = [1,1,1 , 1,1,1, 1,1,1,1]
        self.base_pose = [0.190, 0.000, -0.120, np.pi, 0.000, 0.000] #floating base mount
        self.world_pose = [0.0, 0.0, 0, 0, 0, 0]
        self.vec_g = [0, 0, 9.81]

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.path_publisher = self.node.create_publisher(Path, f'/{self.prefix}desiredPath', qos_profile)
        self.trajectory_path_publisher = self.node.create_publisher(Path, f'/{self.prefix}robotPath', qos_profile)
        self.gt_trajectory_path_publisher = self.node.create_publisher(Path, f'/{self.prefix}gtPath', qos_profile)

        self.path_ops_publisher = self.node.create_publisher(Path, f'/{self.prefix}desiredOpsPath', qos_profile)
        self.trajectory_path_ops_publisher = self.node.create_publisher(Path, f'/{self.prefix}robotOpsPath', qos_profile)

        self.mountPitch_publisher_ = self.node.create_publisher(Float32, '/alpha/cameraMountPitch', 10)
        self.light_publisher_ = self.node.create_publisher(Float32, '/alpha/lights', 10)

        self.vehicle_effort_command_publisher = self.node.create_publisher(
            DynamicInterfaceGroupValues,
            f"vehicle_effort_controller_{prefix}/commands",
            qos_profile
        )
        self.vehicle_pwm_command_publisher = self.node.create_publisher(
            Float64MultiArray,
            f'vehicle_thrusters_pwm_controller_{prefix}/commands',
            qos_profile
        )    
        self.manipulator_effort_command_publisher = self.node.create_publisher(
            Float64MultiArray,
            f"manipulation_effort_controller_{prefix}/commands",
            qos_profile
        )        
        self.ref_acc = np.zeros(10)
        self.ref_vel = np.zeros(10)
        self.ref_pos = initial_ref_pos

       # Initialize path poses
        self.path_poses = []
        self.traj_path_poses = []
        self.gt_traj_path_poses = []


        self.MAX_POSES = 10000

        # robot trajectory
        self.trajectory_twist = []
        self.trajectory_poses = []

        self.record = record

        self.initiaize_data_writer()

         # Estimation logging placeholders
        self.manipulator_estimation_info = None
        self.vehicle_estimation_info = None

        self.node_name = node.get_name()
        # if self.node_name in ['joystick_controller','']:
        # Search for joystick device in /dev/input
        device_interface = f"/dev/input/js{self.k_robot}"
        self.has_joystick_interface = False
        joystick_device = glob.glob(device_interface)

        if device_interface in joystick_device:
            self.node.get_logger().info(f"Found joystick device: {device_interface}")
            self.start_joystick(device_interface)
            self.has_joystick_interface = True
        else:
            self.node.get_logger().info(f"No joystick device found for robot {self.k_robot}.")

    def set_robot_command_status(self):
        state = self.get_state()
        execute_plan = True
        vehicle_current_pos = np.array(state['pose'])
        vehicle_target_pos = np.array(self.pose_command)
        vehicle_error = np.linalg.norm(vehicle_current_pos - vehicle_target_pos)
        
        manipulator_current_pos = np.array(state['q'])
        manipulator_target_pos = np.array(self.arm.q_command)
        manipulator_error = np.linalg.norm(manipulator_current_pos - manipulator_target_pos)
                        
        # If both vehicle and manipulator are within tolerance, we can finish execution.
        if vehicle_error < 0.01 and manipulator_error < 0.01:
            self.node.get_logger().info("Target reached; resetting execution flag.")
            execute_plan = False
        else:
            # Compute the vehicle (position) error in x, y, z only.
            pos_error = np.linalg.norm(np.array(state['pose'][:3]) - np.array(self.pose_command[0:3]))

            # Define a threshold error at which we start blending.
            pos_blend_threshold = 1.1  # Adjust based on your system's scale

            # Calculate the blend factor.
            # When pos_error >= pos_blend_threshold, blend_factor will be 0 (full velocity_yaw).
            # When pos_error == 0, blend_factor will be 1 (full target_yaw).
            blend_factor = np.clip((pos_blend_threshold - pos_error) / pos_blend_threshold, 0.0, 1.0)

            # Get the velocity-based yaw.
            velocity_yaw = self.orient_towards_velocity()

            # If velocity_yaw is not available, simply use the target yaw.
            if velocity_yaw is None:
                final_yaw = self.pose_command[5]
            else:
                # Blend the yaw values: more weight to target_yaw as the position error decreases.
                final_yaw = (1 - blend_factor) * velocity_yaw + blend_factor * self.pose_command[5]

            self.node.get_logger().debug(f"pos_error: {pos_error:.3f}, blend_factor: {blend_factor:.3f}, final_yaw: {final_yaw:.3f}")

            # Update the yaw in the command message with the blended value.
            self.pose_command[5] = final_yaw
        return execute_plan


    def start_joystick(self, device_interface):
        # Shared variables updated by the PS4 controller callbacks.
        self.controller_lock = threading.Lock()
        self.rov_surge = 0.0      # Left stick horizontal (sway)
        self.rov_sway = 0.0      # Left stick vertical (surge)
        self.rov_z = 0.0      # Heave from triggers
        self.rov_roll = 0.0   # roll
        self.rov_pitch = 0.0  # Right stick vertical (pitch)
        self.rov_yaw = 0.0    # Right stick horizontal (yaw)

        self.jointe = 0.0
        self.jointd = 0.0
        self.jointc = 0.0
        self.jointb = 0.0
        self.jointa = 0.0

        # Instantiate the PS4 controller.
        # If you are not receiving analog stick events, try adjusting the event_format.
        self.ps4_controller = PS4Controller(
            ros_node=self,
            prefix=self.prefix,
            interface=device_interface,
            connecting_using_ds4drv=False,
            event_format="3Bh2b"  # Try "LhBB" if you experience mapping issues.
        )
        # Enable debug mode to print raw event data.
        self.ps4_controller.debug = True

        # Start the PS4 controller listener in a separate (daemon) thread.
        self.controller_thread = threading.Thread(target=self.ps4_controller.listen, daemon=True)
        self.controller_thread.start()

        self.node.get_logger().info(f"PS4 Teleop node initialized for robot {self.k_robot} to be control with js{self.k_robot}.")


    def update_state(self, msg: DynamicJointState):
        self.arm.update_state(msg)
        self.ned_pose = self.get_interface_value(
            msg,
            [self.floating_base_IOs] * 6,
            [
                Axis_Interface_names.floating_base_x,
                Axis_Interface_names.floating_base_y,
                Axis_Interface_names.floating_base_z,
                Axis_Interface_names.floating_base_roll,
                Axis_Interface_names.floating_base_pitch,
                Axis_Interface_names.floating_base_yaw
            ]
        )


        self.body_vel = self.get_interface_value(
            msg,
            [self.floating_base_IOs] * 6,
            [
                Axis_Interface_names.floating_dx,
                Axis_Interface_names.floating_dy,
                Axis_Interface_names.floating_dz,
                Axis_Interface_names.floating_roll_vel,
                Axis_Interface_names.floating_pitch_vel,
                Axis_Interface_names.floating_yaw_vel
            ]
        )

        self.ned_vel = self.to_ned_velocity(self.body_vel, self.ned_pose)
        
        self.sensor_reading = self.get_interface_value(
            msg,
            [self.floating_base_IOs] * len(self.sensors),
            self.sensors
        )

        self.prediction_readings = self.get_interface_value(
            msg,
            [self.floating_base_IOs] * len(self.prediction_interfaces),
            self.prediction_interfaces
        )
        self.state_estimate_readings = self.get_interface_value(
            msg,
            [self.floating_base_IOs] * len(self.state_estimate_interfaces),
            self.state_estimate_interfaces
        )

        self.payload_state_readings = self.get_interface_value(
            msg,
            [self.arm_IOs] * len(self.payload_state_interfaces),
            self.payload_state_interfaces
        )

        self.body_forces = self.get_interface_value(
            msg,
            [self.floating_base_IOs] * 6,
            [
            Axis_Interface_names.floating_force_x,
            Axis_Interface_names.floating_force_y, 
            Axis_Interface_names.floating_force_z,
            Axis_Interface_names.floating_torque_x,
            Axis_Interface_names.floating_torque_y,
            Axis_Interface_names.floating_torque_z
            ]
        )
   
        dynamics_sim_time = self.get_interface_value(msg,[self.floating_base_IOs],[Axis_Interface_names.sim_time])[0]
        if self.status == 'inactive':
            self.start_time = copy.copy(dynamics_sim_time)
            self.status = 'active'
        elif self.status == 'active':
            self.sim_time = dynamics_sim_time - self.start_time

    def get_state(self) -> Dict:
        xq = self.arm.get_state()
        xq['name'] = self.prefix
        xq['pose'] = self.ned_pose
        xq['body_vel'] = self.body_vel
        xq['ned_vel'] = self.ned_vel
        xq['body_forces'] = self.body_forces
        xq['status'] = self.status
        xq['sim_time'] = self.sim_time
        xq['prefix'] = self.prefix
        xq['raw_sensor_readings'] = self.sensor_reading
        # self.node.get_logger().info(f"body forces {xq['raw_sensor_readings']}")
        return xq

    def to_body_velocity(self, ned_vel, pose):
        velocity_body = copy.copy(ned_vel)
        J_UV_REF = self.vehicle_J(pose[3:6])
        velocity_body[:6] = np.linalg.inv(J_UV_REF.full())@ned_vel[:6]
        return velocity_body

    def to_ned_velocity(self, body_vel, pose):
        velocity_ned = copy.copy(body_vel)
        J_UV_REF = self.vehicle_J(pose[3:6])
        velocity_ned[:6] = J_UV_REF.full()@body_vel[:6]
        return velocity_ned
        
    def set_robot_goals(self, desired_ned_vel, desired_ned_pos):
        self.ref_ned_vel = desired_ned_vel
        self.ref_vel = self.to_body_velocity(desired_ned_vel, desired_ned_pos)
        self.ref_pos = desired_ned_pos

        # Accumulate reference trajectory
        self.trajectory_twist.append(self.ref_vel.tolist().copy())  # Append a copy of the reference velocity
        self.trajectory_poses.append(self.ref_pos.copy())

        self.goal = dict()
        self.goal['ref_acc'] = self.ref_acc.tolist()
        self.goal['ref_vel'] = self.trajectory_twist[-1]
        self.goal['ref_pos'] = self.trajectory_poses[-1]

    def get_robot_goals(self, ref_type):
        return self.goal.get(ref_type)

    def publish_reference_path(self):
        # Publish the reference path to RViz
        path_msg = Path()
        path_msg.header.stamp = self.node.get_clock().now().to_msg()
        path_msg.header.frame_id = f"{self.prefix}map"  # Set to robot map frame

        # Create PoseStamped from ref_pos
        pose = PoseStamped()
        pose.header = path_msg.header
        pose.pose.position.x = float(self.ref_pos[0])
        pose.pose.position.y = -float(self.ref_pos[1])
        pose.pose.position.z = -float(self.ref_pos[2])
        pose.pose.orientation.w = 1.0  # No rotation
        pose.pose.orientation.x = 0.0  # No rotation
        pose.pose.orientation.y = 0.0  # No rotation
        pose.pose.orientation.z = 0.0  # No rotation

        # Accumulate poses
        self.path_poses.append(pose)
        path_msg.poses = self.path_poses

        # Limit the number of poses
        if len(self.path_poses) > self.MAX_POSES:
            self.path_poses.pop(0)
        self.path_publisher.publish(path_msg)

    def publish_robot_path(self):
        # Publish the robot trajectory path to RViz
        tra_path_msg = Path()
        tra_path_msg.header.stamp = self.node.get_clock().now().to_msg()
        tra_path_msg.header.frame_id = f"{self.prefix}map"  # Set to your appropriate frame

        # Create PoseStamped from ref_pos
        traj_pose = PoseStamped()
        traj_pose.header = tra_path_msg.header
        traj_pose.pose.position.x = float(self.ned_pose[0])
        traj_pose.pose.position.y = -float(self.ned_pose[1])
        traj_pose.pose.position.z = -float(self.ned_pose[2])
        traj_pose.pose.orientation.w = 1.0  # No rotation

        # Accumulate poses
        self.traj_path_poses.append(traj_pose)
        tra_path_msg.poses = self.traj_path_poses

        self.trajectory_path_publisher.publish(tra_path_msg)

    def publish_gt_path(self):
        gt_info = self.gt_measurements
        # Publish the gt trajectory path to RViz
        gt_tra_path_msg = Path()
        gt_tra_path_msg.header.stamp = self.node.get_clock().now().to_msg()
        gt_tra_path_msg.header.frame_id = f"{self.prefix}map"  # Set to your appropriate frame

        # Create PoseStamped from ref_pos
        gt_traj_pose = PoseStamped()
        gt_traj_pose.header = gt_tra_path_msg.header
        gt_traj_pose.pose.position.x = float(gt_info[0])
        gt_traj_pose.pose.position.y = -float(gt_info[1])
        gt_traj_pose.pose.position.z = -float(gt_info[2])

        r = gt_info[3]
        p = gt_info[4]
        y = gt_info[5]

        # Convert RPY to quaternion using SciPy
        rotation = R.from_euler('xyz', [r, p, y], degrees=False)
        quat = rotation.as_quat()  # SciPy returns [x, y, z, w]

        # Assign quaternion to the pose orientation
        gt_traj_pose.pose.orientation.x = quat[0]
        gt_traj_pose.pose.orientation.y = quat[1]
        gt_traj_pose.pose.orientation.z = quat[2]
        gt_traj_pose.pose.orientation.w = quat[3]

        # Accumulate poses
        self.gt_traj_path_poses.append(gt_traj_pose)
        gt_tra_path_msg.poses = self.gt_traj_path_poses

        self.gt_trajectory_path_publisher.publish(gt_tra_path_msg)

    def orient_towards_velocity(self):
        """
        Orient the robot to face the direction of its current positive velocity.
        This updates the robot's reference orientation based on its body velocity.
        """
        vx = self.ned_vel[0]
        vy = self.ned_vel[1]

        # Compute the magnitude of the horizontal velocity
        horizontal_speed = np.hypot(vx, vy)

        # Threshold to avoid undefined behavior when velocity is near zero
        velocity_threshold = 1e-3

        if horizontal_speed > velocity_threshold:
            desired_yaw = np.arctan2(vy, vx)

            # -- Get the CURRENT yaw from the last pose in trajectory_poses
            current_yaw = self.ned_pose[5]
            # -- Compute the shortest-path yaw
            adjusted_yaw = self.normalize_angle(desired_yaw, current_yaw)

            return adjusted_yaw

            # self.node.get_logger().info(f"Orienting towards velocity:current yaw={current_yaw} radians  desired yaw={desired_yaw} radians adjusted yaw={adjusted_yaw} radians")


    def normalize_angle(self, desired_yaw, current_yaw):
        # Compute the smallest angular difference
        angle_diff = desired_yaw - current_yaw
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to (-π, π)

        # Adjust desired_yaw to ensure the shortest rotation path
        adjusted_desired_yaw = current_yaw + angle_diff

        return adjusted_desired_yaw

    def quaternion_to_euler(self, orientation):
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        r = R.from_quat(quat)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        return roll, pitch, yaw


    def initiaize_data_writer(self):
        if self.record:
            # Create a timestamp string
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create a folder with the timestamp as its name (in the current working directory)
            folder_path = os.path.join(os.getcwd(), timestamp_str)
            os.makedirs(folder_path, exist_ok=True)
            
            # Create a timestamped filename for the CSV
            filename = f"{timestamp_str}_{self.prefix}.csv"
            file_path = os.path.join(folder_path, filename)
            
            # Open the CSV file and prepare to write data
            self.csv_file = open(file_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)

            # Write a header row for clarity
            columns = [
                'timestamp',
                'ros_time',
                'base_x_force', 'base_y_force', 'base_z_force', 'base_x_torque', 'base_y_torque', 'base_z_torque',
                'base_x', 'base_y', 'base_z', 'base_roll', 'base_pitch', 'base_yaw',
                'base_dx', 'base_dy', 'base_dz', 'base_vel_roll', 'base_vel_pitch', 'base_vel_yaw',
                
                'effort_alpha_axis_e', 'effort_alpha_axis_d', 'effort_alpha_axis_c', 'effort_alpha_axis_b',
                'q_alpha_axis_e', 'q_alpha_axis_d', 'q_alpha_axis_c', 'q_alpha_axis_b',
                'dq_alpha_axis_e', 'dq_alpha_axis_d', 'dq_alpha_axis_c', 'dq_alpha_axis_b',

                'imu_roll', 'imu_pitch', 'imu_yaw',
                'imu_roll_unwrap', 'imu_pitch_unwrap', 'imu_yaw_unwrap',
                'imu_q_w', 'imu_q_x', 'imu_q_y', 'imu_q_z',
                'imu_ang_vel_x', 'imu_ang_vel_y','imu_ang_vel_z',
                'imu_linear_acc_x', 'imu_linear_acc_y','imu_linear_acc_z',
                'depth_from_pressure2',
                'dvl_roll', 'dvl_pitch', 'dvl_yaw',
                'dvl_speed_x', 'dvl_speed_y', 'dvl_speed_z',

                'base_x_ref', 'base_y_ref', 'base_z_ref', 'base_roll_ref', 'base_pitch_ref', 'base_yaw_ref',
                'q_alpha_axis_e_ref', 'q_alpha_axis_d_ref', 'q_alpha_axis_c_ref', 'q_alpha_axis_b_ref', 'q_alpha_axis_a_ref',

                "position.x", "position.y", "position.z", "roll", "pitch", "yaw",
                "orientation.w", "orientation.x", "orientation.y", "orientation.z", 
                "velocity.x", "velocity.y", "velocity.z", 
                "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",

                "position_estimate.x", "position_estimate.y", "position_estimate.z",
                "roll_estimate", "pitch_estimate", "yaw_estimate",
                "orientation_estimate.w", "orientation_estimate.x", "orientation_estimate.y", "orientation_estimate.z",
                "velocity_estimate.x", "velocity_estimate.y", "velocity_estimate.z",
                "angular_velocity_estimate.x", "angular_velocity_estimate.y", "angular_velocity_estimate.z",
                "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z",
                "angular_acceleration.x", "angular_acceleration.y", "angular_acceleration.z",
                "P_x_x", "P_y_y", "P_z_z", "P_roll_roll", "P_pitch_pitch", "P_yaw_yaw",
                "P_u_u", "P_v_v", "P_w_w", "P_p_p", "P_q_q", "P_r_r",

                "payload.mass", "payload.Ixx", "payload.Iyy", "payload.Izz"
            ]

            self.csv_writer.writerow(columns)
    
    def write_data_to_file(self, ref=[0,0,0, 0,0,0, 0,0,0,0, 0]):
        if self.record:
            t_ros = float(self.node.get_clock().now().nanoseconds) * 1e-9
            row_data = []
            info = self.get_state()
            
            row_data.extend([info['sim_time']])
            row_data.extend([t_ros])
            
            row_data.extend(info['body_forces'])
            row_data.extend(info['pose'])
            row_data.extend(info['body_vel'])

            row_data.extend(info['arm_effort'])
            row_data.extend(info['q'])
            row_data.extend(info['dq'])
            
            row_data.extend(info['raw_sensor_readings'])
            row_data.extend(ref)
            row_data.extend(self.prediction_readings)
            row_data.extend(self.state_estimate_readings)
            row_data.extend(self.payload_state_readings)

            if all(value == 0 for value in row_data):
                return

            """Write a single row of data to the CSV file."""
            self.csv_writer.writerow(row_data)
            self.csv_file.flush()

    def publish_vehicle_and_arm(
        self,
        wrench_body_6: Sequence[float],
        arm_effort_5: Sequence[float],
    ) -> None:
        container = FullRobotMsg(prefix=self.prefix)
        container.set_vehicle_wrench(wrench_body_6)
        container.set_arm_effort(arm_effort_5)

        veh_msg = container.to_vehicle_dynamic_group(self.node.get_clock().now().to_msg())
        arm_msg = container.to_arm_effort_array()

        self.vehicle_effort_command_publisher.publish(veh_msg)
        self.manipulator_effort_command_publisher.publish(arm_msg)
    
    # ForwardCommandController
    def publish_commands(self, wrench_body_6: Sequence[float], arm_effort_5: Sequence[float]):
        # Vehicle, DynamicInterfaceGroupValues payload
        self.publish_vehicle_and_arm(wrench_body_6, arm_effort_5)

    def publish_vehicle_pwms(self,
                             pwm_thruster_8: Sequence[float]):
        container = FullRobotMsg(prefix=self.prefix)
        container.set_vehicle_pwm(pwm_thruster_8)
        vehicle_pwm = container.to_vehicle_pwm()
        self.vehicle_pwm_command_publisher.publish(vehicle_pwm)

    def close_csv(self):
        # Close the CSV file when the node is destroyed
        self.csv_file.close()

    def listener_callback(self, msg: DynamicJointState):
        self.update_state(msg)

    def estimate_manipulator_parameter(self):
        try:
            Y_row = self.manipulator_Y(
                self.arm.filtered_q,
                self.arm.filtered_dq,
                self.arm.estimated_ddq,
                self.vec_g,
                self.base_pose,
                self.world_pose
            ).full()  # expected 4 x p
            tau_row = np.array(self.arm.effort).reshape(-1, 1)   # 4 x 1  
            # Append this time step
            Y_was_full = self.arm.Y_rows_buffer.append(Y_row)
            tau_was_full = self.arm.tau_rows_buffer.append(tau_row)

            Y_big = self.arm.Y_rows_buffer.as_vstack()
            tau_big = self.arm.tau_rows_buffer.as_vstack().ravel()
            assert Y_big.shape[0] == tau_big.size, "tau size must equal Y rows"

            # self.node.get_logger().info(f"Y_big shape {Y_big.shape} {Y_was_full}, tau_big shape {tau_big.shape} {tau_was_full}")
            if Y_was_full and tau_was_full:
                theta_hat, v, w, solve_time = self.arm.manipulator_estimator.estimate_link_physical_parameters(Y_big, tau_big, True)
                self.manipulator_estimation_info = (Y_big, tau_big, theta_hat, v, w, solve_time)
                pi_prev = getattr(self, "last_manip_theta", theta_hat).ravel()
                self.arm.manip_uncert.update(pi_prev=pi_prev, w_t=w.ravel(), pi_t=theta_hat.ravel())
                self.last_manip_theta = theta_hat.ravel()
                if not self.arm.has_intialize_manipulator_estimator:
                    self.node.get_logger().info(f"\033[35mManipulator Warm Start complete\033[0m")
                    self.arm.has_intialize_manipulator_estimator = True
        except Exception as e:
            self.node.get_logger().warn(f"Failed to compute theta from Y or tau for {self.robot_name}: {e}")

    def _manip_param_names(self):
        """Order matches your estimator layout, 4 joints, 12 params each."""
        base = []
        for j in range(4):
            base += [
                f"m{j}", f"m*rcx{j}", f"m*rcy{j}", f"m*rcz{j}",
                f"Ixx{j}", f"Iyy{j}", f"Izz{j}", f"Ixy{j}", f"Ixz{j}", f"Iyz{j}",
                f"fv{j}", f"fs{j}",
            ]
        return base

    def _with_suffix(self, names, suffix):
        return [f"{n}_{suffix}" for n in names]

    def joint_eigs(self, th_block):
        m, mcx, mcy, mcz = th_block[0], th_block[1], th_block[2], th_block[3]
        Ixx, Iyy, Izz = th_block[4], th_block[5], th_block[6]
        Ixy, Ixz, Iyz = th_block[7], th_block[8], th_block[9]

        mc = np.array([mcx, mcy, mcz], float)
        I_bar = np.array([[Ixx, Ixy, Ixz],
                        [Ixy, Iyy, Iyz],
                        [Ixz, Iyz, Izz]], float)

        J_ul = 0.5 * np.trace(I_bar) * np.eye(3) - I_bar
        J = np.block([[J_ul, mc.reshape(3, 1)],
                    [mc.reshape(1, 3), np.array([[m]])]])

        eigJ = np.linalg.eigvalsh(J)   # 4 eigenvalues, ascending
        return eigJ

    def initialize_manipulator_estimator_writer(self):
        # Create a timestamp string
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create a folder with the timestamp as its name
        folder_path = os.path.join(os.getcwd(), timestamp_str)
        os.makedirs(folder_path, exist_ok=True)

        # Create a timestamped filename for the CSV
        filename = f"{timestamp_str}_{self.prefix}_manipulator_estimates.csv"
        file_path = os.path.join(folder_path, filename)

        # Open the CSV file and prepare to write data
        self.manipulator_estimates_csv_file = open(file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.manipulator_estimates_csv_file)

        # time columns
        time_cols = ["t", "ros_time"]
        
        # Build header: theta, sigma, ci lower, ci upper
        base = self._manip_param_names()                         # length 48
        cols_sigma = self._with_suffix(base, "sigma")            # length 48
        cols_ci_lo = self._with_suffix(base, "ci_lo")            # length 48
        cols_ci_hi = self._with_suffix(base, "ci_hi")            # length 48

        # error metrics per joint
        metrics = ["mse", "mae", "rmse"]
        solve_time_col = ["solve_time"]
        tau_trajectory_cols = ["tau0", "tau1", "tau2", "tau3", "tau_fit0", "tau_fit1", "tau_fit2", "tau_fit3"]
        eig_cols = [f"eigJ_joint{j}_e{k}" for j in range(4) for k in range(1, 5)]
        metric_cols = []
        unc_cols = [f"tau_fit_sigma{j}" for j in range(4)] + [f"tau_pred_sigma{j}" for j in range(4)]

        for j in range(4):
            for m in metrics:
                metric_cols.append(f"{m}_joint{j}")

        columns = (
            time_cols
              + base + cols_sigma + cols_ci_lo + cols_ci_hi 
              + metric_cols + solve_time_col 
              + tau_trajectory_cols + unc_cols + eig_cols)
        self.csv_writer.writerow(columns)

    def pretty_log_manipulator_params(self):
        Y_big, tau_big, theta_hat, v, w, solve_time = self.manipulator_estimation_info

        # Fit on the stacked block
        tau_fit_big = Y_big @ theta_hat                  # shape (T*4, 1)
        T = self.arm.horizon_steps
        tau_fit = tau_fit_big.reshape(T, 4)
        tau = tau_big.reshape(T, 4)

        # latest residual for online metrics
        r_last = tau[-1, :] - tau_fit[-1, :]
        self.arm.manip_err_metrics.update(r_last)
        m_summ = self.arm.manip_err_metrics.summary()
        mse_ew, mae_ew, rmse_ew = m_summ["mse"], m_summ["mae"], m_summ["rmse"]

        # Uncertainty summary
        summ = self.arm.manip_uncert.summary(z=1.96)
        sigma = np.asarray(summ["sigma"]).reshape(-1)     # (48,)
        ci = np.asarray(summ["ci95"])                     # (48, 2)
        ci_lower = ci[:, 0].reshape(-1)
        ci_upper = ci[:, 1].reshape(-1)

        # latest 4 rows of the stacked regressor correspond to the current sample for joints 0..3
        # Y_big shape is (T*4, p). Take the last 4 rows
        Y_last = Y_big[-4:, :]                       # shape (4, p)

        # parameter covariance
        Sigma_theta = self.arm.manip_uncert.parameter_covariance()   # shape (p, p)

        # propagate to torque fit covariance, 4x4
        Sigma_tau_fit = Y_last @ Sigma_theta @ Y_last.T

        # standard deviation per joint from the diagonal
        tau_fit_sigma = np.sqrt(np.clip(np.diag(Sigma_tau_fit), 0.0, np.inf))   # shape (4,)

        # predictive torque sigma adds residual noise variance from online metrics
        sigma_noise = np.sqrt(np.maximum(m_summ["mse"], 0.0))                   # shape (4,)
        tau_pred_sigma = np.sqrt(tau_fit_sigma**2 + sigma_noise**2)             # shape (4,)

        # Parameters
        theta_row = np.asarray(theta_hat).reshape(-1)     # (48,)

        tau_last     = tau[-1, :].reshape(-1)        # shape (4,)
        tau_fit_last = tau_fit[-1, :].reshape(-1)    # shape (4,)

        th = np.asarray(theta_hat, float).reshape(-1)
        group_len = 12
        eig_all = []
        for j in range(4):
            s, e = j * group_len, (j + 1) * group_len
            eigJ = self.joint_eigs(th[s:e])
            eig_all.extend([float(x) for x in eigJ])
        # times
        t_sim = float(self.sim_time)
        t_ros = float(self.node.get_clock().now().nanoseconds) * 1e-9
        # Assemble row in the same order as header,
        # where header has: params, sigma, ci_lo, ci_hi, mse_joint0..3, mae_joint0..3, rmse_joint0..3
        row_data = (
            [t_sim, t_ros]
            + theta_row.tolist()
            + sigma.tolist()
            + ci_lower.tolist()
            + ci_upper.tolist()
            + mse_ew.tolist()
            + mae_ew.tolist()
            + rmse_ew.tolist()
            + [solve_time]
            + tau_last.tolist()
            + tau_fit_last.tolist()
            + tau_fit_sigma.tolist()
            + tau_pred_sigma.tolist()
            + eig_all   # 16 values total, 4 per joint
        )

        if all(val == 0 for val in row_data):
            return

        self.csv_writer.writerow(row_data)
        self.manipulator_estimates_csv_file.flush()


    def close_manipulator_estimates_csv(self):
        # Close the CSV file when the node is destroyed
        self.manipulator_estimates_csv_file.close()

    def estimate_vehicle_parameter(self):
        try:
            eul = self.state_estimate_readings[3:6]
            x_nb = self.state_estimate_readings[10:16]
            dx_nb = self.state_estimate_readings[16:22]

            Y_row = self.vehicle_Y(eul, x_nb, dx_nb, self.v_c).full()  # expected 4 x p

            τ_row = np.array(self.body_forces).reshape(-1, 1)   # 4 x 1 
            # Append this time step
            Y_was_full = self.vehicle_Y_rows_buffer.append(Y_row)
            tau_was_full = self.vehicle_tau_rows_buffer.append(τ_row)

            Y_big = self.vehicle_Y_rows_buffer.as_vstack()
            tau_big = self.vehicle_tau_rows_buffer.as_vstack().ravel()
            assert Y_big.shape[0] == tau_big.size, "tau size must equal Y rows"

            # self.node.get_logger().info(f"Y_big shape {Y_big.shape} {Y_was_full}, tau_big shape {tau_big.shape} {tau_was_full}")
            if Y_was_full and tau_was_full:
                theta_hat, v, w, solve_time = self.vehicle_estimator.estimate_vehicle_physical_parameters(Y_big, tau_big, True)
                self.vehicle_estimation_info = (Y_big, tau_big, theta_hat, v, w, solve_time)

                pi_prev = getattr(self, "last_vehicle_theta", theta_hat).ravel()
                self.vehicle_uncert.update(pi_prev=pi_prev, w_t=w.ravel(), pi_t=theta_hat.ravel())
                self.last_vehicle_theta = theta_hat.ravel()

                if not self.has_intialize_vehicle_estimator:
                    self.has_intialize_vehicle_estimator = True
        except Exception as e:
            self.node.get_logger().warn(f"Failed to compute theta from Y or tau for {self.robot_name}: {e}")

    def initialize_vehicle_estimator_writer(self):
        # Create a timestamp string
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create a folder with the timestamp as its name
        folder_path = os.path.join(os.getcwd(), timestamp_str)
        os.makedirs(folder_path, exist_ok=True)

        # Create a timestamped filename for the CSV
        filename = f"{timestamp_str}_{self.prefix}_vehicle_estimates.csv"
        file_path = os.path.join(folder_path, filename)

        # Open the CSV file and prepare to write data
        self.vehicle_estimates_csv_file = open(file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.vehicle_estimates_csv_file)

        # time columns
        time_cols = ["t", "ros_time"]
        
        # Build header: theta, sigma, ci lower, ci upper
        base = [ 'm-X_du', 'm-Y_dv', 'm-Z_dw',
            'm*z_g-X_dq', '-m*z_g+Y_dp', '-m*z_g+K_dv', 'm*z_g-M_du',
            'I_x-K_dp', 'I_y-M_dq', 'I_z-N_dr',
            'W', 'B', 'x_gW-x_bB', 'y_gW-y_bB', 'z_gW-z_bB',
            'X_u', 'Y_v', 'Z_w', 'K_p', 'M_q', 'N_r',
            'X_uu', 'Y_vv', 'Z_ww', 'K_pp', 'M_qq', 'N_rr' ]
        
        # Build header: theta, sigma, ci lower, ci upper
        cols_sigma = self._with_suffix(base, "sigma")            # length 27
        cols_ci_lo = self._with_suffix(base, "ci_lo")            # length 27
        cols_ci_hi = self._with_suffix(base, "ci_hi")            # length 27

        # error metrics per joint
        metrics = ["mse", "mae", "rmse"]
        solve_time_col = ["solve_time"]
        dofs = ["x", "y", "z", "roll", "pitch", "yaw"]
        tau_trajectory_cols = ["F_x", "F_y", "F_z", "T_roll", "T_pitch", "T_yaw",
                               "F_x_fit", "F_y_fit", "F_z_fit", "T_roll_fit", "T_pitch_fit", "T_yaw_fit"]
        metric_cols = []
        unc_cols_fit_sigma = ["F_x_fit_sigma", "F_y_fit_sigma", "F_z_fit_sigma", "T_roll_fit_sigma", "T_pitch_fit_sigma", "T_yaw_fit_sigma"]
        unc_cols_pred_sigma = ["F_x_pred_sigma", "F_y_pred_sigma", "F_z_pred_sigma", "T_roll_pred_sigma", "T_pitch_pred_sigma", "T_yaw_pred_sigma"]
        unc_cols = unc_cols_fit_sigma + unc_cols_pred_sigma

        for dof in dofs:
            for m in metrics:
                metric_cols.append(f"{m}_dof_{dof}")

        columns = (
            time_cols
              + base + cols_sigma + cols_ci_lo + cols_ci_hi 
              + metric_cols + solve_time_col 
              + tau_trajectory_cols + unc_cols)

        self.csv_writer.writerow(columns)

    def pretty_print_vehicle_params(self, title="Identified vehicle parameters"):
        """
        Pretty print and sanity check the lumped vehicle parameters used in your regressor.
        Expects the parameter order:
            [ m-X_du, m-Y_dv, m-Z_dw,
            m*z_g-X_dq, -m*z_g+Y_dp, -m*z_g+K_dv, m*z_g-M_du,
            I_x-K_dp, I_y-M_dq, I_z-N_dr,
            W, B, x_gW-x_bB, y_gW-y_bB, z_gW-z_bB,
            X_u, Y_v, Z_w, K_p, M_q, N_r,
            X_uu, Y_vv, Z_ww, K_pp, M_qq, N_rr ]
        """
        Y_big, tau_big, theta_hat, v, w, solve_time = self.vehicle_estimation_info
        tau_fit_big = Y_big @ theta_hat
        # reshape back to per joint series for plotting
        tau_fit = tau_fit_big.reshape(self.vehicle_n_horizon_steps, 6)
        tau = tau_big.reshape(self.vehicle_n_horizon_steps, 6)
        # latest residual for online metrics
        r_last = tau[-1, :] - tau_fit[-1, :]
        self.vehicle_err_metrics.update(r_last)
        m_summ = self.vehicle_err_metrics.summary()
        mse_ew, mae_ew, rmse_ew = m_summ["mse"], m_summ["mae"], m_summ["rmse"]

        # Uncertainty summary
        summ = self.vehicle_uncert.summary(z=1.96)
        sigma = np.asarray(summ["sigma"]).reshape(-1)     # (27,)
        ci = np.asarray(summ["ci95"])                     # (27, 2)
        ci_lower = ci[:, 0].reshape(-1)
        ci_upper = ci[:, 1].reshape(-1)

        # latest 6 rows of the stacked regressor correspond to the current sample for joints 0..3
        # Y_big shape is (T*6, p). Take the last 6 rows
        Y_last = Y_big[-6:, :]                       # shape (6, p)

        # parameter covariance
        Sigma_theta = self.vehicle_uncert.parameter_covariance()   # shape (p, p)

        # propagate to torque fit covariance, 6x6
        Sigma_tau_fit = Y_last @ Sigma_theta @ Y_last.T

        # standard deviation per joint from the diagonal
        tau_fit_sigma = np.sqrt(np.clip(np.diag(Sigma_tau_fit), 0.0, np.inf))   # shape (6,)

        # predictive torque sigma adds residual noise variance from online metrics
        sigma_noise = np.sqrt(np.maximum(m_summ["mse"], 0.0))                   # shape (6,)
        tau_pred_sigma = np.sqrt(tau_fit_sigma**2 + sigma_noise**2)             # shape (6,)

        # Parameters
        theta_row = np.asarray(theta_hat).reshape(-1)     # (27,)

        tau_last     = tau[-1, :].reshape(-1)        # shape (6,)
        tau_fit_last = tau_fit[-1, :].reshape(-1)    # shape (6,)

        # times
        t_sim = float(self.sim_time)
        t_ros = float(self.node.get_clock().now().nanoseconds) * 1e-9
        # Assemble row in the same order as header,
        # where header has: params, sigma, ci_lo, ci_hi, mse_joint0..3, mae_joint0..3, rmse_joint0..3
        row_data = (
            [t_sim, t_ros]
            + theta_row.tolist()            
            + sigma.tolist()
            + ci_lower.tolist()
            + ci_upper.tolist()
            + mse_ew.tolist()
            + mae_ew.tolist()
            + rmse_ew.tolist()
            + [solve_time]
            + tau_last.tolist()
            + tau_fit_last.tolist()
            + tau_fit_sigma.tolist()
            + tau_pred_sigma.tolist()
        )

        if all(val == 0 for val in row_data):
            return

        self.csv_writer.writerow(row_data)
        self.vehicle_estimates_csv_file.flush()

    def close_vehicle_estimates_csv(self):
        # Close the CSV file when the node is destroyed
        self.vehicle_estimates_csv_file.close()
