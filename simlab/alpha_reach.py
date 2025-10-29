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
import casadi as cs
import itertools

class Params:
    root = "base_link"
    tip = "alpha_standard_jaws_base_link"

    relative_urdf_path = f"/resources/urdf/alpha_5_robot.urdf"

    joint_min = np.array([1.00, 0.01, 0.01, 0.01])
    joint_max = np.array([5.50, 3.40, 3.40, 5.70])

    joint_limits = list(zip(joint_min.tolist(), joint_max.tolist()))
    joint_limits_configurations = np.array(list(itertools.product(*joint_limits)))

    u_min = np.array([-1.5, -1, -1, -0.54])
    u_max = np.array([1.5, 1, 1, 0.54])
    
    Kp = cs.vertcat(8.0, 8.0, 8.0, 8.0)
    Ki = cs.vertcat(0.0, 0.0, 0.0, 0.0)
    Kd = cs.vertcat(0.0, 0.0, 0.0, 0.0)

    # reducing model parameters by assuming non rotating axis are inertialess or inactivity
    Gear_p = cs.vertcat(2253.54, 2253.54, 2253.54, 340.4)
    rigid_p0 = cs.vertcat( 1e-06, 1e-06, 1e-06, 1e-06, 
                            0.0, 0.0, 0.0, 0.0, 
                            2.5, 2.6, 1.7, 0.2,
                            0.0, 0.0, 0.0, 0.0,
                            4.0, 1.9, 1.3, 1.0)
                                                      
    sim_p = cs.vertcat(Gear_p, rigid_p0).full()

    gravity = -9.81
    base_gravity = 0.0 #9.81
    base_T0 = [3.142, 0.000, 0.000, 0.190, 0.000, -0.120] #transformation of uv body frame to manipulator base

    sim_n = 20 # time horizon
    delta_t = 0.035
    N = int(sim_n/delta_t) # number of control intervals