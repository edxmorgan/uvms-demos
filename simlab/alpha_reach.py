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

    u_min = np.array([-1.4, -0.629139, -0.518764, -0.54])
    u_max = np.array([1.4, 0.629139, 0.518764, 0.54])
    
    Kp = cs.vertcat(1.0, 1.0, 1.0, 1.0)
    Ki = cs.vertcat(1e-3, 1e-3, 1e-3, 1e-3)
    Kd = cs.vertcat(1e-3, 1e-3, 1e-3, 1e-3)

    # rho = 1 #kg/L

    # mc = cs.SX.sym('mc', 7)

    # mc_v = [7e-6, 0.032, 1716e-6, 0.017, 0.201, 2443e-6, 0.226]

    # M_A_0 = cs.vertcat(mc[0]*rho, mc[0]*rho, 0, mc[1]*rho, mc[1]*rho, mc[3]*rho)
    # M_A_1 = cs.vertcat(0, mc[2]*rho, mc[2]*rho, mc[3]*rho, mc[4]*rho, mc[4]*rho)
    # M_A_2 = cs.vertcat(mc[0]*rho, 0, mc[0]*rho, mc[1]*rho, mc[3]*rho, mc[1]*rho)
    # M_A_3 = cs.vertcat(mc[5]*rho, mc[5]*rho, 0, mc[6]*rho, mc[6]*rho, mc[3]*rho)
    # MA__cof = cs.vertcat(M_A_0, M_A_1, M_A_2, M_A_3)

    # D_u_0 = cs.DM([0, 0, 0, 0, 0, 0])
    # D_u_1 = cs.DM([0, 0, 0, 0, 0, 0])
    # D_u_2 = cs.DM([0, 0, 0, 0, 0, 0])
    # D_u_3 = cs.DM([0, 0, 0, 0, 0, 0])
    # Du__cof = cs.vertcat(D_u_0, D_u_1, D_u_2, D_u_3)

    # du = cs.SX.sym('du', 4)

    # du_v = [0.26, 0.3, 1.6, 1.8]

    # D_uu_0 = cs.vertcat(0, 0, 0, du[0]*rho, du[0]*rho, du[1]*rho)
    # D_uu_1 = cs.vertcat(0, 0, 0, du[1]*rho, du[2]*rho, du[2]*rho)
    # D_uu_2 = cs.vertcat(0, 0, 0, du[0]*rho, du[1]*rho, du[0]*rho)
    # D_uu_3 = cs.vertcat(0, 0, 0, du[3]*rho, du[3]*rho, du[1]*rho)
    # Duu__cof = cs.vertcat(D_uu_0, D_uu_1, D_uu_2, D_uu_3)

    # COB_m0 = cs.DM([-1e-03, -2e-03, -32e-03])
    # COB_m1 = cs.DM([73e-3, 0, -2e-3])
    # COB_m2 = cs.DM([3e-3, 1e-3, -17e-3])
    # COB_m3 = cs.DM([0e-3, 3e-3, -98e-3])
    # COB__m = cs.vertcat(COB_m0, COB_m1, COB_m2, COB_m3)

    # volume__ = cs.vertcat(1.8e-5, 0.000203, 2.5e-5, 0.000155) #m3

    # p0_Hyd__sx = cs.vertcat(MA__cof, Du__cof, Duu__cof, volume__ ,COB__m, rho)
    # p0_Hyd__fun = cs.Function('p0Hydsx', [mc, du],[p0_Hyd__sx])
    # arm_Hyd__p_est = p0_Hyd__fun(mc_v, du_v)
    # arm_Hyd__p_est

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

    # trivial_sim_p = [0, 0, 0 , 0, 0, 0, 0]
