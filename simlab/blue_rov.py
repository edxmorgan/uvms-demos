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
import casadi as ca
import matplotlib.pyplot as plt
import pandas as pd

class Params:

    # Ocean current velocities. 
    v_flow = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # (m/s). Assume irrotational, constant.

    # Thrust configuration matrix by BlueROV2 Heavy. Converts thrust force to body and vice versa. 
    # TODO Check Calcs to determine how to find. 
    # thrust_config = np.array([
    #     [-0.7070, -0.7070,  0.7070,  0.7070,  0.0000,  0.0000,  0.0000, 0.0000],
    #     [ 0.7070, -0.7070,  0.7070, -0.7070,  0.0000,  0.0000,  0.0000, 0.0000],
    #     [ 0.0000,  0.0000,  0.0000,  0.0000,  1.0000, -1.0000, -1.0000, 1.0000],
    #     [ 0.0000,  0.0000,  0.0000,  0.0000, -0.2180, -0.2180,  0.2180, 0.2180],
    #     [ 0.0000,  0.0000,  0.0000,  0.0000, -0.1200,  0.1200, -0.1200, 0.1200],
    #     [ 0.1888, -0.1888, -0.1888,  0.1888,  0.0000,  0.0000,  0.0000, 0.0000]])
    
    # Alternative thrust config matrix. 
    thrust_config = np.array([[0.707, 0.707, -0.707, -0.707, 0.0, 0.0, 0.0, 0.0],
                            [-0.707, 0.707, -0.707, 0.707, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0,-1.0],
                            [0.06, -0.06, 0.06, -0.06, -0.218, -0.218, 0.218, 0.218],
                            [0.06, 0.06, -0.06, -0.06, 0.12, -0.12, 0.12, -0.12],
                            [-0.1888, 0.1888, 0.1888, -0.1888, 0.0, 0.0, 0.0, 0.0]])
    
    # force coefficient matrix
    # f_K_diag = np.array([1, 1, 1, 1, 1, 1])
    # T_db = np.array([0, 0, 0, 0, 0, 0])
    # B_eps = np.array([3])
    # W_B_bias  = np.array([0.0]) # weight and buoyancy bias.
    ### Parameters in rigid body dynamics and restoring forces
    # Based on BlueRobotics 2018b technical specs. 
    # Based on Table 5.1
    m = 11.5 #(kg)
    W = m*9.81 #(N). 112.8 N. Weight. 
    B = 114.8 #(N). bouyancy 
    rb  = np.array([0, 0, 0]) #(m). center of buoyancy (CoB) coincides with the center of origin
    rg = np.array([0, 0, 0.02]) #(m). 

    # Axis inertias. 
    # BAsed on Table 5.1.
    I_x = 0.16 #(kg m2)
    I_y = 0.16 #(kg m2)
    I_z = 0.16 #(kg m2)
    I_xz = 0
    I_yz = 0
    I_xy = 0
    Io = np.array([I_x, I_y, I_z, I_xz])

    Ib_b = np.zeros((3,3))
    Ib_b[0, :] = [I_x, -I_xy, -I_xz]
    Ib_b[1, :] = [-I_xy, I_y, -I_yz]
    Ib_b[2, :] = [-I_xz, -I_yz, I_z]

    # Added mass parameters.
    # Based on Table 5.2. 
    X_du = -5.5 #(kg). Surge. 
    Y_dv = -12.7 #(kg). Sway. 
    Z_dw = -14.57 #(kg). Heave. 
    K_dp = -0.12 #(kg m2/rad). Roll.
    M_dq = -0.12 #(kg m2/rad). Pitch. 
    N_dr = -0.12 #(kg m2/rad). Yaw. 
    added_m = np.array([X_du, Y_dv, Z_dw, K_dp, M_dq, N_dr])

    _MA = np.zeros((6, 6))
    _MA[0, :] = [X_du, 0, 0, 0, 0, 0]
    _MA[1, :] = [0, Y_dv, 0, 0, 0, 0]
    _MA[2, :] = [0, 0, Z_dw, 0, 0, 0]
    _MA[3, :] = [0, 0, 0, K_dp, 0, 0]
    _MA[4, :] = [0, 0, 0, 0, M_dq, 0]
    _MA[5, :] = [0, 0, 0, 0, 0, N_dr]
    MA = -_MA

    coupl_added_m = np.array([0, 0, 0, 0]) # ASSUMING decoupling motion

    # Linear damping coeffs. 
    Xu = -4.03 #(Ns/m). Surge. 
    Yv = -6.22 #(Ns/m). Sway.
    Zw = -5.18 #(Ns/m). Heave.  
    Kp = -0.07 #(Ns/rad). Roll.
    Mq = -0.07 #(Ns/rad). Pitch.
    Nr = -0.07 #(Ns/rad). Yaw. 
    linear_dc = np.array([Xu, Yv, Zw, Kp,  Mq, Nr])

    # Quadratic damping coeffs. 
    Xuu = -18.18 #(Ns2/m2). Surge. 
    Yvv = -21.66 #(Ns2/m2). Sway. 
    Zww = -36.99 #(Ns2/m2). Heave. 
    Kpp = -1.55 #(Ns2/rad2). Roll. 
    Mqq = -1.55 #(Ns2/rad2). Pitch. 
    Nrr = -1.55 #(Ns2/rad2). Yaw. 
    quadratic_dc = np.array([Xuu, Yvv, Zww, Kpp, Mqq, Nrr])


    rg = np.array([0, 0, 0.02]) #(m). 
    rb = np.array([0, 0, 0]) #(m). center of buoyancy (CoB) coincides with the center of origin

    T = 20 # time horizon in seconds
    N = 1600 # number of control intervals
    at_surface = 0.0
    below_surface = 1.0 #random
    dt_s = T/N

    kp = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    ki = np.array([0, 0, 0, 0, 0, 0])
    kd = np.array([2, 2, 2, 2, 2, 2])
    
    sim_params = np.concatenate(( np.array([m]) , np.array([W]), np.array([B]), 
                                           rg, rb, Io, added_m, coupl_added_m, linear_dc, quadratic_dc, v_flow))
    
