import cvxpy as cp
import numpy as np

class SerialLinksDynamicEstimator:
    def __init__(self, dof, horizon_steps, n_params):
        self.n = horizon_steps * dof
        self.p = n_params
        self.link_group_len = 12  # [m, m*rcx, m*rcy, m*rcz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz, fv, fs]
        self.masses = [0.194, 0.429, 0.115, 0.333]

        # ----- bounds for theta ----- 
        self.n_joints = self.p // self.link_group_len       
        if dof != self.n_joints and self.p % self.link_group_len != 0:
            raise ValueError("p is not divisible by 12, check regressor layout.")
        
        # masses at indices 0, 12, 24, 36, ...
        self.mass_idx = [j*self.link_group_len + 0 for j in range(self.n_joints)]
        self.fv_idx = [j*self.link_group_len + 10 for j in range(self.n_joints)]
        self.fs_idx = [j*self.link_group_len + 11 for j in range(self.n_joints)]

    #  Helper, pseudo inertia LMI per joint
    def J_from_block(self, th, s):
        m   = th[s + 0]
        mcx = th[s + 1];  mcy = th[s + 2];  mcz = th[s + 3]
        Ixx = th[s + 4];  Iyy = th[s + 5];  Izz = th[s + 6]
        Ixy = th[s + 7];  Ixz = th[s + 8];  Iyz = th[s + 9]
        # 3x3 inertia about the same origin as c
        I_bar = cp.bmat([[Ixx, Ixy, Ixz],
                        [Ixy, Iyy, Iyz],
                        [Ixz, Iyz, Izz]])
        mc = cp.vstack([mcx, mcy, mcz])
        J_ul = 0.5 * cp.trace(I_bar) * np.eye(3) - I_bar
        J = cp.bmat([[J_ul, mc],
                    [mc.T,  cp.reshape(m, (1, 1), order="F")]])  # or order="C"
        return J

    def estimate_link_physical_parameters(self, Y_big, tau_big, x_prev):
        x = cp.Variable(shape=(self.p,))
        # x_prev = cp.Parameter(shape=(self.p,))
        w = cp.Variable(shape=(self.p,))
        v = cp.Variable(shape=(self.n,))

        tau = 2
        rho = 2

        eps = 1e-6
        scales = np.maximum(np.abs(x_prev), eps)  # elementwise
        Q = np.diag(1.0 / (scales**2))            # penalize relative change #positive definite
        # obj = cp.quad_form(w, Q)
        # obj += tau*cp.huber(cp.norm(v),rho)

        obj = cp.sum_squares(w)
        obj += tau*cp.sum_squares(v)
        obj = cp.Minimize(obj)

        constr = []

        # friction bounds
        constr += [x[self.fv_idx]   >= 0.0,
                        x[self.fs_idx]   >= 0.0]

        # LMI, J_j >> delta*I to avoid near singularities
        for j in range(self.n_joints):
            s = j * self.link_group_len
            m   = x[s + 0]
            mcx = x[s + 1]
            mcy = x[s + 2]
            mcz = x[s + 3]

            r = 0.1  # 0.1 m CoM bound per axis
            constr += [
                -r * m <= mcx, mcx <= r * m,
                -r * m <= mcy, mcy <= r * m,
                -r * m <= mcz, mcz <= r * m,
                m == self.masses[j]
            ]

            Jj = self.J_from_block(x, s)
            constr += [Jj >> 0 * np.eye(4)]

        constr += [ x == x_prev + w,
                    tau_big == Y_big@x + v]

        cp.Problem(obj, constr).solve(solver=cp.MOSEK, verbose=False, warm_start=True)

        x = np.array(x.value)
        w = np.array(w.value)
        v = np.array(v.value)

        return x, v, w
    

class MarineVehicleEstimator:
    def __init__(self, n_params, n_horizon):
        self.p = n_params
        self.n = n_horizon

        # β = [m-X_du, m-Y_dv, m-Z_dw, m*z_g-X_dq, -m*z_g+Y_dp, -m*z_g+K_dv, m*z_g-M_du, I_x-K_dp, I_y-M_dq, I_z-N_dr ,
        #  W, B, x_g*W - x_b*B, y_g*W - y_b*B , z_g*W - z_b*B, 
        # X_u, Y_v, Z_w, K_p, M_q, N_r, X_uu, Y_vv, Z_ww, K_pp, M_qq, N_rr]
    
        # parameter layout from your build_sys_regressor
        # beta = [inertia 10, W, B, x_gW_x_bB, y_gW_y_bB, z_gW_z_bB, linear 6, quad 6]
        # indices
        self.idx_inertia = list(range(0, 10))
        self.idx_WB = [10, 11]
        self.idx_moment = [12, 13, 14]
        self.idx_linear = list(range(15, 21))
        self.idx_quad = list(range(21, 27))
    
    #  Helper, pseudo inertia LMI per joint
    def M_from_uv(self, th):
        m_X_du = th[0]
        m_Y_dv = th[1]
        m_Z_dw = th[2]
        mz_g_X_dq = th[3]
        mz_g_Y_dp = th[4]
        mz_g_K_dv = th[5]
        mz_g_M_du = th[6]
        I_x_K_dp = th[7]
        I_y_M_dq = th[8]
        I_z_N_dr = th[9]
        zero = cp.Constant(0)
        M = cp.bmat([[m_X_du, zero, zero, zero, mz_g_X_dq, zero],
                     [zero, m_Y_dv, zero, mz_g_Y_dp, zero, zero],
                     [zero, zero, m_Z_dw, zero, zero, zero],
                     [zero, mz_g_K_dv, zero, I_x_K_dp, zero, zero],
                     [mz_g_M_du, zero, zero, zero, I_y_M_dq, zero],
                     [zero, zero, zero, zero, zero, I_z_N_dr]])
        return M
    
    def estimate(self, Y_big, tau_big, x_prev, W_min=100.0, W_max=150.0):
        x = cp.Variable(shape=(self.p,))
        # x_prev = cp.Parameter(shape=(self.p,))
        w = cp.Variable(shape=(self.p,))
        v = cp.Variable(shape=(self.n,))

        tau = 2
        rho = 2

        eps = 1e-6
        scales = np.maximum(np.abs(x_prev), eps)  # elementwise
        Q = np.diag(1.0 / (scales**2))            # penalize relative change #positive definite
        # obj = cp.quad_form(w, Q)
        # obj += tau*cp.huber(cp.norm(v),rho)

        obj = cp.sum_squares(w)
        obj += tau*cp.sum_squares(v)
        obj = cp.Minimize(obj)

        constr = []
        
        Muv = self.M_from_uv(x)
        constr += [Muv >> 0 * np.eye(6)]
        constr += [Muv - Muv.T == 0 * np.eye(6)]

        constr += [ x == x_prev + w,
                    tau_big == Y_big@x + v]
        # weight box constraint, W is x[self.idx_WB[0]]
        W_idx = self.idx_WB[0]
        constr += [x[W_idx] >= W_min, x[W_idx] <= W_max]
        cp.Problem(obj, constr).solve(solver=cp.MOSEK, verbose=False, warm_start=True)

        x = np.array(x.value)
        w = np.array(w.value)
        v = np.array(v.value)

        return x, v, w