import cvxpy as cp
import numpy as np


class SerialLinksDynamicEstimator:
    def __init__(self, dof, horizon_steps, n_params, theta_prev,
                 fixed_blocks=None):
        self.theta_prev = theta_prev
        self.n = horizon_steps * dof
        self.p = n_params
        self.link_group_len = 12
        self.masses = [0.194, 0.429, 0.115, 0.333]

        self.n_joints = self.p // self.link_group_len
        if dof != self.n_joints or self.p % self.link_group_len != 0:
            raise ValueError("p is not divisible by 12, check regressor layout")

        def block_slice(j):
            s = j * self.link_group_len
            return slice(s, s + self.link_group_len)
        self.block_slice = block_slice

        self.mass_idx = [j*self.link_group_len + 0 for j in range(self.n_joints)]
        self.fv_idx   = [j*self.link_group_len + 10 for j in range(self.n_joints)]
        self.fs_idx   = [j*self.link_group_len + 11 for j in range(self.n_joints)]

        # which joints are fixed as full blocks
        self._fixed_blocks = {}
        fixed_joint_set = set()
        if fixed_blocks:
            for j, vec in fixed_blocks.items():
                v = np.asarray(vec, dtype=float).reshape(self.link_group_len)
                self._fixed_blocks[j] = v
                fixed_joint_set.add(int(j))

        self.x = cp.Variable(shape=(self.p,))
        self.w = cp.Variable(shape=(self.p,))
        self.v = cp.Variable(shape=(self.n,))

        self.A = cp.Parameter(shape=(self.n, self.p))
        self.b = cp.Parameter(shape=(self.n,))
        self.x_hat_prev = cp.Parameter(shape=(self.p,))

        tau = 2.0
        obj = cp.Minimize(cp.sum_squares(self.w) + tau * cp.sum_squares(self.v))
        constr = []

        # friction bounds only for joints that are not fixed
        fv_idx_free = [self.fv_idx[j] for j in range(self.n_joints) if j not in fixed_joint_set]
        fs_idx_free = [self.fs_idx[j] for j in range(self.n_joints) if j not in fixed_joint_set]
        if fv_idx_free:
            constr += [self.x[fv_idx_free] >= 0.0]
        if fs_idx_free:
            constr += [self.x[fs_idx_free] >= 0.0]

        # per joint constraints, skip all prior constraints for fixed joints
        for j in range(self.n_joints):
            s = j * self.link_group_len
            bs = self.block_slice(j)

            if j in fixed_joint_set:
                # hard pin entire block, no other constraints on this block
                constr += [self.x[bs] == self._fixed_blocks[j]]
                continue

            # prior constraints for free joints only
            m   = self.x[s + 0]
            mcx = self.x[s + 1]
            mcy = self.x[s + 2]
            mcz = self.x[s + 3]

            r = 0.1
            constr += [
                -r*m <= mcx, mcx <= r*m,
                -r*m <= mcy, mcy <= r*m,
                -r*m <= mcz, mcz <= r*m,
                m == self.masses[j]
            ]

            Jj = self.J_from_block(self.x, s)
            constr += [Jj >> 0 * np.eye(4)]

        # model equations
        constr += [
            self.x == self.x_hat_prev + self.w,
            self.b == self.A @ self.x + self.v
        ]

        self.prob = cp.Problem(obj, constr)

    def J_from_block(self, th, s):
        m   = th[s + 0]
        mcx = th[s + 1];  mcy = th[s + 2];  mcz = th[s + 3]
        Ixx = th[s + 4];  Iyy = th[s + 5];  Izz = th[s + 6]
        Ixy = th[s + 7];  Ixz = th[s + 8];  Iyz = th[s + 9]
        I_bar = cp.bmat([[Ixx, Ixy, Ixz],
                         [Ixy, Iyy, Iyz],
                         [Ixz, Iyz, Izz]])
        mc = cp.vstack([mcx, mcy, mcz])
        J_ul = 0.5 * cp.trace(I_bar) * np.eye(3) - I_bar
        J = cp.bmat([[J_ul, mc],
                     [mc.T, cp.reshape(m, (1, 1), order="F")]])
        return J

    def estimate_link_physical_parameters(self, Y_big, tau_big, warm_start):
        self.A.value = Y_big
        self.b.value = tau_big
        self.x_hat_prev.value = self.theta_prev
        self.prob.solve(solver=cp.MOSEK, verbose=False, warm_start=warm_start, ignore_dpp=True)
        x = np.array(self.x.value)
        self.theta_prev = x
        w = np.array(self.w.value)
        v = np.array(self.v.value)
        solve_time = self.prob.solver_stats.solve_time
        return x, v, w, solve_time
    

class MarineVehicleEstimator:
    def __init__(self, dof, n_params, n_horizon, theta_prev):
        self.theta_prev = theta_prev
        self.p = n_params
        self.n = n_horizon * dof

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

        self.x = cp.Variable(shape=(self.p,))
        # x_prev = cp.Parameter(shape=(self.p,))
        self.w = cp.Variable(shape=(self.p,))
        self.v = cp.Variable(shape=(self.n,))
        
        self.A = cp.Parameter(shape=(self.n, self.p))
        self.b = cp.Parameter(shape=(self.n,))
        self.x_hat_prev = cp.Parameter(shape=(self.p,))
        W_min=105.0,
        W_max=150.0
        tau = 2
        # rho = 2

        # eps = 1e-6
        # scales = np.maximum(np.abs(x_prev), eps)  # elementwise
        # Q = np.diag(1.0 / (scales**2))            # penalize relative change #positive definite
        # obj = cp.quad_form(w, Q)
        # obj += tau*cp.huber(cp.norm(v),rho)

        obj = cp.sum_squares(self.w)
        obj += tau*cp.sum_squares(self.v)
        obj = cp.Minimize(obj)

        constr = []
        
        Muv = self.M_from_uv(self.x)
        constr += [Muv >> 0 * np.eye(6)]
        constr += [Muv - Muv.T == 0 * np.eye(6)]

        constr += [ self.x == self.x_hat_prev + self.w,
                    self.b == self.A@self.x + self.v]
        # weight box constraint, W is x[self.idx_WB[0]]
        W_idx = self.idx_WB[0]
        constr += [self.x[W_idx] >= W_min, self.x[W_idx] <= W_max]
        self.prob = cp.Problem(obj, constr)

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
    
    def estimate_vehicle_physical_parameters(self, Y_big, tau_big, warm_start):
        self.A.value = Y_big
        self.b.value = tau_big
        self.x_hat_prev.value = self.theta_prev
        
        self.prob.solve(solver=cp.MOSEK, verbose=False, warm_start=warm_start, ignore_dpp = True)

        x = np.array(self.x.value)
        self.theta_prev = x
        w = np.array(self.w.value)
        v = np.array(self.v.value)
        solve_time = self.prob.solver_stats.solve_time

        return x, v, w, solve_time
    

class EWUncertainty:
    """
    Exponentially weighted covariance of parameter changes w_t,
    with scale normalization s_{t-1} = max(|pi_{t-1}|, eps) elementwise.
    """
    def __init__(self, dim, alpha=0.05, eps=1e-8, jitter=1e-12):
        assert 0 < alpha <= 1.0
        self.dim = dim
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.jitter = float(jitter)

        # Normalized stats
        self.mu_tilde = np.zeros(dim, dtype=float)
        self.Sigma_tilde = np.eye(dim, dtype=float) * (jitter)

        # Caches of last scale and mapped covariance
        self._s_prev = np.ones(dim, dtype=float)
        self._cov_w = np.eye(dim, dtype=float) * (jitter)
        self._sigma = np.sqrt(np.diag(self._cov_w))
        self._pi_t = np.zeros(dim, dtype=float)

        # Optional mask for fixed parameters, True means fixed
        self._fixed_mask = None

    def set_fixed_mask(self, fixed_mask: np.ndarray):
        """
        fixed_mask is a boolean vector of length dim.
        Entries that are True are treated as equality constrained.
        """
        assert fixed_mask.shape == (self.dim,)
        self._fixed_mask = fixed_mask.astype(bool)

    def _map_back(self, s_prev: np.ndarray):
        """
        Map Sigma_tilde back to original units,
        Sigma_w = S * Sigma_tilde * S, with S = diag(s_prev).
        Use elementwise scaling to avoid forming S explicitly.
        """
        # Efficient diag scaling, row by s, col by s
        cov_w = (self.Sigma_tilde * s_prev[None, :]) * s_prev[:, None]

        if self._fixed_mask is not None:
            m = self._fixed_mask
            cov_w[m, :] = 0.0
            cov_w[:, m] = 0.0

        # Ensure numerical PD on the diagonal
        cov_w.flat[:: self.dim + 1] += self.jitter
        self._cov_w = cov_w
        self._sigma = np.sqrt(np.clip(np.diag(cov_w), 0.0, np.inf))

    def update(self, pi_prev: np.ndarray = None, w_t: np.ndarray = None, pi_t: np.ndarray = None):
        """
        Provide either:
          1) pi_prev and w_t, or
          2) pi_prev and pi_t, in which case w_t = pi_t - pi_prev is computed.

        Updates the exponentially weighted mean and covariance of normalized changes.
        """
        if pi_prev is None:
            raise ValueError("pi_prev is required")
        pi_prev = np.asarray(pi_prev, dtype=float).reshape(-1)
        if pi_prev.size != self.dim:
            raise ValueError(f"pi_prev size {pi_prev.size} does not match dim {self.dim}")

        if w_t is None:
            if pi_t is None:
                raise ValueError("Provide either w_t, or pi_t together with pi_prev")
            pi_t = np.asarray(pi_t, dtype=float).reshape(-1)
            if pi_t.size != self.dim:
                raise ValueError(f"pi_t size {pi_t.size} does not match dim {self.dim}")
            w_t = pi_t - pi_prev
        else:
            w_t = np.asarray(w_t, dtype=float).reshape(-1)
            if w_t.size != self.dim:
                raise ValueError(f"w_t size {w_t.size} does not match dim {self.dim}")
            if pi_t is None:
                pi_t = pi_prev + w_t

        # Scale from previous parameters
        s_prev = np.maximum(np.abs(pi_prev), self.eps)

        # Normalize
        w_tilde = w_t / s_prev

        # EW mean and covariance updates, rank one form
        a = self.alpha
        mu_prev = self.mu_tilde.copy()
        self.mu_tilde = (1.0 - a) * self.mu_tilde + a * w_tilde
        self.Sigma_tilde = (
            (1.0 - a) * self.Sigma_tilde
            + a * np.outer(w_tilde - mu_prev, w_tilde - self.mu_tilde)
        )

        # PD jitter
        self.Sigma_tilde.flat[:: self.dim + 1] += self.jitter

        # Map back
        self._s_prev = s_prev
        self._pi_t = pi_t.copy()
        self._map_back(s_prev)

    def parameter_covariance(self):
        """
        Approximate covariance of theta using the running increment covariance
        and the effective memory length. Returns a copy.
        """
        L = 2.0 / self.alpha - 1.0
        return L * self._cov_w.copy()

    def summary(self, z=1.96):
        """
        Returns dict with covariance in original units, per parameter sigma,
        and 95 percent intervals at the current time t.
        """
        ci_lower = self._pi_t - z * self._sigma
        ci_upper = self._pi_t + z * self._sigma
        return {
            "pi_t": self._pi_t.copy(),
            "sigma": self._sigma.copy(),
            "covariance": self._cov_w.copy(),
            "ci95": np.vstack([ci_lower, ci_upper]).T,
            "alpha": self.alpha,
            "effective_memory_L": 2.0 / self.alpha - 1.0,
        }

class EWErrorMetrics:
    """
    Online exponentially weighted metrics per output channel.
    Keeps running MSE and MAE, RMSE is derived.
    """
    def __init__(self, n_outputs, alpha=0.05, eps=1e-12):
        self.n = int(n_outputs)
        self.a = float(alpha)
        self.eps = float(eps)
        self.mse = np.zeros(self.n, dtype=float)
        self.mae = np.zeros(self.n, dtype=float)

    def update(self, residual_vec):
        r = np.asarray(residual_vec, dtype=float).reshape(-1)
        if r.size != self.n:
            raise ValueError(f"residual_vec length {r.size} does not match n_outputs {self.n}")
        a = self.a
        self.mse = (1.0 - a) * self.mse + a * (r ** 2)
        self.mae = (1.0 - a) * self.mae + a * np.abs(r)

    def summary(self):
        rmse = np.sqrt(np.maximum(self.mse, 0.0))
        return dict(mse=self.mse.copy(), mae=self.mae.copy(), rmse=rmse)
