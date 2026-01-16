#!/usr/bin/env python
__author__ = "Arian Kourangi"
__contact__ = "arianke@kth.se"

import numpy as np
import casadi as cs
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import time

#from inter_stl.models.spacecraft_rate_model import SpacecraftRateModel

class SpacecraftInterMPC():
    def __init__(self, model, Tf=1.0, N=10, add_cbf=False):
        """ Model predictive controller for the spacecraft rate model.
        Args:
            model (SpacecraftRateModel): spacecraft rate model
            Tf (float): time horizon
            N (int): number of discretization steps
            add_cbf (bool): whether to add control barrier function constraints
        """
        self.model = model
        self.Tf = Tf
        self.N = N # number of discretization steps in the horizon
        self.dt = self.Tf/self.N

        self.add_cbf = add_cbf

        self.nx = 10 # Number of state variables
        self.nu = 6 # Number of control inputs

        self.params = {}
        self.vars = {}
        self.r_robot = 0.20
        self.r_object = 0.20

        # cost matrices (SITL)
        # self.Q = np.diag([10e0, 10e0, 10e0, 10e-1, 10e-1, 10e-1, 8e-1])
        # self.Q_e = 10 * self.Q
        # self.R = 2*np.diag([1e-2] * 6)

        # cost matrices (HW)
        # self.Q = np.diag([10e1, 10e1, 10e1, 10e-1, 10e-1, 10e-1, 8e0])
        # self.Q_e = 10 * self.Q
        # self.R = np.diag([1e-3] * 6)

        # px4-mpc 
        self.Q = np.diag([5e1, 5e1, 5e1, 5e1, 5e1, 5e1, 8e2, 8e2, 8e2, 8e2])
        self.Q_e = 10 * self.Q
        self.R = 2*np.diag([1e-2, 1e-2, 1e-2, 2e0, 2e0, 2e0])

        
        self.solver = self.setup()

    def setup(self):
        # State and input variables (over the whole horizon)
        x = cs.SX.sym('x',self.nx*2)
        u = cs.SX.sym('u',self.nu)
        xdot = cs.SX.sym('xdot', self.nx*2) 
        
        x_robot = x[0:self.nx]
        x_object = x[self.nx:self.nx*2]

        #f_alt= self.model.get_casadi_dx(x, u, True)
        f_robot = self.model.get_casadi_dx(x_robot, u, inter=True)

        dist = x_robot[0:3] - x_object[0:3]
        dist_norm = cs.sqrt(cs.dot(dist, dist)) + 1e-6
        contact_norm = dist / dist_norm


        k = 10
        activate = 1/(1 + cs.exp(k*(dist_norm - (self.r_robot + self.r_object))))
        alpha = 2

        proj_scalar = cs.mtimes(contact_norm.T, f_robot[3:6])   # 1x1
        a_o = alpha * activate * cs.mtimes(contact_norm, proj_scalar)  # 3x1

        f_object = cs.vertcat(
            x_object[3:6],
            a_o,
            cs.SX.zeros(4)
        )
        f_alt = cs.vertcat(f_robot, f_object)



        f_expl =f_alt 
        f_impl = xdot - f_expl

        model_ac = AcadosModel()
        model_ac.f_expl_expr = f_expl
        model_ac.f_impl_expr = f_impl
        model_ac.x = x
        model_ac.u = u
        model_ac.xdot = xdot
        model_ac.name = 'spacecraft_inter_model_acados'
        xref_r = cs.SX.sym('xref_r', self.nx)
    
        v_des = cs.SX.sym('v_des', 3)
        model_ac.p = cs.vertcat(xref_r, v_des)

        
        delta_V = v_des - x_object[3:6]
        des_norm = delta_V / (cs.sqrt(cs.dot(delta_V, delta_V)) + 1e-6)
        d_c = self.r_robot + self.r_object

        xref_o = cs.vertcat( xref_r[0:3] + des_norm * d_c,
                                     xref_r[3:10] )
        

        # Robot error
        e_robot = x_robot - xref_r
        cost_p_r =  cs.mtimes([e_robot[0:6].T, self.Q[0:6,0:6], e_robot[0:6]])
        
        q_r = x_robot[6:10]
        qref_r = xref_r[6:10]
        eq_r = 1 - (q_r.T @ qref_r)**2 
        cost_eq_r = eq_r.T @ self.Q[6,6].reshape((1, 1)) @ eq_r

        # Object error
        e_object = x_object - xref_o
        cost_p_o =  cs.mtimes([e_object[0:6].T, 10*self.Q[0:6,0:6], e_object[0:6]])
        q_o = x_object[6:10]
        qref_o = xref_o[6:10]
        eq_o = 1 - (q_o.T @ qref_o)**2
        cost_eq_o = eq_o.T @ self.Q[6,6].reshape((1, 1)) @ eq_o

        cost_u = cs.mtimes([u.T, self.R, u])

        cost_p_r_e = cs.mtimes([e_robot[0:6].T, self.Q_e[0:6,0:6], e_robot[0:6]])
        cost_eq_r_e = eq_r.T @ self.Q_e[6,6].reshape((1, 1)) @ eq_r
        cost_p_o_e = cs.mtimes([e_object[0:6].T, self.Q_e[0:6,0:6], e_object[0:6]])
        cost_eq_o_e = eq_o.T @ self.Q_e[6,6].reshape((1, 1)) @ eq_o

        tangent = cs.SX.eye(3) - cs.mtimes(contact_norm, contact_norm.T)
        tangent_cost= 1e1 *cs.dot(cs.mtimes(tangent, f_robot[3:6]), cs.mtimes(tangent, f_robot[3:6])) 
        
        model_ac.cost_expr_ext_cost = cost_p_r + cost_eq_r + cost_p_o + cost_eq_o + cost_u + tangent_cost
        model_ac.cost_expr_ext_cost_e = cost_p_r_e + cost_eq_r_e + cost_p_o_e + cost_eq_o_e


        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.Tf
        ocp.parameter_values = np.zeros(ocp.model.p.size()[0])
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        #g_pull = cs.mtimes(contact_norm.T, f_robot[3:6]) 
        model_ac.con_h_expr = proj_scalar
        
        ocp.constraints.lh = np.array([0.0]) # lower bound >= 0
        ocp.constraints.uh = np.array([1e6]) # big upper bound

        """
        #set cost matrices
        W = np.zeros((self.nx + self.nu, self.nx + self.nu))
        W[:self.nx, :self.nx] = self.Q
        W[self.nx:, self.nx:] = self.R
        ocp.cost.W = W
        ocp.cost.W_e = self.Q_e

        Vx = np.zeros((self.nx + self.nu, self.nx))
        Vx[0:self.nx, 0:self.nx] = np.eye(self.nx)     # all states go straight in

        Vu = np.zeros((self.nx + self.nu, self.nu))
        Vu[self.nx:, 0:self.nu] = np.eye(self.nu)      # all controls go straight in
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.Vx_e = np.eye(self.nx)
        ocp.cost.yref = np.zeros((self.nx + self.nu,))
        ocp.cost.yref_e = np.zeros((self.nx,))
        """
        # set constraints
        ocp.constraints.x0 = np.zeros((self.nx*2,))
        ocp.constraints.idxbu = np.arange(self.nu)
        ocp.constraints.lbu = self.model.u_lb
        ocp.constraints.ubu = self.model.u_ub


        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.print_level = 0
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.nlp_solver_tol_stat = 1e-6
        ocp.solver_options.nlp_solver_tol_eq   = 1e-6
        ocp.solver_options.nlp_solver_tol_ineq = 1e-6
        ocp.solver_options.nlp_solver_max_iter = 50
        
        ocp.solver_options.qp_solver_tol_stat = 1e-8
        ocp.solver_options.qp_solver_tol_eq   = 1e-8
        ocp.solver_options.qp_solver_tol_ineq = 1e-8
        ocp.solver_options.qp_solver_iter_max = 200

        # regularization helps numeric stability
        ocp.solver_options.levenberg_marquardt = 1e-4
        solver = AcadosOcpSolver(ocp, json_file='inter_rate_mpc.json')
        return solver
    
    def solve(self, x0, setpoints=None,
              weights={'Q': None, 'Q_e': None, 'R': None},
              initial_guess={'X': None, 'U': None},
              xobj=None,
              logger=None, verbose=False, v_des = None):
        t0 = time.time()

        x_0 = np.concatenate((x0.ravel(), xobj.ravel()))  # initial state for both spacecraft
        assert x_0.size == self.nx*2
        self.solver.set(0, "lbx", x_0)
        self.solver.set(0, "ubx", x_0)

        xref = np.hstack(setpoints)
        for k in range(self.N+1):
            yref = xref[:, k].ravel()
            v_des = v_des.ravel()
            p_stacked = np.concatenate((yref, v_des))

            self.solver.set(k, "p", p_stacked)
        
        
        # set initial guess if we are getting any
        if initial_guess['X'] is not None:
            for k in range(self.N+1):
                self.solver.set(k, "x", initial_guess['X'][:, k])    # guessed states

        if initial_guess['U'] is not None:
            for k in range(self.N):
                self.solver.set(k, "u", initial_guess['U'][:, k])    # guessed controls

        # set setpoints parameter
        try:
            X_pred = np.zeros((self.nx*2, self.N+1))
            U_pred = np.zeros((self.nu, self.N))
            sol = self.solver.solve()
            for k in range(self.N+1):
                X_pred[:, k] = self.solver.get(k, "x")
            for k in range(self.N):
                U_pred[:, k] = self.solver.get(k, "u")

        except Exception as e:
            print(f"Optimization failed: {e}")
            X_pred = np.zeros((self.nx*2, self.N+1))
            U_pred = np.zeros((self.nu, self.N))

        return X_pred, U_pred


