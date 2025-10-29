#!/usr/bin/env python
__author__ = "Joris Verhagen"
__contact__ = "jorisv@kth.se"

import numpy as np
import casadi as cs
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import time

#from impact_stl.models.spacecraft_rate_model import SpacecraftRateModel

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
        self.Q = np.diag([5e1, 5e1, 5e1, 5e1, 5e1, 5e1, 8e3, 8e3, 8e3, 8e3])
        self.Q_e = 10 * self.Q
        self.R = 2*np.diag([1e-2, 1e-2, 1e-2, 2e0, 2e0, 2e0])
        self.pos_w = np.diag([1e5, 1e5, 1e5])
        self.acc_w = 1e5
        
        self.solver = self.setup()

    def setup(self):
        # State and input variables (over the whole horizon)
        x = cs.SX.sym('x',self.nx)
        u = cs.SX.sym('u',self.nu)
        xdot = cs.SX.sym('xdot', self.nx) 

        f_alt= self.model.get_casadi_dx(x, u, True)

        f_expl =f_alt 
        f_impl = xdot - f_expl

        model_ac = AcadosModel()
        model_ac.f_expl_expr = f_expl
        model_ac.f_impl_expr = f_impl
        model_ac.x = x
        model_ac.u = u
        model_ac.xdot = xdot
        model_ac.name = 'spacecraft_inter_model_acados'
        yref = cs.SX.sym('yref', self.nx)
        x_obj = cs.SX.sym('x_obj', self.nx)
        v_des = cs.SX.sym('v_des', 3)
        model_ac.p = cs.vertcat(yref, x_obj, v_des)

        error = x - yref
        
        delta_V = v_des- x_obj[3:6]
        u_push = delta_V / (cs.norm_2(delta_V))
        des_pos = x_obj[0:3] - (self.r_robot + self.r_object) * u_push
        
        pos_error = x[0:3] - des_pos
        pos_cost = cs.mtimes([pos_error.T, self.pos_w, pos_error])

        acc = self.model.get_casadi_dx(x, u, True)[3:6]

        acc_cost = self.acc_w * cs.sumsqr((cs.DM.eye(3)- u_push@u_push.T) @ acc)

        #_new = self.R.copy()
        #_new[3:,3:] = 100 * R_new[3:,3:] # Don't want it to rotate during interaction
        model_ac.cost_expr_ext_cost_0 =  cs.mtimes([u.T, self.R, u])  + cs.mtimes([error.T, self.Q, error]) #pos_cost + acc_cost +
        model_ac.cost_expr_ext_cost = cs.mtimes([error.T, self.Q, error]) + cs.mtimes([u.T, self.R, u]) 
        model_ac.cost_expr_ext_cost_e = cs.mtimes([error.T, self.Q_e, error])


        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.Tf
        ocp.parameter_values = np.zeros(ocp.model.p.size()[0])
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

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
        ocp.constraints.x0 = np.zeros((self.nx,))
        ocp.constraints.idxbu = np.arange(self.nu)
        ocp.constraints.lbu = self.model.u_lb
        ocp.constraints.ubu = self.model.u_ub


        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.print_level = 0
        #ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.nlp_solver_tol_stat = 1e-6
        ocp.solver_options.nlp_solver_tol_eq   = 1e-6
        ocp.solver_options.nlp_solver_tol_ineq = 1e-6
        ocp.solver_options.nlp_solver_max_iter = 50
        
        ocp.solver_options.qp_solver_tol_stat = 1e-8
        ocp.solver_options.qp_solver_tol_eq   = 1e-8
        ocp.solver_options.qp_solver_tol_ineq = 1e-8
        ocp.solver_options.qp_solver_iter_max = 200

        # regularization helps numeric stability
        ocp.solver_options.levenberg_marquardt = 1e-6
        solver = AcadosOcpSolver(ocp, json_file='inter_rate_mpc.json')
        return solver
    
    def solve(self, x0, setpoints=None,
              weights={'Q': None, 'Q_e': None, 'R': None},
              initial_guess={'X': None, 'U': None},
              xobj=None,
              logger=None, verbose=False, v_des = None):
        t0 = time.time()

        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        xref = np.hstack(setpoints)
        for k in range(self.N+1):
            yref = xref[:, k].ravel()
            x_obj = xobj.ravel()
            v_des = v_des.ravel()
            p_stacked = np.concatenate((yref, x_obj, v_des))

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
            X_pred = np.zeros((self.nx, self.N+1))
            U_pred = np.zeros((self.nu, self.N))
            sol = self.solver.solve()
            for k in range(self.N+1):
                X_pred[:, k] = self.solver.get(k, "x")
            for k in range(self.N):
                U_pred[:, k] = self.solver.get(k, "u")

        except Exception as e:
            print(f"Optimization failed: {e}")
            X_pred = np.zeros((self.nx, self.N+1))
            U_pred = np.zeros((self.nu, self.N))

        return X_pred, U_pred


