#!/usr/bin/env python
__author__ = "Joris Verhagen"
__contact__ = "jorisv@kth.se"

import numpy as np
import casadi as cs
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import time

#from impact_stl.models.spacecraft_rate_model import SpacecraftRateModel

class SpacecraftRateMPC():
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

        
        self.solver = self.setup()

    def setup(self):
        # State and input variables (over the whole horizon)
        x = cs.SX.sym('x',self.nx)
        u = cs.SX.sym('u',self.nu)
        xdot = cs.SX.sym('xdot', self.nx) 
        s = cs.SX.sym('s') # selector for switching between two sets of dynamics

        f_normal = self.model.get_casadi_ode(x, u, self.dt)
        f_alt    = self.model.get_casadi_ode(x, u, self.dt, True)

        f_expl = f_normal * (1 - s) + f_alt * s
        f_impl = xdot - f_expl

        model_ac = AcadosModel()
        model_ac.f_expl_expr = f_expl
        model_ac.f_impl_expr = f_impl
        model_ac.x = x
        model_ac.u = u
        model_ac.xdot = xdot
        model_ac.name = 'spacecraft_rate_model_acados'
        model_ac.p = s

        ocp = AcadosOcp()
        ocp.parameter_values = np.zeros(1)
        ocp.model = model_ac
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.Tf

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

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

        # set constraints
        ocp.constraints.x0 = np.zeros((self.nx,))
        ocp.constraints.idxbu = np.arange(self.nu)
        ocp.constraints.lbu = self.model.u_lb
        ocp.constraints.ubu = self.model.u_ub


        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
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
        ocp.solver_options.levenberg_marquardt = 1e-6
        solver = AcadosOcpSolver(ocp, json_file='simple_rate_mpc.json')
        return solver
    
    def solve(self, x0, setpoints=None,
              weights={'Q': None, 'Q_e': None, 'R': None},
              initial_guess={'X': None, 'U': None},
              xobj=None, enable_cbf=True,
              logger=None, verbose=False, selectors=None):
        t0 = time.time()

        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        xref = np.hstack(setpoints)
        for k in range(self.N):
            yref = np.concatenate([xref[:, k], np.zeros(self.nu)])  # we only penalize state deviation
            self.solver.set(k, "yref", yref)

        # set terminal cost reference
        yref_e = xref[:, self.N]
        self.solver.set(self.N, "y_ref", yref_e)

        # set weights if provided
        
        
        # set initial guess if we are getting any
        if initial_guess['X'] is not None:
            for k in range(self.N+1):
                self.solver.set(k, "x", initial_guess['X'][:, k])    # guessed states

        if initial_guess['U'] is not None:
            for k in range(self.N):
                self.solver.set(k, "u", initial_guess['U'][:, k])    # guessed controls

        #set selectors for a setpoint being on the an inter curve or not
        if selectors is None:
            selectors = np.zeros((self.N+1))
        
        for k in range(self.N):
            self.solver.set(k, "p", np.array([selectors[k]]))

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


