#!/usr/bin/env python
__author__ = "Arian Kourangi"
__contact__ = "arianke@kth.se"

import numpy as np
import casadi as cs
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import time

#from impact_stl.models.spacecraft_rate_model import SpacecraftRateModel

class SpacecraftRateMPC():
    def __init__(self, model, Tf=1.0, N=10):
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

        self.nx = 10 # Number of state variables
        
        self.nu = 6   # keep physical input size
    
        self.Q = np.diag([5e1, 5e1, 5e1, 5e1, 5e1, 5e1, 8e2, 8e2, 8e2, 8e2])
        self.Q_e = 10 * self.Q
        self.R = 2*np.diag([1e-2, 1e-2, 1e-2, 2e0, 2e0, 2e0])       
        self.solver = self.setup()

    def setup(self):
        # State and input variables (over the whole horizon)
        x = cs.SX.sym('x',self.nx)
        u = cs.SX.sym('u',self.nu)
        xdot = cs.SX.sym('xdot', self.nx) 


        # With inter = True this now outputs the force instead of acceleration
        f_robot_t = self.model.get_casadi_dx(x, u, inter=False)

        f_expl = f_robot_t
        f_impl = xdot - f_expl

        model_ac = AcadosModel()
        model_ac.f_expl_expr = f_expl
        model_ac.f_impl_expr = f_impl
        model_ac.x = x
        model_ac.u = u
        model_ac.xdot = xdot
        model_ac.name = 'spacecraft_model'
        s = cs.SX.sym('s')  # interaction flag parameter
        xref_r = cs.SX.sym('yref', self.nx)
        p = cs.vertcat(xref_r, s)
        model_ac.p = cs.vertcat(p)

        # Robot error
        e_robot = x - xref_r
        # Just velocity cost
        cost_p_r_init =  cs.mtimes([e_robot[0:6].T, self.Q[0:6,0:6], e_robot[0:6]])
        cost_p_r =cs.mtimes([e_robot[3:6].T, self.Q[3:6,3:6], e_robot[3:6]])
        
        # q_r = x[6:10]
        # qref_r = xref_r[6:10]
        # eq_r = 1 - (q_r.T @ qref_r)**2 
        # cost_eq_r = eq_r.T @ self.Q[6,6].reshape((1, 1)) @ eq_r
    
        q_r = x[6:10] / cs.sqrt(cs.sumsqr(x[6:10]) + 1e-15)
        qref_r = xref_r[6:10] / cs.sqrt(cs.sumsqr(xref_r[6:10]) + 1e-15)

        q_error = self.quat_mult(qref_r, cs.vertcat(q_r[0], -q_r[1], -q_r[2], -q_r[3]))
        q_error = q_error * cs.sign(q_error[0])  # shortest rotation
        q_error = q_error[1:4]  # vector part
        cost_eq_r = cs.mtimes([q_error.T, self.Q[6:9,6:9], q_error])

        cost_p_r_e_init = cs.mtimes([e_robot[0:6].T, self.Q_e[0:6,0:6], e_robot[0:6]])
        cost_p_r_e = cs.mtimes([e_robot[3:6].T, self.Q_e[3:6,3:6], e_robot[3:6]])
        cost_eq_r_e = cs.mtimes([q_error.T, self.Q_e[6:9,6:9], q_error])
        #cost_eq_r_e = eq_r.T @ self.Q_e[6,6].reshape((1, 1)) @ eq_r
        # Control error
        cost_u = cs.mtimes([u.T, self.R, u])

        # non-colinear terminal velocity cost, only used for straight pushes


        model_ac.cost_expr_ext_cost = s*cost_p_r + (1-s)*cost_p_r_init \
            + cost_u + (1-s)*cost_eq_r
        
        model_ac.cost_expr_ext_cost_e = s*cost_p_r_e + (1-s)*cost_p_r_e_init + (1-s)*cost_eq_r_e

        ocp = AcadosOcp()
        ocp.model = model_ac
        # initialize parameters


        ocp.parameter_values = np.zeros(ocp.model.p.size()[0])
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.Tf

        #ocp.cost.cost_type_0 = "EXTERNAL"
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        # set constraints
        ocp.constraints.x0 = np.zeros((self.nx,))

        idxbu = np.arange(self.nu) 
        ocp.constraints.idxbu = idxbu

        lbu = np.zeros((self.nu,))
        ubu = np.zeros((self.nu,))

        # set physical bounds on first nu_phys elements:
        lbu = self.model.u_lb.ravel()
        ubu = self.model.u_ub.ravel()
        
        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu

        # Turn of from here if funky
        # idxbx = np.array([0,1,2,3,4,5])
        # ocp.constraints.idxbx = idxbx
        # lbx = np.zeros((6,))
        # ubx = np.zeros((6,))
        # lbx[0] = 0.0
        # ubx[0] = 3.5
        # lbx[1] = -1.75
        # ubx[1] = 1.75
        # lbx[2] = -1e2
        # ubx[2] = 1e2
        # lbx[3] = -0.3
        # ubx[3] = 0.3
        # lbx[4] = -0.3
        # ubx[4] = 0.3
        # lbx[5] = -0.3
        # ubx[5] = 0.3
  

        # ocp.constraints.lbx = lbx
        # ocp.constraints.ubx = ubx
        ####

        
        # set solver options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.print_level = 0
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.nlp_solver_tol_stat = 1e-6
        ocp.solver_options.nlp_solver_tol_eq   = 1e-6
        ocp.solver_options.nlp_solver_tol_ineq = 1e-6
        ocp.solver_options.nlp_solver_max_iter = 200
        
        ocp.solver_options.qp_solver_tol_stat = 1e-8
        ocp.solver_options.qp_solver_tol_eq   = 1e-8
        ocp.solver_options.qp_solver_tol_ineq = 1e-8
        ocp.solver_options.qp_solver_iter_max = 500

        # regularization helps numeric stability
        ocp.solver_options.levenberg_marquardt = 1e-4
        import os
        base_dir = os.getcwd()
        # MPC B folder
        dir = os.path.join(base_dir, "MPC_B")
        os.makedirs(dir, exist_ok=True)
        ocp.code_export_directory = dir
        solver = AcadosOcpSolver(ocp, json_file='vel_keeping_mpc.json', build =True)
        return solver
    def solve(self, x0, setpoints=None,initial_guess={'X': None, 'U': None},started=False):
        
        x_0 = x0.ravel()
        assert x_0.size == self.nx
        self.solver.set(0, "lbx", x_0)
        self.solver.set(0, "ubx", x_0)

        # set initial guess if we are getting any
        if initial_guess['X'] is not None:
            for k in range(self.N+1):
                self.solver.set(k, "x", initial_guess['X'][:, k])    # guessed states

        if initial_guess['U'] is not None:
            for k in range(self.N):
                self.solver.set(k, "u", initial_guess['U'][:, k])    # guessed controls


        xref = np.hstack(setpoints)
        if started:
            s = np.array([1])
        else:
            s = np.array([0])
        s = s.ravel()     
        for k in range(self.N+1):
            yref = xref[:, k].ravel()
            p_stacked = np.concatenate((yref, s))
            self.solver.set(k, "p", p_stacked)

               
        # set setpoints parameter
        try:
            X_pred = np.zeros((self.nx, self.N+1))
            U_pred = np.zeros((self.nu, self.N))
            sol = self.solver.solve()
            if self.solver.get_status() == 0:
                for k in range(self.N+1):
                    X_pred[:, k] = self.solver.get(k, "x")
                for k in range(self.N):
                    U_pred[:, k] = self.solver.get(k, "u") # including slack if any
            else:
                U_pred = None
                X_pred = None

        except Exception as e:
            print(f"Optimization failed: {e}")
            X_pred = np.zeros((self.nx, self.N+1))
            U_pred = np.zeros((self.nu, self.N))

        return X_pred, U_pred, self.solver.get_status()
    def quat_mult(self, q1, q2):
        return cs.vertcat(
                q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
                q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
                q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
                q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
            )