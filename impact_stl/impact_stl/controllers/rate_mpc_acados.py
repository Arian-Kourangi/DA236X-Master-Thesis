#!/usr/bin/env python
__author__ = "Arian Kourangi"
__contact__ = "arianke@kth.se"

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
        self.nu_phys = self.nu   # keep physical input size
        if self.add_cbf:
            #Add another input for slack variable
            self.nu = self.nu_phys + 1 

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
        
        if self.add_cbf:
            p_r = cs.SX.sym('p_r', 3)
            v_r = cs.SX.sym('v_r', 3)
            q_r = cs.SX.sym('q_r', 4)
            u_r = cs.SX.sym('u_r', self.nu_phys)
            p_o = cs.SX.sym('p_o', 3)
            v_o = cs.SX.sym('v_o', 3)
            q_o = cs.SX.sym('q_o', 4)
            u_o = cs.SX.sym('u_o', self.nu_phys)

            h = cs.sumsqr(p_r[0:2] - p_o[0:2]) - (0.2 + 0.2 + 0.05)**2
            x = cs.vertcat(p_r, p_o)
            dx = cs.vertcat(v_r, v_o)

            X_r = cs.vertcat(p_r, v_r, q_r)
            X_o = cs.vertcat(p_o, v_o, q_o)

            dh = cs.jacobian(h, x) @ dx
            ddh = cs.jacobian(dh, x) @ dx + cs.jacobian(dh, dx) @ cs.vertcat(
                self.model.get_casadi_dx(X_r, u_r)[3:6],
                self.model.get_casadi_dx(X_o, u_o)[3:6]
            )

            self.h = cs.Function('h', [X_r, X_o], [h])
            self.dh = cs.Function('dh', [X_r, X_o], [dh])
            self.ddh = cs.Function('ddh', [X_r, X_o, u_r, u_o], [ddh])
            self.alpha = 2.0
            self.beta = 1.0
        
        self.solver = self.setup()

    def setup(self):
        # State and input variables (over the whole horizon)
        x = cs.SX.sym('x',self.nx)
        u = cs.SX.sym('u',self.nu)
        xdot = cs.SX.sym('xdot', self.nx) 
        s = cs.SX.sym('s') # selector for switching between two sets of dynamics

        # split u into physics and slack
        u_phys = u[:self.nu_phys]               # first 6 elements
        u_delta = u[self.nu_phys] if self.add_cbf else None               # slack delta (scalar)


        f_normal = self.model.get_casadi_dx(x, u_phys)
        f_alt    = self.model.get_casadi_dx(x, u_phys, True)
        f_expl = f_normal * (1 - s) + f_alt * s
        f_impl = xdot - f_expl

        model_ac = AcadosModel()
        model_ac.f_expl_expr = f_expl
        model_ac.f_impl_expr = f_impl
        model_ac.x = x
        model_ac.u = u
        model_ac.xdot = xdot
        model_ac.name = 'spacecraft_rate_model_acados'

        if self.add_cbf:
            # symbolic parameter placeholders for object state/input and OffSwitch
            X_o = cs.SX.sym('X_o', self.nx)
            U_o = cs.SX.sym('U_o', self.nu_phys)
            OffSwitch = cs.SX.sym('OffSwitch')

            # CBF expression
            cbf_stage = self.ddh(x, X_o, u_phys, U_o)[0] \
                        + self.alpha * self.dh(x, X_o)[0] \
                        + self.beta * self.h(x, X_o)[0] \
                        + u_delta + OffSwitch   # δ enters additively

            # attach to model as a path constraint expression (h >= 0)
            model_ac.con_h_expr = cbf_stage
            # allow no terminal constraint:
            model_ac.con_h_expr_e = cs.SX.zeros(0)

            # append the new parameters to model_ac.p
            # order: s, X_o, U_o, OffSwitch
            y_ref = cs.SX.sym('yref', self.nx)
            model_ac.p = cs.vertcat(s, X_o, U_o, OffSwitch, y_ref)
        else:
            y_ref = cs.SX.sym('yref', self.nx)
            model_ac.p = cs.vertcat(s,y_ref)
        
        error = x - y_ref
        cost_p = cs.mtimes([error[0:6].T, self.Q[0:6,0:6], error[0:6]])
        q = x[6:10].reshape((4,1))
        q_ref = y_ref[6:10].reshape((4,1))
        
        eq = 1 - (q.T @ q_ref)**2 
        cost_eq = eq.T @ self.Q[6,6].reshape((1, 1)) @ eq

        cost_u = cs.mtimes([u_phys.T, self.R, u_phys])
        if self.add_cbf:
            cost_delta = 1e2 * u_delta  # penalize slack
        else:
            cost_delta = 0.0
        
        #model_ac.cost_expr_ext_cost_0 =  cost_u + cost_p + cost_delta
        model_ac.cost_expr_ext_cost = cost_p + cost_u + cost_eq + cost_delta
        model_ac.cost_expr_ext_cost_e = cs.mtimes([error[0:6].T, self.Q_e[0:6,0:6], error[0:6]]) + eq.T @ self.Q_e[6,6].reshape((1, 1)) @ eq
        
        ocp = AcadosOcp()
        ocp.model = model_ac
        if self.add_cbf:
            ocp.constraints.lh = np.array([0.0])
            ocp.constraints.uh = np.array([1e6])   # large upper bound (not very important)
        
        # initialize parameters
        ocp.parameter_values = np.zeros(ocp.model.p.size()[0])
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.Tf

        #ocp.cost.cost_type_0 = "EXTERNAL"
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        # set constraints
        ocp.constraints.x0 = np.zeros((self.nx,))

        idxbu = np.arange(self.nu)   # includes slack at the end when add_cbf True
        ocp.constraints.idxbu = idxbu

        lbu = np.zeros((self.nu,))
        ubu = np.zeros((self.nu,))

        # set physical bounds on first nu_phys elements:
        lbu[:self.nu_phys] = self.model.u_lb.ravel()
        ubu[:self.nu_phys] = self.model.u_ub.ravel()

        if self.add_cbf:
            # slack delta bound: delta >= 0, and give reasonable upper bound (avoid extreme values)
            lbu[self.nu_phys] = 0.0
            ubu[self.nu_phys] = 1e6  # finite upper bound; small enough to keep numerics sane
        else:
            # if no cbf, there is no extra slack dimension and we already set bounds above
            pass
        
        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu

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
        solver = AcadosOcpSolver(ocp, json_file='simple_rate_mpc.json')
        return solver
    
    def solve(self, x0, setpoints=None,
              weights={'Q': None, 'Q_e': None, 'R': None},
              initial_guess={'X': None, 'U': None},
              xobj=None,
              logger=None, verbose=False, selectors=None):
        t0 = time.time()

        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
           
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
        
        xref = np.hstack(setpoints)     
        if not self.add_cbf:
            # If no CBF our parameter vector is just the selector
            for k in range(self.N+1):
                s = np.array([selectors[k]]).ravel()
                yref = xref[:, k].ravel()
                p_stacked = np.concatenate((s, yref))
                self.solver.set(k, "p", p_stacked)
        
        elif xobj is not None and self.add_cbf:
            X_o = xobj  # object state
            U_o = np.zeros((self.nu_phys,)) # assume object is not actuated
            
            for k in range(self.N+1):
                # Check if setpoint is on inter curve or not, if not we turn On the CBF constraint (offswitch =0)
                sel = np.array([selectors[k]]).ravel()
                if sel[0] == 0:
                    OffSwitch = 0.0
                else:
                    OffSwitch = 10000.0  # large value to turn off the CBF constraint
                # Make sure everything is 1D
                X_o_flat = np.array(X_o).ravel()
                U_o_flat = np.array(U_o).ravel()
                off = np.array([OffSwitch]).ravel()
                yref = xref[:, k].ravel()
                p_vec = np.concatenate((sel, X_o_flat, U_o_flat, off, yref))
                self.solver.set(k, "p", p_vec)
            
            # Only set the cbf for the first 1 stages, the rest we relax. (lh and uh are set for all stages at setup)
            for k in range(2,self.N):
                self.solver.constraints_set(k, "lh", np.array([-1e6]))
                self.solver.constraints_set(k, "uh", np.array([1e6]))
               
        # set setpoints parameter
        try:
            X_pred = np.zeros((self.nx, self.N+1))
            U_pred = np.zeros((self.nu, self.N))
            sol = self.solver.solve()
            for k in range(self.N+1):
                X_pred[:, k] = self.solver.get(k, "x")
            for k in range(self.N):
                U_pred[:, k] = self.solver.get(k, "u") # including slack if any

        except Exception as e:
            print(f"Optimization failed: {e}")
            X_pred = np.zeros((self.nx, self.N+1))
            U_pred = np.zeros((self.nu, self.N))

        return X_pred, U_pred


