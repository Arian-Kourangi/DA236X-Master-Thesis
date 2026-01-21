#!/usr/bin/env python
__author__ = "Arian Kourangi"
__contact__ = "arianke@kth.se"

import numpy as np
import casadi as cs
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import time

#from inter_stl.models.spacecraft_rate_model import SpacecraftRateModel

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
        
        self.nu_phys = 6   # keep physical input size

        #Add another input for slack variable
        self.nu = self.nu_phys + 2 

        self.r_robot = 0.21
        self.r_object = 0.21
        # cost matrices (SITL)
        # self.Q = np.diag([10e0, 10e0, 10e0, 10e-1, 10e-1, 10e-1, 8e-1])
        # self.Q_e = 10 * self.Q
        # self.R = 2*np.diag([1e-2] * 6)

        # cost matrices (HW)
        # self.Q = np.diag([10e1, 10e1, 10e1, 10e-1, 10e-1, 10e-1, 8e0])
        # self.Q_e = 10 * self.Q
        # self.R = np.diag([1e-3] * 6)

        # px4-mpc
        # To test the normal MPC without the reactive part, double the weights 
        #self.Q = 10*np.diag([5e1, 5e1, 5e1, 5e1, 5e1, 5e1, 8e2, 8e2, 8e2, 8e2])
        self.Q = np.diag([5e1, 5e1, 5e1, 5e1, 5e1, 5e1, 8e2, 8e2, 8e2, 8e2])
        self.Q_e = 10 * self.Q
        self.R = 2*np.diag([1e-2, 1e-2, 1e-2, 2e0, 2e0, 2e0])
        

        p_r = cs.SX.sym('p_r', 3)
        v_r = cs.SX.sym('v_r', 3)
        q_r = cs.SX.sym('q_r', 4)
        f_r = cs.SX.sym('u_r', self.nx)
        p_o = cs.SX.sym('p_o', 3)
        v_o = cs.SX.sym('v_o', 3)
        q_o = cs.SX.sym('q_o', 4)
        f_o = cs.SX.sym('u_o', self.nx)
        dist = cs.SX.sym('dist')
        h = cs.sumsqr(p_r[0:2] - p_o[0:2]) - (self.r_object + self.r_robot + dist)**2
        x = cs.vertcat(p_r, p_o)
        dx = cs.vertcat(v_r, v_o)
        X_r = cs.vertcat(p_r, v_r, q_r)
        X_o = cs.vertcat(p_o, v_o, q_o)
        dh = cs.jacobian(h, x) @ dx
        ddh = cs.jacobian(dh, x) @ dx + cs.jacobian(dh, dx) @ cs.vertcat(
            f_r[3:6],
            f_o[3:6]
        )
        self.h = cs.Function('h', [X_r, X_o, dist], [h])
        self.dh = cs.Function('dh', [X_r, X_o, dist], [dh])
        self.ddh = cs.Function('ddh', [X_r, X_o, f_r, f_o, dist], [ddh])
        self.alpha = 2
        self.beta = 2
        
        self.solver = self.setup()

    def setup(self):
        # State and input variables (over the whole horizon)
        x = cs.SX.sym('x',self.nx*2)
        u = cs.SX.sym('u',self.nu)
        xdot = cs.SX.sym('xdot', self.nx*2) 
        s = cs.SX.sym('s') # selector for switching between two sets of dynamics

        # split u into physics and slack
        u_phys = u[:self.nu_phys]               # first 6 elements
        u_delta = u[self.nu_phys]              # slack delta (scalar)
        u_delta2 = u[self.nu_phys + 1]              # slack delta for aggressive CBF (scalar)

        x_robot = x[0:self.nx]
        x_object = x[self.nx:self.nx*2]


        # With inter = True this now outputs the force instead of acceleration
        f_robot_t = self.model.get_casadi_dx(x_robot, u_phys, inter=True)

        dist = x_object[0:3] - x_robot[0:3]  

        dist_norm = cs.sqrt(cs.dot(dist, dist)+ 1e-15)
        contact_norm = dist / dist_norm

        k = 100 # Gain for sigmoid
        delta = dist_norm - (self.r_robot + self.r_object + 0.005)
        switch = k * delta  # no need to clamp
        # acitvate can only be on when we are on interaction curve
        activate = s * self.safe_sigmoid(switch)  # smooth activation between 0 and 1

        # projection of robot force in contact normal direction, only when in contact
        proj_scalar = activate*cs.dot(contact_norm, f_robot_t[3:6])   # 1x1
        
        # only consider positive projections (pushing into the object), negative projections set to zero
        proj_plus = 0.5 * (proj_scalar + cs.sqrt(proj_scalar**2 + 1e-15))
        F_contact = contact_norm* proj_plus



        f_object = cs.vertcat(
            x_object[3:6],
            F_contact/self.model.mass,
            cs.SX.zeros(4)
        )
        f_robot = cs.vertcat(
            f_robot_t[0:3],
            (f_robot_t[3:6] - F_contact)/self.model.mass,
            f_robot_t[6:10]
        )

        f_alt = cs.vertcat(f_robot, f_object)

        f_expl = f_alt
        f_impl = xdot - f_expl

        model_ac = AcadosModel()
        model_ac.f_expl_expr = f_expl
        model_ac.f_impl_expr = f_impl
        model_ac.x = x
        model_ac.u = u
        model_ac.xdot = xdot
        model_ac.name = 'spacecraft_rate_model_acados'

        #CBF setup, is only active when s = 1

        # symbolic parameter placeholders for object state/input and OffSwitch
        # CBF expression
        cbf_stage = self.ddh(x_robot,x_object, f_robot, f_object,0.03)[0] \
                    + self.alpha * self.dh(x_robot, x_object,0.03)[0] \
                    + self.beta * self.h(x_robot, x_object,0.03)[0] \
                    + u_delta  # δ enters additively
        # For test1 set the distance to 1 m instead of 2 # 0.1 for hwtest3
        cbf_stage_aggresive = self.ddh(x_robot,x_object, f_robot, f_object,1)[0] \
                    + self.alpha * self.dh(x_robot, x_object,1)[0] \
                    + self.beta * self.h(x_robot, x_object,1)[0] \
                    + u_delta2

        model_ac.con_h_expr_0 = cs.vertcat(cbf_stage, cbf_stage_aggresive)


        v_des = cs.SX.sym('v_des', 3)
        delta_V = cs.SX.sym('delta_V', 3)
        s_k = cs.SX.sym('s_k')  # selector for if its straight pushing or not
        s_z = cs.SX.sym('s_z')  # selector for if end of interaction or not
        xref_r = cs.SX.sym('yref', self.nx)
        model_ac.p = cs.vertcat(s, xref_r, delta_V, v_des, s_k, s_z)


        
        des_norm = delta_V / (cs.sqrt(cs.dot(delta_V, delta_V)) + 1e-15)
        d_c = self.r_robot + self.r_object

        xref_o = cs.vertcat( xref_r[0:3] + des_norm * d_c,
                                     xref_r[3:10] )
        

        # Robot error
        e_robot = x_robot - xref_r
        cost_p_r =  cs.mtimes([e_robot[0:6].T, self.Q[0:6,0:6], e_robot[0:6]])
        
        q_r = x_robot[6:10] / cs.sqrt(cs.sumsqr(x_robot[6:10]) + 1e-15)
        qref_r = xref_r[6:10] / cs.sqrt(cs.sumsqr(xref_r[6:10]) + 1e-15)

        q_error = self.quat_mult(qref_r, cs.vertcat(q_r[0], -q_r[1], -q_r[2], -q_r[3]))
        q_error = q_error * cs.sign(q_error[0])  # shortest rotation
        q_error = q_error[1:4]  # vector part
        cost_eq_r = cs.mtimes([q_error.T, self.Q[6:9,6:9], q_error])


        #eq_r = 1 - (q_r.T @ qref_r)**2 
        #cost_eq_r = eq_r.T @ self.Q[6,6].reshape((1, 1)) @ eq_r

        # Object error
        e_object = x_object - xref_o
        cost_p_o =  cs.mtimes([e_object[0:6].T, 25*self.Q[0:6,0:6], e_object[0:6]])
        q_o = x_object[6:10]
        qref_o = xref_o[6:10]
        eq_o = 1 - (q_o.T @ qref_o)**2
        cost_eq_o = eq_o.T @ self.Q[6,6].reshape((1, 1)) @ eq_o


        cost_p_r_e = cs.mtimes([e_robot[0:6].T, self.Q_e[0:6,0:6], e_robot[0:6]])
        cost_eq_r_e = cs.mtimes([q_error.T, self.Q_e[6:9,6:9], q_error])
        #cost_eq_r_e = eq_r.T @ self.Q_e[6,6].reshape((1, 1)) @ eq_r
        cost_p_o_e = cs.mtimes([e_object[0:6].T,self.Q_e[0:6,0:6], e_object[0:6]])
        cost_eq_o_e = eq_o.T @ self.Q_e[6,6].reshape((1, 1)) @ eq_o
        # Control error
        cost_u = cs.mtimes([u_phys.T, self.R, u_phys])
        # Slack error only when s = 0
        cost_delta = 1e4 * u_delta  # penalize slack
        cost_delta += 1e4 * u_delta2  # penalize slack

        # Tangent acceleration cost, only active when s = 1
        tangent = cs.SX.eye(3) - cs.mtimes(contact_norm, contact_norm.T)
        # Tangen cost is only active if we are close to tne object, otherwise go ahead and use tangential forces so you can move around
        tangent_cost= (activate)*1e-1 * cs.dot(cs.mtimes(tangent, f_robot_t[3:6]), cs.mtimes(tangent, f_robot_t[3:6])) 
        
        tangent_cost = 2*s_k * tangent_cost  + (1 - s_k) * tangent_cost * 7

        # non-colinear terminal velocity cost, only used for straight pushes
        tmp = cs.cross(x_object[3:6], v_des)
        v_cost_e = s_z*1e5*cs.dot(tmp, tmp)

        model_ac.cost_expr_ext_cost = cost_p_r + cost_eq_r \
            + cost_u + s*cost_p_o + s*cost_eq_o + s*tangent_cost + (1-s)*cost_delta
        model_ac.cost_expr_ext_cost_e = (1-s)*cost_p_r_e + s*cost_p_r_e*(s_k*0.2 + (1-s_k)*0.0) + cost_eq_r_e + s*cost_p_o_e + s*cost_eq_o_e + s*v_cost_e

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
        ocp.constraints.x0 = np.zeros((self.nx*2,))

        idxbu = np.arange(self.nu) 
        ocp.constraints.idxbu = idxbu

        lbu = np.zeros((self.nu,))
        ubu = np.zeros((self.nu,))

        # set physical bounds on first nu_phys elements:
        lbu[:self.nu_phys] = self.model.u_lb.ravel()
        ubu[:self.nu_phys] = self.model.u_ub.ravel()

        # slack delta bound: delta >= 0
        lbu[self.nu_phys] = 0.0
        ubu[self.nu_phys] = 1e2  # finite upper bound; small enough to keep numerics sane
        lbu[self.nu_phys + 1] = 0.0
        ubu[self.nu_phys + 1] = 1e2  # finite upper bound; small enough to keep numerics sane
        
        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu

        ### Turn off from here if funky
        # idxbx = np.array([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15])
        # ocp.constraints.idxbx = idxbx
        # lbx = np.zeros((12,))
        # ubx = np.zeros((12,))
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
        
        # lbx[6] = 0.0
        # ubx[6] = 3.5
        # lbx[6+1] = -1.75
        # ubx[6+1] = 1.75
        # lbx[6+2] = -1e2
        # ubx[6+2] = 1e2
        # lbx[6+3] = -0.3
        # ubx[6+3] = 0.3
        # lbx[6+4] = -0.3
        # ubx[6+4] = 0.3
        # lbx[6+5] = -0.3
        # ubx[6+5] = 0.3
        # ocp.constraints.lbx = lbx
        # ocp.constraints.ubx = ubx

        #### 

       
        # Setting the CBF / interaction constraint bounds

        ocp.constraints.lh_0 = np.array([-1e6,-1e6])
        ocp.constraints.uh_0 = np.array([1e6,1e6])

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
        # MPC A folder
        dir = os.path.join(base_dir, "MPC_A")
        os.makedirs(dir, exist_ok=True)
        ocp.code_export_directory = dir
        solver = AcadosOcpSolver(ocp, json_file='simple_rate_mpc.json', build =True)
        return solver
    
    def solve(self, x0, setpoints=None,
              weights={'Q': None, 'Q_e': None, 'R': None},
              initial_guess={'X': None, 'U': None},
              xobj=None,
              logger=None, verbose=False, selectors=None, delta_V = None, v_des=None, straight = None, col_avoid = None, approach = None, end_of_int = False):
        t0 = time.time()
        
        x_0 = np.concatenate((x0.ravel(), xobj.ravel()))  # initial state for both spacecraft
        assert x_0.size == self.nx*2
        self.solver.set(0, "lbx", x_0)
        self.solver.set(0, "ubx", x_0)

        # set initial guess if we are getting any
        if initial_guess['X'] is not None:
            for k in range(self.N+1):
                self.solver.set(k, "x", initial_guess['X'][:, k])    # guessed states

        if initial_guess['U'] is not None:
            for k in range(self.N):
                self.solver.set(k, "u", initial_guess['U'][:, k])    # guessed controls

        #set selectors for a setpoint being on the an inter curve or not
        if selectors is None: # We haven't started yet, and the object and robot might be on top of eachother so don't enforce any constraint
            selectors = np.zeros((self.N+1))
        else:
            if all(k == 0 for k in selectors):
                if col_avoid:
                    #Enforce aggressive CBF constraint to avoid collision
                    self.solver.constraints_set(0, "lh", np.array([-1e6,0]))
                    self.solver.constraints_set(0, "uh", np.array([1e6, 1e6]))
                elif approach:
                    self.solver.constraints_set(0, "lh", np.array([0, -1e6]))
                    self.solver.constraints_set(0, "uh", np.array([1e6, 1e6]))
                else:
                    # No interaction, no CBF
                    self.solver.constraints_set(0, "lh", np.array([-1e6, -1e6]))
                    self.solver.constraints_set(0, "uh", np.array([1e6, 1e6]))
            else:
                # Interaction step, enforce interaction constraint and not CBF
                self.solver.constraints_set(0, "lh", np.array([-1e6, -1e6]))
                self.solver.constraints_set(0, "uh", np.array([1e6, 1e6]))

        xref = np.hstack(setpoints)     

        # Constants
        v_des = v_des.ravel()
        delta_V = delta_V.ravel()
        if end_of_int:
            s_z = 1
        else:
            s_z = 0
        if straight:
            s_k = 1
            s_z = 1
        else:
            s_k = 0
        s_k = np.array([s_k]).ravel()
        s_z = np.array([s_z]).ravel()
        
        for k in range(self.N+1):
            s = np.array([selectors[k]]).ravel()
            # To turn off the reactive MPC, use s = 0 for all k
            #s = np.array([0.0]).ravel()
            yref = xref[:, k].ravel()
            p_stacked = np.concatenate((s, yref, delta_V, v_des, s_k,s_z))
            self.solver.set(k, "p", p_stacked)

               
        # set setpoints parameter
        try:
            X_pred = np.zeros((self.nx*2, self.N+1))
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
            X_pred = np.zeros((self.nx*2, self.N+1))
            U_pred = np.zeros((self.nu, self.N))

        return X_pred, U_pred, self.solver.get_status()

    def safe_sigmoid(self,x):
        return 0.5 * (1 - cs.tanh(0.5 * x))
    def quat_mult(self, q1, q2):
        return cs.vertcat(
                q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
                q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
                q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
                q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
            )