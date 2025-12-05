#!/usr/bin/env python
__author__ = "Joris Verhagen"
__contact__ = "jorisv@kth.se"

import casadi as cs
import numpy as np
import copy

def skew_symmetric(v):
    """ Returns the skew symmetric matrix of a 3D vector.
    Args:
        v (cs.SX or cs.MX): 3D vector (3x1)
    Returns:
        S (cs.SX or cs.MX): skew symmetric matrix (3x3)
    """
    return cs.vertcat(cs.horzcat(0, -v[0], -v[1], -v[2]),
        cs.horzcat(v[0], 0, v[2], -v[1]),
        cs.horzcat(v[1], -v[2], 0, v[0]),
        cs.horzcat(v[2], v[1], -v[0], 0))

def q_to_rot_mat(q):
    """ Converts a quaternion to a rotation matrix.
    Args:
        q (cs.SX or cs.MX): quaternion (4x1) [qw, qx, qy, qz]
    Returns:
        rot_mat (cs.SX or cs.MX): rotation matrix (3x3)
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    rot_mat = cs.vertcat(
        cs.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
        cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
        cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))

    return rot_mat

def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)

    return cs.mtimes(rot_mat, v)



class SpacecraftRateModel():
    def __init__(self):
        self.name = 'spacecraft_rate_model'
        # TODO: Add selector for doubling the mass, ahve to create on MPC for the interaction and one for the standard just to change the mass
        # The rest can stay the same.
        # constants
        self.mass = 16.8
        self.object_mass = 16.8
        self.max_thrust = 1.86
        self.max_rate = 0.5

        self.u_ub = np.array([self.max_thrust,self.max_thrust,self.max_thrust,
                              self.max_rate,self.max_rate,0.5*self.max_rate])
        self.u_lb = -self.u_ub
    
    def get_casadi_dx(self,x,u, inter = False):
        """ Returns the time derivative of the state for given state and input 
        using CasADi symbols.
        Args:
            x (cs.SX or cs.MX): state vector (10x1) [px,py,pz,vx,vy,vz,qw,qx,qy,qz]
            u (cs.SX or cs.MX): input vector (6x1) [Fx,Fy,Fz,wx,wy,wz]
        Returns:
            dx (cs.SX or cs.MX): time derivative of the state (10x1)
        
        """
        if inter:
            mass = 1
        else:
            mass = copy.deepcopy(self.mass)
        p = x[0:3]
        v = x[3:6]
        q = x[6:10]

        F = u[0:3]
        w = u[3:6]

        # kinematics velocity, acceleration, quaternion rate
        dx = cs.vertcat(v,
                        v_dot_q(F,q)/mass,
                        1/2 * cs.mtimes(skew_symmetric(w),q))
        return dx

    def get_casadi_ode(self,x,u,dt, inter = False):
        dx = self.get_casadi_dx(x,u,inter)
        return x + dx*dt
    
    def get_casadi_rk4(self,x,u,dt):
        k1 = self.get_casadi_dx(x, u)
        k2 = self.get_casadi_dx(x + dt / 2 * k1, u)
        k3 = self.get_casadi_dx(x + dt / 2 * k2, u)
        k4 = self.get_casadi_dx(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
