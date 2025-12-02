#!/usr/bin/env python
import numpy as np

from rclpy.clock import Clock
from impact_stl.helpers.beziers import value_bezier, eval_t
from geometry_msgs.msg import PoseStamped


def Quaternion2Euler(q):
    # quaternion in shape [w,x,y,z]
    # euler angles in [pitch,roll,yaw]
    return np.array([
        np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2)),
        np.arcsin(2*(q[0]*q[2] - q[3]*q[1])),
        np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))
    ])

def Euler2Quaternion(euler):
    # quaternion in shape [w,x,y,z]
    # euler angles in [pitch,roll,yaw]
    #! UNTESTED
    (pitch, roll, yaw) = euler
    return np.array([
        np.cos(roll/2)*np.cos(pitch/2)*np.cos(yaw/2) + np.sin(roll/2)*np.sin(pitch/2)*np.sin(yaw/2),
        np.sin(roll/2)*np.cos(pitch/2)*np.cos(yaw/2) - np.cos(roll/2)*np.sin(pitch/2)*np.sin(yaw/2),
        np.cos(roll/2)*np.sin(pitch/2)*np.cos(yaw/2) + np.sin(roll/2)*np.cos(pitch/2)*np.sin(yaw/2),
        np.cos(roll/2)*np.cos(pitch/2)*np.sin(yaw/2) - np.sin(roll/2)*np.sin(pitch/2)*np.cos(yaw/2)
    ])

def vector2PoseMsg(frame_id, position, attitude):
    pose_msg = PoseStamped()
    pose_msg.header.stamp = Clock().now().to_msg()
    pose_msg.header.frame_id = frame_id
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.orientation.w = attitude[0]
    pose_msg.pose.orientation.x = attitude[1]
    pose_msg.pose.orientation.y = attitude[2]
    pose_msg.pose.orientation.z = attitude[3]
    pose_msg.pose.position.x = float(position[0])
    pose_msg.pose.position.y = float(position[1])
    pose_msg.pose.position.z = float(position[2])
    return pose_msg

def BezierCurve2NumpyArray(bezier_curve):
    # BezierCurve consists of an array of points. point.x, point.y, and point.z are the 
    # control point. the array gives all the control points. Pack this in a numpy array
    # print(f"bezier_curve {bezier_curve}")
    ncp = len(bezier_curve.x_cp)
    # hvar is a Bezier curve but only has values in x_cp, so we need to check if the z_cp is empty
    if len(bezier_curve.y_cp) != ncp:
        control_points = np.zeros((1,ncp))
        for i in range(ncp):
            control_points[0,i] = bezier_curve.x_cp[i]
        return control_points
    else:
        control_points = np.zeros((3,ncp))
        for i in range(ncp):
            control_points[0,i] = bezier_curve.x_cp[i]
            control_points[1,i] = bezier_curve.y_cp[i]
            control_points[2,i] = bezier_curve.z_cp[i] # orientation
        return control_points

def BezierPlan2NumpyArray(bezier_plan):
    """
    Converts a BezierPlan message to a dictionary of numpy arrays.
    bezier_plan: BezierPlan message

    returns: dictionary with keys 'rvar', 'drvar', 'hvar', 'dhvar'
    each value is a list of numpy arrays, one for each bezier curve. Each curve is of shape (d, n) where n is the number of control points
    d=3 for rvars, d=1 for hvars
    
    """

    # create a tmp function to get the control points of a bezier, for all bezier segments
    def convert_curves(curves):
        return [BezierCurve2NumpyArray(curve) for curve in curves]
    
    return {
        'rvar': convert_curves(bezier_plan.rvar),
        'drvar': convert_curves(bezier_plan.drvar),
        'hvar': convert_curves(bezier_plan.hvar),
        'dhvar': convert_curves(bezier_plan.dhvar)
    }

def VerboseBezierPlan2NumpyArray(bezier_plan):
    """
    Converts a VerboseBezierPlan message to a dictionary of numpy arrays.
    Args:
        bezier_plan: VerboseBezierPlan message

    returns: 
        dictionary with keys 'rvar', 'drvar', 'hvar', 'dhvar', 'ids', 'other_names'
        each value is a list of numpy arrays, one for each bezier curve. Each curve is of shape (d, n) where n is the number of control points
        d=3 for rvars, d=1 for hvars
    
    """
    plan = BezierPlan2NumpyArray(bezier_plan)
    plan['ids'] = bezier_plan.ids
    plan['other_names'] = bezier_plan.other_names
    return plan

def interpolate_bezier(plan, t):
    """
    Interpolates a bezier plan at time t.
    Args:
        plan: dictionary with keys 'rvar', 'drvar', 'hvar', 'dhvar', 'ids', 'other_names'
            each value is a list of numpy arrays, one for each bezier curve. Each curve is of shape (d, n) where n is the number of control points
            d=3 for rvars, d=1 for hvars
        t: time to interpolate at
    Returns:
        dictionary with keys 'q', 'dq', 'h', 'dh', 'id', 'other_name'
        q: position at time t
        dq: velocity at time t
        h: time at time t
        dh: time derivative at time t
        id: id of the bezier curve at time t
        other_name: other name of the bezier curve at time t
    """
    
    # idx i segment index, s is the local parameter in [0,1] for the bezier segment (how far along the segment we are)
    idx, s = eval_t(plan['hvar'], t)
    return {
        'q': value_bezier(plan['rvar'][idx], s),
        'dq': value_bezier(plan['drvar'][idx], s)/value_bezier(plan['dhvar'][idx], s),
        'h': value_bezier(plan['hvar'][idx], s),
        'dh': value_bezier(plan['dhvar'][idx], s),
        'id': plan['ids'][idx],
        'other_name': plan['other_names'][idx]
    }