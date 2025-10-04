import casadi as ca
from planner.utilities.beziers import get_derivative_control_points_gurobi, eval_bezier, value_bezier
from planner.utilities.read_write_plan import csv_to_plan, plan_to_csv


rvars,hvars,idvars,other_names = csv_to_plan(robot_name='crackle',
                                                                 scenario_name='minimal_test_throw_and_catch',
                                                                 path='/home/px4space/space_ws/src/impact_stl/planner/plans')  
print("rvars:", type(rvars), len(rvars))
print("hvars:", type(hvars), len(hvars))
print("idvars:", type(idvars), len(idvars))
print("other_names:", type(other_names), len(other_names))

print("rvar[0]:", type(rvars[0]), rvars[0].shape)