#Timeshift.msg

#Used to propogate a timeshift in the plan from a replanner of different robot. As well as update the object plan

#Time shift in seconds
float64 time_shift

# object verboseplan
VerboseBezierPlan object_plan

#Name of the robot who replanned the time shift
string robot_name