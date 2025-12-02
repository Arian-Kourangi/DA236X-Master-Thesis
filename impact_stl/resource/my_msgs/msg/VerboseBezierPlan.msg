# BezierPlan.msg
uint64 timestamp

# an array of BezierCurve messages
BezierCurve[] rvar
BezierCurve[] drvar

BezierCurve[] hvar
BezierCurve[] dhvar

# an array of strings to send the ids (none, pre, or post) 
string[] ids
# an array of strings to send the other agent associated with pre or post
string[] other_names
