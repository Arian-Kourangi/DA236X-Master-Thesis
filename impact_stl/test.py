import numpy as np
import casadi as cs

x = [0]*10
x[0] = 1
x[1] = 1
x[9] = 1
print(x)
N = 10

ocp = cs.Opti()
S = ocp.parameter(N)
ocp.set_value(S,x)

print(ocp.debug.value(S))