import numpy as np
import casadi as cs

x = np.zeros((10))
x[0] = 1
x[1] = 1
x[9] = 1

N = 10

ocp = cs.Opti()
S = ocp.parameter(N)
ocp.set_value(S,x)
#ocp.set_value(S, np.zeros((N,)))
print(ocp.debug.value(S))