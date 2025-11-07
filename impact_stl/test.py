import numpy as np
import casadi as cs

#x = [0]*10
#x[0] = 1
#x[1] = 1
#x[9] = 1
#print(x)
#N = 10
#
#ocp = cs.Opti()
#S = ocp.parameter(N)
#ocp.set_value(S,x)
#
#print(ocp.debug.value(S))

#x = [0]*10
#x[0] = 1
#x[1] = 1
#
#print(x)
#if any(x[i]==1 for i in range(len(x)-1)) and x[-1]==0:
#    print("yes")
#idx = next(i for i in range(len(x)) if x[i]==0)
#print(idx)
#x = [1]*10
#x[-1] = 0
#if all(x[i]==1 for i in range(len(x))):
#    print("all ones")

x = 1/(1+ np.exp(1000*(0.40-0.45)))
print(x)
y = 0.5 * (1 - cs.tanh(0.5 * 1000*(0.40-0.45)))
print(y)