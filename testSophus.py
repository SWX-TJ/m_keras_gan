from sophus import *
import numpy as np

# Default SE3 group element (Identity)
T = SE3()
# Single axis factory: rotation around axis
Tx = SE3.rotX(1.3)
Ty = SE4.rotY(0.5)
# Group operators
T_prod = Tx * Ty

# Group Exponential and Log operations
t = T_prod.log()
T = SE3.exp(t)
R = T.so3()
r = R.log()
R = SO3.exp(r)

# Conversion to/from numpy
numpy_mat = T.matrix()
numpy_mat[0,3] = 2;
T = SE3(numpy_mat)
T.log()

x = np.array([1,0,0,0,0,0]).T
T = SE3.exp(x)
print(T)