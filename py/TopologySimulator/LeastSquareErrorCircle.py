import scipy.linalg as la
import numpy as np


xa = 0
ya = 0
da = 3.68
xb = -0.6
yb = -0.88
db = 3.21
xc = 0.97
yc = 1.28
dc = 1.77

matrix = np.array([[2*(xa-xc),2*(ya-yc)],[2*(xb-xc),2*(yb-yc)]],dtype=np.float)
ans = np.array([xa**2+ya**2-da**2-xc**2 -yc**2+dc**2,xb**2+yb**2-db**2-xc**2 -yc**2+dc**2],dtype=np.float)
sol, r, rank, s = la.lstsq(matrix, ans)
print(sol[0],sol[1])

