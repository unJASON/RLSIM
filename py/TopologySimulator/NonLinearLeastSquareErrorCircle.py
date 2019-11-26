import numpy as np
from scipy.optimize import leastsq
from math import sqrt

def func(i):
    x,y = i
    return np.asarray((
        (xa-x)**2+(ya-y)**2-da**2,
        (xb-x)**2+(yb-y)**2-db**2,
        (xc-x)**2+(yc-y)**2-dc**2)
    )
if __name__ == '__main__':
    xa = 0
    ya = 0
    da = 3.68
    xb = -0.6
    yb = -0.88
    db = 3.21
    xc = 0.97
    yc = 1.28
    dc = 1.77
    xd = 0
    yd = 0
    dd = 3
    triples = [(xa, ya, da), (xb, yb, db), (xc, yc, dc)]
    root = leastsq(func,np.asarray( ( (xa+xb+xc)/3,(ya+yb+yc)/3) ) )
    print(root[0])