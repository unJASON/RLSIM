import sympy
import numpy as np
from scipy.optimize import leastsq
def multiPosition(triples):
    x,y = sympy.symbols('x y')
    functions = []
    final = triples[-1]
    for idx,triple in enumerate(triples):
        if idx != len(triples) -1:
            f = 2*x*(triple[0]-final[0])+ np.square(final[0])-np.square(triple[0]) + 2*y*(triple[1]-final[1])+np.square(final[1]) - np.square(triple[1]) -(np.square(final[2]) - np.square(triple[2]))
            functions.append(f)
        else:
            pass
    result = sympy.solve(functions,[x,y])
    locx,locy = result[x],result[y]
    return [locx,locy]
def multiPosition2(i,triples):
    x, y = i
    formulation =[]
    for ele in triples:
        xi = ele[0]
        yi = ele[1]
        di = ele[2]
        formulation.append((xi-x)**2+(yi-y)**2-di**2)
    return np.asarray(formulation)

#非线性最小二乘多点定位
def nonLinearLeastSquareMultiPosition(triples):
    # print( triples.mean(axis=0) )
    # print(triples.mean(axis=1) )
    root=leastsq(multiPosition2, np.asarray((triples.mean(axis=0)[0],triples.mean(axis=0)[1])),args=(triples))
    return root[0]
if __name__ == '__main__':
    # xa = 0
    # ya = 0
    # da = 3.68
    # xb = -0.6
    # yb = -0.88
    # db = 3.21
    # xc = 0.97
    # yc = 1.28
    # dc = 1.77
    xa = 0
    ya = 0
    da = 2**0.5
    xb = 1
    yb = 0
    db = 1
    xc = 0
    yc = 1
    dc = 1

    triples = np.array([(xa,ya,da),(xb,yb,db),(xc,yc,dc)])
    print(nonLinearLeastSquareMultiPosition(triples))