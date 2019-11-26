from scipy.optimize import leastsq
import numpy as np
import sympy
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import time
from matplotlib.patches import Circle
# Kabsch W . A Solution of the Best Rotation to Relate Two Sets of Vectors[J]. Acta Crystallographica Section A Foundations of Crystallography, 1976, A32(5)
import rmsd.calculate_rmsd as c_rmsd

#建立文件夹
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def showAns(prediction,ground_truth,isSave = False,path='test',name='ans'):
    # prepare_result
    fig = plt.figure()
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    plt.scatter(x=ground_truth[:, 0], y=ground_truth[:, 1], c='r')
    plt.scatter(x=prediction[:, 0], y=prediction[:, 1], c='g', marker='*')
    # plt.show()
    if isSave:
        mkdir(path)
        fig.savefig(path + "/" + name + ".png")
def getLinkMatrix(distMatrix,max_range_length):
    linkMatarix = distMatrix.copy()
    linkMatarix[linkMatarix > max_range_length] = 0  #
    linkMatarix[linkMatarix != 0] = 1
    return linkMatarix

def getOneHopNeighbor(linkMatarix):
    oneHopNeighbor = {}
    for i in  range(linkMatarix.shape[0]):
        for idx,ele in enumerate(linkMatarix[i]):
            if ele == 1:
                if not oneHopNeighbor.keys().__contains__(i) :
                    oneHopNeighbor[i] = []
                oneHopNeighbor[i].append(idx)
            else:
                pass
    return oneHopNeighbor

def getTwoHopNeighbor(oneHopNeighbor):
    twoHopNeighbor = {}
    for k,v in oneHopNeighbor.items():
        twoHopNeighbor[k] = []
        for ele in v:
            twoHopNeighbor[k] = twoHopNeighbor[k] + oneHopNeighbor[ele]
        twoHopNeighbor[k] = list(set(twoHopNeighbor[k]))
    return twoHopNeighbor

# test the cluster
def getClusters(oneHopNeighbor,twoHopNeighbor):
    clusters = {}
    for k,v in oneHopNeighbor.items():
        clusters[k] = []
        for ele in v:
            if twoHopNeighbor[k].__contains__(ele):
                clusters[k].append(ele)
        if clusters[k].__contains__(k):
            clusters[k].remove(k)
    # {center:its cluster without center}
    return clusters

#计算定位坐标
#后面需要改成多边定位并使用最小二乘估计
def triposition(xa,ya,da,xb,yb,db,xc,yc,dc):
    xa = round(xa, 4)
    ya = round(ya, 4)
    da = round(da, 4)
    xb = round(xb, 4)
    yb = round(yb, 4)
    db = round(db, 4)
    xc = round(xc, 4)
    yc = round(yc, 4)
    dc = round(dc, 4)
    x,y = sympy.symbols('x y')
    f1 = 2*x*(xa-xc)+np.square(xc)-np.square(xa)+2*y*(ya-yc)+np.square(yc)-np.square(ya)-(np.square(dc)-np.square(da))
    f2 = 2*x*(xb-xc)+np.square(xc)-np.square(xb)+2*y*(yb-yc)+np.square(yc)-np.square(yb)-(np.square(dc)-np.square(db))
    result = sympy.solve([f1,f2],[x,y])

    locx,locy = result[x],result[y]
    return [locx,locy]

# 计算两圆交点
def caculatePointsInTwoCircle(xa,ya,ra,xb,yb,rb):
    xa = round(xa, 4)
    ya = round(ya, 4)
    ra = round(ra, 4)
    xb = round(xb, 4)
    yb = round(yb, 4)
    rb = round(rb, 4)
    distance = math.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)
    min_rad = min(ra, rb)
    max_rad = max(ra, rb)
    if distance > ra + rb:
        return ((xa + (xb - xa) * ra / distance, ya + (yb - ya) * ra / distance), (xb + (xa - xb) * rb / distance, yb + (ya - yb) * rb / distance) )
    if distance + min_rad <= max_rad:
        if ra < rb:
            return [(xb + (xa - xb) * (distance+ra) / distance, yb + (ya - yb) * (distance+ra) / distance),(xb + (xa - xb) * (distance+ra) / distance, yb + (ya - yb) * (distance+ra) / distance)]
        else:
            return [(xa + (xb - xa) * (distance+rb) / distance, ya + (yb - ya) *  (distance+rb) / distance), (xa + (xb - xa) * (distance+rb) / distance, ya + (yb - ya) *  (distance+rb) / distance)]
    x, y = sympy.symbols('x y')
    f1 = (xa-x)**2 + (ya-y)**2 - ra**2
    f2 = (xb-x)**2 + (yb-y)**2 - rb**2
    result = sympy.solve([f1,f2],[x,y])
    return [float(result[0][0]),float(result[0][1])],[float(result[1][0]),float(result[1][1])]

def multiPosition(i,triples):
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
    root=leastsq(multiPosition, np.asarray((triples.mean(axis=0)[0],triples.mean(axis=0)[1])),args=(triples))
    return root[0]
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
    circle_a =Circle(xy=(xa,ya),radius=da,fill=False)
    circle_b =Circle(xy=(xb,yb),radius=db,fill=False)
    circle_c = Circle(xy=(xc,yc),radius=dc,fill=False)
    fig, ax = plt.subplots()
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    plt.scatter(x=[xa,xb,xc], y=[ya,yb,yc], c='r')
    prediction = triposition(xa, ya, da, xb, yb, db, xc, yc, dc)
    # plt.scatter(x=[prediction[0]],y=[prediction[1]])
    ax.add_patch(circle_a)
    ax.add_patch(circle_b)
    ax.add_patch(circle_c)
    plt.show()

    print( triposition(xa,ya,da,xb,yb,db,xc,yc,dc) )