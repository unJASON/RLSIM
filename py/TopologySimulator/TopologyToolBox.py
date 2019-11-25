
import numpy as np
import sympy
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import time
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
    xa = round(xa, 2)
    ya = round(ya, 2)
    da = round(da, 2)
    xb = round(xb, 2)
    yb = round(yb, 2)
    db = round(db, 2)
    xc = round(xc, 2)
    yc = round(yc, 2)
    dc = round(dc, 2)
    x,y = sympy.symbols('x y')
    f1 = 2*x*(xa-xc)+np.square(xc)-np.square(xa)+2*y*(ya-yc)+np.square(yc)-np.square(ya)-(np.square(dc)-np.square(da))
    f2 = 2*x*(xb-xc)+np.square(xc)-np.square(xb)+2*y*(yb-yc)+np.square(yc)-np.square(yb)-(np.square(dc)-np.square(db))
    result = sympy.solve([f1,f2],[x,y])

    locx,locy = result[x],result[y]
    return [locx,locy]

# 计算两圆交点
def caculatePointsInTwoCircle(xa,ya,ra,xb,yb,rb):
    xa = round(xa, 2)
    ya = round(ya, 2)
    ra = round(ra, 2)
    xb = round(xb, 2)
    yb = round(yb, 2)
    rb = round(rb, 2)
    distance = math.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)
    min_rad = min(ra, rb)
    max_rad = max(ra, rb)
    if distance > ra + rb:
        return ((xa + (xb - xa) * ra / distance, ya + (yb - ya) * ra / distance), (xb + (xa - xb) * rb / distance, yb + (ya - yb) * rb / distance) )
    if distance + min_rad <= max_rad:
        if ra < rb:
            return ((xb + (xa - xb) * (distance+ra) / distance, yb + (ya - yb) * (distance+ra) / distance),(xb + (xa - xb) * (distance+ra) / distance, yb + (ya - yb) * (distance+ra) / distance))
        else:
            return ((xa + (xb - xa) * (distance+rb) / distance, ya + (yb - ya) *  (distance+rb) / distance), (xa + (xb - xa) * (distance+rb) / distance, ya + (yb - ya) *  (distance+rb) / distance))
    x, y = sympy.symbols('x y')
    f1 = (xa-x)**2 + (ya-y)**2 - ra**2
    f2 = (xb-x)**2 + (yb-y)**2 - rb**2
    result = sympy.solve([f1,f2],[x,y])
    return [float(result[0][0]),float(result[0][1])],[float(result[1][0]),float(result[1][1])]

#input list[(x,y,d)]
def multiposition(triples):
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