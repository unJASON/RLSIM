import numpy as np
import sympy
import math
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
# Kabsch W . A Solution of the Best Rotation to Relate Two Sets of Vectors[J]. Acta Crystallographica Section A Foundations of Crystallography, 1976, A32(5)
import rmsd.calculate_rmsd as c_rmsd
max_range_length = 5
max_distance = 4.1
ground_truth = [[0., 0.]]
INF = 10000

for i in range(10):
    temp = [ground_truth[-1][0] + random.uniform(0, 2), ground_truth[-1][1] + random.uniform(0, 2)]
    ground_truth.append(list(temp))
# hex
# ground_truth.clear()
# ground_truth.append([0,0])
# ground_truth.append([3.9,0])
# ground_truth.append([-3.9,0])
# ground_truth.append([3.9*math.cos(math.pi/3),3.9*math.sin(math.pi/3)])
# ground_truth.append([-3.9*math.cos(math.pi/3),3.9*math.sin(math.pi/3)])
# ground_truth.append([3.9*math.cos(math.pi/3),-3.9*math.sin(math.pi/3)])
# ground_truth.append([-3.9*math.cos(math.pi/3),-3.9*math.sin(math.pi/3)])

#
# ground_truth.clear()
# ground_truth.append([0,0])
# ground_truth.append([4.0/math.sqrt(2),0])
# ground_truth.append([0,4.0/math.sqrt(2)])
# ground_truth.append([4.0/math.sqrt(2),4.0/math.sqrt(2)])
# ground_truth.append([0.5,0.5])
# ground_truth.append([0,8.0/math.sqrt(2)])



ground_truth = np.array(ground_truth)



distance_matrix = np.zeros(shape=[ground_truth.shape[0], ground_truth.shape[0]], dtype=float)
for i in range(distance_matrix.shape[0]):
    for j in range(i):
        if i == j:
            distance_matrix[i][j] = 0
        else:
            distance_matrix[i][j] = math.sqrt((ground_truth[i][0] - ground_truth[j][0]) ** 2 + (
                        ground_truth[i][1] - ground_truth[j][1]) ** 2 )
            distance_matrix[j][i] = math.sqrt((ground_truth[i][0] - ground_truth[j][0]) ** 2 + (
                        ground_truth[i][1] - ground_truth[j][1]) ** 2 )

def getLinkMatrix(distMatrix):
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
    x,y = sympy.symbols('x y')
    f1 = 2*x*(xa-xc)+np.square(xc)-np.square(xa)+2*y*(ya-yc)+np.square(yc)-np.square(ya)-(np.square(dc)-np.square(da))
    f2 = 2*x*(xb-xc)+np.square(xc)-np.square(xb)+2*y*(yb-yc)+np.square(yc)-np.square(yb)-(np.square(dc)-np.square(db))
    result = sympy.solve([f1,f2],[x,y])

    locx,locy = result[x],result[y]
    return [locx,locy]


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

def getPositions(center, cluster, distMatarix,oneHopNeighbor):
    cluster_set = set(cluster)
    position = {}
    vertex_coord = {} #stores neighbors that have position info
    position_queue = [] #stores index
    vertex_queue =[center,cluster[0],cluster[1]]
    idx = 0
    while len(vertex_queue) != 0:
        # caculate position
        ele = vertex_queue.pop(0)
        if position.keys().__contains__(ele):
            continue
        if idx == 0:
            # center
            position[ele] = [0, 0]
            # establish index
            position_queue.append(ele)
        elif idx == 1:
            # second vertex
            position[ele] = [(distMatarix[center][ele] + distMatarix[ele][center]) / 2, 0]
            position_queue.append(ele)
        elif idx == 2:
            # third vertex
            a = (distMatarix[position_queue[1]][center] + distMatarix[center][position_queue[1]]) / 2
            b = (distMatarix[center][ele] + distMatarix[ele][center]) / 2
            c = (distMatarix[position_queue[1]][ele] + distMatarix[ele][position_queue[1]]) / 2
            Cos_angle = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
            # the range error affect the result
            if Cos_angle > 1 or Cos_angle < -1:
                return position
            Sin_angle = math.sqrt(1 - Cos_angle)
            position[ele] = [b * Cos_angle, b * Sin_angle]
            position_queue.append(ele)
        else:
            # others
            coord_neighbor_position =  vertex_coord[ele]
            pos_0 = coord_neighbor_position[0]
            pos_1 = coord_neighbor_position[1]
            pos_2 = coord_neighbor_position[2]
            position[ele] = triposition(position[pos_0][0], position[pos_0][1],
                                        (distMatarix[pos_0][ele] + distMatarix[ele][pos_0]) / 2,
                                        position[pos_1][0], position[pos_1][1],
                                        (distMatarix[pos_1][ele] + distMatarix[ele][pos_1]) / 2,
                                        position[pos_2][0], position[pos_2][1],
                                        (distMatarix[pos_2][ele] + distMatarix[ele][pos_2]) / 2)
            # triples = []
            # for vertex_coord_ele in vertex_coord[ele]:
            #     triples.append((position[vertex_coord_ele][0],position[vertex_coord_ele][1], (distMatarix[vertex_coord_ele][ele] + distMatarix[ele][vertex_coord_ele]) / 2))
            # position[ele] = multiposition(triples)
            position_queue.append(ele)
            # update  vertex information
        for neighbor in oneHopNeighbor[ele]:
            if cluster_set.__contains__(neighbor) :
                if not vertex_coord.keys().__contains__(neighbor):
                    vertex_coord[neighbor] = []
                vertex_coord[neighbor].append(ele)
                if  len( vertex_coord[neighbor] ) >= 3 and (not position.keys().__contains__(neighbor) ) :
                    # insert vertex
                    vertex_queue.append(neighbor)
        idx = idx + 1
    return position




def reEstabilish(distMatarix):

    distMatarix[distMatarix < 0] = 0.01  #get ride of error range
    distMatarix[distMatarix > max_range_length] = INF  # exceed range
    linkMatarix = getLinkMatrix(distMatarix)
    oneHopNeighbor=getOneHopNeighbor(linkMatarix)
    twoHopNeighbor=getTwoHopNeighbor(oneHopNeighbor)
    clusters = getClusters(oneHopNeighbor,twoHopNeighbor)

    max_cluster_num = 2
    ans_cluster = None
    final_ans = None
    final_score = 1000
    for k,cluster in clusters.items():
        if len(cluster) >=2:
            start =time.time()
            clusterPositions = getPositions(k,cluster,distMatarix,oneHopNeighbor)
            print("keys:",clusterPositions.keys())
            print("cost time:",time.time() - start)
            if len( clusterPositions ) >= 3:
                # print(clusterPositions)
                temp = []
                temp2 =[]
                temp3 = []
                temp.append(ground_truth[k].tolist())
                temp2.append(clusterPositions[k])
                temp3.append([clusterPositions[k][0], -clusterPositions[k][1]])
                for ele in cluster:
                    if clusterPositions.keys().__contains__(ele):
                        temp.append(ground_truth[ele].tolist())
                        temp2.append(clusterPositions[ele])
                        temp3.append([clusterPositions[ele][0],-clusterPositions[ele][1]])
                temp_np = np.array(temp,dtype=np.float)
                temp_np_center = c_rmsd.centroid(temp_np)
                temp_np = temp_np -temp_np_center

                temp2_np = np.array(temp2, dtype=np.float)
                temp2_np_center = c_rmsd.centroid(temp2_np)
                temp2_np = temp2_np - temp2_np_center

                temp3_np = np.array(temp3, dtype=np.float)
                temp3_np_center = c_rmsd.centroid(temp3_np)
                temp3_np = temp3_np - temp3_np_center

                ans = c_rmsd.kabsch_rotate(temp2_np , temp_np)
                ans2 = c_rmsd.kabsch_rotate(temp3_np,temp_np)
                if len(clusterPositions) >= max_cluster_num:
                    max_cluster_num = len(clusterPositions)
                    ans_cluster = cluster

                    ans_score = c_rmsd.rmsd(ans, temp_np)
                    ans2_score = c_rmsd.rmsd(ans2, temp_np)
                    print(ans_score,ans2_score)
                    if ans_score < 0.5 or ans2_score < 0.5:
                        if ans_score < ans2_score:
                            final_ans = ans
                            final_score = ans_score
                        else:
                            final_ans = ans2
                            final_score = ans2_score
                        final_ans = final_ans + temp_np_center
        else:
            pass
    return final_ans,ans_cluster,final_score



if __name__ == '__main__':
    prediction,cluster,score = reEstabilish(distance_matrix)
    print(cluster)
    print("score:",score)
    #prepare_result
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    plt.scatter(x=ground_truth[:,0],y=ground_truth[:,1],c='r')
    plt.scatter(x=prediction[:,0],y= prediction[:,1],c='g',marker='*')
    plt.show()

