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
import TopologyToolBox as ttb
max_range_length = 5.1    #最大测距距离
max_distance = 4.1      #无人机间最大允许距离
ground_truth = [[0., 0.]]
INF = 10000

for i in range(30):
    temp = [ground_truth[-1][0] + random.uniform(-1, 2), ground_truth[-1][1] + random.uniform(-1, 2)]
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

#triangle
# ground_truth.clear()
# ground_truth.append([0,0])
# ground_truth.append([5,0])
# ground_truth.append([2.5,2.5*math.sqrt(3)])
# ground_truth.append([0,2.5*math.sqrt(3)])
# ground_truth.append([0,5*math.sqrt(3)])

# ground_truth.clear()
# ground_truth.append([0,0])
# ground_truth.append([2.5*2/math.sqrt(3),2.5])
# ground_truth.append([2.5*math.sqrt(3),2.5])
# ground_truth.append([2.5*math.sqrt(3) + 2.5*2/math.sqrt(3) ,5])
# ground_truth.append([2.5*math.sqrt(3),0])
# ground_truth.append([2.5*2/math.sqrt(3),5])



ground_truth = np.array(ground_truth)
if False:
    ground_truth = np.load('./problem.npy')

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



def getPositions(center, cluster, distMatarix,oneHopNeighbor):
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
            Sin_angle = math.sqrt(1 - Cos_angle**2)
            position[ele] = [b * Cos_angle, b * Sin_angle]
            position_queue.append(ele)
        else:
            # others
            coord_neighbor_position =  vertex_coord[ele]
            if len(coord_neighbor_position) >= 3:
                #use triposition
                pos_0 = coord_neighbor_position[0]
                pos_1 = coord_neighbor_position[1]
                pos_2 = coord_neighbor_position[2]
                position[ele] = ttb.triposition(position[pos_0][0], position[pos_0][1],
                                            (distMatarix[pos_0][ele] + distMatarix[ele][pos_0]) / 2,
                                            position[pos_1][0], position[pos_1][1],
                                            (distMatarix[pos_1][ele] + distMatarix[ele][pos_1]) / 2,
                                            position[pos_2][0], position[pos_2][1],
                                            (distMatarix[pos_2][ele] + distMatarix[ele][pos_2]) / 2)
                position_queue.append(ele)
            else:
                # 2 degree vertexes
                # use conditional exclusion
                # use cache to get remove duplicate situation
                pos_0 = coord_neighbor_position[0]
                pos_1 = coord_neighbor_position[1]
                p1,p2 = ttb.caculatePointsInTwoCircle(position[pos_0][0], position[pos_0][1],
                                            (distMatarix[pos_0][ele] + distMatarix[ele][pos_0]) / 2,
                                            position[pos_1][0], position[pos_1][1],
                                            (distMatarix[pos_1][ele] + distMatarix[ele][pos_1]) / 2)
                flag = False
                choose_pos = None
                for k,v in position.items():
                    #filter
                    if k != pos_0 and k!= pos_1:
                        if v[0]>=p1[0]-max_range_length and v[0] <= p1[0]+max_range_length  and v[1]>=p1[1]-max_range_length and v[1] <= p1[1]+max_range_length:
                            # circle_filter
                            if (v[0]-p1[0])**2 +(v[1]-p1[1])**2 <= max_range_length**2:
                                flag = True
                                choose_pos = p2
                                break
                            else:
                                pass
                        if v[0]>=p2[0]-max_range_length and v[0] <= p2[0]+max_range_length  and v[1]>=p2[1]-max_range_length and v[1] <= p2[1]+max_range_length:
                            # circle_filter
                            if (v[0]-p2[0])**2 +(v[1]-p2[1])**2 <= max_range_length**2:
                                flag = True
                                choose_pos = p1
                                break
                            else:
                                pass
                if flag:
                    position[ele] = choose_pos
                    position_queue.append(ele)

        # update  vertex information
        if position.keys().__contains__(ele):
            for neighbor in oneHopNeighbor[ele]:
                if not vertex_coord.keys().__contains__(neighbor):
                    vertex_coord[neighbor] = []
                vertex_coord[neighbor].append(ele)
                if  len( vertex_coord[neighbor] ) >= 2 and (not position.keys().__contains__(neighbor) ) :
                    # insert vertex
                    vertex_queue.append(neighbor)
        idx = idx + 1
    return position

#modify process
def getPositionsV2(center, cluster, distMatarix,oneHopNeighbor):
    position = {}
    vertex_coord = {} #stores neighbors that have position info
    position_queue = [] #stores index
    vertex_queue =[center,cluster[0],cluster[1]]
    idx = 0
    possible_position = {}
    while len(vertex_queue) != 0:
        # caculate position
        flag = False
        choose_pos = None
        ele = vertex_queue.pop(0)
        if position.keys().__contains__(ele):
            continue
        if idx == 0:
            flag = True
            # center
            choose_pos = [0, 0]
            # establish index
        elif idx == 1:
            flag = True
            # second vertex
            choose_pos = [(distMatarix[center][ele] + distMatarix[ele][center]) / 2, 0]
        elif idx == 2:
            flag = True
            # third vertex
            a = (distMatarix[position_queue[1]][center] + distMatarix[center][position_queue[1]]) / 2
            b = (distMatarix[center][ele] + distMatarix[ele][center]) / 2
            c = (distMatarix[position_queue[1]][ele] + distMatarix[ele][position_queue[1]]) / 2
            Cos_angle = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
            # the range error affect the result
            if Cos_angle > 1 or Cos_angle < -1:
                return position
            Sin_angle = math.sqrt(1 - Cos_angle**2)
            choose_pos = [b * Cos_angle, b * Sin_angle]
        else:
            # others
            coord_neighbor_position = vertex_coord[ele]
            if len(coord_neighbor_position) >= 3:
                flag = True
                #use triposition
                pos_0 = coord_neighbor_position[0]
                pos_1 = coord_neighbor_position[1]
                pos_2 = coord_neighbor_position[2]
                choose_pos = ttb.triposition(position[pos_0][0], position[pos_0][1],
                                            (distMatarix[pos_0][ele] + distMatarix[ele][pos_0]) / 2,
                                            position[pos_1][0], position[pos_1][1],
                                            (distMatarix[pos_1][ele] + distMatarix[ele][pos_1]) / 2,
                                            position[pos_2][0], position[pos_2][1],
                                            (distMatarix[pos_2][ele] + distMatarix[ele][pos_2]) / 2)
            else:
                # 2 degree vertexes
                # use conditional exclusion
                # use cache to get remove duplicate situation
                pos_0 = coord_neighbor_position[0]
                pos_1 = coord_neighbor_position[1]
                p1,p2 = ttb.caculatePointsInTwoCircle(position[pos_0][0], position[pos_0][1],
                                            (distMatarix[pos_0][ele] + distMatarix[ele][pos_0]) / 2,
                                            position[pos_1][0], position[pos_1][1],
                                            (distMatarix[pos_1][ele] + distMatarix[ele][pos_1]) / 2)
                for k,v in position.items():
                    #filter
                    if k != pos_0 and k!= pos_1:
                        if v[0]>=p1[0]-max_range_length and v[0] <= p1[0]+max_range_length  and v[1]>=p1[1]-max_range_length and v[1] <= p1[1]+max_range_length:
                            # circle_filter
                            if (v[0]-p1[0])**2 +(v[1]-p1[1])**2 <= max_range_length**2:
                                flag = True
                                choose_pos = p2
                                break
                            else:
                                pass
                        if v[0]>=p2[0]-max_range_length and v[0] <= p2[0]+max_range_length  and v[1]>=p2[1]-max_range_length and v[1] <= p2[1]+max_range_length:
                            # circle_filter
                            if (v[0]-p2[0])**2 +(v[1]-p2[1])**2 <= max_range_length**2:
                                flag = True
                                choose_pos = p1
                                break
                            else:
                                pass
                possible_position[ele] = [p1, p2]
        if flag:
            position[ele] = choose_pos
            # update vertex information and insert queue
            for neighbor in oneHopNeighbor[ele]:
                if not vertex_coord.keys().__contains__(neighbor):
                    vertex_coord[neighbor] = []
                vertex_coord[neighbor].append(ele)
                if len(vertex_coord[neighbor]) >= 2 and (not position.keys().__contains__(neighbor)):
                    # insert vertex
                    vertex_queue.append(neighbor)
            position_queue.append(ele)

            # remove keys in possible_position
            possible_list = [(ele,choose_pos)]
            while len( possible_list ) != 0:
                (choose_ele,choose_ele_pos) = possible_list.pop(0)
                if possible_position.keys().__contains__(choose_ele):
                    del possible_position[choose_ele]
                for k, posible_pos_list in possible_position.items():
                    pos_temp_1 = posible_pos_list[0]
                    pos_temp_2 = posible_pos_list[1]
                    if (pos_temp_1[0]-choose_ele_pos[0])**2 + (pos_temp_1[1]-choose_ele_pos[1])**2<= max_range_length**2:
                        position[k] = pos_temp_2
                        possible_list.append((k, pos_temp_2))
                    if (pos_temp_2[0]-choose_ele_pos[0])**2 + (pos_temp_2[1]-choose_ele_pos[1])**2<= max_range_length**2:
                        position[k] = pos_temp_1
                        possible_list.append((k,pos_temp_1))

        else:
            pass




        idx = idx + 1
    return position


def reEstabilish(distMatarix):
    global ground_truth
    distMatarix[distMatarix < 0] = 0.01  #get ride of error range
    distMatarix[distMatarix > max_range_length] = INF  # exceed range
    linkMatarix = ttb.getLinkMatrix(distMatarix,max_range_length)
    oneHopNeighbor=ttb.getOneHopNeighbor(linkMatarix)
    twoHopNeighbor=ttb.getTwoHopNeighbor(oneHopNeighbor)
    clusters = ttb.getClusters(oneHopNeighbor,twoHopNeighbor)

    max_cluster_num = 2
    ans_cluster = None
    final_ans = None
    final_score = 1000
    np.save('./problem.npy', ground_truth)
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
                for ele,pos in clusterPositions.items():
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
                    if ans_score < 2 or ans2_score < 2:
                        if ans_score < ans2_score:
                            final_ans = ans
                            final_score = ans_score
                        else:
                            final_ans = ans2
                            final_score = ans2_score
                        final_ans = final_ans + temp_np_center
                        ttb.showAns(final_ans,ground_truth,isSave=True,name=str(k))
                    else:
                        final_ans = ans
                        final_ans = final_ans + temp_np_center
                        ttb.showAns(final_ans, ground_truth, isSave=True, name=str(k))
        else:
            pass
    return final_ans,ans_cluster,final_score



if __name__ == '__main__':
    prediction,cluster,score = reEstabilish(distance_matrix)
    print(cluster)
    print("score:",score)


