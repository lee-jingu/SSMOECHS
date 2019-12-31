import numpy as np
import config as cf
import networkx as nx
import matplotlib.pyplot as plt
from network import Energy
from network import Network
from network import Node
from scipy.spatial import distance
import math
import networkx as nx
from random import *
import pandas as pd
# def Get_Fitness(network,SM_Array,Alive_Node):
#     Fitness = 0
#     f1 = 0
#     f2 = 0
#     f3 = 0
#     f4 = 0
#     CHID = np.where(SM==np.max(SM))[0] + 1
#     potf3 = np.zeros(cf.N_NODE+1)
#     potf4 = np.zeros(cf.N_NODE+1)

#     for i in Alive_Node:
#         x,y = network.node[i]['pos']
#         NNDist = 1000
#         NNID = 0
#         for j in CHID:
#             if i == j:
#                 d = network.node[j]['RTBS']
#                 if d > cf.D_o:
#                     f1 += cf.E_ELEC*cf.L*cf.E_MP*(d**4) + cf.E_DA
#                 else:
#                     f1 += cf.E_ELEC*cf.L*cf.E_FS*(d**2) + cf.E_DA
#                 continue
#             if i in CHID:
#                 continue
#             x2,y2 = network.node[j]['pos']
#             NewDist = math.sqrt((x-x2)**2+(y-y2)**2)
#             if NewDist<NNDist:
#                 NNDist = NewDist
#                 NNID = j
#         if NNDist > cf.D_o:
#             f1 += cf.NCH_L * (2*cf.E_ELEC + cf.E_MP*(NNDist**4))
#             f2 += NNDist
#             potf3[NNID] += network.node[i]['res_energy']
#             potf4[NNID] += 1
#         else:
#             f1 += cf.NCH_L * (2*cf.E_ELEC + cf.E_FS*(NNDist**2))
#             f2 += NNDist
#             potf3[NNID] += network.node[i]['res_energy']
#             potf4[NNID] += 1

#     MinDist = 1000
#     for i in CHID:
#         for j in CHID:
#             if i==j:
#                 continue
#             x,y = network.node[i]['pos']
#             x2,y2 = network.node[j]['pos']
#             NewMinDist = math.sqrt((x-x2)**2+(y-y2)**2)
#             if NewMinDist < MinDist:
#                 MinDist = NewMinDist
#     for i in CHID:
#         potf3[i] = potf3[i]/network.node[i]['res_energy']
#     f2 = f2/MinDist
#     f3 = np.sum(potf3)
#     f4 = np.where(np.max(potf4)==potf4)[0][0]
#     Fitness = (f1+f2+f3+f4) *0.25
#     return Fitness

# network, Alive_Node, R = Network.init_network()
# BSMO_NET = nx.create_empty_copy(network)
# BSMO_CHID = []
# Swarm_Size = 40
# MIR = 100

# ##Initializing
# SM_Arr = []
# FIT = []
# for i in range(0,Swarm_Size):
#     SM = []
#     for j in Alive_Node:
#         if random()<=cf.P_CH:
#             SM.append(1)
#         else:
#             SM.append(0)
#     SM_Arr.append(SM)
#     FIT.append(Get_Fitness(BSMO_NET,SM_Arr[i],Alive_Node))
# 
# a = [67, 89, 101, 103, 106, 112, 122, 126, 130, 137, 149]
# b = [93, 105, 113, 118, 122, 127, 131, 135, 139, 142, 146]
# c = [58, 89, 104, 111, 120, 125, 131, 137, 144, 151, 157]
# d = [94, 105, 110, 114, 119, 127, 132, 137, 141, 148, 154]
# e = [101, 108, 112, 118, 122, 129, 137, 142, 148, 153, 161]
# data1 = []
# FHL = []
# FHL.append(a)
# FHL.append(b)
# FHL.append(c)
# FHL.append(d)
# FHL.append(e)
# print(FHL)
# INDEX = []

# for i in range(1,len(cf.Optimizer)):
#     INDEX.append(cf.Optimizer[i])

# IFHL = []
# for i in range(0,len(FHL[0])):
#     temp = []
#     for j in range(0,len(FHL)):
#         temp.append(FHL[j][i])
#     IFHL.append(temp)
# df = pd.DataFrame(IFHL,index = ["99(FND)","90","80","70","60","50(HND)","40","30","20","10","0(LND)"], columns=pd.Index(INDEX,name = "Alive Nodes[%]"))
# df.plot(kind="bar",grid=True)
# plt.show()

A = [0.5, 0.5, 0]
B = np.exp(A)/np.sum(np.exp(A))
print(B)
print(B/np.sum(B))