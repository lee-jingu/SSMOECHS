import numpy as np
import config as cf
import networkx as nx
import matplotlib.pyplot as plt
from network import Energy
from network import Network
from Optimizer import LEACH, PSO, HEED, KSMO
from utils import Plot
import math

# pos1 = np.random.uniform(low=0, high=cf.AREA_H, size=(5,2))
# pos2 = np.around(pos1*4, decimals=0)/4
# print(pos1,pos2)

G, Alive_Node,R = Network.init_network()
# print(cf.D_o)
# G, CHID = HEED.Optimizer(G, Alive_Node)
# for i in range(0,300):
#     G, Alive_Node, Res = Network.Run_Round(G,Alive_Node)
G, CHID,R = KSMO.Optimizer(G,Alive_Node,R)
print(CHID)
# A = [1,2,3]
# B = [5,6,7]
# C = []

# C = np.append(C,A)
# C = np.append(C,B)
# print(C)
# value = []
# sortvec = []
# dtype = [('idx',int),('x', float), ('y',float), ('dist',float)]
# for i in Alive_Node:
#     x,y = G.node[i]['pos']
#     dist = math.sqrt((x-50)**2+(y-50)**2)
#     value.append((i,x,y,dist))
# a = np.array(value,dtype=dtype)
# xy_sortvec = np.sort(a,order=['x','y'])
# yx_sortvec = np.sort(a,order=['y','x'])
# sortvec = np.sort(a,order=['dist'])


# med = []
# for i in range(0, 30):
#     med.append(i)
# median = np.median(med, axis=0)
# print(median)

# N = int(len(Alive_Node)*cf.P_CH*((2**0.5)/(1+2**0.5)))
# N2 = int(len(Alive_Node)*cf.P_CH-N)
# N3 = len(Alive_Node)*cf.P_CH
# print(N,N2,N3)
# print(cf.D_o)

# for i in range(0,len(CHID)):
#     print("{} : {}".format(CHID[i],G.out_edges(CHID[i])))

# t = 0
# for i in CHID:
#     chk = Network.get_MSG(G,i)
#     print(i,chk)
#     t += chk
# print(t)


# Death = [3]
# Death.append(3)
# Alive = list(set(Alive)-set(Death))

# print(Alive[3])

# P = 0.05
# for i in range(0, 100):
#     print(i, P/(1-P*(i%(1/P))))

# A = np.array([[1,2,3,5],[3,0,1,4]])
# row,col = np.where(A==4)
# print(row)