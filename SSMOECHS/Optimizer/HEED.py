import copy
import networkx as nx
import numpy as np
import config as cf
import random
import math
from network import Network


def Optimizer(network, Alive_Node, Residual=False, R=30, IN_Median=False):
    HEED_NET=nx.create_empty_copy(network)
    HEED_CHID=[]
    P = cf.P_CH

    ## CH Selection
    for i in Alive_Node:
        r = HEED_NET.node[i]['round']
        CH_Prob = P/(1-P*(r%(1/P)))
        if Residual == True:
            MAX = np.amax(Residual,axis=0)[1]
            row,col = np.where(Residual==i)
            CH_Prob *= (Residual[row][1]/MAX)
        if random.random()<CH_Prob:
            HEED_CHID.append(i)
            HEED_NET.node[i]['round'] = 0
            HEED_NET.node[i]['N_Packet']=cf.L
        else:
            HEED_NET.node[i]['round'] += 1
            HEED_NET.node[i]['N_Packet']=cf.NCH_L

    ## Clustering
    for i in Alive_Node:
        x1,y1 = HEED_NET.node[i]['pos']
        NN_Dist = math.sqrt((x1-50)**2+(y1-50)**2)
        NNID = 0
        for NN in HEED_CHID:
            x2,y2 = HEED_NET.node[NN]['pos']
            new_dist = math.sqrt((x1-x2)**2+(y1-y2)**2)
            if new_dist == 0:
                continue
            if new_dist<NN_Dist:
                NNID = NN
                NN_Dist = new_dist
        HEED_NET.node[i]['Next']=NNID



    # if two nodes are linked each other,one of them have to link to BS
    for CH in HEED_CHID:
        NEXT_NODE = HEED_NET.node[CH]['Next']
        CHK = HEED_NET.node[NEXT_NODE]['Next']
        if HEED_NET.node[NEXT_NODE]['Next'] == CH:
            ## To BS distance
            x1,y1 = HEED_NET.node[CH]['pos']
            x2,y2 = HEED_NET.node[NEXT_NODE]['pos']
            dist1 = math.sqrt((x1-50)**2 + (y1-50)**2)
            dist2 = math.sqrt((x2-50)**2 + (y2-50)**2)
            if dist1 > dist2:                       ##NNID to BS is more near than nodei to BS
                HEED_NET.node[NEXT_NODE]['Next'] = 0  ##BSID
            else:
                HEED_NET.node[CH]['Next'] = 0


    ## add_Edge 
    for i in Alive_Node:
        HEED_NET.add_edge(i,HEED_NET.node[i]['Next'])

    return HEED_NET, HEED_CHID, R