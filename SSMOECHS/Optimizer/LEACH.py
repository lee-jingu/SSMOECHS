import copy
import networkx as nx
import numpy as np
import config as cf
import random
import math
from network import Network


def Optimizer(network, Alive_Node, Update=False, R=30, IN_Median=False, First=False):
    LEACH_NET=nx.create_empty_copy(network)
    LEACH_CHID=[]
    P = cf.P_CH
    r = LEACH_NET.node[0]['round']
    ## CH Selection
    for i in Alive_Node:
        if r%(1/P) == 0 and r>1:
            LEACH_NET.node[i]['round'] = 1
        r0 = LEACH_NET.node[i]['round']
        if random.random()<r0*P/(1-P*(r%(1/P))):
            LEACH_CHID.append(i)
            LEACH_NET.node[i]['round'] = 0


    ## Clustering
    for i in Alive_Node:
        if i in LEACH_CHID:
            LEACH_NET.node[i]['Next'] = 0
            continue
        x1,y1 = LEACH_NET.node[i]['pos']
        NN_Dist = 1000
        NNID=0
        for NN in LEACH_CHID:
            if i in LEACH_CHID:
                continue
            x2,y2 = LEACH_NET.node[NN]['pos']
            new_dist = math.sqrt((x1-x2)**2+(y1-y2)**2)
            if new_dist == 0:
                continue
            if new_dist<NN_Dist:
                NNID = NN
                NN_Dist = new_dist
        LEACH_NET.node[i]['Next'] = NNID
        
    if First == True:
        ## add_Edge
        for i in Alive_Node:
            NEXT = LEACH_NET.node[i]['Next']
            if NEXT != 0:
                LEACH_NET.add_edge(i,NEXT)
    LEACH_NET.node[0]['round'] += 1
    return LEACH_NET, LEACH_CHID, R, IN_Median