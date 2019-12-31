import copy
import networkx as nx
import numpy as np
import config as cf
import random
import math
from network import Network


def Optimizer(network, Alive_Node, Update=False, R=30, IN_Median=False, First=False,a=False):
    LEACHC_NET=nx.create_empty_copy(network)
    LEACHC_CHID=[]
    k = len(Alive_Node) * cf.P_CH
    ETotal = 0
    for i in Alive_Node:
        ETotal += network.node[i]['res_energy']
    ## CH Selection
    for i in Alive_Node:
        P = min(network.node[i]['res_energy']/ETotal*k,1)
        if random.random()<P:
            LEACHC_CHID.append(i)


    ## Clustering
    for i in Alive_Node:
        if i in LEACHC_CHID:
            LEACHC_NET.node[i]['Next'] = 0
            continue
        x1,y1 = LEACHC_NET.node[i]['pos']
        NN_Dist = 1000
        NNID=0
        for NN in LEACHC_CHID:
            if i in LEACHC_CHID:
                continue
            x2,y2 = LEACHC_NET.node[NN]['pos']
            new_dist = math.sqrt((x1-x2)**2+(y1-y2)**2)
            if new_dist == 0:
                continue
            if new_dist<NN_Dist:
                NNID = NN
                NN_Dist = new_dist
        LEACHC_NET.node[i]['Next'] = NNID
        
    if First == True:
        ## add_Edge
        for i in Alive_Node:
            NEXT = LEACHC_NET.node[i]['Next']
            if NEXT != 0:
                LEACHC_NET.add_edge(i,NEXT)

    return LEACHC_NET, LEACHC_CHID, R, IN_Median