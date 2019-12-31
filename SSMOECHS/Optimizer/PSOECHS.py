import copy
import networkx as nx
import numpy as np
import config as cf
import random
import math
from network import Network
from network import Energy
import matplotlib.pyplot as plt
from random import *

def Get_Fitness(network,CH,Alive_Node):
    fitness = 0
    a = 0.3
    CH_ARR = np.zeros(len(CH))
    DIST_ARR = np.zeros(len(CH))
    TEMP_DIST = np.zeros(len(CH))
    COUNT_ARR = np.ones(len(CH))
    for i in Alive_Node:
        x,y = network.node[i]['pos']
        for j in range(0,len(CH)):
            RES = network.node[CH[j]]['res_energy']
            x2,y2 = network.node[CH[j]]['pos']
            dist = math.sqrt((x-x2)**2+(y-y2)**2)
            if dist == 0:
                continue
            dis2 = network.node[CH[j]]['RTBS']
            CH_ARR[j] = RES/(dist*dis2*COUNT_ARR[j])
            TEMP_DIST[j] = dist
        idx = np.where(CH_ARR == np.max(CH_ARR))
        DIST_ARR[idx] += TEMP_DIST[idx]
        COUNT_ARR[idx] += 1
            
    f1 = 0
    f2 = 0
    for i in range(0,len(CH)):
        x,y = network.node[CH[i]]['pos']
        # RES = res_arr[i]
        BSDist = network.node[CH[i]]['RTBS']
        f1 += (DIST_ARR[i]+BSDist)/(COUNT_ARR[i]+1)
        f2 += RES
    fitness = a*f1 + (1-a)*(1/f2)
    return fitness



def Optimizer(network, Alive_Node, Update=False, R=30, In_Median=30,First = False):
    PSO_NET = nx.create_empty_copy(network)
    PSO_CHID = []
    M = max(round(cf.P_CH*len(Alive_Node)),1)
    SN = 40
    MIR = 10
    CH = []
    v_arr = []
    x_arr = []
    RES_ARR = []
    FIT = []
    ## initializing
    for i in range(0,SN):
        choice = np.random.choice(Alive_Node,M,replace = False)
        choice_x = []
        choice_v = []
        for j in choice:
            x,y = PSO_NET.node[j]['pos']
            choice_x.append([x,y])
            choice_v.append([0,0])
        CH.append(choice)
        x_arr.append(choice_x)
        v_arr.append(choice_v)
        FIT.append(Get_Fitness(PSO_NET,choice,Alive_Node))
    v_arr = np.array(v_arr)
    x_arr = np.array(x_arr)    
    Gbest = np.where(np.min(FIT)==FIT)[0][0]
    w = np.array([0.9,0.9])
    ##update
    for Iter in range(0,MIR):
        PGD = x_arr[Gbest]
        for i in range(0,SN):
            if FIT[i] == FIT[Gbest]:
                continue
            for j in range(0,len(CH[i])):
                v_arr[i][j] =  w * v_arr[i][j] + 2*uniform(0,1)*(PGD[j] - x_arr[i][j])
                x_arr[i][j] += v_arr[i][j]
                NNDist = 10000
                NNID = 0
                x1,y1= x_arr[i][j]
                for T in Alive_Node:
                    x2,y2 = PSO_NET.node[T]['pos']
                    NewDist = math.sqrt((x1-x2)**2+(y1-y2)**2)
                    if NewDist < NNDist:
                        NNDist = NewDist
                        NNID = T
                CH[i][j] = NNID
            FIT[i] = Get_Fitness(PSO_NET,CH[i],Alive_Node)
            if FIT[i] < FIT[Gbest]:
                Gbest = i
        w = [w[0] - (0.9-0.4)/MIR, w[1] - (0.9-0.4)/MIR]


    PSO_CHID = CH[Gbest]
    for i in Alive_Node:
        if i in PSO_CHID:
            PSO_NET.node[i]['Next'] = 0
            continue
        NNDist = 1000
        CH_ARR = np.zeros(len(PSO_CHID))
        x,y = PSO_NET.node[i]['pos']
        COUNT_ARR = np.ones(len(PSO_CHID))
        for j in range(0,len(PSO_CHID)):
            if i == j:
                continue
            RES = PSO_NET.node[PSO_CHID[j]]['res_energy']
            x2,y2 = PSO_NET.node[PSO_CHID[j]]['pos']
            dist = math.sqrt((x-x2)**2+(y-y2)**2)
            dis2 = math.sqrt((x2-50)**2+(y2-50)**2)
            CH_ARR[j] = RES/(dist*dis2*COUNT_ARR[j])
        idx = np.where(CH_ARR == np.max(CH_ARR))[0][0]
        COUNT_ARR[idx] += 1
        PSO_NET.node[i]['Next'] = PSO_CHID[idx]

    if First == True:
        for i in Alive_Node:
            PSO_NET.add_edge(i,PSO_NET.node[i]['Next'])

    return PSO_NET, PSO_CHID, R, In_Median