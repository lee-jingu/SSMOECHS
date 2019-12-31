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

def Get_Fitness(network,CHID,Alive_Node):
    Fitness = 0
    CHID_ARR = np.zeros(len(CHID),dtype=np.int32)
    DIST_ARR = np.zeros(len(CHID))
    RES_CHID = 0
    RES_ALL = 0
    for i in Alive_Node:
        RES_ALL += network.node[i]['res_energy']
        NNID = 0
        NNDist = 1000
        x1,y1 = network.node[i]['pos']
        for j in range(0,len(CHID)):
            if i == CHID[j]:
                RES_CHID += network.node[CHID[j]]['res_energy']
                continue
            x2,y2 = network.node[CHID[j]]['pos']
            NewDist = math.sqrt((x1-x2)**2+(y1-y2)**2)
            if NewDist < NNDist:
                NNDist = NewDist
                NNID = j
        DIST_ARR[NNID]+= NNDist
        CHID_ARR[NNID]+= 1
    FIT1 = np.max(DIST_ARR/(1+CHID_ARR))
    FIT2 = RES_ALL/RES_CHID

    Fitness = (FIT1 + FIT2)*0.5
    return Fitness


def Optimizer(network, Alive_Node, Update=False, R=30, In_Median=30,First = False,a=False):
    PSO_NET = nx.create_empty_copy(network)
    PSO_CHID = []
    M = max(round(cf.P_CH*len(Alive_Node)),1)
    SN = 40
    MIR = 100
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
    ##clustering
    for i in Alive_Node:
        x,y = PSO_NET.node[i]['pos']
        NNDist = 10000
        NNID = 0
        for j in PSO_CHID:           
            x2,y2 = PSO_NET.node[j]['pos']
            NewDist = math.sqrt((x-x2)**2+(y-y2)**2)
            if NNDist > NewDist:
                NNDist = NewDist
                NNID = j
        PSO_NET.node[i]['Next'] = NNID

    for i in PSO_CHID:
        PSO_NET.node[i]['Next']=0
            
    if First == True:
        for i in Alive_Node:
            if i in PSO_CHID:
                continue
            PSO_NET.add_edge(i,PSO_NET.node[i]['Next'])

    return PSO_NET, PSO_CHID, R, In_Median