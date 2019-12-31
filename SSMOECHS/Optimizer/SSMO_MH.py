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

def Get_Fitness(network,SMID,Alive_Node):
    Fitness = 0
    Consume = 0
    Cover = np.zeros(cf.N_NODE+1)
    INNER = []
    OUTER = []
    RTBS = []
    for T in SMID:
        RTBS.append(network.node[T]['RTBS'])
    CENTER = np.median(RTBS)
    for i in Alive_Node:
        x1,y1 = network.node[i]['pos']
        NNDist = 1000
        NNID = 0
        for j in SMID:
            if i == j:
                if network.node[i]['RTBS']<CENTER:
                    INNER.append(i)
                    Consume += Energy.ETX(network,i,0,cf.L) + cf.E_DA
                else:
                    OUTER.append(i)
                continue
            x2,y2 = network.node[j]['pos']
            NewDist = math.sqrt((x1-x2)**2+(y1-y2)**2)
            if NewDist < NNDist:
                NNDist = NewDist
                NNID = j
        Cover[NNID] += 1
        Consume += Energy.ETX(network,i,NNID,cf.NCH_L)+cf.E_DA
    for k in OUTER:
        NNID = 0
        NNDist = 1000
        xo,yo = network.node[k]['pos']
        for j in INNER:
            xi,yi = network.node[k]['pos']
            NewDist = math.sqrt((xi-xo)**2+(yi-yo)**2)
            if NewDist <= NNDist:
                NNID = j
                NNDist = NewDist
        Consume += Energy.ETX(network,k,NNID,cf.NCH_L) + cf.E_DA
    f1 = Consume
    f2 = np.max(Cover) - np.min(Cover)

    Fitness = 1/(f1+f2)
    
    return Fitness

def Optimizer(network, Alive_Node, Update=False, R=30, In_Median=30, First=False):
    NET_MAX = 0
    SSMO_NET = nx.create_empty_copy(network)
    SSMO_CHID = []
    NB_Cluster = max(round(cf.P_CH*len(Alive_Node)),1)
    update = 0
    if Update == True:
        Rmax = 0
        for i in Alive_Node:
            R_tmp = math.sqrt((SSMO_NET.node[i]['RTBS']**2)/NB_Cluster)
            if R_tmp > Rmax:
                Rmax = R_tmp
                
            if Rmax != R:
                R = Rmax
                update = 1

    
    if update == 1:
        INNER = []
        for i in Alive_Node:
            if SSMO_NET.node[i]['RTBS'] < R:
                INNER.append(i)
            SSMO_NET.node[i]['Cover'] = []
            for j in Alive_Node:
                if i == j:
                    continue
                x1,y1 = SSMO_NET.node[i]['pos']
                x2,y2 = SSMO_NET.node[j]['pos']
                D = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                if D < R:
                    SSMO_NET.node[i]['Cover'].append(j)
        In_Median = np.median(INNER)
        if len(INNER) ==0:
            In_Median = 0
            
        

    ## Initializing Phase
    SM_Arr = []
    MG = 5
    MIR = 100
    Swarm_Size = 40
    FIT = []
    
    MGLL = 20
    MLLL = 8
    Group0 = []
    Group1 = []
    Group2 = []
    Group3 = []

    for i in range(0,Swarm_Size):
        choice = np.random.choice(Alive_Node,NB_Cluster,replace = False)
        SM_Arr.append(choice)
        Group0.append(i)
        FIT.append(Get_Fitness(SSMO_NET,choice,Alive_Node))
    
    Group = 1
    GLID = np.where(FIT==np.max(FIT))[0][0]
    LLID_ARR = np.zeros(MG,dtype=np.int32)
    LLID_ARR[0] = GLID
    Pr = 0.1
    GLL = 0
    for Iter in range(0,MIR):
        LLL = 0

        ## Local Leader Phase
        Pr += (0.4-0.1)/MIR
        for i in range(0,Group):
            if i == 0:
                temp = Group0
            if i == 1:
                temp = Group1
            if i == 2:
                temp = Group2
            if i == 3:
                temp = Group3

            LLID = LLID_ARR[i]
            LLMAX = FIT[LLID]
            LMAX = FIT[LLID]
            MAXFIT = FIT[LLID]
            
            for j in temp:
                if j == LLID or j == GLID:
                    continue

                if random() < Pr:
                    Prob_Arr = []
                    LL = SM_Arr[LLID]
                    SM = SM_Arr[j]
                    Rand = np.random.choice(temp,1)[0]
                    SMR = SM_Arr[Rand]
                    ARANGE = np.hstack([SM,LL,SMR])
                    b = uniform(0,1)
                    d = uniform(-1,1)
                    PROBSM = np.ones(NB_Cluster) * (1-b-d)
                    PROBLL = np.ones(NB_Cluster) * (b)
                    PROBSMR = np.ones(NB_Cluster) * (d)
                    Prob_Arr = np.hstack([PROBSM,PROBLL,PROBSMR])
                    Prob_Arr = np.exp(Prob_Arr)/np.sum(np.exp(Prob_Arr))
                    choice = np.random.choice(ARANGE,NB_Cluster,replace = False, p = Prob_Arr/np.sum(Prob_Arr))
                    SM_Arr[j] = choice
                    FIT[j] = Get_Fitness(SSMO_NET,choice,Alive_Node)
                    if LMAX < FIT[j]:
                        LMAX = FIT[j]
                        LLID_ARR[i] = j
            if LLMAX == LMAX:
                LLL += 1


        ## Global Leader Phase
        GLID = np.where(FIT==np.max(FIT))[0][0]
        for i in range(0,Swarm_Size-1):
            
            GGLMAX = FIT[GLID]
            GLMAX = FIT[GLID]
            if i == GLID:
                continue            
            Prob = 0.9*(FIT[i]/FIT[GLID]) + 0.1
            if Prob > random():
                GL = SM_Arr[GLID]
                SM = SM_Arr[i]
                Rand = np.random.choice(Group0,1)[0]
                SMR = SM_Arr[Rand]
                ARANGE = np.hstack([SM,GL,SMR])
                b = uniform(0,1)
                d = uniform(-1,1)
                PROBSM = np.ones(NB_Cluster) * (1-b-d)
                PROBGL = np.ones(NB_Cluster) * (b)
                PROBSMR = np.ones(NB_Cluster) * (d)
                Prob_Arr = np.hstack([PROBSM,PROBGL,PROBSMR])
                Prob_Arr = np.exp(Prob_Arr)/np.sum(np.exp(Prob_Arr))
                choice = np.random.choice(ARANGE,NB_Cluster,replace = False, p = Prob_Arr/np.sum(Prob_Arr))
                SM_Arr[i] = choice
                FIT[i] = Get_Fitness(SSMO_NET,choice,Alive_Node)
                if FIT[i]>GLMAX:
                    GLMAX = FIT[i]
                    GLID = i
        if GLMAX == GGLMAX:
            GLL += 1


        ## Local Decision Phase
        # if LLL == MLLL:

        ## Global Decision Phase
        if GLL == MGLL:
            GLL = 0
            Group += 1
            Choice_Node = np.arange(0,Swarm_Size,1)
            if Group == 2:
                Group0 = np.random.choice(Choice_Node,int(len(Choice_Node)/Group),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group0))
                Group1 = np.array(Choice_Node)
            if Group == 3:
                Group0 = np.random.choice(Choice_Node,int(len(Choice_Node)/Group),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group0))
                Group1 = np.random.choice(Choice_Node,int(len(Choice_Node)/Group),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group1))
                Group2 = np.array(Choice_Node)
            if Group == 4:
                Group0 = np.random.choice(Choice_Node,int(len(Choice_Node)/Group),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group0))
                Group1 = np.random.choice(Choice_Node,int(len(Choice_Node)/Group),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group1))
                Group2 = np.random.choice(Choice_Node,int(len(Choice_Node)/Group),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group2))
                Group3 = np.array(Choice_Node)
            if Group == 5:
                SSMO_CHID = SM_Arr[GLID]

    SSMO_CHID = SM_Arr[GLID]
    INNER = []
    OUTER = []
    RTBS = []
    for i in SSMO_CHID:
        RTBS.append(SSMO_NET.node[i]['RTBS'])
    CENTER = np.median(RTBS)
    for i in Alive_Node:
        if i in SSMO_CHID:
            if network.node[i]['RTBS']>CENTER:
                OUTER.append(i)
                continue
            else:
                INNER.append(i)
                SSMO_NET.node[i]['Next'] = 0
                continue
        x1,y1 = SSMO_NET.node[i]['pos']
        NNID = 0
        NN_Dist = 1000
        for NN in SSMO_CHID:
            x2,y2 = SSMO_NET.node[NN]['pos']
            new_dist = math.sqrt((x1-x2)**2+(y1-y2)**2)
            if new_dist<NN_Dist:
                NNID = NN
                NN_Dist = new_dist
        SSMO_NET.node[i]['Next']=NNID

    for i in OUTER:   
        NNID = 0
        NN = SSMO_NET.node[i]['RTBS']
        x,y = SSMO_NET.node[i]['pos']
        for j in INNER:
            x2,y2 = SSMO_NET.node[j]['pos']
            Dist = math.sqrt((x-x2)**2+(y-y2)**2)
            if Dist < NN:
                NNID = j
                NN = Dist
        SSMO_NET.node[i]['Next'] = NNID

    if First==True:
        ## add_Edge 
        for i in Alive_Node:
            SSMO_NET.add_edge(i,SSMO_NET.node[i]['Next'])
        
    return SSMO_NET, SSMO_CHID, R, In_Median