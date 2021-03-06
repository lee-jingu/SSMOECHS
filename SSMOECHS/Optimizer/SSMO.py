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

def Get_Fitness(network,SMID,R,IN_MEDIAN,MAX,Alive_Node):
    Fitness = 0
    SUM = 0
    Cover = {0}
    RTBS = []
    INNER = []
    OUTER = []
    CNARR = []
    for i in SMID:
        RTBS.append(network.node[i]['RTBS'])
    MED = min(np.median(RTBS),cf.D_o)
    for i in SMID:
        Consume = 0
        RES = network.node[i]['res_energy']
        New_COVER = set(network.node[i]['Cover'])
        if network.node[i]['RTBS'] < MED:
            INNER.append(i)
            Consume += Energy.ETX(network,i,0,cf.L)
        else:
            OUTER.append(i)
        CN = len(New_COVER)
        CNARR.append(CN)
        Consume += CN *cf.NCH_L * cf.E_ELEC + cf.E_DA
        SUM += RES - Consume
        Cover.update(New_COVER)
    for i in OUTER:
        NNID = 0
        NNDist = network.node[i]['RTBS']
        x,y = network.node[i]['pos']
        for j in INNER:
            x2,y2 = network.node[j]['pos']
            NewDist = math.sqrt((x-x2)**2+(y-y2)**2)
            if NewDist < NNDist:
                NNID = j
        SUM -= Energy.ETX(network,i,NNID,cf.NCH_L)

    f1 = SUM / len(SMID)
    f2 = len(Cover)/len(Alive_Node)
    
    return f1,f2

def Get_MAX(network,SM_Array,R,IN_MEDIAN):
    FIT = []
    for i in range(0,len(SM_Array)):
        SUM = 0
        for j in range(0,len(SM_Array[i])):
            RES = network.node[SM_Array[i][j]]['res_energy']
            CN = len(network.node[SM_Array[i][j]]['Cover'])
            d = network.node[SM_Array[i][j]]['RTBS']
            if d > R:
                d -= IN_MEDIAN
            Consume = CN *cf.NCH_L * cf.E_ELEC + cf.L*(cf.E_ELEC + cf.E_FS*(d**2))+cf.E_DA
            SUM += RES - Consume
        FIT.append(SUM/len(SM_Array[i]))
    return np.max(FIT)


def Optimizer(network, Alive_Node, Update=False, R=30, In_Median=30, First=False):
    SSMO_NET = nx.create_empty_copy(network)
    SSMO_CHID = []
    NB_Cluster = max(1,round(len(Alive_Node)*cf.P_CH))
    update = 0
    a = 0.5
    if Update == True:
        Rmax = 0
        for i in Alive_Node:
            x,y = SSMO_NET.node[i]['pos']
            R_tmp = math.sqrt(((x-50)**2+(y-50)**2))/math.sqrt(NB_Cluster)
            if R_tmp > Rmax:
                Rmax = R_tmp
                
            if Rmax != R:
                R = Rmax
                update = 1

    
    if update == 1:
        for i in Alive_Node:
            SSMO_NET.node[i]['Cover'] = []
            for j in Alive_Node:
                if i == j:
                    continue
                x1,y1 = SSMO_NET.node[i]['pos']
                x2,y2 = SSMO_NET.node[j]['pos']
                D = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                if D < R:
                    SSMO_NET.node[i]['Cover'].append(j)
            
        

    ## Initializing Phase
    SM_Arr = []
    MG = 5
    MIR = 100
    Swarm_Size = 40
    FIT = []
    FIT1 = []
    FIT2 = []
    MGLL = 10
    MLLL = 20
    Group0 = []
    Group1 = []
    Group2 = []
    Group3 = []

    for i in range(0,Swarm_Size):
        choice = np.random.choice(Alive_Node,NB_Cluster,replace = False)
        SM_Arr.append(choice)
        Group0.append(i)

    # NET_MAX = Get_MAX(SSMO_NET,SM_Arr,R,In_Median)
    NET_MAX = 0
    w1 = []
    w2 = []
    for i in range(0,Swarm_Size):
        f1 ,f2 = Get_Fitness(SSMO_NET,SM_Arr[i],R,In_Median,NET_MAX,Alive_Node)
        FIT1.append(f1)
        w1.append(a)
        FIT2.append(f2)
        w2.append(1-a)
    MAXFIT = np.max(FIT1)
    FIT1 = np.array(FIT1)
    FIT2 = np.array(FIT2)
    w1 = np.array(w1)
    w2 = np.array(w2)
    FIT = w1*FIT1/MAXFIT + w2*FIT2
    
    Group = 1
    GLID = np.where(FIT==np.max(FIT))[0][0]
    LLID_arr = np.zeros(MG,dtype=np.int32)
    LLID_arr[0] = GLID
    LLL = np.zeros(MG, dtype=np.int32)
    Pr = 0.1
    GLL = 0
    for Iter in range(0,MIR):
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

            LLID = LLID_arr[i]
            LLMAX = FIT[LLID]
            LMAX = FIT[LLID]
            MAXFIT = FIT[LLID]
            
            Prob_Arr = np.zeros(len(Alive_Node))
            for j in temp:
                if j in LLID_arr:
                    continue
                if j == GLID:
                    continue

                if random() < Pr:
                    LL = SM_Arr[LLID]
                    SM = SM_Arr[j]
                    Rand = np.random.choice(temp,1)[0]
                    SMR = SM_Arr[Rand]
                    ARANGE = np.hstack([SM,LL,SMR])
                    b = uniform(0,1)
                    d = uniform(-1,1)
                    PROBSM = np.ones(len(SM)) * (1-b-d)
                    PROBLL = np.ones(len(LL)) * (b)
                    PROBSMR = np.ones(len(SMR)) * (d)
                    Prob_Arr = np.hstack([PROBSM,PROBLL,PROBSMR])
                    Prob_Arr = np.exp(Prob_Arr)/np.sum(np.exp(Prob_Arr))
                    choice = list(set(np.random.choice(ARANGE,NB_Cluster,replace = False, p = Prob_Arr/np.sum(Prob_Arr))))
                    SM_Arr[j] = choice
                    FIT1[j], FIT2[j] = Get_Fitness(SSMO_NET,choice,R,In_Median,FIT[LLID],Alive_Node)
                    FIT[j] = a*FIT1[j]/MAXFIT + (1-a)*FIT2[j]
                    if LMAX < FIT[j]:
                        LMAX = FIT[j]
                        LLID_arr[i] = j
            if LLMAX == LMAX:
                LLL[i] += 1

            ## Local Leader Decision
            if LLL[i] == MLLL:
                LLL[i] = 0
                for j in temp:
                    if j in LLID_arr:
                        continue
                    if j == GLID:
                        continue
                    if random() < Pr:
                        LL = SM_Arr[LLID]
                        GL = SM_Arr[GLID]
                        SM = SM_Arr[j]
                        ARANGE = np.hstack([SM,LL,GL])
                        b = uniform(0,1)
                        c = uniform(0.2,1)
                        PROBSM = np.ones(len(SM)) * (1-b-c)
                        PROBLL = np.ones(len(LL)) * (b)
                        PROBGL = np.ones(len(GL)) * (c)
                        Prob_Arr = np.hstack([PROBSM,PROBLL,PROBGL])
                        Prob_Arr = np.exp(Prob_Arr)/np.sum(np.exp(Prob_Arr))
                        choice = list(set(np.random.choice(ARANGE,NB_Cluster,replace = False, p = Prob_Arr/np.sum(Prob_Arr))))
                    else:
                        choice = np.random.choice(Alive_Node,NB_Cluster,replace=False)
                    SM_Arr[j] = choice
                    FIT1[j], FIT2[j] = Get_Fitness(SSMO_NET,choice,R,In_Median,FIT[LLID],Alive_Node)
                    FIT[j] = a*FIT1[j]/MAXFIT + (1-a)*FIT2[j]
                    if LMAX < FIT[j]:
                        LMAX = FIT[j]
                        LLID_arr[i] = j


        ## Global Leader Phase
        for i in range(0,Swarm_Size-1): 
            GGLMAX = FIT[GLID]
            GLMAX = FIT[GLID]
            if i == GLID:
                FIT1[i] = 1
                FIT[i] = FIT1[i] + FIT2[i]
                continue
            if i in LLID_arr:
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
                PROBSM = np.ones(len(SM)) * (1-b-d)
                PROBGL = np.ones(len(GL)) * (b)
                PROBSMR = np.ones(len(SMR)) * (d)
                Prob_Arr = np.hstack([PROBSM,PROBGL,PROBSMR])
                Prob_Arr = np.exp(Prob_Arr)/np.sum(np.exp(Prob_Arr))
                choice = list(set(np.random.choice(ARANGE,NB_Cluster,replace = False, p = Prob_Arr/np.sum(Prob_Arr))))
                FIT1[i], FIT2[i] = Get_Fitness(SSMO_NET,choice,R,In_Median,FIT[LLID],Alive_Node)
                FIT[i] = a*FIT1[i]/MAXFIT + (1-a)*FIT2[i]
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
            Group = min(Group+1,MG-1)
            Choice_Node = np.arange(0,Swarm_Size,1)
            if Group == 2:
                Group0 = np.random.choice(Choice_Node,int(Swarm_Size/Group),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group0))
                Group1 = np.array(Choice_Node)
            if Group == 3:
                Group0 = np.random.choice(Choice_Node,int(Swarm_Size/Group),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group0))
                Group1 = np.random.choice(Choice_Node,int(Swarm_Size/Group),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group1))
                Group2 = np.array(Choice_Node)
            if Group == 4:
                Group0 = np.random.choice(Choice_Node,int(Swarm_Size/Group),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group0))
                Group1 = np.random.choice(Choice_Node,int(Swarm_Size/Group),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group1))
                Group2 = np.random.choice(Choice_Node,int(Swarm_Size/Group),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group2))
                Group3 = np.array(Choice_Node)


    SSMO_CHID = SM_Arr[GLID]
    INNER = []
    OUTER = []
    RTBS = []
    for i in SSMO_CHID:
        RTBS.append(SSMO_NET.node[i]['RTBS'])
    CENTER = min(np.average(RTBS),cf.D_o)
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
            if i in SSMO_CHID:
                continue
            SSMO_NET.add_edge(i,SSMO_NET.node[i]['Next'])
        
    return SSMO_NET, SSMO_CHID, R, In_Median
