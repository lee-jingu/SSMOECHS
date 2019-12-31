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

def Get_Fitness(network,SM,Alive_Node):
    Fitness = 0
    f1 = 0
    f2 = 0
    f3 = 0
    f4 = 0
    CHID = np.where(SM==np.max(SM))[0]+1
    potf3 = np.zeros(cf.N_NODE+1)
    potf4 = np.zeros(cf.N_NODE+1)

    for i in Alive_Node:
        x,y = network.node[i]['pos']
        NNDist = 1000
        NNID = 0
        for j in CHID:
            if i == j:
                d = network.node[j]['RTBS']
                if d > cf.D_o:
                    f1 += cf.E_ELEC*cf.L*cf.E_MP*(d**4) + cf.E_DA
                else:
                    f1 += cf.E_ELEC*cf.L*cf.E_FS*(d**2) + cf.E_DA
                continue
            if i in CHID:
                continue
            x2,y2 = network.node[j]['pos']
            NewDist = math.sqrt((x-x2)**2+(y-y2)**2)
            if NewDist<NNDist:
                NNDist = NewDist
                NNID = j
        if NNDist > cf.D_o:
            f1 += cf.NCH_L * (2*cf.E_ELEC + cf.E_MP*(NNDist**4))
            f2 += NNDist
            potf3[NNID] += network.node[i]['res_energy']
            potf4[NNID] += 1
        else:
            f1 += cf.NCH_L * (2*cf.E_ELEC + cf.E_FS*(NNDist**2))
            f2 += NNDist
            potf3[NNID] += network.node[i]['res_energy']
            potf4[NNID] += 1

    MinDist = 1000
    for i in CHID:
        potf3[i] = potf3[i]/network.node[i]['res_energy']
        for j in CHID:
            if i==j:
                continue
            x,y = network.node[i]['pos']
            x2,y2 = network.node[j]['pos']
            NewMinDist = math.sqrt((x-x2)**2+(y-y2)**2)
            if NewMinDist < MinDist:
                MinDist = NewMinDist

    f2 = f2/MinDist
    f3 = np.sum(potf3)
    f4 = np.max(potf4)+1
    Fitness = (f1+f2+f3+f4) *0.25
    Fitness = 1/(1+Fitness)
    return Fitness
            
    


def Optimizer(network, Alive_Node, Update=False, R=30, In_Median=30,First = False):
    BSMO_NET = nx.create_empty_copy(network)
    BSMO_CHID = []
    Swarm_Size = 40
    MIR = 100

    if Update == True:
        MAX_X = 0
        MAX_Y = 0
        for i in Alive_Node:
            x,y =BSMO_NET.node[i]['pos']
            if x > MAX_X:
                MAX_X = x
            if y > MAX_Y:
                MAX_Y = y
        
        R = math.sqrt(MAX_X**2 + MAX_Y**2)/4

    ##Initializing
    SM_Arr = []
    FIT = []
    MG = 4
    Group0 = []
    Group1 = []
    Group2 = []
    Group3 = []
    NGroup = 1
    LLL = np.zeros(MG)
    GLL = 0
    MLLL = 10
    MGLL = 20
    NB_Cluster = max(round(cf.P_CH*len(Alive_Node)),1)
    for i in range(0,Swarm_Size):
        SM = []
        for j in Alive_Node:
            if random()<=cf.P_CH:
                SM.append(1)
            else:
                SM.append(0)
        SM_Arr.append(SM)
        FIT.append(Get_Fitness(BSMO_NET,SM,Alive_Node))
        Group0.append(i)


    Pr = 0.1
    LLID = np.where(np.max(FIT)==FIT)[0][0]
    GLID = np.where(np.max(FIT)==FIT)[0][0]

    for Iter in range(0,MIR):
        ## Local Leader Phase
        Pr = Pr + (0.4-0.1)/MIR
        for i in range(0,MG):
            if i == 0:
                temp = Group0
            if i == 1:
                temp = Group1
            if i == 2:
                temp = Group2
            if i == 3:
                temp = Group3
            
            ## find LLID
            MAXFIT = 0
            count = 0
            for ID in temp:
                TMPFIT = FIT[ID]
                if TMPFIT > MAXFIT:
                    LLID = ID
                    MAXFIT = TMPFIT

            for j in temp:
                if FIT[j] == FIT[LLID]:
                    continue
                if FIT[j] == FIT[GLID]:
                    continue
                if Pr > random():
                    SM = SM_Arr[j]
                    LL = SM_Arr[LLID]
                    Rand = np.random.choice(temp,1)[0]
                    SMR = SM_Arr[Rand]
                    b = randint(0,1)
                    d = randint(-1,1)
                    SM_Arr[j] = np.bitwise_xor(SM,np.bitwise_or(np.bitwise_and(b,np.bitwise_xor(LL,SM)),np.bitwise_and(d,np.bitwise_xor(SMR,SM))))
                    FIT[j] = Get_Fitness(BSMO_NET,SM_Arr[j],Alive_Node)
                if FIT[j] > FIT[LLID]:
                    count = 1
                    LLIDPOT = j
            if count == 0:
                LLL[i] += 1
            else:
                count = 0
                LLID = LLIDPOT
            
            ## Local Leader Decision
            if LLL[i] == MLLL:
                LLL[i] = 0
                for TT in temp:
                    if FIT[TT] == FIT[LLID]:
                        continue
                    if FIT[TT] == FIT[GLID]:
                        continue
                    if Pr > random():
                        SM = SM_Arr[TT]
                        LL = SM_Arr[LLID]
                        GL = SM_Arr[GLID]
                        b = randint(0,1)
                        SM_Arr[TT] = np.bitwise_xor(SM,np.bitwise_or(np.bitwise_and(b,np.bitwise_xor(LL,SM)),np.bitwise_and(b,np.bitwise_xor(GL,SM))))
                        FIT[TT] = Get_Fitness(BSMO_NET,SM_Arr[TT],Alive_Node)
                    
                    else:
                        SM = []
                        for KT in Alive_Node:
                            if random()<cf.P_CH:
                                SM.append(KT)
                            else:
                                SM.append(KT)
                        SM_Arr[TT] = SM
                        FIT[TT] = Get_Fitness(BSMO_NET,SM_Arr[TT],Alive_Node)
    


        ## Global Leader Phase
        count = 0
        GLID = np.where(np.max(FIT)==FIT)[0][0]
        for i in range(0,len(SM_Arr)):
            if FIT[i] == FIT[GLID]:
                continue
            Prob =  0.9 * (FIT[i]/FIT[GLID]) + 0.1
            if Prob > random():
                GL = SM_Arr[GLID]
                SM = SM_Arr[i]
                Rand = randint(0,Swarm_Size-1)
                SMR = SM_Arr[Rand]
                b = randint(0,1)
                d = randint(-1,1)
                SM_Arr[i] = np.bitwise_xor(SM,np.bitwise_or(np.bitwise_and(b,np.bitwise_xor(GL,SM)),np.bitwise_and(d,np.bitwise_xor(SMR,SM))))
                FIT[i] = Get_Fitness(BSMO_NET,SM_Arr[i],Alive_Node)
                if FIT[i] > FIT[GLID]:
                    count = 1
        if count == 0:
            GLL += 1
        else:
            count = 0
            GLID = np.where(np.max(FIT)==FIT)[0][0]
        
        ## Global Desision
        if GLL == MGLL:
            GLL = 0
            NGroup += 1
            Choice_Node = np.arange(0,Swarm_Size,1)
            if NGroup == 2:
                Group0 = np.random.choice(Choice_Node,int(len(Choice_Node)/NGroup),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group0))
                Group1 = np.array(Choice_Node)
            if NGroup == 3:
                Group0 = np.random.choice(Choice_Node,int(len(Choice_Node)/NGroup),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group0))
                Group1 = np.random.choice(Choice_Node,int(len(Choice_Node)/NGroup),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group1))
                Group2 = np.array(Choice_Node)
            if NGroup == 4:
                Group0 = np.random.choice(Choice_Node,int(len(Choice_Node)/NGroup),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group0))
                Group1 = np.random.choice(Choice_Node,int(len(Choice_Node)/NGroup),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group1))
                Group2 = np.random.choice(Choice_Node,int(len(Choice_Node)/NGroup),replace=False)
                Choice_Node = list(set(Choice_Node)-set(Group2))
                Group3 = np.array(Choice_Node)
            if NGroup == 5:
                BSMO_CHID = SM_Arr[GLID]

    INNER = []
    OUTER = []
    BSMO_CHID = np.where(SM_Arr[GLID]==np.max(SM_Arr[GLID]))[0] + 1
    for i in BSMO_CHID:
        if BSMO_NET.node[i]['RTBS'] < R:
            INNER.append(i)
            BSMO_NET.node[i]['Next'] = 0
        else:
            OUTER.append(i)
    
    for i in Alive_Node:
        if i in BSMO_CHID:
            continue
        x,y = BSMO_NET.node[i]['pos']
        NNDist = 1000
        NNID = 0
        for j in BSMO_CHID:
            if i == j:
                continue
            x2,y2 = BSMO_NET.node[j]['pos']
            NewDist = math.sqrt((x-x2)**2+(y-y2)**2)
            if NNDist > NewDist:
                NNID = j
                NNDist = NewDist
        BSMO_NET.node[i]['Next'] = NNID
    
    for i in OUTER:
        NNID = 0
        NNDist = 1000
        x,y = BSMO_NET.node[i]['pos']
        for j in INNER:
            x2,y2 = BSMO_NET.node[j]['pos']
            NewDist = math.sqrt((x-x2)**2+(y-y2)**2)
            if NNDist > NewDist:
                NNID = j
                NNDist = NewDist
        BSMO_NET.node[i]['Next'] = NNID

    if First == True:
        ## add_Edge 
        for i in Alive_Node:
            BSMO_NET.add_edge(i,BSMO_NET.node[i]['Next'])

    return BSMO_NET, BSMO_CHID, R