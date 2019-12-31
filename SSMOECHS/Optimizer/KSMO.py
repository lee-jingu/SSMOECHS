import copy
import networkx as nx
import numpy as np
import config as cf
import random
import math
from network import Network
from network import Energy
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
from random import *

def Kriging(network, Alive_Node, R, IN_MEDIAN,Plot=False):
    X = []
    Y = []
    PREDICT = []
    data = []

    for i in range(1,len(network.node)):
        x,y = network.node[i]['pos']
        X.append(x)
        Y.append(y)
        RES = network.node[i]['res_energy']
        if RES == cf.E_INIT:
            CN = len(network.node[i]['Cover'])
            d = network.node[i]['RTBS']
            Consume = cf.L*(cf.E_ELEC + cf.E_FS*(d**2))
            PREDICT.append(RES-Consume)
            continue
        PREDICT.append(RES)
    
    OK = OrdinaryKriging(X, Y, PREDICT, variogram_model='gaussian')

    xgrid = np.arange(0,cf.AREA_W,0.25)
    ygrid = np.arange(0,cf.AREA_H,0.25)
    z, ss = OK.execute('grid', xgrid,ygrid)

    if Plot ==True:
        # if Plot == True:
        X = np.array(X)
        Y = np.array(Y)
        fig, ax = plt.subplots()
        im = ax.imshow(z, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
        ax.scatter(X, Y, c=PREDICT)
        fig.colorbar(im)
        ax.set(xlabel='X', ylabel='Y', title = "Next Potential Energy Predicted by Kriging")
        plt.show()
    return z

def Get_Fitness(network,CH,Alive_Node,Kriging_Map):
    fitness = 0
    a = 0.3
    CH_ARR = np.zeros(len(CH))
    DIST_ARR = np.zeros(len(CH))
    TEMP_DIST = np.zeros(len(CH))
    COUNT_ARR = np.ones(len(CH))
    for i in Alive_Node:
        x,y = network.node[i]['pos']
        for j in range(0,len(CH)):
            x2,y2 = CH[j]
            Kx2 = int(x2*4)
            if Kx2 >= 400:
                Kx2 = 399
            if Kx2 <=0:
                Kx2 = 1
            Ky2 = int(y2*4)
            if Ky2 >= 400:
                Ky2 = 399
            if Ky2 <=0:
                Ky2 = 1
            RES = Kriging_Map[Kx2,Ky2]
            dist = math.sqrt((x-x2)**2+(y-y2)**2)
            dis2 = math.sqrt((x2-50)**2+(y2-50)**2)
            CH_ARR[j] = RES/(dist*dis2*COUNT_ARR[j])
            TEMP_DIST[j] = dist
        idx = np.where(CH_ARR == np.max(CH_ARR))
        DIST_ARR[idx] += TEMP_DIST[idx]
        COUNT_ARR[idx] += 1
            
    f1 = 0
    f2 = 0
    for i in range(0,len(CH)):
        x,y = CH[i]
        if i >0:
            x2,y2 = CH[i-1]
        Kx = int(x*4)
        if Kx >= 400:
            Kx = 399
        if Kx <= 0:
            Kx = 1
        Ky = int(y*4)
        if Ky >= 400:
            Ky = 399
        if Ky <= 0:
            Ky = 1
        RES = Kriging_Map[Kx,Ky]
        BSDist = math.sqrt((x-50)**2+(y-50)**2)
        f1 += (DIST_ARR[i]+BSDist)/(COUNT_ARR[i])
        f2 += RES
   
    fitness = a*f1 + (1-a)*(1/f2)
    return fitness

def Optimizer(network, Alive_Node, Residual=False, R=30, In_Median=30, First=False):
    KSMO_NET = nx.create_empty_copy(network)
    KSMO_CHID = []
    
    ## find kriging map z
    z = Kriging(KSMO_NET, KSMO_CHID,R,In_Median, Plot=False)

    ## Initializing Phase
    CH = []
    FIT = []
    MG = 4
    MIR = 10
    Swarm_Size = 40
    NB_Cluster = round(cf.P_CH*len(Alive_Node))
    if NB_Cluster == 0:
        NB_Cluster = 1
    MGLL = 3
    MLLL = 20
    Group0 = []
    Group1 = []
    Group2 = []
    Group3 = []
    for i in range(0,Swarm_Size):
        pos = np.random.uniform(low=0, high=cf.AREA_H, size=(NB_Cluster,2))
        CH.append(pos)
        FIT.append(Get_Fitness(KSMO_NET,CH[i],Alive_Node,z))
    Group = 1
    GLID = np.where(FIT==np.min(FIT))[0][0]
    LLID = np.where(FIT==np.min(FIT))[0][0]
    Pr = 0.1
    Group0 = np.arange(0,len(CH),1)

    for Iter in range(0,MIR):
        GLL = 0
        LLL = 0

        ## Local Leader Phase
        Pr += (0.4-0.1)/MIR
        for i in range(0,Group):
            LLMAX = FIT[LLID]
            LMAX = FIT[LLID]
            if i == 0:
                temp = Group0
            if i == 1:
                temp = Group1
            if i == 2:
                temp = Group2
            if i == 3:
                temp = Group3
            if i == LLID:
                continue
            for T in temp:
                if Pr >random():
                    SM = CH[T]
                    LL = CH[LLID]
                    SMR = CH[randint(0,len(CH)-1)]
                    b = uniform(0,1)
                    d = 0
                    for j in range(0,len(SM)):
                        x1,y1 = SM[j]
                        x2,y2 = LL[j]
                        x3,y3 = SMR[j]
                        X_pot = x1 + b*(x2-x1) + d*(x3-x1)
                        Y_pot = y1 + b*(y2-y1) + d*(y3-y1)
                        SM[j] = [X_pot,Y_pot]
                    CH[T] = SM
                    FIT[T] = Get_Fitness(KSMO_NET,CH[T],Alive_Node,z)
                    if FIT[T] < LLMAX:
                        LLMAX = FIT[T]
                        LLID = T

                    if LLMAX == LMAX:
                        LLL += 1

        GLID = np.where(FIT==np.min(FIT))[0][0]
        ## Global Leader Phase
        for i in range(0,len(CH)):
            GGLMAX = FIT[GLID]
            GLMAX = FIT[GLID]
            if i == GLID:
                continue
            
            Prob = 0.9*(FIT[i]/FIT[GLID]) + 0.1
            if Prob > random():
                GL = CH[GLID]
                SM = CH[i]
                SMR = CH[randint(0,len(CH)-1)]
                b = uniform(0,1)
                d = 0
                for j in range(0,len(SM)):
                    x1,y1 = SM[j]
                    x2,y2 = GL[j]
                    x3,y3 = SMR[j]
                    X_pot = x1 + b*(x2-x1) + d*(x3-x1)
                    Y_pot = y1 + b*(y2-y1) + d*(y3-y1)
                    SM[j] = [X_pot,Y_pot]
                CH[i] = SM
                FIT[i] = Get_Fitness(KSMO_NET,CH[i],Alive_Node,z)
                if FIT[i]<GLMAX:
                    GLMAX = FIT[i]
                    GLID = i
        if GLMAX == GGLMAX:
            GLL += 1

        ## Local Decision Phase
        # if LLL == MLLL:

        ## Global Decision Phase
        


    for i in range(0,len(CH[GLID])):
        x,y = CH[GLID][i]
        CHID = 0
        NNDist = 1000
        for j in Alive_Node:
            x2,y2 = KSMO_NET.node[j]['pos']
            NewDist = math.sqrt((x-x2)**2+(y-y2)**2)
            if NNDist>NewDist:
                NNDist = NewDist
                CHID = j
        if CHID in KSMO_CHID:
            continue
        KSMO_CHID.append(CHID)

    if GLL == MGLL:
        Group += 1
        Choice_Node = copy.deepcopy(Alive_Node)
        if Group == 2:
            Group0 = np.random.choice(Choice_Node,int(len(Alive_Node)/Group),replace=False)
            Group1 = np.random.choice(Choice_Node,int(len(Alive_Node)/Group),replace=False)
            MGLL = 0
        if Group == 3:
            Group0 = np.random.choice(Choice_Node,int(len(Alive_Node)/Group),replace=False)
            Group1 = np.random.choice(Choice_Node,int(len(Alive_Node)/Group),replace=False)
            Group2 = np.random.choice(Choice_Node,int(len(Alive_Node)/Group),replace=False)
            MGLL = 0
        if Group == 4:
            Group0 = np.random.choice(Choice_Node,int(len(Alive_Node)/Group),replace=False)
            Group1 = np.random.choice(Choice_Node,int(len(Alive_Node)/Group),replace=False)
            Group2 = np.random.choice(Choice_Node,int(len(Alive_Node)/Group),replace=False)
            Group3 = np.random.choice(Choice_Node,int(len(Alive_Node)/Group),replace=False)
            MGLL = 0
        if Group == 5:
            KSMO_CHID = CH[GLID]
    ## Clustering
    for i in Alive_Node:
        if i in KSMO_CHID:
            KSMO_NET.node[i]['Next'] = 0
        else:
            NNDist = 1000
            CH_ARR = np.zeros(len(KSMO_CHID))
            x,y = KSMO_NET.node[i]['pos']
            COUNT_ARR = np.ones(len(KSMO_CHID))
            for j in range(0,len(KSMO_CHID)):
                x2,y2 = KSMO_NET.node[KSMO_CHID[j]]['pos']
                RES = KSMO_NET.node[KSMO_CHID[j]]['res_energy']
                dist = math.sqrt((x-x2)**2+(y-y2)**2)
                if dist==0:
                    continue
                dis2 = math.sqrt((x2-50)**2+(y2-50)**2)
                CH_ARR[j] = RES/(dist*dis2*COUNT_ARR[j])
            idx = np.where(CH_ARR == np.max(CH_ARR))[0][0]
            COUNT_ARR[idx] += 1
            KSMO_NET.node[i]['Next'] = KSMO_CHID[idx]

    if First ==True:
        ## add_Edge 
        for i in Alive_Node:
            KSMO_NET.add_edge(i,KSMO_NET.node[i]['Next'])

    return KSMO_NET, KSMO_CHID, R