import config as cf
import numpy as np
import math
import random
from random import *
import networkx as nx
from network import Node
from network import Energy

def init_network():
    print('INITIALIGING NETWORK...')
    network = nx.DiGraph()
    position = np.random.uniform(low=0, high=cf.AREA_H, size=(cf.N_NODE+1,2))
    network.add_node(0,pos=(cf.SINK_X,cf.SINK_Y), res_energy=5000,round=0, Next=False, N_Packet = 0, Dist=[], Cover = []) ##BS
    Alive_Node = []
    
    R_MAX = 0
    for i in range(1,cf.N_NODE+1):
        # uniform(cf.E_INIT*0.5,cf.E_INIT)
        R_tmp = math.sqrt((position[i][0]-cf.SINK_X)**2 + (position[i][1]-cf.SINK_Y)**2)
        network.add_node(i, pos=position[i], res_energy=uniform(0.5,cf.E_INIT), round=1, Next=0, N_Packet = cf.L, Dist=[], Cover = [], RTBS = R_tmp)
        network.add_edge(i,0)
        Alive_Node.append(i)
        if R_tmp > R_MAX:
            R_MAX = R_tmp
        

    R = R_MAX/math.sqrt(cf.NB_Cluster)
    for i in range(1,cf.N_NODE+1):
        x,y = position[i]
        for j in range(1,cf.N_NODE+1):
            if j == i:
                continue
            x2,y2 = position[j]
            dist = math.sqrt((x-x2)**2 + (y-y2)**2)
            if dist < R:
                network.node[i]['Cover'].append(j)
                network.node[i]['Dist'].append(dist)      

    return network, Alive_Node, R

def Run_Round(network,Alive_Node,CHID,Optimizer):
    Death_Node=[]
    Res_Energy = []

    ## Recieve from BS(CH info)
    for i in CHID:
        IS_DEATH=0
        network, IS_DEATH = Node.Receive(network, i,cf.L)
        if IS_DEATH == 1:
            Death_Node.append(i)

    ## TX & RX
    Rx_Node = []
    for i in Alive_Node:
        Rx_Node = network.node[i]['Next']
        if Rx_Node == 0:
            MSG = cf.L
            network.node[Rx_Node]['N_Packet'] += MSG
            if Optimizer == 'SSMOECHS':
                network.node[Rx_Node]['N_Packet'] += MSG
        else:
            MSG = cf.NCH_L
        IS_DEATH = 0
        network, IS_DEATH = Node.Transmit(network,i,Rx_Node,MSG)
        

        ## Data Tx and Rx
        if IS_DEATH == 1:  ##Transmit Fail
            Death_Node.append(i)
        else: 
            network, IS_DEATH = Node.Receive(network,Rx_Node,MSG)
            if IS_DEATH == 1:  ##Recieve Fail
                Death_Node.append(Rx_Node)
    Alive_Node = list(set(Alive_Node) - set(Death_Node))

    for i in Alive_Node:
        Res_Energy.append([i,network.node[i]['res_energy']])


    return network, Alive_Node, Res_Energy


# def add_edge(network,Start_NODE,Goal_NODE):
#     # if network.node[Start_NODE]['Head'] == 'CH':
#     # N = len(network.in_edges(Start_NODE))+1
#     # cost = Energy.ETX(network,Start_NODE,Goal_NODE) + cf.E_DA + N*cf.E_RX
#     # else:
#     cost = Energy.ETX(network,Start_NODE,Goal_NODE)
#     network.add_edge(Start_NODE,Goal_NODE)
#     network.node[Start_NODE]['Cost']= cost
#     return network