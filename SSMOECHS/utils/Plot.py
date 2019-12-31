import networkx as nx
import matplotlib.pyplot as plt
import config as cf
import numpy as np
from network import Energy
from network import Network
from network import Node
from Optimizer import LEACH,DC
import copy
import time
import math
def Topology(network_array, CHID_Array, Plot_array):
    a,b = Plot_array
    
    
    for j in range(0,len(cf.Optimizer)):
        index = a*100 + b*10 + j+1
        pos = nx.get_node_attributes(network_array[j],'pos')
        _title = cf.Optimizer[j]

        if _title == 'LEACH-C':
            index = 232
        if _title == 'PSO-C':
            index = 233
        if _title == 'SMOTECP':
            index = 235
        if _title == 'SSMOECHS':
            index = 236
        if _title == 'INITIAL':
            index = 231
        network = network_array[j]
        BS = network.subgraph([0])
        CH = network.subgraph(CHID_Array[j])
        CH_edge = []

        plt.subplot(index)
        plt.title(_title)
        Node = np.arange(1,cf.N_NODE+1,1,dtype=np.int32)
        if _title != 'INITIAL':
            # plt.subplot(122)
            # plt.subplot(index2)
            nx.draw_networkx(network, pos=pos, node_color = 'blue', node_size = 20, with_labels = False, label = 'Node')
            nx.draw_networkx_nodes(BS, pos=pos, node_color = 'green', node_size=100, node_shape='X',with_labels='pos', label = 'BS')
            for i in range(0,len(CHID_Array[j])):
                Next_Node = network.node[CHID_Array[j][i]]['Next']
                CH_edge.append((CHID_Array[j][i],Next_Node))
            nx.draw_networkx_nodes(CH, pos=pos, node_color = 'red', node_size=35, node_shape='s',with_labels='pos', label = 'CH')
            nx.draw_networkx_edges(network, pos=pos, edgelist = CH_edge, width=2.0, edge_color='r', alpha=0.7)

        else:
            nx.draw_networkx_nodes(Node, pos=pos, node_color = 'blue', node_size=20, with_labels=False, label = 'Node')
            nx.draw_networkx_nodes(BS, pos=pos, node_color = 'green', node_size=100, node_shape='X',with_labels='pos', label = 'BS')
            
        plt.legend(loc = 'upper left')
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')

    return

def EnergyConsume(Init_network, _Optimizer, Alive_Node, Label, Alive = 0, Total = 0, Average = 1, R=30, In_Median=30):
    Order = Alive + Total + Average
    if Order == 0 :
        return
    ## initial settings
    X = []
    Alive_N = []
    Total_Energy = []
    Average_Energy = []
    Res_Energy = []
    FHL = []
    PKT = []
    Time =[]
    ## Initializing
    Alive_Node = np.arange(1,cf.N_NODE+1,1)
    network, CHID,R,In_Median = _Optimizer(Init_network,Alive_Node,R,In_Median)
    for RI in Alive_Node:
        Res_Energy.append([RI,cf.E_INIT])
    update = True
    Consume = 0
    ## Consume and Plot Phase        
    for j in range(1,cf.ROUND+1):
        print(Label)
        start_time = time.time()
        FHLCHK = np.arange(cf.N_NODE,-10,-10)
        PREALIVE = len(Alive_Node)
        ## Optimizing
        network, CHID, R,In_Median = _Optimizer(network,Alive_Node,update,R,In_Median)
        # Energy Consume Phase
        PRESUM = np.sum(Res_Energy,axis=0)[1]
        network, Alive_Node, Res_Energy = Network.Run_Round(network,Alive_Node,CHID,Label)
        NEXTALIVE = len(Alive_Node)
        if PREALIVE == NEXTALIVE:
            update = False
        else:
            chker = len(FHL)
            if NEXTALIVE <= FHLCHK[chker]:
                FHL.append(j)
                PKT.append(network.node[0]['N_Packet'])
            update = True
        if len(Alive_Node) == 0:
            break
        else:
            Sum = np.sum(Res_Energy,axis=0)[1]
            Consume += (PRESUM - Sum)
            Var = np.var(Res_Energy,axis=0)[1]
        if Alive == 1:
            Alive_N.append(len(Alive_Node))
            print("{} Round.... Alive Node : {}".format(j,len(Alive_Node)))
        if Total == 1:
            Total_Energy.append(Var)
            print("{} Round.... Energy Variance : {}".format(j,Var))
        if Average == 1:
            Average_Energy.append(Consume)
            print("{} Round.... Total Consume Energy : {}".format(j,Consume))
        print("{} Round.... CHID : {}".format(j,CHID))
        X.append(j)
        end_time = round(time.time() - start_time,3)
        Time.append(end_time)
        print("{} Round.... Time Consume : {}s".format(j,end_time))
    X.append(j)
    Time.append(end_time)
    if Alive == 1:
        Alive_N.append(0)
        print("{} Round.... Alive Node : {}".format(j,0))
        index = Order *100 + 10 + Alive
        plt.subplot(index)
        plt.plot(X,Alive_N, label = Label)
        plt.xlabel('Round')
        plt.ylabel('Number of Alive Nodes')
        plt.ylim(0,cf.N_NODE*1.05)
        plt.legend(loc = 'lower left')
    if Total == 1:
        Total_Energy.append(0)
        print("{} Round.... Energy Variance: {}".format(j,0))
        index = Order * 100 + 10 + Alive+Total
        plt.subplot(index)
        plt.plot(X,Total_Energy, label = Label)
        plt.xlabel('Round')
        plt.ylabel('Energy Variance')
        plt.legend(loc = 'upper right')
    if Average == 1:
        print(Label)

        Average_Energy.append(cf.N_NODE*cf.E_INIT)
        print("{} Round.... Total Consume Energy : {}".format(j,cf.N_NODE*cf.E_INIT))
        index = Order * 100 + 10 + Alive + Average + Total
        plt.subplot(index)
        if Label == "LEACH-C":
            plt.plot(X,Time, label = Label, marker="o", markersize=5, color='blue')
        if Label== "PSO-C":
            plt.plot(X,Time, label = Label, marker="s", markersize=5, color='orange')
        if Label== "SMOTECP":
            plt.plot(X,Time, label = Label, marker="P", markersize=8, color='green')
        if Label== "SSMOECHS":
            plt.plot(X,Time, label = Label, marker="*", markersize=8, color='red')
        plt.xlabel('Round')
        plt.ylabel('Time')
        plt.legend(loc = 'upper left')
    return FHL, PKT, Time