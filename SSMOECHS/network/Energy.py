import config as cf
import numpy as np
import networkx as nx
import config as cf
import math

def ETX(network, node1, node2,MSG_L):
    x1,y1 = network.node[node1]['pos']
    x2,y2 = network.node[node2]['pos']
    d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    # MSG_L = (network.node[node1]['N_Packet']+1)*cf.L
    if d > cf.D_o:
        return MSG_L*cf.E_ELEC+MSG_L*cf.E_MP*(d**4) + cf.E_DA
    else:
        return MSG_L*cf.E_ELEC+MSG_L*cf.E_FS*(d**2) + cf.E_DA

def ERX(network, Node,MSG_L):
    # MSG_L = network.node[Node]['N_Packet']*cf.L
    return MSG_L * cf.E_ELEC

def Consume(network, NodeID, Energy):
    IS_DEATH=0
    if network.node[NodeID]['res_energy'] >= Energy:
        network.node[NodeID]['res_energy'] -= Energy
    else:
        network.node[NodeID]['res_energy'] = -1
        IS_DEATH = 1

    return network, IS_DEATH