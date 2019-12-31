from network import Network
import config as cf
import networkx as nx

def Optimizer(network, Alive_Node, Residual=False, R=30, IN_Median=False):
    DC_NET=nx.create_empty_copy(network)
    CHID=[]
    for i in Alive_Node:
        CHID.append(i)
        DC_NET.node[i]['Next']=0
        DC_NET.add_edge(i,0)
        DC_NET.node[i]['N_Packet']=cf.L

    return DC_NET,CHID,R