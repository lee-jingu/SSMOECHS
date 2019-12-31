import config as cf
import networkx as nx
import config as cf
from network import Energy

def Transmit(network,NodeID,RXID,MSG):
    IS_DEATH = 0
    ETX = Energy.ETX(network,NodeID,RXID,MSG) + cf.E_DA
    network,IS_DEATH = Energy.Consume(network,NodeID,ETX)
    return network, IS_DEATH

def Receive(network,NodeID,MSG):
    IS_DEATH=0
    ERX = cf.E_ELEC * MSG
    network, IS_DEATH = Energy.Consume(network,NodeID,ERX)
    return network, IS_DEATH
    