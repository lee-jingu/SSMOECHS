import numpy as np
import config as cf
import networkx as nx
import matplotlib.pyplot as plt
from network import Energy
from network import Network
from network import Node
from Optimizer import LEACH, PSOECHS, SSMO_MH, SSMO, BSMO, LEACH_C, SSMO2,PSO_C
from utils import Plot
from scipy.spatial import distance
import math
import pandas as pd

## Initializing Network
INIT_NET, Alive_Node, R = Network.init_network()
Optimizer_Array = [LEACH_C.Optimizer,PSO_C.Optimizer,BSMO.Optimizer,SSMO.Optimizer]
INIT_CHID = []
Trash = 0
# ## Plot Initial Network Topology
LEACHC_NET, LEACHC_CHID,R, Trash = Optimizer_Array[0](INIT_NET, Alive_Node,R=R, First=True, Update=True)
PSOC_NET,PSOC_CHID,R,Trash = Optimizer_Array[1](INIT_NET, Alive_Node, R=R, First=True, Update=True)
BSMO_NET, BSMO_CHID,R,Trash = Optimizer_Array[2](INIT_NET, Alive_Node,R=R, First=True, Update=True)
SSMOMH_NET, SSMOMH_CHID,R,Trash = Optimizer_Array[3](INIT_NET, Alive_Node,R=R, First=True, Update=True)

CHID_Arr = [INIT_CHID,LEACHC_CHID,PSOC_CHID,BSMO_CHID,SSMOMH_CHID]

network_array = [INIT_NET,LEACHC_NET,PSOC_NET,BSMO_NET,SSMOMH_NET]
plt.figure(1)
Plot.Topology(network_array, CHID_Arr, [2,3])


## Plot Energy Consume Data
plt.figure(2)
FHL_ALL = []
PACKET_ALL = []
Time =[]
for i in range(0,len(Optimizer_Array)):
    FHL = []
    FHL,PACKET,_TIME = Plot.EnergyConsume(INIT_NET,Optimizer_Array[i],Alive_Node, Label = cf.Optimizer[i+1],Alive=1, Total=0, Average=1, R=R, In_Median=30)
    PACKET_ALL.append(PACKET)
    FHL_ALL.append(FHL)
    Time.append(_TIME)

## Plot and Print Result
INDEX = []
for i in range(1,len(cf.Optimizer)):
    INDEX.append(cf.Optimizer[i])

IFHL = []
IPKT = []
for i in range(0,len(FHL_ALL[0])):
    temp = []
    temp2 = []
    for j in range(0,len(FHL_ALL)):
        temp.append(FHL_ALL[j][i])
        temp2.append(PACKET_ALL[j][i])
    IFHL.append(temp)
    IPKT.append(temp2)
df = pd.DataFrame(IFHL,index = ["99(FND)","90","80","70","60","50(HND)","40","30","20","10","0(LND)"], columns=pd.Index(INDEX,name = "Alive Nodes[ROUND]"))
df2 = pd.DataFrame(IPKT,index = ["99(FND)","90","80","70","60","50(HND)","40","30","20","10","0(LND)"], columns=pd.Index(INDEX,name = "TRANSMIT PACKET"))
df.plot(kind="bar",grid=True)
print(df)
print(df2)

plt.show()