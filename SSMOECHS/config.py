import math
## MAXIMUM_ROUND
ROUND = 10000

## searching area
AREA_W = 100
AREA_H = 100

## sink position(or BS position)
SINK_X = 50        #sink position x
SINK_Y = 150         #sink position y

## a number of node and cluster head
N_NODE = 100        #a number of nodes
P_CH = 0.05          #CH percentage
NB_Cluster = round(N_NODE*P_CH)
CH_N = int(N_NODE*P_CH)

## simulator parameter
E_INIT = 1     #init energy
E_ELEC = 50e-9      #energy consume on circuit
E_DA = 5e-9         #5nJ/bit
E_FS = 10e-12   #0.0013pJ/bit
E_MP = 0.0013e-12       #multi-path channel param.
L = 6400            #packet length
NCH_L = 2800
D_o = math.sqrt(E_FS/E_MP) #Do

#tx param
TX_PARAM = []

#Optimizer
Optimizer = ['INITIAL','LEACH-C','PSO-C','SMOTECP','SSMOECHS']

#Aggregation function
Agg_Func = 0