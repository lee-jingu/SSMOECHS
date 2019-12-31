import numpy as np
import copy
from scipy.spatial import distance

def Node(network,CHID, CHID_POS,NODE_POS):
    Dist = []
    for i in range(0,len(CHID_POS)):
        d = distance.euclidean(NODE_POS,CHID_POS[i])
        Dist.append(d)
    point = np.argmin(Dist)

    return CHID[point]