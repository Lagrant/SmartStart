"""
Empty Space Search (ES) algorithm.
"""

import numpy as np

def empty_center(coor, data, neigh, movestep, iternum, bounds=np.array([[-1, 1]])):
    """
    Empty center search process.
    """
    
    es_configs = []
    for i in range(iternum):
        adjs_,  distances_= neigh.query(coor)
        
        direc = elastic(coor, data[adjs_[0]], distances_[0])
        mag = np.linalg.norm(direc)
        if mag < 1e-7:
            break
        direc /= mag
        coor += direc * movestep

        if (coor < bounds[:, 0]).any() or (coor > bounds[:, 1]).any():
            np.clip(coor, bounds[:, 0], bounds[:, 1], out=coor)
            es_configs.extend(coor.tolist())
            break

    return coor

def force(sigma, d):
    """
    Optimized Force function.
    """
    ratio = sigma / d  # Reuse this computation
    ratio = np.clip(ratio, a_min=None, a_max=3.1622)  # Avoids overflow
    attrac = ratio ** 6
    attrac = np.clip(attrac, a_min=None, a_max=1000)  # Avoids overflow
    
    return 6 * (2 * attrac ** 2 - attrac) / d

def elastic(es, neighbors, neighbors_dist):
    """
    Optimized Elastic force with vectorization.
    """
    sigma = np.mean(neighbors_dist) / 2
    neighbors_dist = np.clip(neighbors_dist, a_min=0.001, a_max=None)  # Avoids distances < 0.001

    # Vectorized force computation
    forces = force(sigma, neighbors_dist)

    # Vectorized displacement computation
    vecs = (es - neighbors) / neighbors_dist[:, np.newaxis]
    
    # Compute the directional force
    direc = np.sum(vecs * forces[:, np.newaxis], axis=0)
    
    return direc
