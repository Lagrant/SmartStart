"""
Population initialization methods enhanced by empty space search.
"""
import joblib
import numpy as np
import random
import hnswlib
import pyBlindOpt.utils as utils
from src.empty_space_search import empty_center
from src.approximated_nearest_neighbor import NN
import pyBlindOpt.utils as utils

def scale(arr, min_val=None, max_val=None):
    scl_arr = (arr - min_val) / (max_val - min_val)
    return scl_arr, min_val, max_val
def inv_scale(scl_arr, min_val, max_val):
    return scl_arr*(max_val - min_val) + min_val

def es_opposition_based(objective:callable, bounds:np.ndarray,
    population:np.ndarray=None, n_pop:int=20, n_jobs:int=-1, seed:int=None) -> np.ndarray:
    '''
    Opposition-based population initialization.
    '''

    # set the random seed
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # check if the initial population is given
    if population is None:
        # initial population of random bitstring
        pop = [utils.get_random_solution(bounds) for _ in range(n_pop)]
    else:
        # initialise population of candidate and validate the bounds
        pop = [utils.check_bounds(p, bounds) for p in population]
        # overwrite the n_pop with the length of the given population
        n_pop = len(population)

    # compute the fitness of the initial population
    # scores = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective)(c) for c in pop)
    scores = utils.vectorized_evaluate(pop, objective)

    # compute the opposition population
    a = bounds[:,0]
    b = bounds[:,1]
    pop_opposition = [a+b-p for p in pop]

    # compute the fitness of the opposition population
    scores_opposition = utils.vectorized_evaluate(pop_opposition, objective)

    # merge the results and filter
    results = list(zip(scores, pop)) + list(zip(scores_opposition, pop_opposition))
    results.sort(key=lambda x: x[0])

    results = np.array([results[i][1] for i in range (len(results))])
    results, _, _ = scale(results, bounds[:,0], bounds[:,1])

    es_params = []
    neigh = NN('hnswlib', results.shape[1], n_neighbors=results.shape[1] + 1)
    neigh.fit(results)
    for res in results:
        es_param = empty_center(res.reshape(1, -1).copy(), results, \
                                neigh, movestep=0.01, iternum=100, \
                                    bounds=bounds)
        es_params.append(es_param)
    results = np.concatenate((results, np.concatenate(es_params)))
    results = inv_scale(results, bounds[:,0], bounds[:,1])
    scores_results = utils.vectorized_evaluate(results, objective)
    results = results[np.argsort(scores_results)[:n_pop]]
    return results