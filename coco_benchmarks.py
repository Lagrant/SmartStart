import time
import argparse
import numpy as np

import cocoex  # experimentation module
import cocopp  # post-processing module (not strictly necessary)

import pyBlindOpt.de as de
import pyBlindOpt.egwo as egwo
import pyBlindOpt.init as init

from src.es_opposition_learning import es_opposition_based


def acceleration_rate(reference:float, improved:float)->float:
    return 1.0 - (improved/(reference+np.finfo(float).eps))


def main(args, n_pop:int=100):

    blind_optimizers = {
        'de': de.differential_evolution,
        'egwo': egwo.grey_wolf_optimization
    }

    optimizer=blind_optimizers[args.o]

    ### input
    suite_name = "bbob"
    budget_multiplier = 20  # x dimension, increase to 3, 10, 30,...

    ### prepare
    for method in ['rnd', 'obl', 'oblesa']:
        suite = cocoex.Suite(suite_name, "", "")  # see https://numbbo.github.io/coco-doc/C/#suite-parameters
        output_folder = '{}_seed{}'.format(
                method, args.s)
        observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
        repeater = cocoex.ExperimentRepeater(budget_multiplier)  # 0 == no repetitions
        minimal_print = cocoex.utilities.MiniPrint()

        ### go
        while not repeater.done():  # while budget is left and successes are few
            for problem in suite:  # loop takes 2-3 minutes x budget_multiplier
                if repeater.done(problem):
                    continue  # skip this problem
                problem.observe_with(observer)  # generate data for cocopp
                problem(problem.dimension * [0])  # for better comparability
                repeater.track(problem)  # track evaluations and final_target_hit
                bounds = np.stack([problem.lower_bounds, problem.upper_bounds]).transpose()

                if method == 'rnd': 
                    start_time = time.process_time()
                    population = init.random(bounds, n_pop, seed=args.s)
                    end_time = time.process_time()
                    time_init = round(end_time - start_time, 3)
                elif method == 'obl':
                    start_time = time.process_time()
                    population = init.opposition_based(problem, bounds, n_pop=n_pop, seed=args.s)
                    end_time = time.process_time()
                    time_init = round(end_time - start_time, 3)
                elif method == 'oblesa':
                    start_time = time.process_time()
                    population = es_opposition_based(problem, bounds=bounds, n_pop=n_pop, seed=args.s)
                    end_time = time.process_time()
                    time_init = round(end_time - start_time, 3)
            
                result, objective = optimizer(problem, bounds, population=population,
                    n_iter=args.e, verbose=False)
                problem(result)  # make sure the returned solution is evaluated
                repeater.track(problem)  # track evaluations and final_target_hit
                minimal_print(problem)  # show progress
    # ### post-process data
    # cocopp.main(observer.result_folder + ' bfgs!');  # re-run folders look like "...-001" etc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the agents')
    
    parser.add_argument('-o', type=str, help='Optimization algorithm', choices=['de', 'egwo'], default='egwo')

    parser.add_argument('-e', type=int, help='Number of optimization epochs', default=500)

    parser.add_argument('-s', type=int, help='Random seed', default=1)
    
    args = parser.parse_args()

    main(args)