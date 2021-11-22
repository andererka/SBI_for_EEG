from data_load_writer import load_from_file as lf
from data_load_writer import write_to_file


from summary_features.calculate_summary_features import calculate_summary_stats, calculate_summary_statistics_alternative

import numpy as np
import torch

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# sbi
from sbi import utils as utils
from sbi import analysis as analysis


import pickle
import sys

## defining neuronal network model

def main(argv):
    """

    description: assuming to already have a posterior from previous simulations, this function is 
    drawing samples with respect to an observation (synthetic or real) and is saving the pairplot figures
    in the result file of the previous gained posterior
    
    arg 1: file directory to the results file with the posterior pt file
    arg 2: number of samples that one wants to draw from the posterior
    arg 3: number of rounds; how many times the 'true' observation should be simulated 
            (only for the case where we do not have a real observation, but take a simulated one where we 
            set the parameter values to the same values all the times)
    """
    try:
        file = argv[0]
    except:
        file = 'results/ERP_nsf_num_params:3_11-20-2021_12:19:49/class'
    try:
        num_samples = int(argv[1])
    except:
        num_samples = 100

    try: 
        num_rounds = int(argv[2])
    except:
        num_rounds = 1



    ### loading the class:
    with open(file, 'rb') as pickle_file:
        file_writer = pickle.load(pickle_file)


    ##loading theta and x from results
    theta = lf.load_thetas(file_writer.folder)
    x_without = lf.load_obs_without(file_writer.folder)

    ###calculating summary statistics:
    x = calculate_summary_stats(torch.from_numpy(x_without))
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()
    

    posterior = inference.build_posterior(density_estimator)


    true_params = lf.load_true_params(file_writer.folder)

    from utils import inference

    for i in range(num_rounds):
        obs_real = inference.run_only_sim(true_params)

        samples = posterior.sample((num_samples,), 
                                x=obs_real)




        fig, axes = analysis.pairplot(samples,
                                #limits=[[.5,80], [1e-4,15.]],
                                #ticks=[[.5,80], [1e-4,15.]],
                                figsize=(5,5),
                                points=true_params,
                                points_offdiag={'markersize': 6},
                                points_colors='r');

        file_writer.save_fig(fig)

    


if __name__ == "__main__":
   main(sys.argv[1:])






