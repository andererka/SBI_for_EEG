#!/usr/bin/env python
# coding: utf-8


import os.path as op
import tempfile
import datetime

import matplotlib.pyplot as plt


import hnn_core
from hnn_core import simulate_dipole, jones_2009_model
from hnn_core.viz import plot_dipole
from summary_features.calculate_summary_features import calculate_summary_stats, calculate_summary_statistics_alternative

import numpy as np
import torch

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer



from utils.simulation_wrapper import event_seed, simulation_wrapper
from utils.helpers import get_time



from utils import inference
import sys


def main(argv):
    """
    arg 1: number of simulations; default is 50
    arg 2: density estimator; default is nsf
    arg 3: number of workers; should be set to the number of available cpus; default is 8
    arg 4: number of samples that should be drawn from posterior; default is 100
    
    """
    start_time = get_time()


    ##defining the prior lower and upper bounds
    prior_min = [43.8, 3.01, 11.364, 1.276, 89.49, 5.29]   # 't_evdist_1', 'sigma_t_evdist_1', 't_evprox_1', 'sigma_t_evprox_1', 't_evprox_2', 'sigma_t_evprox_2'

    prior_max = [79.9, 9.03, 26.67, 3.828, 152.96, 15.87]  

    prior = utils.torchutils.BoxUniform(low=prior_min, 
                                        high=prior_max)
    try:
        number_simulations = int(argv[0])
    except:
        number_simulations = 50
    try:
        density_estimator = argv[1]
    except:
        density_estimator ='nsf'
    try:
        num_workers = int(argv[2])
    except:
        num_workers = 8
    try: 
        num_samples = int(argv[3])
    except:
        num_samples = 100
    posterior, theta, x = inference.run_sim_inference(prior, simulation_wrapper, number_simulations, num_workers =num_workers, density_estimator=density_estimator)

    # next two lines are not necessary if we have a real observation from experiment
    # here we simulate this 'real observation' by simulation

    true_params = torch.tensor([61.89, 6.022, 19.01, 2.55, 121.23, 10.58])    
    # 't_evdist_1', 'sigma_t_evdist_1', 't_evprox_1', 'sigma_t_evprox_1', 't_evprox_2', 'sigma_t_evprox_2'


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


    from data_load_writer import write_to_file
    import pickle

    file_writer = write_to_file.WriteToFile(experiment='ERP_{}num_params:{}'.format(density_estimator, torch.Size(true_params, dim=0)), num_sim=number_simulations,
                    true_params=true_params, density_estimator=density_estimator)



    finish_time = get_time()
    file_writer.save_all(posterior, prior, theta=theta, x =x, fig=fig, start_time=start_time, finish_time=finish_time)

    ##save class 
    with open('{}/class'.format(file_writer.folder), 'wb') as pickle_file:
        pickle.dump(file_writer, pickle_file)
    


if __name__ == "__main__":
   main(sys.argv[1:])


