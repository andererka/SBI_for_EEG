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
    start_time = get_time()

    window_len = 30

    ##defining the prior lower and upper bounds
    prior_min = [43.8, 89.49]   # 't_evdist_1', 'sigma_t_evdist_1', 't_evprox_2', 'sigma_t_evprox_2'

    prior_max = [79.9, 152.96]  

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
    posterior, theta, x = inference.run_sim_inference(prior, simulation_wrapper, number_simulations, num_workers =num_workers, density_estimator=density_estimator)

    window_len, scaling_factor = 30, 3000



    ## defining neuronal network model

    ## defining weights
    ## set params as the 'true parameters'

    #net = jones_2009_model()
    #net._params['t_evdist_1'] = 63.53
    #net._params['sigma_t_evdist_1'] = 3.85
    #net._params['t_evdist_2'] = 137.12
    #net._params['sigma_t_evprox_2'] = 8.33

    #dpls = simulate_dipole(net, tstop=170., n_trials=1)
    #for dpl in dpls:
    #    obs = dpl.smooth(window_len).scale(scaling_factor).data['agg']

    obs_real = inference.run_only_sim([63.53, 137.12])

    samples = posterior.sample((100,), 
                            x=obs_real)




    true_params = torch.Tensor([63.53, 137.12])




    fig, axes = analysis.pairplot(samples,
                            #limits=[[.5,80], [1e-4,15.]],
                            #ticks=[[.5,80], [1e-4,15.]],
                            figsize=(5,5),
                            points=true_params,
                            points_offdiag={'markersize': 6},
                            points_colors='r');


    from data_load_writer import write_to_file
    import pickle

    file_writer = write_to_file.WriteToFile(experiment='ERP_{}'.format(density_estimator), num_sim=number_simulations,
                    true_params=true_params, density_estimator=density_estimator)



    finish_time = get_time()
    file_writer.save_all(posterior, prior, theta=theta, x =x, fig=fig, start_time=start_time, finish_time=finish_time)

    ##save class 


    with open('{}/class'.format(file_writer.folder), 'wb') as pickle_file:
        pickle.dump(file_writer, pickle_file)
    


if __name__ == "__main__":
   main(sys.argv[1:])


