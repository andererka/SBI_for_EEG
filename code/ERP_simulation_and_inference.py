#!/usr/bin/env python
# coding: utf-8


import os.path as op
import tempfile

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



window_len = 30

##defining the prior lower and upper bounds
prior_min = [43.8, 89.49]   # 't_evdist_1', 'sigma_t_evdist_1', 't_evprox_2', 'sigma_t_evprox_2'

prior_max = [79.9, 152.96]  

prior = utils.torchutils.BoxUniform(low=prior_min, 
                                    high=prior_max)

number_simulations = 3
density_estimator = 'nsf'

from utils import inference

posterior, theta, x = inference.run_sim_inference(prior, simulation_wrapper, number_simulations, density_estimator=density_estimator)

window_len, scaling_factor = 30, 3000



## defining neuronal network model


#net.plot_cells()
#net.cell_types['L5_pyramidal'].plot_morphology()

## defining weights
## set params as the 'true parameters'

net = jones_2009_model()
net._params['t_evdist_1'] = 63.53
net._params['sigma_t_evdist_1'] = 3.85
net._params['t_evdist_2'] = 137.12
net._params['sigma_t_evprox_2'] = 8.33

dpls = simulate_dipole(net, tstop=170., n_trials=1)
for dpl in dpls:
    obs = dpl.smooth(window_len).scale(scaling_factor).data['agg']

obs_real = calculate_summary_stats(torch.from_numpy(obs))

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


file_writer.save_all(posterior, prior, theta=theta, x =x, fig=fig)

##save class 


with open('{}/class'.format(file_writer.folder), 'wb') as pickle_file:
    pickle.dump(file_writer, pickle_file)
