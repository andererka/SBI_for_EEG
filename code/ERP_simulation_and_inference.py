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



## defining neuronal network model

net = jones_2009_model()
#net.plot_cells()
#net.cell_types['L5_pyramidal'].plot_morphology()

## defining weights

from utils.simulation_wrapper import event_seed

weights_ampa_d1 = {'L2_basket': 0.006562, 'L2_pyramidal': .000007,
                   'L5_pyramidal': 0.142300}
weights_nmda_d1 = {'L2_basket': 0.019482, 'L2_pyramidal': 0.004317,
                   'L5_pyramidal': 0.080074}
synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}
net.add_evoked_drive(
    'evdist1', mu=63.53, sigma=3.85, numspikes=1, weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1, location='distal',
    synaptic_delays=synaptic_delays_d1, event_seed=event_seed())




weights_ampa_p1 = {'L2_basket': 0.08831, 'L2_pyramidal': 0.01525,
                   'L5_basket': 0.19934, 'L5_pyramidal': 0.00865}
synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                        'L5_basket': 1., 'L5_pyramidal': 1.}

# all NMDA weights are zero; pass None explicitly
net.add_evoked_drive(
    'evprox1', mu=26.61, sigma=2.47, numspikes=1, weights_ampa=weights_ampa_p1,
    weights_nmda=None, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=event_seed())

# Second proximal evoked drive. NB: only AMPA weights differ from first
weights_ampa_p2 = {'L2_basket': 0.000003, 'L2_pyramidal': 1.438840,
                   'L5_basket': 0.008958, 'L5_pyramidal': 0.684013}
# all NMDA weights are zero; omit weights_nmda (defaults to None)
net.add_evoked_drive(
    'evprox2', mu=137.12, sigma=8.33, numspikes=1,
    weights_ampa=weights_ampa_p2, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=event_seed())


from utils import simulation_wrapper


simulation_wrapper = simulation_wrapper.simulation_wrapper

window_len = 30

##defining the prior lower and upper bounds
prior_min = [43.8, 89.49]   # 't_evdist_1', 'sigma_t_evdist_1', 't_evprox_2', 'sigma_t_evprox_2'

prior_max = [79.9, 152.96]  

prior = utils.torchutils.BoxUniform(low=prior_min, 
                                    high=prior_max)

number_simulations = 500
density_estimator = 'nsf'

from utils import inference

posterior, theta, x = inference.run_sim_inference(prior, simulation_wrapper, number_simulations, density_estimator=density_estimator)

window_len, scaling_factor = 30, 3000


## set params as the 'true parameters'
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