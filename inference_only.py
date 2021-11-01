from dataloads_storage import load_from_file as lf
from dataloads_storage import write_to_file
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
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi


## defining neuronal network model

net = jones_2009_model()
import os

print(os.getcwd())
prior = lf.load_prior('results/ERP_results10-31-2021_11:27:06')

posterior = lf.load_posterior('results/ERP_results10-31-2021_11:27:06')

x = lf.load_obs('results/ERP_results10-31-2021_11:27:06')

theta = lf.load_thetas('results/ERP_results10-31-2021_11:27:06')



# set params as the 'true parameters'
net._params['t_evdist_1'] = 63.53
net._params['sigma_t_evdist_1'] = 3.85
net._params['t_evdist_2'] = 137.12
net._params['sigma_t_evprox_2'] = 8.33


window_len = 30
scaling_factor = 3000
dpls = simulate_dipole(net, tstop=170., n_trials=4)
for dpl in dpls:
    obs = dpl.smooth(window_len).scale(scaling_factor).data['agg']

obs_real = calculate_summary_stats(torch.from_numpy(obs))

samples = posterior.sample((1000,), 
                           x=obs_real)




true_params = torch.Tensor([63.53, 3.85, 137.12, 8.33])




fig, axes = analysis.pairplot(samples,
                           #limits=[[.5,80], [1e-4,15.]],
                           #ticks=[[.5,80], [1e-4,15.]],
                           figsize=(5,5),
                           points=true_params,
                           points_offdiag={'markersize': 6},
                           points_colors='r');


file_writer = write_to_file.WriteToFile(experiment='ERP_results')

file_writer.save_fig(fig)