from data_load_writer import load_from_file as lf
from data_load_writer import write_to_file
import matplotlib.pyplot as plt


import hnn_core
from hnn_core import simulate_dipole, jones_2009_model
from hnn_core.viz import plot_dipole

# import the summary statistics that you want to investigate
from summary_features.calculate_summary_features import calculate_summary_statistics_alternative as extract_sumstats
#from summary_features.calculate_summary_features import calculate_summary_stats_temporal as extract_sumstats
from summary_features.calculate_summary_features import calculate_summary_stats_number as number_sumstats


import numpy as np
import torch

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.plot import cov, compare_vars, plot_varchanges

from utils.plot import plot_KLs

# sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE_C, prepare_for_sbi, simulate_for_sbi

from utils import inference

from utils.helpers import SummaryNet
import sys

from data_load_writer import write_to_file

from utils.simulation_wrapper import event_seed, set_network_default, simulation_wrapper,simulation_wrapper_obs


import os
import pickle

import sys

#sys.path.append('/mnt/qb/work/macke/kanderer29/')

from utils.helpers import get_time

## defining neuronal network model


def main(argv):
    """

    description: assuming to already have a posterior from previous simulations, this function is 
    drawing samples with respect to an observation (synthetic or real) and is saving the pairplot figures
    in the result file of the previous gained posterior
    
    arg 1: file directory to the results file with the posterior pt file
    arg 2: number of samples that one wants to draw from the posterior
    arg 3: number of rounds; how many times to 'true' observation should be simulated 
            (only for the case where we do not have a real observation, but take a simulated one where we 
            set the parameter values to the same values all the times)
    """

    try:
        num_sim = int(argv[0])
    except:
        num_sim = 1000
    try:
        experiment_name = argv[1]
    except:
        experiment_name = 'sim_1000'
    try:
        num_workers = int(argv[2])
    except:
        num_workers = 8

    start_time = get_time()

    true_params = torch.tensor([[26.61, 63.53,  137.12]])
    sim_wrapper = simulation_wrapper_obs


    file_writer = write_to_file.WriteToFile(
    experiment=experiment_name,
    num_sim=num_sim,
    true_params=true_params,
    density_estimator='nsf',
    num_params=3,
    num_samples=None,
    slurm=True
    )

    try:
        os.mkdir(file_writer.folder)
    except:
        print('file exists')
        exit

    os.chdir(file_writer.folder)

    prior_min = [7.9, 43.8,  89.49] 

    prior_max = [30, 79.9, 152.96]

    prior = utils.torchutils.BoxUniform(low=prior_min, high=prior_max)

    inf = SNPE_C(prior, density_estimator="nsf")


    theta, x_without = inference.run_sim_theta_x(
        prior,
        simulation_wrapper=sim_wrapper,
        num_simulations=num_sim,
        num_workers=num_workers)
    ## save thetas and x_without to file_writer:
    file_writer.save_obs_without(x_without)
    file_writer.save_thetas(theta)
    


    torch.save(file_writer, "class.pt")
    torch.save(true_params, "true_params.pt")

    finish_time = get_time()

    file_writer.save_all(
    posterior=None,
    prior=prior,
    start_time=start_time,
    finish_time=finish_time,
)



if __name__ == "__main__":
    main(sys.argv[1:])
