from utils.simulation_wrapper import SimulationWrapper, simulation_wrapper_all, simulation_wrapper_all_small_steps, simulation_wrapper_obs
from data_load_writer import load_from_file as lf
from data_load_writer import write_to_file


from summary_features.calculate_summary_features import (
    calculate_summary_stats_temporal

)

import numpy as np
import torch
import json
import pandas as pd
import seaborn as sns

import datetime

from utils.helpers import get_time

from utils.sbi_modulated_functions import Combined

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE_C, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn

from utils import inference


import pickle
import sys

import os

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
        num_sim = int(argv[0])
    except:
        num_sim = 100
    try:
        num_samples = int(argv[1])
    except:
        num_samples = 20

    try:
        num_workers = int(argv[2])
    except:
        num_workers = 4
    try:
        experiment_name = argv[3]
    except:
        experiment_name = "All_params_ERP"
    try:
        slurm = int(argv[4])
    except:
        slurm = 1

    if (slurm==0):
        slurm = True
    else:
        slurm = False

    ## using a density estimator with only 1 transform (which should be enough for the 1D case)
    #dens_estimator = posterior_nn(model='nsf', hidden_features=60, num_transforms=1)


    sim_wrapper = simulation_wrapper_all_small_steps


    prior_min = [0, 0, 0, 0, 0, 0, 0, 0, 17.3,    # prox1 weights
                0, 0, 0, 0, 0, 0, 51.980,            # distal weights
                0, 0, 0, 0, 0, 0, 0, 0, 112.13]       # prox2 weights

    prior_max = [0.927, 0.160, 2.093, 0.0519, 1.0, 1.0, 1.0, 1.0, 35.9, 
                0.0394, 0.000042, 0.039372, 0.854, 0.117,  0.480, 75.08, 
                0.000018, 8.633, 0.05375, 4.104, 1.0, 1.0, 1.0, 1.0, 162.110]

    #true_params = torch.tensor([[26.61, 63.53,  137.12]])
    true_params = torch.tensor([[0.277, 0.0399, 0.6244, 0.034, 0.3739, 0.0, 0.3739, 0.0, 18.977, 
                    0.011467, 0.000012, 0.466095, 0.06337, 0.013407, 0.0767, 63.08, 
                    0.000005, 4.6729, 0.011468, 2.33, 0.116706,  0.016733, 0.061556, 0.0679, 120.86]])

    
    #parameter_names = ["t_evprox_1", "t_evdist_1", "t_evprox_2"]

    parameter_names = ["prox1_ampa_l2_bas","prox1_ampa_l2_pyr","prox1_ampa_l5_bas","prox1_nmda_l5_bas", "prox1_nmda_l5_pyr",
     "t_prox1",
     "dist_ampa_l2_pyr","dist_ampa_l2_bas","dist_nmda_l2_pyr",
     "dist_nmda_l5_pyr","dist_nmda_l2_bas",
     "t_dist", 
     "prox2_ampa_l2_pyr","prox2_ampa_l5_pyr","prox2_nmda_l2_pyr","prox2_nmda_l5_pyr",
     "t_prox2"]

    ###### starting with P50 parameters/summary stats:
    #prior1 = utils.torchutils.BoxUniform(low=[prior_min[0]], high=[prior_max[0]])


    file_writer = write_to_file.WriteToFile(
    experiment=experiment_name,
    num_sim=num_sim,
    density_estimator='nsf',
    num_params=len(prior_max),
    num_samples=num_samples,
    slurm=slurm,
    )

    print(file_writer.folder)


    try:
        os.mkdir(file_writer.folder)
    except:
        print('file exists')
    

    os.chdir(file_writer.folder)

    print(file_writer.folder)

    prior_i = utils.torchutils.BoxUniform(low=prior_min[0:2], high=prior_max[0:2])

    inf = SNPE_C(prior_i, density_estimator='nsf')

    ##define list of number of parameters inferred in each incremental round:
    range_list = [2,4,6,8,9,10,12,14,15,16,18,20,22,24,25]

    for index in range(len(range_list)):

        ## i defines number of parameters to be inferred, j indicates how many parameters 
        #to come in the next round
        i = range_list[index]
        j = range_list[index+1]


        start_time = datetime.datetime.now()

        theta, x_without = inference.run_sim_theta_x(
            prior_i, 
            sim_wrapper,
            num_simulations=num_sim,
            num_workers=num_workers
        )

        obs_real = inference.run_only_sim(
            torch.tensor([list(true_params[0][0:i])]), 
            simulation_wrapper = sim_wrapper, 
            num_workers=1
        )  

        sim_len = obs_real[0].shape[0]

        x_without = x_without[:,:sim_len]

        x_P50 = calculate_summary_stats_temporal(x_without)
        inf = inf.append_simulations(theta, x_P50)
        neural_dens = inf.train()

        posterior = inf.build_posterior(neural_dens)

        obs_real_stat = calculate_summary_stats_temporal(obs_real)

        proposal1 = posterior.set_default_x(obs_real_stat)

        next_prior = utils.torchutils.BoxUniform(low=prior_min[i:j], high=prior_max[i:j])

        combined_prior = Combined(proposal1, next_prior, number_params_1=i)

        ## set inf for next round:
        inf = SNPE_C(combined_prior, density_estimator="nsf")

    
        ## set combined prior to be the new prior_i:
        prior_i = combined_prior

        finish_time = datetime.datetime.now()

        diff_time = finish_time - start_time

        json_dict = {
        "CPU time for step:": str(diff_time)}
        with open( "meta_{}.json".format(i), "a") as f:
            json.dump(json_dict, f)
            f.close()

        torch.save(x_without, 'x_without{}.pt'.format(i))
        torch.save(theta, 'thetas{}.pt'.format(i))



    file_writer.save_posterior(posterior)
    file_writer.save_obs_without(x_without)
    file_writer.save_prior(prior_i)
    file_writer.save_thetas(theta)

    os.chdir(file_writer.folder)

    ##save class
    with open("class", "wb") as pickle_file:
        pickle.dump(file_writer, pickle_file)


if __name__ == "__main__":
    main(sys.argv[1:])
