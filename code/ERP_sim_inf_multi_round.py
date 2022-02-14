#!/usr/bin/env python
# coding: utf-8


import os.path as op
import tempfile
import datetime


import numpy as np
from summary_features.calculate_summary_features import calculate_summary_stats_temporal
import torch
import os
import json
from data_load_writer import write_to_file
import pickle

# visualization
import matplotlib.pyplot as plt

# sbi
from sbi import utils as utils
from sbi import analysis as analysis


from utils.simulation_wrapper import SimulationWrapper
from utils.helpers import get_time


from utils import inference
import sys

##sbi
from sbi.inference import SNPE_C


def main(argv):
    """

    description: is simulating an event related potential with the hnn-core package and then uses sbi to
    infer parameters and draw samples from parameter posteriors. Special here is that we use a multi-round approach 
    that can make training more efficient, but not amortized anymore because it is optimized regarding one single observation
    every round.
    
    One can choose the following
    argument settings:

    arg 1: number of simulations; default is 50
    arg 2: density estimator; default is nsf
    arg 3: number of workers; should be set to the number of available cpus; default is 8
    arg 4: number of samples that should be drawn from posterior; default is 100
    
    """
    start_time = get_time()

    try:
        number_simulations = int(argv[0])
    except:
        number_simulations = 50
    try:
        density_estimator = argv[1]
    except:
        density_estimator = "nsf"
    try:
        num_workers = int(argv[2])
    except:
        num_workers = 8
    try:
        num_samples = int(argv[3])
    except:
        num_samples = 100

    try:
        num_params = int(argv[4])
    except:
        num_params = 6
    try:
        sample_method = argv[5]
    except:
        sample_method = "rejection"

    sim_wrapper = SimulationWrapper(num_params=num_params, small_steps=False)

   
    ##defining the prior lower and upper bounds
    if num_params == 6:
        prior_min = [0.0, 11.3, 0.0, 43.8, 0.0, 89.491]
        prior_max = [0.160, 35.9, 0.821, 79.0, 8.104, 162.110]
        #true_params = torch.tensor([[26.61, 63.53,  137.12]])
        true_params = torch.tensor([[0.0274, 19.01, 0.1369, 61.89, 0.1435, 120.86]])


        parameter_names = ["prox_1_ampa_l2_pyr",
        "t_evprox_1",
        "dist_nmda_l2_pyr",
        "t_evdist_1", 
        "prox_2_ampa_l5_pyr",
        "t_evprox_2"]

    elif num_params == 3:
        prior_min = [43.8, 7.9, 89.49]
        prior_max = [79.9, 30, 152.96]

        true_params = torch.tensor([[63.53, 18.97, 137.12]])

        parameter_names = ["t_evdist_1", "t_evprox_1", "t_evprox_2"]

    elif num_params == 2:
        prior_min = [43.8, 89.49]
        prior_max = [79.9, 152.96]

        true_params = torch.tensor([[63.53, 137.12]])

        parameter_names = ["t_evdist_1", "t_evprox_1"]

    elif num_params == 17:


        prior_min = [0, 0, 0, 0, 0, 17.3,  0, 0, 0, 0, 0, 51.980, 0, 0, 0, 0, 112.13]
        prior_max = [0.927, 0.160, 2.093, 1.0, 1.0, 35.9, 0.000042, 0.039372, 0.025902,  0.480, 0.117, 75.08, 8.633, 4.104, 1.0, 1.0, 162.110]

        #true_params = torch.tensor([[26.61, 63.53,  137.12]])
        true_params = torch.tensor([[0.277, 0.0399, 0.6244, 0.3739, 0.0, 18.977, 0.000012, 0.0115, 0.0134,  0.0767, 0.06337, 63.08, 4.6729, 2.33, 0.016733, 0.0679, 120.86]])

        
        #parameter_names = ["t_evprox_1", "t_evdist_1", "t_evprox_2"]

        parameter_names = ["prox1_ampa_l2_bas","prox1_ampa_l2_pyr","prox1_ampa_l5_bas","prox1_nmda_l5_bas", "prox1_nmda_l5_pyr",
        "t_prox1",
        "dist_ampa_l2_pyr","dist_ampa_l2_bas","dist_nmda_l2_pyr",
        "dist_nmda_l5_pyr","dist_nmda_l2_bas",
        "t_dist", 
        "prox2_ampa_l2_pyr","prox2_ampa_l5_pyr","prox2_nmda_l2_pyr","prox2_nmda_l5_pyr",
        "t_prox2"]

    elif num_params == 25:
        prior_min = [0, 0, 0, 0, 0, 0, 0, 0, 17.3,    # prox1 weights
                    0, 0, 0, 0, 0, 0, 51.980,            # distal weights
                    0, 0, 0, 0, 0, 0, 0, 0, 112.13]       # prox2 weights
    

        prior_max = [0.927, 1.0, 0.160, 1.0,  2.093, 1.0, 0.0519, 1.0, 35.9,
                    0.0394, 0.854, 0.000042, 0.117, 0.039372, 0.480, 75.08, 
                    0.000018, 1.0, 8.633, 1.0, 0.05375, 1.0, 4.104,  1.0, 162.110]

        true_params = torch.tensor([[0.277, 0.3739, 0.0399, 0.0, 0.6244, 0.3739, 0.034, 0.0, 18.977, 
                        0.011467, 0.06337, 0.000012, 0.013407, 0.466095, 0.0767, 63.08, 
                        0.000005, 0.116706, 4.6729, 0.016733, 0.011468, 0.061556, 2.33, 0.0679, 120.86]])

        parameter_names = ["prox1_ampa_l2_bas","prox1_nmda_l2_bas","prox1_ampa_l2_pyr", "prox1_nmda_l2_pyr", "prox1_ampa_l5_bas", "prox1_nmda_l5_bas", "prox1_ampa_l5_pyr", "prox1_nmda_l5_pyr",
        "t_prox1",
        "dist_ampa_l2_bas", "dist_nmda_l2_bas", "dist_ampa_l2_pyr", "dist_nmda_l2_pyr", "dist_ampa_l5_pyr","dist_nmda_l5_pyr",
        "t_dist", 
        "prox2_ampa_l2_bas","prox2_nmda_l2_bas","prox2_ampa_l2_pyr", "prox2_nmda_l2_pyr", "prox2_ampa_l5_bas", "prox2_nmda_l5_bas", "prox2_ampa_l5_pyr", "prox2_nmda_l5_pyr",
        "t_prox2"]


    elif num_params == None:
        print("number of parameters must be defined in the arguments")
        sys.exit()

    prior = utils.torchutils.BoxUniform(low=prior_min, high=prior_max)

    print(torch.tensor([list(true_params[0])]))

    obs_real = inference.run_only_sim(
        torch.tensor([list(true_params[0])]), simulation_wrapper = sim_wrapper, num_workers=1) 

    obs_real_stat = calculate_summary_stats_temporal(obs_real)

    posteriors = []
    proposal = prior


    file_writer = write_to_file.WriteToFile(
    experiment="{}_multi_round_num_params_{}newparams".format(
        number_simulations, num_params
    ),
    num_sim=number_simulations,
    true_params=true_params,
    density_estimator=density_estimator,
    num_params=num_params,
    num_samples=num_samples,
    slurm=True
)

    try:
        os.mkdir(file_writer.folder)

        print(file_writer.folder)
        print('mkdir file')
    except:
        print('file exists')

    os.chdir(file_writer.folder)



    for i in range(3):

        start_time = datetime.datetime.now()
        start_time_str = get_time()

        theta, x_without = inference.run_sim_theta_x(
        prior = proposal, 
        simulation_wrapper = sim_wrapper,
        num_simulations=number_simulations,
        num_workers=num_workers
        )

        x = calculate_summary_stats_temporal(x_without)
        
        density_estimator = 'nsf'


        inf = SNPE_C(prior=prior, density_estimator = density_estimator)


        inf = inf.append_simulations(theta, x)

        neural_dens= inf.train()

        posterior = inf.build_posterior(neural_dens)

        posteriors.append(posterior)
        proposal = posterior.set_default_x(obs_real_stat)

        finish_time_str = get_time()
        finish_time = datetime.datetime.now()

        diff_time = finish_time - start_time


        json_dict = {
        "start time:": start_time_str,
        "finish time": finish_time_str,
        'total CPU time:': str(diff_time),
        'parameter names': parameter_names}

        filename = 'meta_round_' + str(i) + '.json'

        with open( filename, "a") as f:
            json.dump(json_dict, f)
            f.close()



    samples = posterior.sample((num_samples,), x=obs_real_stat, sample_with = sample_method)

    s_x = inference.run_only_sim(samples, simulation_wrapper = sim_wrapper, num_workers=num_workers)


    file_writer.save_posterior(posterior)
    file_writer.save_obs_without(x_without)
    file_writer.save_prior(prior)
    file_writer.save_thetas(theta)

    torch.save(obs_real, 'obs_real.pt')





    ##save class
    with open("class", "wb") as pickle_file:
        pickle.dump(file_writer, pickle_file)

    ##save simulations from samples
    with open("sim_from_samples", "wb") as pickle_file:
        pickle.dump(s_x, pickle_file)


if __name__ == "__main__":
    main(sys.argv[1:])
