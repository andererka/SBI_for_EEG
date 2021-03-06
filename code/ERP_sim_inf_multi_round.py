#!/usr/bin/env python
# coding: utf-8


import os.path as op
import tempfile
import datetime

import shutil


import numpy as np
from summary_features.calculate_summary_features import (
    calculate_summary_stats_temporal,
    calculate_summary_statistics_alternative,
)
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
import pandas as pd

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
    arg 2: density estimator; default is maf
    arg 3: number of workers; should be set to the number of available cpus; default is 8
    arg 4: number of samples that should be drawn from posterior; default is 100

    arg 7: observation: can be 'fake' from chosen 'true parameters' or 'supra' for data from an experimental paradigm where 100% of stimuli were detected, or 'threshold' were only 50% were detected.
        See: https://jonescompneurolab.github.io/hnn-tutorials/optimization/optimization for further explanation of the data.
    
    """

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
        num_params = int(argv[3])
    except:
        num_params = 6
    try:
        slurm = bool(int(argv[4]))
    except:
        slurm = True
    try:
        experiment_name = argv[5]
    except:
        experiment_name = "multi_round"
    try:
        set_proposal = bool(int(argv[6]))
    except:
        set_proposal = True
    try:
        observation = argv[7]
    except:
        observation = "fake"

    set_std = False
    if num_params == 20:
        set_std = True

    sim_wrapper = SimulationWrapper(num_params=num_params, noise=True, set_std=set_std)

    ##defining the prior lower and upper bounds
    if num_params == 6:
        prior_min = [0.0, 11.3, 0.0, 43.8, 0.0, 89.491]
        prior_max = [0.160, 35.9, 0.821, 79.0, 8.104, 162.110]
        # true_params = torch.tensor([[26.61, 63.53,  137.12]])
        true_params = torch.tensor([[0.0274, 19.01, 0.1369, 61.89, 0.1435, 120.86]])

        parameter_names = [
            "prox_1_ampa_l2_pyr",
            "t_evprox_1",
            "dist_nmda_l2_pyr",
            "t_evdist_1",
            "prox_2_ampa_l5_pyr",
            "t_evprox_2",
        ]

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

        prior_min = [0, 0, 0, 0, 0, 13.3, 0, 0, 0, 0, 0, 51.980, 0, 0, 0, 0, 112.13]
        prior_max = [
            0.927,
            0.160,
            2.093,
            1.0,
            1.0,
            35.9,
            0.000042,
            0.039372,
            0.025902,
            0.480,
            0.117,
            75.08,
            8.633,
            4.104,
            1.0,
            1.0,
            162.110,
        ]

        ## larger priors:
        # prior_min = [0, 0, 0, 0, 0, 13.3,  0, 0, 0, 0, 0, 51.980, 0, 0, 0, 0, 112.13]
        # prior_max = [0.927, 0.160, 2.093, 1.0, 1.0, 44.9, 3.0, 7.0, 0.025902,  0.480, 0.117, 75.08, 8.633, 4.104, 1.0, 1.0, 162.110]

        # true_params = torch.tensor([[26.61, 63.53,  137.12]])
        true_params = torch.tensor(
            [
                [
                    0.277,
                    0.0399,
                    0.6244,
                    0.3739,
                    0.0,
                    18.977,
                    0.000012,
                    0.0115,
                    0.0134,
                    0.0767,
                    0.06337,
                    63.08,
                    4.6729,
                    2.33,
                    0.016733,
                    0.0679,
                    120.86,
                ]
            ]
        )

        # parameter_names = ["t_evprox_1", "t_evdist_1", "t_evprox_2"]

        parameter_names = [
            "prox1_ampa_l2_bas",
            "prox1_ampa_l2_pyr",
            "prox1_ampa_l5_bas",
            "prox1_nmda_l5_bas",
            "prox1_nmda_l5_pyr",
            "t_prox1",
            "dist_ampa_l2_pyr",
            "dist_ampa_l2_bas",
            "dist_nmda_l2_pyr",
            "dist_nmda_l5_pyr",
            "dist_nmda_l2_bas",
            "t_dist",
            "prox2_ampa_l2_pyr",
            "prox2_ampa_l5_pyr",
            "prox2_nmda_l2_pyr",
            "prox2_nmda_l5_pyr",
            "t_prox2",
        ]

    elif num_params == 20:

        prior_min = [
            0,
            0,
            0,
            0,
            0,
            13.3,
            1.276,
            0,
            0,
            0,
            0,
            0,
            51.980,
            3.011,
            0,
            0,
            0,
            0,
            112.13,
            5.29,
        ]
        prior_max = [
            0.927,
            0.160,
            2.093,
            1.0,
            1.0,
            35.9,
            3.828,
            0.000042,
            0.039372,
            0.025902,
            0.480,
            0.117,
            75.08,
            9.034,
            8.633,
            4.104,
            1.0,
            1.0,
            162.110,
            15.87,
        ]

        true_params = torch.tensor(
            [
                [
                    0.277,
                    0.0399,
                    0.6244,
                    0.3739,
                    0.0,
                    18.977,
                    2.55,
                    0.000012,
                    0.0115,
                    0.0134,
                    0.0767,
                    0.06337,
                    63.08,
                    6.02,
                    4.6729,
                    2.33,
                    0.016733,
                    0.0679,
                    120.86,
                    10.57,
                ]
            ]
        )

        parameter_names = [
            "prox1_ampa_l2_bas",
            "prox1_ampa_l2_pyr",
            "prox1_ampa_l5_bas",
            "prox1_nmda_l5_bas",
            "prox1_nmda_l5_pyr",
            "t_prox1",
            "std_prox1",
            "dist_ampa_l2_pyr",
            "dist_ampa_l2_bas",
            "dist_nmda_l2_pyr",
            "dist_nmda_l5_pyr",
            "dist_nmda_l2_bas",
            "t_dist",
            "std_dist",
            "prox2_ampa_l2_pyr",
            "prox2_ampa_l5_pyr",
            "prox2_nmda_l2_pyr",
            "prox2_nmda_l5_pyr",
            "t_prox2",
            "std_prox2",
        ]

    elif num_params == 25:
        prior_min = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            17.3,  # prox1 weights
            0,
            0,
            0,
            0,
            0,
            0,
            51.980,  # distal weights
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            112.13,
        ]  # prox2 weights

        prior_max = [
            0.927,
            1.0,
            0.160,
            1.0,
            2.093,
            1.0,
            0.0519,
            1.0,
            35.9,
            0.0394,
            0.117,
            0.000042,
            0.025902,
            0.854,
            0.480,
            75.08,
            0.000018,
            1.0,
            8.633,
            1.0,
            0.05375,
            1.0,
            4.104,
            1.0,
            162.110,
        ]

        true_params = torch.tensor(
            [
                [
                    0.277,
                    0.3739,
                    0.0399,
                    0.0,
                    0.6244,
                    0.3739,
                    0.034,
                    0.0,
                    18.977,
                    0.011467,
                    0.06337,
                    0.000012,
                    0.013407,
                    0.466095,
                    0.0767,
                    63.08,
                    0.000005,
                    0.116706,
                    4.6729,
                    0.016733,
                    0.011468,
                    0.061556,
                    2.33,
                    0.0679,
                    120.86,
                ]
            ]
        )

        parameter_names = [
            "prox1_ampa_l2_bas",
            "prox1_nmda_l2_bas",
            "prox1_ampa_l2_pyr",
            "prox1_nmda_l2_pyr",
            "prox1_ampa_l5_bas",
            "prox1_nmda_l5_bas",
            "prox1_ampa_l5_pyr",
            "prox1_nmda_l5_pyr",
            "t_prox1",
            "dist_ampa_l2_bas",
            "dist_nmda_l2_bas",
            "dist_ampa_l2_pyr",
            "dist_nmda_l2_pyr",
            "dist_ampa_l5_pyr",
            "dist_nmda_l5_pyr",
            "t_dist",
            "prox2_ampa_l2_bas",
            "prox2_nmda_l2_bas",
            "prox2_ampa_l2_pyr",
            "prox2_nmda_l2_pyr",
            "prox2_ampa_l5_bas",
            "prox2_nmda_l5_bas",
            "prox2_ampa_l5_pyr",
            "prox2_nmda_l5_pyr",
            "t_prox2",
        ]

    elif num_params == None:
        print("number of parameters must be defined in the arguments")
        sys.exit()

    prior = utils.torchutils.BoxUniform(low=prior_min, high=prior_max)

    print(torch.tensor([list(true_params[0])]))

    posteriors = []
    proposal = prior

    file_writer = write_to_file.WriteToFile(
        experiment=experiment_name,
        num_sim=number_simulations,
        true_params=true_params,
        density_estimator=density_estimator,
        num_params=num_params,
        slurm=slurm,
    )
    print(os.getcwd())
    print(file_writer.folder)

    try:
        os.mkdir(file_writer.folder)

        print(file_writer.folder)
        print("mkdir file")
    except:
        print("file exists")

    open("{}/ERP_sim_inf_multi_round.py".format(file_writer.folder), "a").close()

    shutil.copyfile(
        str(os.getcwd() + "/ERP_sim_inf_multi_round.py"),
        str(file_writer.folder + "/ERP_sim_inf_multi_round.py"),
    )

    os.chdir(file_writer.folder)

    if observation == "fake":

        obs_real_complete = inference.run_only_sim(
            torch.tensor([list(true_params[0][0:])]),
            simulation_wrapper=sim_wrapper,
            num_workers=1,
        )

        obs_real = obs_real_complete[0]

    if observation == "threshold":
        os.chdir("..")
        print(os.getcwd())
        trace = pd.read_csv(
            "data/ERPYes3Trials/dpl.txt", sep="\t", header=None, dtype=np.float32
        )
        obs_real = torch.tensor(trace.values, dtype=torch.float32)[:, 1]
        print(obs_real.shape[0])
        plt.plot(obs_real)
        noise = np.random.normal(0, 1, obs_real.shape[0])
        obs_real += noise
        plt.plot(obs_real)
        plt.savefig("obs_real_noise")

    if observation == "default":
        os.chdir("..")
        print(os.getcwd())
        trace = pd.read_csv(
            "data/default/dpl.txt", sep="\t", header=None, dtype=np.float32
        )
        obs_real = torch.tensor(trace.values, dtype=torch.float32)[:, 1]
        noise = np.random.normal(0, 1, obs_real.shape[0])
        obs_real += noise

    if observation == "No":
        os.chdir("..")
        print(os.getcwd())
        trace = pd.read_csv(
            "data/ERPNo100Trials/dpl_1.txt", sep="\t", header=None, dtype=np.float32
        )
        obs_real = torch.tensor(trace.values, dtype=torch.float32)[:, 1]
        noise = np.random.normal(0, 1, obs_real.shape[0])
        obs_real += noise

    os.chdir(file_writer.folder)

    obs_real_stat = calculate_summary_stats_temporal(obs_real)

    json_dict = {"arguments:": str(argv)}
    with open("argument_list.json", "a") as f:
        json.dump(json_dict, f)
        f.close()

    inf = SNPE_C(prior=prior, density_estimator=density_estimator)

    for i in range(3):

        start_time = datetime.datetime.now()
        start_time_str = get_time()

        theta, x_without = inference.run_sim_theta_x(
            prior=proposal,
            simulation_wrapper=sim_wrapper,
            num_simulations=number_simulations,
            num_workers=num_workers,
        )

        inf_start = datetime.datetime.now()

        x = calculate_summary_stats_temporal(x_without)

        if set_proposal:
            inf = inf.append_simulations(theta, x, proposal=proposal)
        else:
            inf = inf.append_simulations(theta, x)

        neural_dens = inf.train()

        posterior = inf.build_posterior(neural_dens)

        posteriors.append(posterior.set_default_x(obs_real_stat))
        proposal = posterior.set_default_x(obs_real_stat)

        inf_end = datetime.datetime.now()

        # time needed for the inference part of the pipeline:
        inf_diff = inf_end - inf_start

        finish_time_str = get_time()
        finish_time = datetime.datetime.now()

        diff_time = finish_time - start_time

        json_dict = {
            "start time:": start_time_str,
            "finish time": finish_time_str,
            "total CPU time:": str(diff_time),
            "parameter names": parameter_names,
            "inference time": str(inf_diff),
        }

        filename = "meta_round_" + str(i) + ".json"

        with open(filename, "a") as f:
            json.dump(json_dict, f)
            f.close()

    ## save posteriors for each round without conditioning on observation
    torch.save(posteriors, "posteriors_each_round.pt")

    file_writer.save_posterior(posterior)
    file_writer.save_obs_without(x_without)
    file_writer.save_prior(prior)
    file_writer.save_thetas(theta)

    torch.save(obs_real, "obs_real.pt")

    ## tries to store posterior without torch.save as there is a known bug that torch.save cannot save attributes of class
    with open("posterior2.pt", "wb") as f:
        pickle.dump(posterior, f)

    ##save class
    with open("class", "wb") as pickle_file:
        pickle.dump(file_writer, pickle_file)


if __name__ == "__main__":
    torch.manual_seed(5)
    np.random.seed(5)
    main(sys.argv[1:])
