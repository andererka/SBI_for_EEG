import datetime
from utils.simulation_wrapper import SimulationWrapper
from data_load_writer import load_from_file as lf
from data_load_writer import write_to_file


from summary_features.calculate_summary_features import (
    calculate_summary_stats_temporal

)

import numpy as np
import torch
import json

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
import shutil

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
        num_workers = int(argv[1])
    except:
        num_workers = 4
    try:
        experiment_name = argv[2]
    except:
        experiment_name = "ERP_sequential"
    try:
        slurm = bool(int(argv[3]))
    except:
        slurm = True
    try:
        density_estimator = argv[4]
    except:
        density_estimator = 'nsf'


    ## using a density estimator with only 1 transform (which should be enough for the 1D case)
    #dens_estimator = posterior_nn(model='nsf', hidden_features=60, num_transforms=1)


    #sim_wrapper = simulation_wrapper_obs
    sim_wrapper = SimulationWrapper(6)


    prior_min = [0.0, 11.3, 0.0, 43.8, 0.0, 89.491]
    prior_max = [0.160, 35.9, 0.821, 79.0, 8.104, 162.110]

    true_params = torch.tensor([[0.0274, 19.01, 0.1369, 61.89, 0.1435, 120.86]])

    
    #parameter_names = ["t_evprox_1", "t_evdist_1", "t_evprox_2"]

    parameter_names = ["prox_1_ampa_l2_pyr",
     "t_evprox_1",
     "dist_nmda_l2_pyr",
     "t_evdist_1", 
     "prox_2_ampa_l5_pyr",
     "t_evprox_2"]

    ###### starting with P50 parameters/summary stats:
    #prior1 = utils.torchutils.BoxUniform(low=[prior_min[0]], high=[prior_max[0]])

    prior1 = utils.torchutils.BoxUniform(low=prior_min[0:2], high=prior_max[0:2])

    inf = SNPE_C(prior1, density_estimator=density_estimator)

    obs_real = inference.run_only_sim(
    true_params, sim_wrapper, num_workers=1)  # first output gives summary statistics, second without

    obs_real_stat = calculate_summary_stats_temporal(obs_real[0], complete=True)



    file_writer = write_to_file.WriteToFile(
    experiment=experiment_name,
    num_sim=num_sim,
    density_estimator=density_estimator,
    num_params=len(prior_max),
    slurm=slurm,
    )

    print(file_writer.folder)



    try:
        os.mkdir(file_writer.folder)
    except:
        print('file exists')
    
    try:
        os.mkdir('{}/step1'.format(file_writer.folder))
        os.mkdir('{}/step2'.format(file_writer.folder))
        os.mkdir('{}/step3'.format(file_writer.folder))
    except:
        print('step files exist')


    # stores the running file into the result folder for later reference:
    open('{}/sequential_inference_6params.py'.format(file_writer.folder), 'a').close()
    shutil.copyfile(str(os.getcwd() + '/sequential_inference_6params.py'), str(file_writer.folder+ '/sequential_inference_6params.py'))



    os.chdir(file_writer.folder)

        

    start_time0 = datetime.datetime.now()
    start_time = get_time()

    
    try:
        theta = torch.load('step1/thetas.pt')
        x_without = torch.load('step1/obs_without.pt')

    except:
        theta, x_without = inference.run_sim_theta_x(
            prior1, 
            sim_wrapper,
            num_simulations=num_sim,
            num_workers=num_workers
        )


        file_writer.save_obs_without(x_without, name='step1')
        file_writer.save_thetas(theta, name='step1')

    x_P50 = calculate_summary_stats_temporal(x_without)

    print('x50 shape 0', x_P50.shape[0], x_P50.shape)

    print('theta shape 0', theta.shape[0], theta.shape)

    inf = inf.append_simulations(theta, x_P50)
    neural_dens = inf.train()

    posterior = inf.build_posterior(neural_dens)


    proposal1 = posterior.set_default_x(obs_real_stat[:,:x_P50.shape[1]])

    start_time1 = datetime.datetime.now()
    diff = start_time1 - start_time0

    step_time = get_time()
    json_dict = {
    "start time:": start_time,
    "round 1 time": step_time,
    "diff": str(diff), }
    with open( "step1/meta.json", "a") as f:
        json.dump(json_dict, f)
        f.close()

    ###### continuing with N100 parameters/summary stats:

    prior2 = utils.torchutils.BoxUniform(low=prior_min[2:4], high=prior_max[2:4])


    combined_prior = Combined(proposal1, prior2, number_params_1=2)

    inf = SNPE_C(combined_prior, density_estimator=density_estimator)


    try:
        theta = torch.load('step2/thetas.pt')
        x_without = torch.load('step2/obs_without.pt')

    except:
        theta, x_without = inference.run_sim_theta_x(
            combined_prior,
            sim_wrapper,
            num_simulations=num_sim,
            num_workers=num_workers
        )
        file_writer.save_obs_without(x_without, name='step2')
        file_writer.save_thetas(theta, name='step2')



    print("second round completed")

    x_N100 = calculate_summary_stats_temporal(x_without)

    inf = inf.append_simulations(theta, x_N100)
    neural_dens = inf.train()

    posterior = inf.build_posterior(neural_dens)


    proposal2 = posterior.set_default_x(obs_real_stat[:,:x_N100.shape[1]])

    start_time2 = datetime.datetime.now()
    step_time2 = get_time()

    diff = start_time2 -start_time1

    json_dict = {
    "start time:": start_time,
    "round 3 time": step_time2}
    with open( "step2/meta.json", "a") as f:
        json.dump(json_dict, f)
        f.close()

    ###### continuing with P200 parameters/summary stats:
    #prior3 = utils.torchutils.BoxUniform(low=[prior_min[2]], high=[prior_max[2]])

    prior3 = utils.torchutils.BoxUniform(low=prior_min[4:], high=prior_max[4:])

    #combined_prior = Combined(proposal2, prior3, number_params_1=2)

    combined_prior = Combined(proposal2, prior3, number_params_1=4)

    inf = SNPE_C(combined_prior, density_estimator=density_estimator)

    try:
        theta = torch.load('step3/thetas.pt')
        x_without = torch.load('step3/obs_without.pt')

    except:
        theta, x_without = inference.run_sim_theta_x(
            combined_prior,
            sim_wrapper,
            num_simulations=num_sim,
            num_workers=num_workers
        )

        file_writer.save_obs_without(x_without, name='step3')
        file_writer.save_thetas(theta, name='step3')


    x_P200 = calculate_summary_stats_temporal(x_without)

    inf = inf.append_simulations(theta, x_P200)
    neural_dens = inf.train()

    posterior = inf.build_posterior(neural_dens)


    posterior.set_default_x(obs_real_stat)

    step_time = get_time()

    diff = datetime.datetime.now() - start_time2

    json_dict = {

    "start time:": start_time,
    "round 3 time": step_time,
    "diff": str(diff),
}
    with open( "step3/meta.json", "a") as f:
        json.dump(json_dict, f)
        f.close()

    torch.save(posterior, 'posterior.pt')

    torch.save(obs_real, 'obs_real.pt')
    torch.save(obs_real_stat, 'obs_real_stat.pt')

if __name__ == "__main__":
    main(sys.argv[1:])
