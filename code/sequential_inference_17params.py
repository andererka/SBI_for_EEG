from utils.simulation_wrapper import SimulationWrapper
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

import shutil

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
        experiment_name = "ERP_sequential"
    try:
        slurm = int(argv[4])
    except:
        slurm = 1

    try:
        slurm = bool(int(argv[4]))
    except:
        slurm = True
    try:
        density_estimator = argv[5]
    except:
        density_estimator = 'nsf'

    ## using a density estimator with only 1 transform (which should be enough for the 1D case)
    #dens_estimator = posterior_nn(model='nsf', hidden_features=60, num_transforms=1)


    start_time_str = get_time()
    start_time = datetime.datetime.now()

    sim_wrapper = SimulationWrapper(num_params=17)



    prior_min = [0, 0, 0, 0, 0, 17.3,  0, 0, 0, 0, 0, 51.980, 0, 0, 0, 0, 112.13]
    prior_max = [0.927, 0.160, 2.093, 1.0, 1.0, 35.9, 0.000042, 0.039372, 0.025902,  0.480, 0.117, 75.08, 8.633, 4.104, 1.0, 1.0, 162.110]

    true_params = torch.tensor([[0.277, 0.0399, 0.6244, 0.3739, 0.0, 18.977, 0.000012, 0.0115, 0.0134,  0.0767, 0.06337, 63.08, 4.6729, 2.33, 0.016733, 0.0679, 120.86]])


    parameter_names = ["prox1_ampa_l2_bas","prox1_ampa_l2_pyr","prox1_ampa_l5_bas","prox1_nmda_l5_bas", "prox1_nmda_l5_pyr",
     "t_prox1",
     "dist_ampa_l2_pyr","dist_ampa_l2_bas","dist_nmda_l2_pyr",
     "dist_nmda_l5_pyr","dist_nmda_l2_bas",
     "t_dist", 
     "prox2_ampa_l2_pyr","prox2_ampa_l5_pyr","prox2_nmda_l2_pyr","prox2_nmda_l5_pyr",
     "t_prox2"]

    ###### starting with P50 parameters/summary stats:
    #prior1 = utils.torchutils.BoxUniform(low=[prior_min[0]], high=[prior_max[0]])

    prior = utils.torchutils.BoxUniform(low=prior_min, high=prior_max)

    prior1 = utils.torchutils.BoxUniform(low=prior_min[0:6], high=prior_max[0:6])

    inf = SNPE_C(prior1, density_estimator=density_estimator)



    file_writer = write_to_file.WriteToFile(
    experiment=experiment_name,
    num_sim=num_sim,
    density_estimator=density_estimator,
    num_params=len(prior_max),
    num_samples=num_samples,
    slurm=slurm,
    )

    print(file_writer.folder)

    print(os.getcwd())


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

    

    open('{}/sequential_inference_17params.py'.format(file_writer.folder), 'a').close()

    shutil.copyfile(str(os.getcwd() + '/sequential_inference_17params.py'), str(file_writer.folder+ '/sequential_inference_17params.py'))


    os.chdir(file_writer.folder)

    obs_real_complete = inference.run_only_sim(
        torch.tensor([list(true_params[0][0:])]), 
        simulation_wrapper = sim_wrapper, 
        num_workers=1
    )

    
    try:
        theta = torch.load('step1/thetas.pt')
        x_without = torch.load('step1/obs_without.pt')

    except:
        theta, x_without = inference.run_sim_theta_x(
            prior1, 
            sim_wrapper,
            num_simulations=int(num_sim*(1/10)),
            num_workers=num_workers
        )

    
        finish_time = datetime.datetime.now()

        diff_time = finish_time - start_time



        step_time_str = get_time()
        json_dict = {
        "start time:": start_time_str,
        "round 1 time": step_time_str,
        "CPU time for step:": str(diff_time)}
        with open( "step1/meta.json", "a") as f:
            json.dump(json_dict, f)
            f.close()


        file_writer.save_obs_without(x_without, name='step1')
        file_writer.save_thetas(theta, name='step1')

    start_time = datetime.datetime.now()

    os.chdir('..')
    os.chdir('..')
    print(os.getcwd())

    #os.chdir('data')

    #trace = pd.read_csv('ERPYes3Trials/dpl.txt', sep='\t', header=None, dtype= np.float32)
    #trace_torch = torch.tensor(trace.values, dtype = torch.float32)

    os.chdir(file_writer.folder)


    x_without = x_without[:,:2700]

    x_P50 = calculate_summary_stats_temporal(x_without)

    print('x_P50',x_P50)

    print('x without', x_without.shape)



    inf = inf.append_simulations(theta, x_P50)
    neural_dens = inf.train()

    posterior = inf.build_posterior(neural_dens)


    #### either simulate 'fake observation' or load data from hnn 

    obs_real = [obs_real_complete[0][:x_without.shape[1]]]

    print('obs real', obs_real)
    obs_real_stat = calculate_summary_stats_temporal(obs_real)

    #samples = posterior.sample((num_samples,), x=obs_real_stat)

    proposal1 = posterior.set_default_x(obs_real_stat)

    ###### continuing with N100 parameters/summary stats:
    prior2 = utils.torchutils.BoxUniform(low=prior_min[6:12], high=prior_max[6:12])

    #combined_prior = Combined(proposal1, prior2, number_params_1=1)
    combined_prior = Combined(proposal1, prior2, number_params_1=6)

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

        finish_time = datetime.datetime.now()

        diff_time = finish_time - start_time



        step_time_str = get_time()
        json_dict = {
        "start time:": start_time_str,
        "round 1 time": step_time_str,
        "CPU time for step:": str(diff_time)}
        with open( "step1/meta.json", "a") as f:
            json.dump(json_dict, f)
            f.close()



    print("second round completed")

    start_time = datetime.datetime.now()

    print(x_without.shape)
    x_without = x_without[:,:4200]

    x_N100 = calculate_summary_stats_temporal(x_without)

    inf = inf.append_simulations(theta, x_N100)
    density_estimator = inf.train()

    posterior = inf.build_posterior(density_estimator)


    obs_real = [obs_real_complete[0][:x_without.shape[1]]]
    obs_real_stat = calculate_summary_stats_temporal(obs_real)



    proposal2 = posterior.set_default_x(obs_real_stat)

    ###### continuing with P200 parameters/summary stats:

    prior3 = utils.torchutils.BoxUniform(low=prior_min[12:], high=prior_max[12:])



    combined_prior = Combined(proposal2, prior3, number_params_1=12)

    inf = SNPE_C(combined_prior, density_estimator=density_estimator)

    try:
        theta = torch.load('step3/thetas.pt')
        x_without = torch.load('step3/obs_without.pt')

    except:
        theta, x_without = inference.run_sim_theta_x(
            combined_prior,
            sim_wrapper,
            num_simulations=int(num_sim*(19/10)),
            num_workers=num_workers
        )

        file_writer.save_obs_without(x_without, name='step3')
        file_writer.save_thetas(theta, name='step3')

        finish_time = datetime.datetime.now()

        diff_time = finish_time - start_time



        step_time_str = get_time()
        json_dict = {
        "start time:": start_time_str,
        "round 1 time": step_time_str,
        "CPU time for step:": str(diff_time)}
        with open( "step1/meta.json", "a") as f:
            json.dump(json_dict, f)
            f.close()


    obs_real = [obs_real_complete[0][:x_without.shape[1]]]

    obs_real_stat = calculate_summary_stats_temporal(obs_real)

    posterior.set_default_x(obs_real_stat)
   

    file_writer.save_posterior(posterior)
    file_writer.save_obs_without(x_without)
    file_writer.save_prior(prior)
    file_writer.save_thetas(theta)

    ## tries to store posterior without torch.save as there is a known bug that torch.save cannot save attributes of class
    with open('posterior2.pt', 'rb') as f:
        pickle.dump(posterior, f)

    os.chdir(file_writer.folder)

    ##save class
    with open("class", "wb") as pickle_file:
        pickle.dump(file_writer, pickle_file)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    main(sys.argv[1:])
