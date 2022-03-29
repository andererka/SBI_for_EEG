from utils.simulation_wrapper import SimulationWrapper
from data_load_writer import load_from_file as lf
from data_load_writer import write_to_file


from summary_features.calculate_summary_features import (
    calculate_summary_stats_temporal, calculate_summary_statistics_alternative

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



def main(argv):
    """

    description: simulating from 17 different parameters, starting with with a set of 6 parameters (all
    associated with the first proximal drive), then combining the posterior of the first set with the
    next 6 parameters (from the subset associated with the parameters of the distal drive), and so on. 

    argument 1: number of simulation
    argument 2: number of cpus available for parallel processing
    argument 3: name of experiment that result folder will be associated with
    argument 4: running on slurm or not. If experiment is running on slurm, it should be 1 (also the default),
                otherwise 0.
    argument 5: density estimator. Default is 'maf'

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
        density_estimator = 'maf'



    start_time_str = get_time()
    start_time = datetime.datetime.now()

    # defining simulation wrapper with the SimulationWrapper class. Takes number of parameters as argument
    sim_wrapper = SimulationWrapper(num_params=17)


    prior_min = [0, 0, 0, 0, 0, 13.3,  0, 0, 0, 0, 0, 51.980, 0, 0, 0, 0, 112.13]
    prior_max = [0.927, 0.160, 2.093, 1.0, 1.0, 35.9, 0.000042, 0.039372, 0.025902,  0.480, 0.117, 75.08, 8.633, 4.104, 1.0, 1.0, 162.110]

    true_params = torch.tensor([[0.277, 0.0399, 0.6244, 0.3739, 0.0, 18.977, 0.000012, 0.0115, 0.0134,  0.0767, 0.06337, 63.08, 4.6729, 2.33, 0.016733, 0.0679, 120.86]])

    # parameter names as reference although not needed here:
    parameter_names = ["prox1_ampa_l2_bas","prox1_ampa_l2_pyr","prox1_ampa_l5_bas","prox1_nmda_l5_bas", "prox1_nmda_l5_pyr",
     "t_prox1",
     "dist_ampa_l2_pyr","dist_ampa_l2_bas","dist_nmda_l2_pyr",
     "dist_nmda_l5_pyr","dist_nmda_l2_bas",
     "t_dist", 
     "prox2_ampa_l2_pyr","prox2_ampa_l5_pyr","prox2_nmda_l2_pyr","prox2_nmda_l5_pyr",
     "t_prox2"]

    obs_real = inference.run_only_sim(
    true_params, sim_wrapper, num_workers=1)  # first output gives summary statistics, second without

    obs_real_stat = calculate_summary_stats_temporal(obs_real[0], complete=True)



    ###### starting with P50 parameters/summary stats:

    prior = utils.torchutils.BoxUniform(low=prior_min, high=prior_max)

    prior1 = utils.torchutils.BoxUniform(low=prior_min[0:6], high=prior_max[0:6])

    inf = SNPE_C(prior1, density_estimator=density_estimator)



    file_writer = write_to_file.WriteToFile(
    experiment=experiment_name,
    num_sim=num_sim,
    density_estimator=density_estimator,
    num_params=len(prior_max),
    slurm=slurm,
    )


    ## create result folder and subfolders for the 3 steps:
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
    open('{}/sequential_inference_17params.py'.format(file_writer.folder), 'a').close()
    shutil.copyfile(str(os.getcwd() + '/sequential_inference_17params.py'), str(file_writer.folder+ '/sequential_inference_17params.py'))


    os.chdir(file_writer.folder)

    ### save all single posteriors after each step without conditioning on observation (for later sbc analysis)
    posteriors = []


    
    try:
        theta = torch.load('step1/thetas.pt')
        x_without = torch.load('step1/obs_without.pt')

    except:
        theta, x_without = inference.run_sim_theta_x(
            prior1, 
            sim_wrapper,
            #num_simulations=int(num_sim*(1/10)),
            num_simulations = num_sim,
            num_workers=num_workers
        )

    
        finish_time = datetime.datetime.now()

        diff_time = finish_time - start_time



        step_time_str = get_time()

        # stores meta information so far:
        json_dict = {
        "start time:": start_time_str,
        "round 1 time": step_time_str,
        "CPU time for step:": str(diff_time)}
        with open( "step1/meta.json", "a") as f:
            json.dump(json_dict, f)
            f.close()

        # stores x and theta into result subfolder 'step1'
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

    theta = theta[:,:6]


    x_P50 = calculate_summary_stats_temporal(x_without)

    print('x_P50',x_P50)

    print('x without', x_without.shape)



    inf = inf.append_simulations(theta, x_P50)
    neural_dens = inf.train()

    posterior = inf.build_posterior(neural_dens)

    posteriors.append(posterior)


    posterior.set_default_x(obs_real_stat[:,:10])

    proposal1 = posterior

    ###### continuing with N100 parameters/summary stats:
    prior2 = utils.torchutils.BoxUniform(low=prior_min[6:12], high=prior_max[6:12])

    combined_prior = Combined([proposal1], prior2, steps = [0, 6, 12])

    inf = SNPE_C(prior2, density_estimator=density_estimator)


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

    theta = theta[:,6:12]
    x_without = x_without[:,:4200]

    x_N100 = calculate_summary_stats_temporal(x_without)

    print('x:N100 shape', x_N100.shape)

    inf = inf.append_simulations(theta, x_N100)
    neural_dens = inf.train()

    posterior = inf.build_posterior(neural_dens)

    posteriors.append(posterior)

    proposal2 = posterior.set_default_x(obs_real_stat[:,:13])

    ###### continuing with P200 parameters/summary stats:

    prior3 = utils.torchutils.BoxUniform(low=prior_min[12:], high=prior_max[12:])



    combined_prior = Combined([proposal1, proposal2], prior3, steps = [0, 6, 12])

    inf = SNPE_C(prior3, density_estimator=density_estimator)

    try:
        theta = torch.load('step3/thetas.pt')
        x_without = torch.load('step3/obs_without.pt')

    except:
        theta, x_without = inference.run_sim_theta_x(
            combined_prior,
            sim_wrapper,
            #num_simulations=int(num_sim*(19/10)),
            num_simulations = num_sim,
            num_workers = num_workers
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


    x = calculate_summary_stats_temporal(x_without)

    theta = theta[:,12:]

    inf = inf.append_simulations(theta, x)
    neural_dens = inf.train()

    proposal3 = inf.build_posterior(neural_dens)

    posteriors.append(proposal3)


    proposal3.set_default_x(obs_real_stat)


    posterior = Combined([proposal1, proposal2, proposal3], None, steps=[0, 6, 12])

    torch.save(posteriors, 'posteriors_each_step.pt')

    file_writer.save_posterior(posterior)
    file_writer.save_prior(prior)
    file_writer.save_thetas(theta)

    torch.save(obs_real, 'obs_real.pt')

    ## tries to store posterior without torch.save as there is a known bug that torch.save cannot save attributes of class
    with open('posterior2.pkl', 'wb') as f:
        pickle.dump(posterior, f)



if __name__ == "__main__":
    torch.manual_seed(5)
    np.random.seed(5)
    main(sys.argv[1:])
