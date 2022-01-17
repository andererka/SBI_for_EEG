from utils.simulation_wrapper import simulation_wrapper_all, simulation_wrapper_obs
from data_load_writer import load_from_file as lf
from data_load_writer import write_to_file


from summary_features.calculate_summary_features import (
    calculate_summary_stats_temporal

)

import numpy as np
import torch

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

    if (slurm==0):
        slurm = True
    else:
        slurm = False

    ## using a density estimator with only 1 transform (which should be enough for the 1D case)
    #dens_estimator = posterior_nn(model='nsf', hidden_features=60, num_transforms=1)


    start_time = get_time()

    sim_wrapper = simulation_wrapper_obs
    #sim_wrapper = simulation_wrapper_all

    prior_min_fix = [7.9, 43.8, 89.49]  # 't_evprox_1', 't_evdist_1', 't_evprox_2'

    prior_max_fix = [30, 79.9,  152.96]

    prior_min = [7.9, 43.8,  89.49] 

    prior_max = [30, 79.9, 152.96]

    ### for also inferring connection weights etc.:

    #prior_min_fix = [0, 0, 0, 0, 17.3, 0, 0, 0, 0, 0, 0, 51.980, 0, 0, 0, 0, 112.13]
    #prior_max_fix = [0.927, 0.160, 2.093, 0.0519, 35.9, 0.039, 0.000042, 0.854, 0.117, 0.0259, 0.480, 75.08, 0.0000018, 8.633, 0.0537, 4.104, 162.110]

    #prior_min = [0, 0, 0, 0, 17.3, 0, 0, 0, 0, 0, 0, 51.980, 0, 0, 0, 0, 112.13]
    #prior_max = [0.927, 0.160, 2.093, 0.0519, 35.9, 0.039, 0.000042, 0.854, 0.117, 0.0259, 0.480, 75.08, 0.0000018, 8.633, 0.0537, 4.104, 162.110]


    true_params = torch.tensor([[26.61, 63.53,  137.12]])
    #true_params = torch.tensor([[0.277, 0.0399, 0.3739, 0.034, 18.977, 0.0115, 0.000012, 0.466, 0.06337, 0.0134, 0.0766, 63.08, 0.000005, 4.6729, 0.0115, 0.3308, 120.86]])

    
    parameter_names = ["t_evprox_1", "t_evdist_1", "t_evprox_2"]

    #parameter_names = ["prox_1_ampa_l2_bas","prox_1_ampa_l2_pyr","prox_1_ampa_l5_bas","prox_1_ampa_l5_pyr",
    # "t_evprox_1", posterior_nn(model='nsf', hidden_features=60, num_transforms=3)

    # "dist_ampa_l2_bas","dist_ampa_l2_pyr","dist_ampa_l5_pyr",
    # "dist_nmda_l2_bas","dist_nmda_l2_pyr","dist_nmda_l5_pyr",
    # "t_evdist_1", 
    # "prox_2_ampa_l2_bas","prox_2_ampa_l2_pyr","prox_2_ampa_l5_bas","prox_2_ampa_l5_pyr",
    # "t_evprox_2"]

    ###### starting with P50 parameters/summary stats:
    prior1 = utils.torchutils.BoxUniform(low=[prior_min[0]], high=[prior_max[0]])

    #prior1 = utils.torchutils.BoxUniform(low=prior_min[0:5], high=prior_max[0:5])

    inf = SNPE_C(prior1, density_estimator='nsf')


    try:

        file_writer = torch.load('results/{}/class.pt'.format(experiment_name))
        
    except:

        print('new experiment')
        file_writer = write_to_file.WriteToFile(
        experiment=experiment_name,
        num_sim=num_sim,
        density_estimator='nsf',
        num_params=3,
        num_samples=num_samples,
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
            exit

        



    
    try:
        theta = torch.load('{}/step1/thetas.pt'.format(file_writer.folder))
        x_without = torch.load('{}/step1/obs_without.pt'.format(file_writer.folder))

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
    density_estimator = inf.train()

    posterior = inf.build_posterior(density_estimator)

    obs_real = inference.run_only_sim(
    torch.tensor([list([true_params[0][0]])]), simulation_wrapper = sim_wrapper, num_workers=num_workers
)  

    #obs_real = inference.run_only_sim(
     #   torch.tensor([list(true_params[0][0:5])]), simulation_wrapper = simulation_wrapper_all, num_workers=num_workers
    #)  # first output gives summary statistics, second without

    print("obs real", obs_real)
    obs_real = calculate_summary_stats_temporal(obs_real)

    samples = posterior.sample((num_samples,), x=obs_real)

    proposal1 = posterior.set_default_x(obs_real)

    ###### continuing with N100 parameters/summary stats:
    prior2 = utils.torchutils.BoxUniform(low=[prior_min[1]], high=[prior_max[1]])
    #prior2 = utils.torchutils.BoxUniform(low=prior_min[5:12], high=prior_max[5:12])

    combined_prior = Combined(proposal1, prior2, number_params_1=1)

    inf = SNPE_C(combined_prior, density_estimator="nsf")


    try:
        theta = torch.load('{}/step2/thetas.pt'.format(file_writer.folder))
        x_without = torch.load('{}/step2/obs_without.pt'.format(file_writer.folder))

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
    density_estimator = inf.train()

    posterior = inf.build_posterior(density_estimator)

    obs_real = inference.run_only_sim(
        torch.tensor([list(true_params[0][0:1])]),
        sim_wrapper,
        num_workers=num_workers
    )
    #obs_real = inference.run_only_sim(
    #    torch.tensor([list(true_params[0][0:12])]),
    #    simulation_wrapper_all,
    #    num_workers=num_workers
    #)  # first output gives summary statistics, second without

    obs_real = calculate_summary_stats_temporal(obs_real)

    print("obs real", obs_real.size())

    samples = posterior.sample((num_samples,), x=obs_real)

    proposal2 = posterior.set_default_x(obs_real)

    ###### continuing with P200 parameters/summary stats:
    prior3 = utils.torchutils.BoxUniform(low=[prior_min[2]], high=[prior_max[2]])
    #prior3 = utils.torchutils.BoxUniform(low=prior_min[12:], high=prior_max[12:])
    combined_prior = Combined(proposal2, prior3, number_params_1=2)

    inf = SNPE_C(combined_prior, density_estimator="nsf")

    try:
        theta = torch.load('{}/step3/thetas.pt'.format(file_writer.folder))
        x_without = torch.load('{}/step3/obs_without.pt'.format(file_writer.folder))

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
    density_estimator = inf.train()

    posterior = inf.build_posterior(density_estimator)

    obs_real = inference.run_only_sim(
        true_params, sim_wrapper, num_workers=num_workers
    )  # first output gives summary statistics, second without

    obs_real = calculate_summary_stats_temporal(obs_real)

    samples = posterior.sample((num_samples,), x=obs_real)

    limits = [list(tup) for tup in zip(prior_min_fix, prior_max_fix)]

    fig, axes = analysis.pairplot(
        samples,
        limits=limits,
        ticks=limits,
        figsize=(5, 5),
        points=true_params,
        points_offdiag={"markersize": 6},
        points_colors="r",
        labels=parameter_names,
    )

    
    fig.savefig('{}/posterior_dens.png'.format(file_writer.folder))

    finish_time = get_time()


    s_x = inference.run_only_sim(samples, simulation_wrapper=sim_wrapper, num_workers=num_workers)

    fig3, ax = plt.subplots(1, 1)
    ax.set_title("Simulating from proposal")
    for x in x_without:
        plt.plot(x)

    file_writer.save_fig(fig3, 'sim_from_proposal')

    fig4, ax = plt.subplots(1, 1)
    ax.set_title("Simulating from posterior")
    for s in s_x:
        plt.plot(s)

    file_writer.save_fig(fig4, 'sim_from_posterior')

    fig3.savefig('{}/from_prior.png'.format(file_writer.folder))
    fig4.savefig('{}/from_posterior_dens.png'.format(file_writer.folder))

    #file_writer.save_posterior(posterior)
    #file_writer.save_prior(combined_prior)

    file_writer.save_all(
        start_time=start_time,
        finish_time=finish_time,
        source='sequential_inference')

if __name__ == "__main__":
    main(sys.argv[1:])
