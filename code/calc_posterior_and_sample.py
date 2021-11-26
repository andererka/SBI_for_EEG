from data_load_writer import load_from_file as lf
from data_load_writer import write_to_file


from summary_features.calculate_summary_features import calculate_summary_stats, calculate_summary_stats9, calculate_summary_statistics_alternative

import numpy as np
import torch

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE_C, prepare_for_sbi, simulate_for_sbi

from utils.helpers import get_time

from utils import inference


import pickle
import sys

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
        file = argv[0]
    except:
        file = 'results/ERP_save_sim_nsf_num_params:3_11-25-2021_21:36:41/class'
    try:
        num_samples = int(argv[1])
    except:
        num_samples = 100

    try: 
        num_rounds = int(argv[2])
    except:
        num_rounds = 1
    try:
        num_workers = int(argv[3])
    except:
        num_workers = 4

    start_time = get_time()

    ### loading the class:
    with open(file, 'rb') as pickle_file:
        file_writer = pickle.load(pickle_file)


    ##loading theta and x from results
    theta = lf.load_thetas(file_writer.folder)
    x_without = lf.load_obs_without(file_writer.folder)
    x = lf.load_obs(file_writer.folder)



    num_params = theta.size(dim=1)
    print(num_params)
    if (num_params==6):
        prior_min = [43.8, 3.01, 11.364, 1.276, 89.49, 5.29]   # 't_evdist_1', 'sigma_t_evdist_1', 't_evprox_1', 'sigma_t_evprox_1', 't_evprox_2', 'sigma_t_evprox_2'

        prior_max = [79.9, 9.03, 26.67, 3.828, 152.96, 15.87]  
        
        true_params = torch.tensor([[61.89, 6.022, 19.01, 2.55, 121.23, 10.58]])   

        parameter_names =  ['t_evdist_1', 'sigma_t_evdist_1', 't_evprox_1', 'sigma_t_evprox_1', 't_evprox_2', 'sigma_t_evprox_2']
    

    if (num_params==3):
        prior_min = [43.8,  7.9, 89.49]    # 't_evdist_1', 't_evprox_1', 't_evprox_2'

        prior_max = [79.9, 30, 152.96]
        true_params = torch.tensor([[63.53, 18.97, 137.12]]) 
        parameter_names =  ['t_evdist_1',  't_evprox_1',  't_evprox_2']

    if (num_params==2):
        prior_min = [43.8, 89.49] 
        prior_max = [79.9, 152.96]

        true_params = torch.tensor([[63.53, 137.12]]) 
        parameter_names =  ['t_evdist_1',  't_evprox_1']

    elif (num_params==None):
        print('number of parameters must be defined in the arguments')
        sys.exit()
        

    ###calculating summary statistics:
    prior = lf.load_prior(file_writer.folder)
    inf = SNPE_C(prior, density_estimator='nsf')
    print('x_without', x_without)
    x_9 = calculate_summary_stats9(x_without)

    inf = inf.append_simulations(theta, x_9)
    density_estimator = inf.train()
    

    posterior = inf.build_posterior(density_estimator)




    true_params = lf.load_true_params(file_writer.folder)
    limits = [list(tup) for tup in zip(prior_min,prior_max)]


    file_writer2 = write_to_file.WriteToFile(experiment='ERP_9stats_num_params:{}_'.format(num_params), num_sim=0,
                    true_params=true_params, density_estimator=density_estimator, num_params=num_params, num_samples=num_samples)

    finish_time = get_time()
    file_writer2.save_all(posterior, prior, theta=theta, x =x, x_without=x_without, fig=None, start_time=start_time, finish_time=finish_time)



    for i in range(num_rounds):
        obs_real = inference.run_only_sim(true_params)

        obs_real = calculate_summary_stats9(obs_real)

        samples = posterior.sample((num_samples,), 
                                x=obs_real)




        fig, axes = analysis.pairplot(samples,
                                limits=limits,
                                ticks=limits,
                                figsize=(5,5),
                                points=true_params,
                                points_offdiag={'markersize': 6},
                                points_colors='r',
                                labels=parameter_names);

        file_writer2.save_fig(fig, figname='summary_stats_9')
        axes[0,0].set_xlabel(parameter_names[0])

    s_x = inference.run_only_sim(samples, num_workers=num_workers)
    s_x_stats = calculate_summary_stats9(s_x)


    fig2, ax = plt.subplots(1,1)
    ax.set_title('Simulating from prior')
    for x_w in x_without:
        im = plt.plot(x_w)

    fig3, ax = plt.subplots(1,1)
    ax.set_title('Simulating from posterior')
    for s in s_x_stats:
        im = plt.plot(s)

    fig4, ax = plt.subplots(1,1)
    ax.set_title('Simulating from posterior')
    for s in s_x:
        im = plt.plot(s)

    #file_writer.save_fig(fig2, figname='With_9_stats')
    file_writer2.save_fig(fig2, figname="from_prior")
    file_writer2.save_fig(fig3, figname="from_posterior_stats")
    file_writer2.save_fig(fig4, figname="from_posterior")

    


if __name__ == "__main__":
   main(sys.argv[1:])






