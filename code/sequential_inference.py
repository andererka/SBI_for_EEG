from data_load_writer import load_from_file as lf
from data_load_writer import write_to_file


from summary_features.calculate_summary_features import calculate_summary_stats_N100, calculate_summary_stats_P200, calculate_summary_stats_P50, calculate_summary_stats_P200, calculate_summary_stats9, calculate_summary_statistics_alternative

import numpy as np
import torch

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE_C, prepare_for_sbi, simulate_for_sbi

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
        num_sim = int(argv[0])
    except:
        num_sim = 100
    try:
        num_samples = int(argv[1])
    except:
        num_samples = 20

    try: 
        num_rounds = int(argv[2])
    except:
        num_rounds = 1
    try:
        num_workers = int(argv[3])
    except:
        num_workers = 4




    prior_min = [43.8, 89.49, 7.9]    # 't_evdist_1', 't_evprox_1', 't_evprox_2'

    prior_max = [79.9, 152.96, 30]

    true_params = torch.tensor([[63.53, 137.12, 18.97]]) 
    parameter_names =  ['t_evdist_1',  't_evprox_1',  't_evprox_2']


        

    ###### starting with P50 parameters/summary stats:
    prior1 = utils.torchutils.BoxUniform(low=[prior_min[1]], 
                                        high=[prior_max[1]])

    inf = SNPE_C(prior1, density_estimator='nsf')

    posterior, theta, _, x_without = inference.run_sim_inference(prior1, num_simulations=100, num_workers =num_workers, density_estimator='nsf')



    x_P50 = calculate_summary_stats_P50(x_without)
   

    inf = inf.append_simulations(theta, x_P50)
    density_estimator = inf.train()
    

    posterior = inf.build_posterior(density_estimator)
    print(true_params, [true_params[0][1]])

    _, obs_real = inference.run_only_sim([true_params[0][1]], num_workers=num_workers)   # first output gives summary statistics, second without

    obs_real = calculate_summary_stats_P50(obs_real)

    samples = posterior.sample((num_samples,), 
                            x=calculate_summary_stats_P50(obs_real))

    P50_sample_mean = torch.mean(samples)
    P50_sample_std = torch.std(samples)

    print('P50 sample mean', P50_sample_mean)


    prior_min[1] = P50_sample_mean-P50_sample_std
    prior_max[1] = P50_sample_mean+P50_sample_std

    print('prior_min', prior_min)
    print('prior max', prior_max)



    ###### continuing with N199 parameters/summary stats:
    prior2 = utils.torchutils.BoxUniform(low=prior_min[0:1], 
                                        high=prior_max[0:1])

    inf = SNPE_C(prior2, density_estimator='nsf')

    posterior, theta, _, x_without = inference.run_sim_inference(prior2, num_simulations=100, num_workers =num_workers, density_estimator='nsf')



    x_N100 = calculate_summary_stats_N100(x_without)
   

    inf = inf.append_simulations(theta, x_N100)
    density_estimator = inf.train()
    

    posterior = inf.build_posterior(density_estimator)

    _, obs_real = inference.run_only_sim(true_params[0][0:1], num_workers=num_workers)   # first output gives summary statistics, second without

    obs_real = calculate_summary_stats_N100(obs_real)

    samples = posterior.sample((num_samples,), 
                            x=calculate_summary_stats_P50(obs_real))

    N100_sample_mean = torch.mean(samples)
    N100_sample_std = torch.std(samples)

    print('N100 sample mean', N100_sample_mean)


    prior_min[0] = N100_sample_mean-N100_sample_std
    prior_max[0] = N100_sample_mean+N100_sample_std

    print('prior_min', prior_min)
    print('prior max', prior_max)



    ###### continuing with P200 parameters/summary stats:
    prior3 = utils.torchutils.BoxUniform(low=prior_min, 
                                        high=prior_max)

    inf = SNPE_C(prior3, density_estimator='nsf')


    posterior, theta, _, x_without = inference.run_sim_inference(prior3, num_simulations=num_sim, num_workers =num_workers, density_estimator='nsf')



    x_P200 = calculate_summary_stats_P200(x_without)
   

    inf = inf.append_simulations(theta, x_P200)
    density_estimator = inf.train()
    

    posterior = inf.build_posterior(density_estimator)

    _, obs_real = inference.run_only_sim(true_params, num_workers=num_workers)   # first output gives summary statistics, second without

    obs_real = calculate_summary_stats_P200(obs_real)

    samples = posterior.sample((num_samples,), 
                            x=calculate_summary_stats_P200(obs_real))




    limits = [list(tup) for tup in zip(prior_min,prior_max)]

    fig, axes = analysis.pairplot(samples,
                            limits=limits,
                            ticks=limits,
                            figsize=(5,5),
                            points=true_params,
                            points_offdiag={'markersize': 6},
                            points_colors='r',
                            label_samples=parameter_names);

    file_writer = write_to_file.WriteToFile(experiment='ERP_sequential_{}'.format(density_estimator), num_sim=num_sim,
                    true_params=true_params, density_estimator=density_estimator, num_params=3, num_samples=num_samples)


    file_writer.save_fig(fig, figname='sequential_approach')


    s_x, s_x_stats = inference.run_only_sim(samples, num_workers=num_workers)


    fig3, ax = plt.subplots(1,1)
    ax.set_title('Simulating from posterior (with summary stats)')
    for s in s_x_stats:
        im = plt.plot(s)

    fig4, ax = plt.subplots(1,1)
    ax.set_title('Simulating from posterior (with summary stats)')
    for s in s_x:
        im = plt.plot(s)

    #file_writer.save_fig(fig2, figname='With_9_stats')
    file_writer.save_fig(fig3, figname="Without_9_stats")
    file_writer.save_fig(fig4, figname="With_9_stats")

    


if __name__ == "__main__":
   main(sys.argv[1:])






