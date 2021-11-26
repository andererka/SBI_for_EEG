from data_load_writer import load_from_file as lf
from data_load_writer import write_to_file


from summary_features.calculate_summary_features import calculate_summary_stats_N100, calculate_summary_stats_P200, calculate_summary_stats_P50, calculate_summary_stats_P200, calculate_summary_stats9, calculate_summary_statistics_alternative

import numpy as np
import torch

from utils.helpers import get_time

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

    start_time = get_time()
    prior_min_fix = [43.8, 7.9, 89.49]    # 't_evdist_1', 't_evprox_1', 't_evprox_2'

    prior_max_fix = [79.9, 30, 152.96]


    prior_min = [43.8, 7.9, 89.49]    # 't_evdist_1', 't_evprox_1', 't_evprox_2'

    prior_max = [79.9, 30, 152.96]

    true_params = torch.tensor([[63.53, 26.61, 137.12]]) 
    parameter_names =  ['t_evdist_1',  't_evprox_1',  't_evprox_2']


        

    ###### starting with P50 parameters/summary stats:
    prior1 = utils.torchutils.BoxUniform(low=[prior_min[1]], 
                                        high=[prior_max[1]])

    inf = SNPE_C(prior1, density_estimator='nsf')

    theta, x_without = inference.run_sim_theta_x(prior1, num_simulations=num_sim, num_workers =num_workers, density_estimator='nsf')

    print('x without', x_without)

    x_P50 = calculate_summary_stats_P50(x_without)
   

    inf = inf.append_simulations(theta, x_P50)
    density_estimator = inf.train()
    

    posterior = inf.build_posterior(density_estimator)
 

    obs_real = inference.run_only_sim(torch.tensor([[true_params[0][1]]]), num_workers=num_workers)   # first output gives summary statistics, second without

    print('obs real', obs_real)
    obs_real = calculate_summary_stats_P50(obs_real)

    samples = posterior.sample((num_samples,), 
                            x=obs_real)


    print('sample shape', samples.size())

    exit()


    print('samples drawn from first sequence')

    P50_sample_mean = torch.mean(samples)
    P50_sample_std = torch.std(samples)

    print('P50 sample mean', P50_sample_mean)
    print('P50 sample std', P50_sample_std)


    prior_min[1] = P50_sample_mean-P50_sample_std
    prior_max[1] = P50_sample_mean+P50_sample_std

    print('prior_min', prior_min)
    print('prior max', prior_max)


    cond_samples = posterior.sample_conditional(
        (500,), condition=samples[0], dims_to_sample=[1]
    )
    fig, _ = analysis.pairplot(
        cond_samples,
        #limits=[[-2, 2], [-2, 2], [-2, 2]],
        fig_size=(2, 2),
        diag="kde",
        upper="kde",
    )




    ###### continuing with N100 parameters/summary stats:
    prior2 = utils.torchutils.BoxUniform(low=prior_min[0:2], 
                                        high=prior_max[0:2])

    inf = SNPE_C(prior2, density_estimator='nsf')

    theta, x_without = inference.run_sim_theta_x(prior2, num_simulations=num_sim, num_workers =num_workers, density_estimator='nsf')



    x_N100 = calculate_summary_stats_N100(x_without)
   

    inf = inf.append_simulations(theta, x_N100)
    density_estimator = inf.train()
    

    posterior = inf.build_posterior(density_estimator)

    obs_real = inference.run_only_sim(torch.tensor([list(true_params[0][0:2])]), num_workers=num_workers)   # first output gives summary statistics, second without

    obs_real = calculate_summary_stats_N100(obs_real)

    samples = posterior.sample((num_samples,), 
                            x=obs_real)

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


    theta, x_without = inference.run_sim_theta_x(prior3, num_simulations=num_sim, num_workers =num_workers, density_estimator='nsf')



    x_P200 = calculate_summary_stats_P200(x_without)
   

    inf = inf.append_simulations(theta, x_P200)
    density_estimator = inf.train()
    

    posterior = inf.build_posterior(density_estimator)

    obs_real = inference.run_only_sim(true_params, num_workers=num_workers)   # first output gives summary statistics, second without

    obs_real = calculate_summary_stats_P200(obs_real)

    samples = posterior.sample((num_samples,), 
                            x=obs_real)




    limits = [list(tup) for tup in zip(prior_min_fix, prior_max_fix)]

    fig, axes = analysis.pairplot(samples,
                            limits=limits,
                            ticks=limits,
                            figsize=(5,5),
                            points=true_params,
                            points_offdiag={'markersize': 6},
                            points_colors='r',
                            labels=parameter_names);

    file_writer = write_to_file.WriteToFile(experiment='ERP_sequential', num_sim=num_sim,
                    true_params=true_params, density_estimator=density_estimator, num_params=3, num_samples=num_samples)


    file_writer.save_fig(fig, figname="conditional on 2.")

    finish_time = get_time()    
    file_writer.save_all(posterior, prior=prior3, theta=theta, x =x_P200, w_without=x_without,  start_time=start_time, finish_time=finish_time)


    file_writer.save_fig(fig, figname='sequential_approach')


    s_x = inference.run_only_sim(samples, num_workers=num_workers)


    fig3, ax = plt.subplots(1,1)
    ax.set_title('Simulating from proposal')
    for x in x_without:
        im = plt.plot(x)

    fig4, ax = plt.subplots(1,1)
    ax.set_title('Simulating from posterior')
    for s in s_x:
        im = plt.plot(s)

    file_writer.save_fig(fig3, figname="from_prior")
    file_writer.save_fig(fig4, figname="from_posterior")

    


if __name__ == "__main__":
   main(sys.argv[1:])






