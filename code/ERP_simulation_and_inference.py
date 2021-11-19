#!/usr/bin/env python
# coding: utf-8


import os.path as op
import tempfile
import datetime


import numpy as np
import torch
from data_load_writer import write_to_file
import pickle

# visualization
import matplotlib.pyplot as plt

# sbi
from sbi import utils as utils
from sbi import analysis as analysis



from utils.helpers import get_time



from utils import inference
import sys


def main(argv):
    """

    description: is simulating an event related potential with the hnn-core package and then uses sbi to
    infer parameters and draw samples from parameter posteriors. One can choose the following
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
        density_estimator ='nsf'
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
        num_params = None
    try:
        sample_method = argv[5]
    except:
        sample_method = 'rejection'

    try:
        prior_check = argv[6]
    except:
        prior_check=False


    print(num_params)
    ##defining the prior lower and upper bounds
    if (num_params==6):
        prior_min = [43.8, 3.01, 11.364, 1.276, 89.49, 5.29]   # 't_evdist_1', 'sigma_t_evdist_1', 't_evprox_1', 'sigma_t_evprox_1', 't_evprox_2', 'sigma_t_evprox_2'

        prior_max = [79.9, 9.03, 26.67, 3.828, 152.96, 15.87]  
        
        true_params = torch.tensor([[61.89, 6.022, 19.01, 2.55, 121.23, 10.58]])   

        parameter_names =  ['t_evdist_1', 'sigma_t_evdist_1', 't_evprox_1', 'sigma_t_evprox_1', 't_evprox_2', 'sigma_t_evprox_2']
    

    if (num_params==3):
        prior_min = [43.8, 89.49, 7.9]    # 't_evdist_1', 't_evprox_1', 't_evprox_2'

        prior_max = [79.9, 152.96, 30]

        true_params = torch.tensor([[63.53, 137.12, 18.97]]) 
        parameter_names =  ['t_evdist_1',  't_evprox_1',  't_evprox_2']

    if (num_params==2):
        prior_min = [43.8, 89.49] 
        prior_max = [79.9, 152.96]

        true_params = torch.tensor([[63.53, 137.12]]) 
        parameter_names =  ['t_evdist_1',  't_evprox_1']

    elif (num_params==None):
        print('number of parameters must be defined in the arguments')
        sys.exit()
        

    prior = utils.torchutils.BoxUniform(low=prior_min, 
                                        high=prior_max)
    if prior_check:
        posterior, theta, x, x_without = inference.run_sim_inference(prior, num_simulations=number_simulations, num_workers =num_workers, density_estimator=density_estimator, prior_check=prior_check)
    else:
        posterior, theta, x = inference.run_sim_inference(prior, num_simulations=number_simulations, num_workers =num_workers, density_estimator=density_estimator, prior_check=prior_check)



    # next two lines are not necessary if we have a real observation from experiment
    # here we simulate this 'real observation' by simulation


    # 't_evdist_1', 'sigma_t_evdist_1', 't_evprox_1', 'sigma_t_evprox_1', 't_evprox_2', 'sigma_t_evprox_2'


    obs_real, _ = inference.run_only_sim(true_params, num_workers=num_workers)   # first output gives summary statistics, second without


    samples = posterior.sample((num_samples,), 
                            x=obs_real[0], sample_with=sample_method)

    s_x, s_x_stats = inference.run_only_sim(samples, num_workers=num_workers)

    limits = [list(tup) for tup in zip(prior_min,prior_max)]

    fig, axes = analysis.pairplot(samples,
                            limits=limits,
                            ticks=limits,
                            figsize=(5,5),
                            points=true_params,
                            points_offdiag={'markersize': 6},
                            points_colors='r',
                            tick_labels=parameter_names);

    corr_matrix_marginal = np.corrcoef(samples.T)
    fig2, ax = plt.subplots(1,1, figsize=(4, 4))
    im = plt.imshow(corr_matrix_marginal, clim=[-1, 1], cmap='PiYG')
    _ = fig2.colorbar(im)


    fig3, ax = plt.subplots(1,1, figsize=(4, 4))
    ax.set_title('Simulating from posterior (with summary stats')
    for s in s_x_stats:
        im = plt.plot(s)


    if prior_check:
        fig5, ax = plt.subplots(1,1, figsize=(4, 4))
        ax.set_title('Simulating from proposal (without summary stats')
        for x_w in x_without:
            im = plt.plot(x_w)

    fig4, ax = plt.subplots(1,1)
    ax.set_title('Simulating from posterior (without summary stats)')
    for s in s_x:
        im = plt.plot(s)




    file_writer = write_to_file.WriteToFile(experiment='ERP_{}_num_params:{}_'.format(density_estimator, num_params), num_sim=number_simulations,
                    true_params=true_params, density_estimator=density_estimator, num_params=num_params, num_samples=num_samples)



    finish_time = get_time()
    file_writer.save_all(posterior, prior, theta=theta, x =x, fig=fig, start_time=start_time, finish_time=finish_time)

    file_writer.save_fig(fig2)
    file_writer.save_fig(fig3)
    file_writer.save_fig(fig4)
    file_writer.save_fig(fig5)
    ##save class 
    with open('{}/class'.format(file_writer.folder), 'wb') as pickle_file:
        pickle.dump(file_writer, pickle_file)
    


if __name__ == "__main__":
   main(sys.argv[1:])


