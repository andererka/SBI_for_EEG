from data_load_writer import load_from_file as lf
from data_load_writer import write_to_file
import matplotlib.pyplot as plt


import hnn_core
from hnn_core import simulate_dipole, jones_2009_model
from hnn_core.viz import plot_dipole

# import the summary statistics that you want to investigate
from summary_features.calculate_summary_features import calculate_summary_statistics_alternative as extract_sumstats
#from summary_features.calculate_summary_features import calculate_summary_stats_temporal as extract_sumstats


import numpy as np
import torch

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi

from utils import inference

from utils.helpers import SummaryNet
import sys

from data_load_writer import write_to_file


import os
import pickle

import sys

from utils.helpers import get_time


## defining neuronal network model

from utils.simulation_wrapper import event_seed, set_network_default, simulation_wrapper,simulation_wrapper_obs

from joblib import Parallel, delayed

import matplotlib.gridspec as gridspec



"""
This file aims to investigate different summary statistics, their time efficiency and how well they represent the data.
"""



def main(argv):

    start = get_time()

    try:
        file = argv[0]
    except:
        file = "results/ERP_sequential_3params/step3"

    try:
        experiment_name = argv[1]
    except:
        experiment_name = '6_sum_stats_500sim_3params'

    ### loading the class:
    #with open('results/{}/class'.format(file), "rb") as pickle_file:
        #file_writer = pickle.load(pickle_file)

    try:
        sim_wrapper = argv[2]
    except:
        sim_wrapper = simulation_wrapper_obs

    try:
        embed_net = argv[3]
    except:
        embed_net = False


    try:
        num_workers = argv[4]
    except:
        num_workers = 8

    try:
        number_stats = argv[5]
    except:
        number_stats = 6

    if embed_net==True:
        embedding_net = SummaryNet()
    else:
        embedding_net = None



    ##loading the prior, thetas and observations for later inference

    #prior = file_writer.prior

    prior_min = [7.9, 43.8,  89.49] 

    prior_max = [30, 79.9, 152.96]

    prior = utils.torchutils.BoxUniform(low=prior_min, high=prior_max)

    print('prior sample', prior.sample())


    #thetas = torch.load('thetas.pt')
    with open('{}/thetas.pt'.format(file), "rb") as torch_file:
        thetas = torch.load(torch_file)

    with open('{}/obs_without.pt'.format(file), "rb") as torch_file:
        x_without = torch.load(torch_file)

    #thetas = torch.load('{}/thetas.pt'.format(file))
    #x_without = torch.load('{}/obs_without.pt'.format(file))

    x = extract_sumstats(x_without, number_stats)

    true_params = torch.load('{}/true_params.pt'.format(file))
    #true_params = torch.tensor([[  18.9700, 63.5300, 137.1200]])
    print(true_params)


    # instantiate the neural density estimator
    if (embed_net==True):
        density_estimator = utils.posterior_nn(model='nsf', 
                                        embedding_net=embedding_net,
                                        hidden_features=10,
                                        num_transforms=2)
    else:
        density_estimator = 'nsf'

    # setup the inference procedure with the SNPE-C procedure

    inf = SNPE(prior=prior, density_estimator = density_estimator)

    #inf = SNPE_C(prior, density_estimator="nsf")

    inf = inf.append_simulations(thetas, x)


    density_estimator = inf.train()

    posterior = inf.build_posterior(density_estimator)


    #os.mkdir('results')


    num_samples = 100
    file_writer = write_to_file.WriteToFile(
        experiment=experiment_name,
        num_sim='from previous',
        true_params=true_params,
        density_estimator='nsf',
        num_params=2,
        num_samples=num_samples,
        )

    os.mkdir('results/{}/step1'.format(experiment_name))


    file_writer.save_posterior(posterior)
    file_writer.save_observations(x, name='step1')

    file_writer.save_thetas(thetas, name='step1')


    #### how would simulation look like under 'true parameters'
    obs_real = inference.run_only_sim(true_params, sim_wrapper)

    ### sample from posterior given the simulation done with 'true parameters'
    print(embed_net)
    if (embed_net==False):
        print('embed net false')
        print('ob real', obs_real)
        obs_real = extract_sumstats(obs_real[0], number_stats)
        print('ob real', obs_real)
        samples = posterior.sample((num_samples,), x=obs_real)

    else:
        samples = posterior.sample((num_samples,), x=obs_real[0][0:6800])



    ### sample from prior now
    samples_prior = []


    for i in range(num_samples):
        sample = prior.sample()
        samples_prior.append(sample)



    ## simulations from prior samples and posterior samples
    s_x_prior = inference.run_only_sim(samples_prior, sim_wrapper, num_workers)
    s_x = inference.run_only_sim(samples, sim_wrapper, num_workers)

    fig1, ax = plt.subplots(1, 1)
    ax.set_title("Simulating from posterior")
    for s in s_x:
        plt.plot(s, alpha=0.1, color='blue')
        #plt.ylim(-30,30)
        plt.xlim(0, 7000)
    plt.plot(obs_real[0], label='Ground truth', color='red')
    #plt.legend()
        
        
        
    fig2, ax = plt.subplots(1, 1)
    ax.set_title("Simulating from prior")
    for x_w in s_x_prior:
        plt.plot(x_w, alpha=0.1, color='blue')

    fig1.savefig('results/{}/from_posterior_samples.png'.format(experiment_name))
    fig2.savefig('results/{}/from_prior_samples.png'.format(experiment_name))


    finish = get_time()

    file_writer.save_meta(start_time=start, finish_time=finish)



    ###  histogram plots:
    """
    histogram plot compares for each summary feature the distribution of values from posterior versus prior and plots the 'true value' as well
    """

    if (embed_net==False):
        s_x_torch = torch.stack(([s_x[i] for i in range(len(s_x))]))

        s_x_prior_torch = torch.stack(([s_x_prior[i] for i in range(len(s_x_prior))]))
        s_x_stat = extract_sumstats(s_x_torch, number_stats)

        s_x_prior_stat = extract_sumstats(s_x_prior_torch, number_stats)

    
    ### if we use an embedding network, we do not need the step to extract the summary statistics
    else:
        s_x_stat = s_x
        s_x_prior_stat = s_x_prior

    fig3 = plt.figure(figsize=(10,10*len(s_x_stat[0])), tight_layout=True)

    gs = gridspec.GridSpec(nrows=len(s_x_stat[0]), ncols=1)


    sum_stats_names = torch.arange(1, len(s_x_stat[0])+1, 1)

    ##save class
    with open("{}/class".format(file_writer.folder), "wb") as pickle_file:
        pickle.dump(file_writer, pickle_file)


    for i in range(len(sum_stats_names)):

        globals()['ax%s' % i] = fig3.add_subplot(gs[i])

        globals()['sum_stats%s' % i] = []
        globals()['x%s' % i] = []

        for j in range(len(s_x)):
            globals()['sum_stats%s' % i].append(s_x_stat[j][i])
            globals()['x%s' % i].append(s_x_prior_stat[j][i])



        globals()['ax%s' % i].hist(globals()['sum_stats%s' % i],  density=False, facecolor='g', alpha=0.75, histtype='barstacked', label='from posterior')
        globals()['ax%s' % i].hist(globals()['x%s' % i],  density=False, facecolor='b', alpha=0.5, histtype='barstacked', label='simulated')
        
    
        globals()['ax%s' % i].set_title('Histogram of summary stat "{}" '.format(sum_stats_names[i]), pad=20)
        #ax0.set(ylim=(-500, 7000))

        globals()['ax%s' % i].axvline(obs_real[i].detach().numpy(), color='red', label='ground truth')
        globals()['ax%s' % i].legend(loc='upper right')





    fig3.savefig('results/{}/histogram.png'.format(experiment_name))

    



if __name__ == "__main__":
    main(sys.argv[1:])
