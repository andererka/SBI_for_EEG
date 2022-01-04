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


import os
import pickle

import sys

from utils.helpers import get_time



## defining neuronal network model

from utils.simulation_wrapper import event_seed, set_network_default, simulation_wrapper,simulation_wrapper_obs

from joblib import Parallel, delayed



def main(argv):

    start = get_time()

    try:
        file = argv[0]
    except:
        file = "name_bad"
    ### loading the class:
    with open('results/{}/class'.format(file), "rb") as pickle_file:
        file_writer = pickle.load(pickle_file)

    try:
        sim_wrapper = argv[1]
    except:
        sim_wrapper = simulation_wrapper_obs
    try:
        embed_net = argv[2]
    except:
        embed_net = False
    try:
        experiment_name = argv[3]
    except:
        experiment_name = 'cnn_sum_stats_500sim_2params'

    try:
        num_workers = argv[4]
    except:
        num_workers = 8

    try:
        number_stats = argv[5]
    except:
        number_stats = 6

    if embed_net==True:
        embed_net = SummaryNet()
    else:
        embed_net = None




    ##loading the prior, thetas and observations for later inference

    prior = file_writer.prior
    file_writer.folder = 'results/name_bad'

    import os
    print(os.getcwd())
    #thetas = torch.load('thetas.pt')


    thetas = lf.load_thetas(file_writer.folder)
    x_without = lf.load_obs(file_writer.folder)

    x = extract_sumstats(x_without, number_stats)

    true_params = lf.load_true_params(file_writer.folder)


    # instantiate the neural density estimator
    if (embed_net==True):
        density_estimator = utils.posterior_nn(model='nsf', 
                                        embedding_net=embed_net,
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

    from data_load_writer import write_to_file

    import os
    
    true_params = torch.tensor([[26.61, 63.53,  137.12]])

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

    #sample from posterior given the simulation done with 'true parameters'

    if (embed_net==False):
        print('embed net false')
        obs_real = extract_sumstats(obs_real[0], number_stats)
        samples = posterior.sample((num_samples,), x=obs_real[0])

    else:
        samples = posterior.sample((num_samples,), x=obs_real[0][0:6800])



    ## sample from prior now
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

if __name__ == "__main__":
    main(sys.argv[1:])
