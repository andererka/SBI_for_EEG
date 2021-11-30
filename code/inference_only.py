from data_load_writer import load_from_file as lf
from data_load_writer import write_to_file
import matplotlib.pyplot as plt


import hnn_core
from hnn_core import simulate_dipole, jones_2009_model
from hnn_core.viz import plot_dipole
from summary_features.calculate_summary_features import (
    calculate_summary_stats,
    calculate_summary_statistics_alternative,
)

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
import sys


import os
import pickle


def main(argv):

    try:
        file = argv[0]
    except:
        file = "results/ERP_results11-02-2021_16:22:47/class"
    ### loading the class:
    with open(file, "rb") as pickle_file:
        file_writer = pickle.load(pickle_file)

    ##loading the prior, thetas and observations for later inference
    prior = lf.load_prior(file_writer.folder)
    thetas = lf.load_thetas(file_writer.folder)
    x = lf.load_obs(file_writer.folder)

    true_params = lf.load_true_params(file_writer.folder)

    posterior = inference.run_only_inference(theta=thetas, x=x, prior=prior)

    obs_real = inference.run_only_sim(true_params)

    samples = posterior.sample((100,), x=obs_real)

    fig, axes = analysis.pairplot(
        samples,
        # limits=[[.5,80], [1e-4,15.]],
        # ticks=[[.5,80], [1e-4,15.]],
        figsize=(5, 5),
        points=true_params,
        points_offdiag={"markersize": 6},
        points_colors="r",
    )

    file_writer.save_posterior(posterior)
    file_writer.save_fig(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
