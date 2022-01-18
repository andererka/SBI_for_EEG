#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=64         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --mem=70G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=hostname_%j.out  # File to which STDOUT will be written
#SBATCH --error=hostname_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=k.anderer@t-online.de   # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID 

# insert your commands he
python3 sequential_inference.py 5000 1000 64 5000_sims_3_params 0
#python3 calc_posterior_and_sample.py results/ERP_save_sim_nsf_num_params:3_11-25-2021_21:36:41/class 1000 1 64 
#python3 ERP_simulation_and_inference.py 500 nsf 64 7 rejection
python3 sequential_inference_6params.py 10000 1000 64 10000_sims_6_params 0
#python3 efficiency_comparison_summary_statistics.py True
#python3 investigating_summary_features.py eval_features 
