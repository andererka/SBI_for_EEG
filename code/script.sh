#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=64         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-08:00            # Runtime in D-HH:MM
#SBATCH --mem=50G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=hostname_%j.out  # File to which STDOUT will be written
#SBATCH --error=hostname_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=k.anderer@t-online.de   # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID 

# insert your commands he
python sequential_inference.py 1000 1000 1 64
