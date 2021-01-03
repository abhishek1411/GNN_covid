#!/bin/bash
#SBATCH --job-name=GNN-covid19    # create a short name for your job
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=3G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --time=00:55:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=a.tomy@student.tudelft.nl

module purge
module use /opt/insy/modulefiles
module load cuda/11.0  
module load cudnn/11.0-8.0.3.33 
module load miniconda
conda activate torch-env

python gnn_seird_cluster.py
