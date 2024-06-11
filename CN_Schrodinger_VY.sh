#!/bin/bash
#SBATCH --time=6:00:00            # time in days-hours:minutes:seconds, COMP PREDICTED 10
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                 # memory
#SBATCH --mail-user=gaianoseworthy@gmail.com
#SBATCH --mail-type=ALL
module load python/3.10
module load scipy-stack
python -u "CN_Schrodinger_VY.py"
