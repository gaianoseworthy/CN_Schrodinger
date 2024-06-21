#!/bin/bash
#SBATCH --time=45:00            # time in days-hours:minutes:seconds, COMP PREDICTED 10
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --array=0-26
#SBATCH --mail-user=gaianoseworthy@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --mem=4G
module load python/3.10
module load scipy-stack
python -u "CN_Schrodinger_New.py" $SLURM_ARRAY_TASK_ID
