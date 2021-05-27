#!/bin/bash
#SBATCH --job-name=mimic_outer_long
#SBATCH --export=ALL
#SBATCH --time=1-00:00:00
#SBATCH --mem=15G
#SBATCH --array=1-10
#SBATCH --gres=gpu


singularity exec --nv --overlay $SCRATCH/pytorch1.7.0-cuda11.0.ext3:ro /scratch/work/public/singularity/centos-8.2.2004.sif /bin/bash -c "
source /ext3/env.sh
which python
$(head -n $SLURM_ARRAY_TASK_ID runs.txt | tail -n 1)
"

# export PATH=/ext3/miniconda3/bin:$PATH
# export PYTHONPATH=/ext3/miniconda3/bin:$PYTHONPATH