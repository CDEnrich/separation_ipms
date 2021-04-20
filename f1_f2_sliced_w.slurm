#!/bin/bash
#SBATCH --job-name=f1_f2_sliced_grid_job
#SBATCH --open-mode=append
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH --export=ALL
#SBATCH --time=10:00:00
#SBATCH --mem=15G

#SBATCH --array=1-60

singularity exec --overlay /scratch/cd2754/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate

python ./f1_f2_sliced_w.py --task_id $SLURM_ARRAY_TASK_ID --n_samples 100000
"

