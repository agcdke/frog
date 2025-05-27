#!/bin/bash
# Example of running python script in a batch mode
#SBATCH --mail-user=name@domain.com
#SBATCH --mail-type=end,fail
#SBATCH --output=slurmop/soybeanstress128/T1_mvit_h4b4/out-T1_mvit_h4b4-run_1
#SBATCH --error=slurmop/soybeanstress128/err-rgb
#SBATCH --job-name="iowa_imgcls"

eval "$(conda shell.bash hook)"
source activate tf2_15
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
echo "Iowa Soybean Plant Stress image classifier slurm job started !"
echo $LD_LIBRARY_PATH
echo $XLA_FLAGS
srun python src/multigpu_imgcls_soybeanplantstress.py
echo "Iowa Soybean Plant Stress image classifier slurm job finished !"
