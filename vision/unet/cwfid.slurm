#!/bin/bash
# Example of running python script in a batch mode
#SBATCH --mail-user=name@domain.com
#SBATCH --mail-type=end,fail
#SBATCH --output=slurmop/cwfid128/out-cwfid-unet3pHybrid-run-1
#SBATCH --error=slurmop/err-cwfid128
#SBATCH --job-name="unet_cwfid"

eval "$(conda shell.bash hook)"
source activate tf2_15
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
echo "CWFID U-Net Semantic Segmentation slurm job started !"
echo $LD_LIBRARY_PATH
echo $XLA_FLAGS
srun python src/mgpu_unet_cwfid.py
echo "CWFID U-Net Semantic Segmentation slurm job finished !"
