#!/bin/bash
# specify a partition
#SBATCH -p batch
# specify number of nodes
#SBATCH -N 1
# specify number of cores
#SBATCH -n 14
# specify memory pool for all cores
##SBATCH --mem 10
# specify the wall clock time limit for the job
#SBATCH -t 72:00:00
# specify the job name
#SBATCH -J NPCESRGAN
# specify the filename to be used for writing output
#SBATCH -o /home-mscluster/mmolefe/output.%N.%j.out
# specify the filename for stderr
#SBATCH -e /home-mscluster/mmolefe/output.%N.%j.err

echo ------------------------------------------------------
echo -n 'Job is running on node ' $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo SLURM: sbatch is running on $SLURM_SUBMIT_HOST
echo SLURM: job ID is $SLURM_JOB_ID
echo SLURM: submit directory is $SLURM_SUBMIT_DIR
echo SLURM: number of nodes allocated is $SLURM_JOB_NUM_NODES
echo SLURM: number of cores is $SLURM_NTASKS
echo SLURM: job name is $SLURM_JOB_NAME
echo ------------------------------------------------------

#cd $SLURM_SUBMIT_DIR
source ~/.bashrc
conda activate py39
nvidia-smi
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -y
python3 -c "import torch;print(torch.version.cuda,torch.cuda.is_available())"
python3 -c 'import torch;print("cuda" if torch.cuda.is_available() else "cpu")'
export CUDA_LAUNCH_BLOCKING=1
export MKL_DEBUG_CPU_TYPE=5
python3 esrgan.py