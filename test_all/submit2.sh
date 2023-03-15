#!/bin/bash                                                             
#SBATCH --gres=gpu:1
#SBATCH --time=01-00:00:00
#SBATCH --mem-per-cpu=5000M
#SBATCH -p  gpu

module load gcc
module load  cuda/11.3.1

python nequip_0314.py 
  
echo "Nequip trained."






