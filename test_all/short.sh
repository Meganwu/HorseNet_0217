#!/bin/bash                                                             
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=1000M
#SBATCH -p  gpushort

module load gcc
module load  cuda/11.3.1

python  nequip_comp_0303.py 
 
echo "Nequip trained."






