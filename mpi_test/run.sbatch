#!/bin/bash

#SBATCH --ntasks=10
#SBATCH --time=00:07:00
#SBATCH --qos=debug
#SBATCH --partition=shas
#SBATCH --job-name=test
#SBATCH --output=test.out

module load python
srun -n $SLURM_NTASKS --mpi=pmi2 python main.py

