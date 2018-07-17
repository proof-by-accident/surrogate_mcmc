#!/bin/bash

#SBATCH --ntasks=5
#SBATCH --time=00:07:00
#SBATCH --qos=normal
#SBATCH --partition=shas
#SBATCH --job-name=aug_data_sampler
#SBATCH --output=output.out

module load python
srun -n $SLURM_NTASKS --mpi=pmi2 python main.py

