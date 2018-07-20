#!/bin/bash

#SBATCH --ntasks=100
#SBATCH --time=23:00:00
#SBATCH --qos=normal
#SBATCH --partition=shas
#SBATCH --job-name=aug_data_sampler
#SBATCH --output=std_out.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=pesh5067@colorado.edu

module load python
srun -n $SLURM_NTASKS --mpi=pmi2 python main.py
