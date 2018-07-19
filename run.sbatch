#!/bin/bash

#SBATCH --ntasks=100
#SBATCH --time=10:00:00
#SBATCH --qos=normal
#SBATCH --partition=shas
#SBATCH --job-name=aug_data_sampler2
#SBATCH --output=std_out2.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=pesh5067@colorado.edu

module load python
srun -n $SLURM_NTASKS --mpi=pmi2 python main.py
