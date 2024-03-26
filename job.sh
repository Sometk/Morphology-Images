#!/bin/bash
#SBATCH --job-name=julia-job
#SBATCH --account=mjlab
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --partition=partition_name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=10:00:00
#SBATCH --mem=500gb
#SBATCH --constraint=mem768

# Load the Julia module
module load julia

# Run your Julia scripts
julia process_images.jl
