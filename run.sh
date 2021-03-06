#!/bin/bash

#SBATCH --job-name=term-prj
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:10:00 
#SBATCH --mem=64000MB 
#SBATCH --cpus-per-task=32
#SBATCH --partition=shpc
#SBATCH --output=log.txt

salloc --nodes=2 --ntasks-per-node=1 --cpus-per-task=64 --partition=shpc --gres=gpu:4 \
mpirun ./facegen_parallel network.bin input3.txt output3.txt output3.bmp

./compare_result output3.txt answer3.txt