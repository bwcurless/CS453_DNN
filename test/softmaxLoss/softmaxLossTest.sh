#!/bin/bash
#SBATCH --job-name=SoftmaxLossTest  #the name of your job

#change to your NAU ID below
#SBATCH --output=/scratch/bc2497/SoftmaxLossTest.out #this is the file for stdout
#SBATCH --error=/scratch/bc2497/SoftmaxLossTest.err #this is the file for stderr

#SBATCH --time=00:10:00		#Job timelimit is 3 minutes
#SBATCH --mem=1000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C a100 #GPU Model: k80, p100, v100, a100
#SBATCH --account=cs453-spr24

module load cuda
nvcc -O3 -arch=compute_80 -code=sm_80 -lcuda -Xcompiler -fopenmp -lineinfo -Xcompiler -rdynamic softmaxLossTests.cu ../../src/softmaxLoss.cu -link -o softmaxLossTests

#srun ./softmaxLossTests
compute-sanitizer --tool=memcheck ./softmaxLossTests
