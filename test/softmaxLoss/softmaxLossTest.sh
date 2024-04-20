#!/bin/bash
# Little script to compile .cu files and run sbatch
# Pass in .cu file to build
source ../../slurm_funcs.sh

# Exit if any part fails
set -e

jobName=$1
outputFile="softmaxLossTest"

# Add a dash on if we are customizing the filename
if [[ -n $jobName ]]; then
	jobPrefix=$jobName-
fi

# Do a test build locally to make sure there aren't errors before waiting in queue
echo "Building executable to $outputFile"
module load "cuda/12.3.2"
nvcc -O3 -arch=compute_80 -code=sm_80 -lcuda -Xcompiler -fopenmp -lineinfo -Xcompiler -rdynamic softmaxLossTests.cu ../../src/softmaxLoss.cu -link -o "$outputFile"

# Define where outputs go
outputPath="/scratch/bc2497/"
errorPath="$outputPath"

echo "Executing..."

jobid=$(sbatch --parsable <<SHELL
#!/bin/bash
#SBATCH --job-name=$jobPrefix$outputFile  #the name of your job

#change to your NAU ID below
#SBATCH --output=$outputPath$jobPrefix$outputFile-%j.out
#SBATCH --error=$errorPath$jobPrefix$outputFile-%j.out

#SBATCH --time=03:00:00
#SBATCH --mem=10000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C a100 #GPU Model: k80, p100, v100, a100
#SBATCH --account=cs453-spr24

# Code will not compile if we don't load the module
module load "cuda/$cudaModule"

# Can do arithmetic interpolation inside of $(( )). Need to escape properly
nvcc -O3 -arch=compute_80 -code=sm_80 -lcuda -lineinfo -Xcompiler -fopenmp -Xcompiler -rdynamic softmaxLossTests.cu ../../src/softmaxLoss.cu -link -o "$outputFile"

srun "./$outputFile"
#compute-sanitizer --tool=memcheck "./$outputFile"
# -f overwrite profile if it exists
#srun ncu -f -o "Softmax_Profile" --clock-control=none --set full "./$outputFile"

echo "----------------- JOB FINISHED -------------"

SHELL
)


waitForJobComplete "$jobid"
printFile "$outputPath$jobPrefix$outputFile-$jobid.out"
#scrollOutput "$outputPath$jobPrefix$outputFile-$jobid.out"
