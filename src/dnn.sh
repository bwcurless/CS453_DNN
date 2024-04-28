#!/bin/bash
# Little script to compile .cu files and run sbatch
# Pass in .cu file to build
source ../slurm_funcs.sh

# Exit if any part fails
set -e

# Input Arguments
target=$1
jobName=$2

# Add a dash on if we are customizing the filename
if [[ -n $jobName ]]; then
	jobPrefix=$jobName-
fi

outputFile="DNN"

# Determine what target to build and run on
case $target in
	a100)
		echo "Running on a100"
		CC=80
		cudaModule=12.3.2
		gpu=a100
		;;

	v100)
		echo "Running on v100"
		CC=70
		cudaModule=12.3.2
		gpu=v100
		;;

	p100)
		echo "Running on p100"
		CC=60
		cudaModule=12.3.2
		gpu=p100
		;;

	k80 | *)
		echo "Default mode k80"
		CC=37
		cudaModule=11.7 # k80's only run on this
		gpu=k80
		;;
esac

# Do a test build locally to make sure there aren't errors before waiting in queue
echo "Building executable to $outputFile"
module load "cuda/$cudaModule"
make release

# Define where outputs go
outputPath="/scratch/$USER/"
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
#SBATCH -C $gpu #GPU Model: k80, p100, v100, a100
#SBATCH --account=cs453-spr24

# Code will not compile if we don't load the module
module load "cuda/$cudaModule"

# Can do arithmetic interpolation inside of $(( )). Need to escape properly
make release

srun "./release/$outputFile" ~/Documents/cifar-10-batches-bin/data_batch_1.bin
#compute-sanitizer --tool=memcheck "./$outputFile"
# -f overwrite profile if it exists
#srun ncu -f -o "affine_profile" --clock-control=none --set full "./$outputFile"

echo "----------------- JOB FINISHED -------------"

SHELL
)


waitForJobComplete "$jobid"
printFile "$outputPath$jobPrefix$outputFile-$jobid.out"
#scrollOutput "$outputPath$jobPrefix$outputFile-$jobid.out"
