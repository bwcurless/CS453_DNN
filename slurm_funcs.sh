#!/bin/bash
# Waits for slurm job to complete

waitForJobRunning () {
	waitForJobState "$1" RUNNING
}

waitForJobComplete () {
	waitForJobState "$1" COMPLETE
}

waitForJobState () {
	jobid=$1
	state=$2

	while true;
	do
		status=$(scontrol show job "$jobid" | grep "JobState=")
		#echo "$status"
		if [[ $status =~ JobState=$state ]]; then
			break;
		else
			sleep 1
		fi
	done
}

#prints output with srun so we get most recent version
printFile () {
	file=$1
	# srun cat to make sure we get the most recent file, should print output to terminal
	printf '%.s─' $(seq 1 $(tput cols))
	srun cat "$file"
	printf '%.s─' $(seq 1 $(tput cols))
}

waitFileExist () {
	file=$1
	until [ -f "$file" ]
	do
		echo "Waiting for file $file to exist"
		sleep 1
	done
}

# Prints only the output file, and auto updates
scrollOutput () {
	outputFile="$1"
	waitFileExist "$outputFile"
	printf '%.s─' $(seq 1 $(tput cols))
	# if we don't srun here, it seems like the file gets locked and output doesn't get written
	srun tail -f "$outputFile"
}
