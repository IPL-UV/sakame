#!/bin/bash

#SBATCH --nodes=2                                   # 2 node
#SBATCH --ntasks=2                                  # 2 tasks
#SBATCH --tasks-per-node=1                          # 1 task per node
#SBATCH --cpus-per-task=16                          # 1 core (cpu) per task
#SBATCH --exclude=nodo17                            # Don't use the GPU nodes for processing
#SBATCH --job-name=sakame_sampling                  # Job Name
#SBATCH --output="/home/emmanuel/projects/2019_sakame/src/experiments/esdc/logs/sakame_sampling.log"        # Path + Log file
#SBATCH --error="/home/emmanuel/projects/2019_sakame/src/experiments/esdc/logs/sakame_sampling.err"         # Path + Error File

# Set Python Path and Output State
export PYTHONUNBUFFERED=TRUE                                    # Ensures print statement to log files
export PYTHONPATH="/home/emmanuel/projects/2019_sakame"         # Ensures path to src file is added

module load Anaconda3
source activate sakame

# same commands
ntrain=5000
nrestarts=0
v1=gross_primary_productivity
v2=land_surface_temperature
file=/home/emmanuel/projects/2019_sakame/src/experiments/esdc/regression.py


# Execute jobs in parallel (variables)
for ispatial in 2 3 4 5 6 7 8 9 10
do 
    srun --nodes 1 --ntasks 1 \
    python -u $file --exp-name trial1 --train-size $ntrain --restarts $nrestarts --spatial $ispatial --variable $v1 &
    srun --nodes 1 --ntasks 1 \
    python -u $file --exp-name trial1 --train-size $ntrain --restarts $nrestarts --spatial $ispatial --variable $v2 
done


