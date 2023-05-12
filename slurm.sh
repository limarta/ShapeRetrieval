#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --partition=sched_any
#SBATCH --output=$HOME/docs/ShapeRetrieval/slurm.log
#SBATCH --time=1:00:00

cd $HOME/docs/ShapeRetrieval
port=1234

echo "Hi"
#ssh -N -f -R $port:localhost:$port log-0
#ssh -N -f -R $port:localhost:$port log-1

#julia -e 'using Pluto; Pluto.run(port=1234)'
