#!/bin/bash
## Modify walltime and account at minimum
#SBATCH --time=00:10:00
#SBATCH --account=bar

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=36

module purge
module load conda


source activate /projects/bar/mchetan/weis-p2/env/weis-viz
port=8050

echo "run the following command on your machine"
echo ""
echo "ssh -L $port:$HOSTNAME:$port $SLURM_SUBMIT_HOST.hpc.nrel.gov"

python simpleDashApp.py