#!/bin/bash
#SBATCH -N 30
#SBATCH -t 5-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevin.devries@student.uva.nl

N=1000
VISC=1
IMIN=2
IMAX=10
NPROCS=24
MODEL_TYPE="external_cluster"

if [[ $VISC == 1 ]]; then
    TYPE="visc"
else
    TYPE="normal"
fi

echo "Model type: $TYPE"

MAINNODE="$(hostname -i)"

echo "Starting script"
date

OUTDIR="$(mktemp -d -p /scratch-shared hemocell_sobol.XXXXX)"

echo "Output directory: $OUTDIR"

cp -r $HOME/IUQ-Project/* $OUTDIR
cd $OUTDIR

python3 sobol_hemocell.py --enableInteriorViscosity $VISC --n_samples $N --imin $IMIN --imax $IMAX --nprocs $NPROCS --model_type ${MODEL_TYPE} &

sleep 10

# Lisa
#node_list="$(python nodelistToTuple.py ${SLURM_JOB_NODELIST})"

# Cartesius
node_list="$(nodeset -e $SLURM_JOB_NODELIST)"

#Loop over the bash array to execute the program on each node in node_list.
for node in ${node_list}; do
    if [[ $node != $SLURMD_NODENAME ]]; then
        echo "ssh to node $node"
        ssh $node "$HOME/IUQ-Project/clusterClient_job.sh $MAINNODE $OUTDIR" &
    else
        echo "Job running on $node"
    fi
done

wait

cp  $OUTDIR/sobol_hemocell_qoi_${TYPE}_${IMIN}_${IMAX}.npy $HOME/results
cp  $OUTDIR/sobol_hemocell_c_err_${TYPE}_${IMIN}_${IMAX}.npy $HOME/results

echo "Ending script"
date
