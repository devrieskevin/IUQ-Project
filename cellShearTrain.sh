#!/bin/bash
#SBATCH -N 2
#SBATCH -t 0-00:05:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=devrieskevin@live.nl

N=100
VISC=1
IMIN=0
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

OUTDIR="$(mktemp -d -p /scratch-shared hemocell_train.XXXXX)"

echo "Output directory: $OUTDIR"

cp -r $HOME/kevin/IUQ-Project/* $OUTDIR
cd $OUTDIR

python3 sampleHemocell_train.py --enableInteriorViscosity $VISC --n_samples $N --imin $IMIN --imax $IMAX --nprocs $NPROCS --model_type ${MODEL_TYPE} &

sleep 3

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

#cp  $OUTDIR/train_hemocell_samples_${TYPE}_${IMIN}_${IMAX}_nsamples_$N.npy $HOME/kevin/results
#cp  $OUTDIR/train_hemocell_qoi_${TYPE}_${IMIN}_${IMAX}_nsamples_$N.npy $HOME/kevin/results
#cp  $OUTDIR/train_hemocell_c_err_${TYPE}_${IMIN}_${IMAX}_nsamples_$N.npy $HOME/kevin/results

echo "Ending script"
date
