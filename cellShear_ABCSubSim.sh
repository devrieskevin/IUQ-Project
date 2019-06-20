#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-00:05:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevin.devries@student.uva.nl

N=500
VISC=1
IMIN=2
IMAX=6
NPROCS=16
MODEL_TYPE="external_cluster"

MAINNODE="$(hostname -i)"

echo "Starting script"
date

cp -r $HOME/IUQ-Project/* $TMPDIR
cd $TMPDIR

python3 sampleHemocell_ABCSubSim.py --enableInteriorViscosity $VISC --n_samples $N --imin $IMIN --imax $IMAX --nprocs $NPROCS --model_type ${MODEL_TYPE} &

sleep 15

# Lisa
node_list="$(python nodelistToTuple.py ${SLURM_JOB_NODELIST})"

# Cartesius
#node_list="$(nodeset -e ${SLURM_JOB_NODELIST})"

#Loop over the bash array to execute the program on each node in node_list.
for node in ${node_list}; do
    if [[ $node != $SLURMD_NODENAME ]]; then
        echo "ssh to node $node"
        ssh $node "$HOME/IUQ-Project/clusterClient_job.sh $MAINNODE" &
    else
        echo "Job running on $node"
    fi
done

wait

if [[ $VISC == 1 ]]; then
    cp  $TMPDIR/ABCSubSim_hemocell_qoi_visc_${IMIN}_${IMAX}.npy $HOME/results
    cp  $TMPDIR/ABCSubSim_hemocell_c_err_visc_${IMIN}_${IMAX}.npy $HOME/results
    cp  $TMPDIR/ABCSubSim_hemocell_samples_visc_${IMIN}_${IMAX}.csv $HOME/results
else
    cp  $TMPDIR/ABCSubSim_hemocell_qoi_normal_${IMIN}_${IMAX}.npy $HOME/results
    cp  $TMPDIR/ABCSubSim_hemocell_c_err_normal_${IMIN}_${IMAX}.npy $HOME/results
    cp  $TMPDIR/ABCSubSim_hemocell_samples_normal_${IMIN}_${IMAX}.csv $HOME/results
fi

echo "Ending script"
date
