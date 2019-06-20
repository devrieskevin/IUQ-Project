#!/bin/bash
#SBATCH -N 30
#SBATCH -t 5-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevin.devries@student.uva.nl

N=1000
LMAX=1
TREATMENT="0"
VISC=1
IMIN=2
IMAX=8
NPROCS=24
MODEL_TYPE="external_cluster"

if [[ $TREATMENT > 0 ]]; then
    CELLHEALTH="treated"
else
    CELLHEALTH="healthy"
fi

if [[ $VISC == 1 ]]; then
    TYPE="visc"
else
    TYPE="normal"
fi

echo "Cell type: $CELLHEALTH"
echo "Model type: $TYPE"

MAINNODE="$(hostname -i)"

echo "Starting script"
date

OUTDIR="$(mktemp -d -p /scratch-shared hemocell_TMCMC.XXXXX)"

echo "Output directory: $OUTDIR"

cp -r $HOME/IUQ-Project/* $OUTDIR
cd $OUTDIR

python3 sampleHemocell_TMCMC.py --enableInteriorViscosity $VISC --n_samples $N --lmax $LMAX --treatment $TREATMENT --imin $IMIN --imax $IMAX --nprocs $NPROCS --model_type ${MODEL_TYPE} &

sleep 10

# Lisa
#node_list="$(python nodelistToTuple.py ${SLURM_JOB_NODELIST})"

# Cartesius
node_list="$(nodeset -e ${SLURM_JOB_NODELIST})"

# Loop over the bash array to execute the program on each node in node_list.
for node in ${node_list}; do
    if [[ $node != $SLURMD_NODENAME ]]; then
        echo "ssh to node $node"
        ssh $node "$HOME/IUQ-Project/clusterClient_job.sh $MAINNODE $OUTDIR" &
    else
        echo "Job running on $node"
    fi
done

wait

cp  $OUTDIR/TMCMC_hemocell_${CELLHEALTH}_qoi_${TYPE}_${IMIN}_${IMAX}_lmax_${LMAX}.npy $HOME/results
cp  $OUTDIR/TMCMC_hemocell_${CELLHEALTH}_c_err_${TYPE}_${IMIN}_${IMAX}_lmax_${LMAX}.npy $HOME/results
cp  $OUTDIR/TMCMC_hemocell_${CELLHEALTH}_samples_${TYPE}_${IMIN}_${IMAX}_lmax_${LMAX}.csv $HOME/results

echo "Ending script"
date
