#!/bin/bash

N=1000
LMAX=1
NBURN=0
TREATMENT="0"
VISC=1
IMIN=2
IMAX=8
NPROCS=2
MODEL_TYPE="external_cluster"

NUMCLIENTS=2
OUTDIR="/home/kevin/master_project/IUQ-Project"

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

echo "Starting script"
date

python3 sampleHemocell_TMCMC.py --enableInteriorViscosity $VISC --n_samples $N --lmax $LMAX --nburn $NBURN --treatment $TREATMENT --imin $IMIN --imax $IMAX --nprocs $NPROCS --model_type ${MODEL_TYPE} &

sleep 10

# Loop over the bash array to execute the program on each node in node_list.
for ((i = 1; i <= ${NUMCLIENTS}; i++))
do
    echo "Starting client: $i"
    python3 clusterClient.py --server_host "localhost" --nprocs $NPROCS &
done

wait

cp  $OUTDIR/TMCMC_hemocell_${CELLHEALTH}_qoi_${TYPE}_${IMIN}_${IMAX}_lmax_${LMAX}_nburn_${NBURN}.npy $HOME/results
cp  $OUTDIR/TMCMC_hemocell_${CELLHEALTH}_c_err_${TYPE}_${IMIN}_${IMAX}_lmax_${LMAX}_nburn_${NBURN}.npy $HOME/results
cp  $OUTDIR/TMCMC_hemocell_${CELLHEALTH}_samples_${TYPE}_${IMIN}_${IMAX}_lmax_${LMAX}_nburn_${NBURN}.csv $HOME/results

echo "Ending script"
date
