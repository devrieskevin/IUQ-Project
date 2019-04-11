#!/bin/bash
#SBATCH -N 1
#SBATCH -t 5-00:00:00

N=500
TMAX=40000
VISC=1
IMIN=3
IMAX=10
NPROCS=16
MODEL_TYPE="external"

echo "Starting script"
date

cp -r $HOME/IUQ-Project/* $TMPDIR
cd $TMPDIR

python sampleHemocell_TMCMC.py --enableInteriorViscosity $VISC --n_samples $N --tmax $TMAX --imin $IMIN --imax $IMAX --nprocs $NPROCS --model_type ${MODEL_TYPE}

if [[ $VISC == 1 ]]; then
    cp  $TMPDIR/TMCMC_hemocell_qoi_visc_${IMIN}_${IMAX}_tmax_${TMAX}.npy $HOME/results
    cp  $TMPDIR/TMCMC_hemocell_samples_visc_${IMIN}_${IMAX}_tmax_${TMAX}.csv $HOME/results
else
    cp  $TMPDIR/TMCMC_hemocell_qoi_normal_${IMIN}_${IMAX}_tmax_${TMAX}.npy $HOME/results
    cp  $TMPDIR/TMCMC_hemocell_samples_normal_${IMIN}_${IMAX}_tmax_${TMAX}.csv $HOME/results
fi

echo "Ending script"
date
