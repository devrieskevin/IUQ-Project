#!/bin/bash
#SBATCH -N 1
#SBATCH -t 5-00:00:00

TMAX=80000
VISC=1
IMIN=8
IMAX=12

cp -r $HOME/IUQ-Project/* $TMPDIR
cd $TMPDIR

python sampleHemocell_TMCMC.py --enableInteriorViscosity $VISC --tmax $TMAX --imin $IMIN --imax $IMAX

if [[ $VISC == 1 ]]; then
    cp  $TMPDIR/TMCMC_hemocell_qoi_visc_${IMIN}_${IMAX}_tmax_${TMAX}.npy $HOME/results
    cp  $TMPDIR/TMCMC_hemocell_samples_visc_${IMIN}_${IMAX}_tmax_${TMAX}.csv $HOME/results
else
    cp  $TMPDIR/TMCMC_hemocell_qoi_normal_${IMIN}_${IMAX}_tmax_${TMAX}.npy $HOME/results
    cp  $TMPDIR/TMCMC_hemocell_samples_normal_${IMIN}_${IMAX}_tmax_${TMAX}.csv $HOME/results
fi
