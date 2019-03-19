#!/bin/bash
#SBATCH -N 1
#SBATCH -t 5-00:00:00

TMAX=80000
VISC=1

cp -r $HOME/IUQ-Project/* $TMPDIR
cd $TMPDIR

python sampleHemocell_TMCMC.py --enableInteriorViscosity $VISC --tmax $TMAX

if [[ $VISC == 1 ]]; then
    cp  $TMPDIR/TMCMC_hemocell_qoi_visc_3_12_tmax_$TMAX.npy $HOME/IUQ-Project
    cp  $TMPDIR/TMCMC_hemocell_samples_visc_3_12_tmax_$TMAX.csv $HOME/IUQ-Project
else
    cp  $TMPDIR/TMCMC_hemocell_qoi_normal_3_12_tmax_$TMAX.npy $HOME/IUQ-Project
    cp  $TMPDIR/TMCMC_hemocell_samples_normal_3_12_tmax_$TMAX.csv $HOME/IUQ-Project
fi
