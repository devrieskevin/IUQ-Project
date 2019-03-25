#!/bin/bash
#SBATCH -N 1
#SBATCH -t 5-00:00:00

TMAX=80000
VISC=0
IMIN=8
IMAX=12

cp -r $HOME/IUQ-Project/* $TMPDIR
cd $TMPDIR

python sampleHemocell_ABCSubSim.py --enableInteriorViscosity $VISC --tmax $TMAX --imin $IMIN --imax $IMAX

if [[ $VISC == 1 ]]; then
    cp  $TMPDIR/ABCSubSim_hemocell_qoi_visc_${IMIN}_${IMAX}_tmax_${TMAX}.npy $HOME/results
    cp  $TMPDIR/ABCSubSim_hemocell_samples_visc_${IMIN}_${IMAX}_tmax_${TMAX}.csv $HOME/results
else
    cp  $TMPDIR/ABCSubSim_hemocell_qoi_normal_${IMIN}_${IMAX}_tmax_${TMAX}.npy $HOME/results
    cp  $TMPDIR/ABCSubSim_hemocell_samples_normal_${IMIN}_${IMAX}_tmax_${TMAX}.csv $HOME/results
fi
