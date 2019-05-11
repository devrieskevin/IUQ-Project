#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-12:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=kevin.devries@student.uva.nl

TMAX=1000000
TMEAS=500
VISC=1
IMIN=0
IMAX=10

cp -r $HOME/IUQ-Project/* $TMPDIR
cd $TMPDIR

echo "Starting script"
date

python testConvergence.py --tmax $TMAX --tmeas $TMEAS --enableInteriorViscosity $VISC --imin $IMIN --imax $IMAX

if [[ $VISC == 1 ]]; then
    mv Convergence_output/ $HOME/results/conv_output_visc
    cp  $TMPDIR/convergence_qoi_visc_${IMIN}_${IMAX}.npy $HOME/results
else
    mv Convergence_output/ $HOME/results/conv_output_normal
    cp  $TMPDIR/convergence_qoi_normal_${IMIN}_${IMAX}.npy $HOME/results
fi

echo "Ending script"
date
