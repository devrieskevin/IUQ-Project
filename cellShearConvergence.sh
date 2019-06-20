#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevin.devries@student.uva.nl

TMAX=1000000
TMEAS=500
VISC=1
IMIN=0
IMAX=10
KLINK=100
KBEND=80
VISCOSITYRATIO=5

if [[ $VISC == 1 ]]; then
    TYPE="visc"
else
    TYPE="normal"
fi

echo "Starting script"
date

OUTDIR="$(mktemp -d -p /scratch-shared hemocell_convergence.XXXXX)"
echo "Output directory: $OUTDIR"

cp -r $HOME/IUQ-Project/* $OUTDIR
cd $OUTDIR

python3 testConvergence.py --tmax $TMAX --tmeas $TMEAS --enableInteriorViscosity $VISC --imin $IMIN --imax $IMAX --kLink $KLINK --kBend $KBEND --viscosityRatio $VISCOSITYRATIO

cp  $OUTDIR/convergence_qoi_${TYPE}_${IMIN}_${IMAX}_${KLINK}_${KBEND}_${VISCOSITYRATIO}.npy $HOME/results

echo "Ending script"
date
