#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevin.devries@student.uva.nl

METHOD="TMCMC"
LMAX=1
TREATMENT="0"
VISC=1
IMIN=2
IMAX=8
MODEL_TYPE="external"
MODEL="GP"

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

OUTDIR="$(mktemp -d -p /scratch-shared hemocell_sample.XXXXX)"
echo "Output directory: $OUTDIR"

cp -r $HOME/IUQ-Project/* $OUTDIR
cd $OUTDIR

python3 runHemocellSample.py --method $METHOD --enableInteriorViscosity $VISC --lmax $LMAX --cellHealth $CELLHEALTH --imin $IMIN --imax $IMAX --model_type ${MODEL_TYPE} --model ${MODEL}

cp  $OUTDIR/${METHOD}_${MODEL}_${CELLHEALTH}_qoi_${TYPE}_${IMIN}_${IMAX}_lmax_${LMAX}_mpe_sample.npy $HOME/results
cp  $OUTDIR/${METHOD}_${MODEL}_${CELLHEALTH}_c_err_${TYPE}_${IMIN}_${IMAX}_lmax_${LMAX}_mpe_sample.npy $HOME/results

cp -r $OUTDIR/sample_output $HOME/results/${METHOD}_${MODEL}_${CELLHEALTH}_${TYPE}_${IMIN}_${IMAX}_lmax_${LMAX}_mpe_sample

echo "Ending script"
date
