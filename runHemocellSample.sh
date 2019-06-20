#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevin.devries@student.uva.nl

METHOD="TMCMC"
LMAX=1
TREATMENT="0.5"
VISC=1
IMIN=4
IMAX=10
MODEL_TYPE="external"

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

python3 runHemocellSample.py --method $METHOD --enableInteriorViscosity $VISC --lmax $LMAX --cellHealth $CELLHEALTH --imin $IMIN --imax $IMAX --model_type ${MODEL_TYPE}

echo "Ending script"
date
