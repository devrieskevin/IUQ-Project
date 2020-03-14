#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=devrieskevin@live.nl

METHOD="TMCMC"
LMAX=1
NBURN=0
NSAMPLES=1000
TREATMENT="0"
VISC=1
IMIN=2
IMAX=8
MODEL_TYPE="external"
MODEL="hemocell"
ERRTYPE="EL_error"

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

### Cluster instructions ###
OUTDIR="$(mktemp -d -p /scratch-shared hemocell_sample.XXXXX)"
echo "Output directory: $OUTDIR"

cp -r $HOME/kevin/IUQ-Project/* $OUTDIR
cd $OUTDIR
############################

python3 runHemocellSample.py --method $METHOD --enableInteriorViscosity $VISC --lmax $LMAX --nburn $NBURN --nsamples $NSAMPLES  --cellHealth $CELLHEALTH --imin $IMIN --imax $IMAX --model_type ${MODEL_TYPE} --model ${MODEL} --errType $ERRTYPE

### Cluster instructions ###
cp -r $OUTDIR/sample_output $HOME/kevin/results/${METHOD}_${MODEL}_${CELLHEALTH}_${ERRTYPE}_${TYPE}_${IMIN}_${IMAX}_lmax_${LMAX}_nburn_${NBURN}_nsamples_${NSAMPLES}_mpe_sample
############################

### Local instructions ###
#mv ./sample_output $HOME/master_project/results/${METHOD}_${MODEL}_${CELLHEALTH}_${ERRTYPE}_${TYPE}_${IMIN}_${IMAX}_lmax_${LMAX}_nburn_${NBURN}_nsamples_${NSAMPLES}_mpe_sample
##########################

echo "Ending script"
date
