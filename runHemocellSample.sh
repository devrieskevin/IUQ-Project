#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevin.devries@student.uva.nl

METHOD="TMCMC"
LMAX=1
NBURN=10
NSAMPLES=10000
TREATMENT="0"
VISC=1
IMIN=2
IMAX=8
MODEL_TYPE="external"
MODEL="single_GP_isotropic"
ERRTYPE="no_EL_error"

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

#OUTDIR="$(mktemp -d -p /scratch-shared hemocell_sample.XXXXX)"
#echo "Output directory: $OUTDIR"

#cp -r $HOME/IUQ-Project/* $OUTDIR
#cd $OUTDIR

python3 runHemocellSample.py --method $METHOD --enableInteriorViscosity $VISC --lmax $LMAX --nburn $NBURN --nsamples $NSAMPLES  --cellHealth $CELLHEALTH --imin $IMIN --imax $IMAX --model_type ${MODEL_TYPE} --model ${MODEL} --errType $ERRTYPE

#cp  $OUTDIR/${METHOD}_${MODEL}_${CELLHEALTH}_${ERRTYPE}_qoi_${TYPE}_${IMIN}_${IMAX}_lmax_${LMAX}_nburn_${NBURN}_nsamples_${NSAMPLES}_mpe_sample.npy $HOME/results
#cp  $OUTDIR/${METHOD}_${MODEL}_${CELLHEALTH}_${ERRTYPE}_c_err_${TYPE}_${IMIN}_${IMAX}_lmax_${LMAX}_nburn_${NBURN}_nsamples_${NSAMPLES}_mpe_sample.npy $HOME/results

#cp -r $OUTDIR/sample_output $HOME/results/${METHOD}_${MODEL}_${CELLHEALTH}_${ERRTYPE}_${TYPE}_${IMIN}_${IMAX}_lmax_${LMAX}_nburn_${NBURN}_nsamples_${NSAMPLES}_mpe_sample

mv ./sample_output $HOME/master_project/results/${METHOD}_${MODEL}_${CELLHEALTH}_${ERRTYPE}_${TYPE}_${IMIN}_${IMAX}_lmax_${LMAX}_nburn_${NBURN}_nsamples_${NSAMPLES}_mpe_sample

echo "Ending script"
date
