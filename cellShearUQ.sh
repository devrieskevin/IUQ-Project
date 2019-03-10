#!/bin/bash
#SBATCH -N 1
#SBATCH -t 5-00:00:00

cp -r $HOME/IUQ-Project/* $TMPDIR
cd $TMPDIR

python sampleHemocell_TMCMC.py

cp  $TMPDIR/hemocell_qoi_normal_3_12.npy $HOME/IUQ-Project
cp  $TMPDIR/hemocell_samples_normal_3_12.csv $HOME/IUQ-Project
