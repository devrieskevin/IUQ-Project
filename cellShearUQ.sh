#!/bin/bash
#SBATCH -N 1
#SBATCH -t 48:00:00

cp -r $HOME/IUQ-Project/* $TMPDIR
cd $TMPDIR

python sampleHemocell.py

cp  $TMPDIR/hemocell_qoi.npy $HOME/IUQ-Project
cp  $TMPDIR/hemocell_samples.csv $HOME/IUQ-Project
cp -r $TMPDIR/TMCMC_output $HOME/IUQ-Project
