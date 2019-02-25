#!/bin/bash
#SBATCH -N 1
#SBATCH -t 48:00:00

cp -r $HOME/IUQ-Project/* $TMPDIR
cd $TMPDIR

echo "Hello one" &
echo "Hello two" &
wait

python sampleHemocell.py

cp  $TMPDIR/hemocell_qoi.npy $HOME/IUQ-Project
cp  $TMPDIR/hemocell_samples.csv $HOME/IUQ-Project
