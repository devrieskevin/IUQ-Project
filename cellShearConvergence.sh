#!/bin/bash
#SBATCH -N 1
#SBATCH -t 5-00:00:00

cp -r $HOME/IUQ-Project/* $TMPDIR
cd $TMPDIR

python testConvergence.py

cp  $TMPDIR/convergence_qoi.npy $HOME/IUQ-Project
cp -r $TMPDIR/Convergence_output $HOME/IUQ-Project
