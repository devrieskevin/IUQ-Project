#!/bin/bash
#SBATCH -N 1
#SBATCH -t 00:05:00

cp -r $HOME/IUQ-Project/* $TMPDIR
cd $TMPDIR

python scheduler.py
