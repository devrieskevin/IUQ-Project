#!/bin/bash
#SBATCH -N 1
#SBATCH -t 48:00:00

python sampleHemocell.py
mkdir /home/kdevries/IUQ-Project/results
mv ./* /home/kdevries/IUQ-Project/results
