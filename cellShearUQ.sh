#!/bin/bash
#SBATCH -N 1
#SBATCH -t 06:00:00

#source /home/czajab/hemocell/scripts/lisa_env.sh
python sampleHemocell.py
