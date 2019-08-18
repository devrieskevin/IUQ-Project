#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevin.devries@student.uva.nl

echo "Starting script"
date

OUTDIR="$(mktemp -d -p /scratch-shared hemocell_sample.XXXXX)"
echo "Output directory: $OUTDIR"

cp -r $HOME/IUQ-Project/* $OUTDIR
cd $OUTDIR

for i in {0..9};do
    time (python3 timeHemocellRuns.py --index $i) &>shear_stress_output_$i.dat  &
done

wait

for i in {0..9};do
    echo "Shear stress #$i:"
    cat shear_stress_output_$i.dat
done

echo "Ending script"
date
