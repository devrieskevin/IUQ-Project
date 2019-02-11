#!/bin/bash
#SBATCH -N 1
#SBATCH -t 06:00:00
source /home/czajab/hemocell/scripts/lisa_env.sh
cd /home/czajab/hemocell/cases/oneCellShear/SR1/KL100/IV1/
../../../oneCellShear config.xml &
cd /home/czajab/hemocell/cases/oneCellShear/SR1/KL100/IV2/
../../../oneCellShear config.xml &
cd /home/czajab/hemocell/cases/oneCellShear/SR1/KL100/IV4/
../../../oneCellShear config.xml &
cd /home/czajab/hemocell/cases/oneCellShear/SR1/KL100/IV6/
../../../oneCellShear config.xml &
cd /home/czajab/hemocell/cases/oneCellShear/SR1/KL100/IV8/
../../../oneCellShear config.xml &
cd /home/czajab/hemocell/cases/oneCellShear/SR1/KL100/IV10/
../../../oneCellShear config.xml &
cd /home/czajab/hemocell/cases/oneCellShear/SR1/KL100/IV12/
../../../oneCellShear config.xml &
cd /home/czajab/hemocell/cases/oneCellShear/SR1/KL100/IV14/
../../../oneCellShear config.xml &
cd /home/czajab/hemocell/cases/oneCellShear/SR1/KL100/IV16/
../../../oneCellShear config.xml &
cd /home/czajab/hemocell/cases/oneCellShear/SR1/KL100/IV18/
../../../oneCellShear config.xml &
cd /home/czajab/hemocell/cases/oneCellShear/SR1/KL100/IV20/
../../../oneCellShear config.xml &
cd /home/czajab/hemocell/cases/oneCellShear/SR1/KL120/IV1/
../../../oneCellShear config.xml &
cd /home/czajab/hemocell/cases/oneCellShear/SR1/KL120/IV2/
../../../oneCellShear config.xml &
cd /home/czajab/hemocell/cases/oneCellShear/SR1/KL120/IV4/
../../../oneCellShear config.xml &
cd /home/czajab/hemocell/cases/oneCellShear/SR1/KL120/IV6/
../../../oneCellShear config.xml &
cd /home/czajab/hemocell/cases/oneCellShear/SR1/KL120/IV8/
../../../oneCellShear config.xml &
wait