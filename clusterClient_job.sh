MAINNODE="$1"
NPROCS=24

# Lisa
#. /sara/sw/modules/module/init/bash

# Cartesius
#. /usr/share/Modules/init/bash

source $HOME/.bash_profile > /dev/null

rm -r /scratch/output_dir
mkdir /scratch/output_dir

cp -r $HOME/IUQ-Project/* /scratch/output_dir
cd /scratch/output_dir

python clusterClient.py --server_host $MAINNODE --nprocs $NPROCS

rm -r /scratch/output_dir
