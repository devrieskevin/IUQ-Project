MAINNODE="$1"
NPROCS=16

. /sara/sw/modules/module/init/bash
source $HOME/.bash_profile > /dev/null

mkdir /scratch/output_dir

cp -r $HOME/IUQ-Project/* /scratch/output_dir
cd /scratch/output_dir

python clusterClient.py --server_host $MAINNODE --nprocs $NPROCS

rm -r /scratch/output_dir
