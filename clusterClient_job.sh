MAINNODE="$1"
OUTDIR="$2"
NPROCS=24

# Lisa
#. /sara/sw/modules/module/init/bash

source $HOME/.bash_profile &> /dev/null

cd $OUTDIR

python3 -u clusterClient.py --server_host $MAINNODE --nprocs $NPROCS
