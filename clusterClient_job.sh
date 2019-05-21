MAINNODE="$1"
NPROCS=16

# Lisa
. /sara/sw/modules/module/init/bash

source $HOME/.bash_profile > /dev/null

# Lisa
TMPDIR="/scratch/kdevries_client"
mkdir $TMPDIR

cp -r $HOME/IUQ-Project/* "$TMPDIR"
cd $TMPDIR

python3 clusterClient.py --server_host $MAINNODE --nprocs $NPROCS
