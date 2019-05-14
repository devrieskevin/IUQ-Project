MAINNODE="$1"
NPROCS=24

# Lisa
#. /sara/sw/modules/module/init/bash

# Cartesius
#. /usr/share/Modules/init/bash

source $HOME/.bash_profile > /dev/null

cp -r $HOME/IUQ-Project/* "$TMPDIR"
cd $TMPDIR

python3 clusterClient.py --server_host $MAINNODE --nprocs $NPROCS
