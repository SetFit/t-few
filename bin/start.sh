conda activate baselines-tfew
export NICL_ROOT=`pwd`
export SCRIPTS_ROOT="$(dirname "$NICL_ROOT")"
export SETFIT_ROOT="$(dirname "$SCRIPTS_ROOT")"
export PYTHONPATH=$SETFIT_ROOT:"$SETFIT_ROOT/src":$NICL_ROOT/t-few:$PYTHONPATH
export OUTPUT_PATH=$NICL_ROOT/results
export CONFIG_PATH=$NICL_ROOT/t-few/configs