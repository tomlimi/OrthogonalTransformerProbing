#!/bin/bash
CUDA_version=10.1
CUDNN_version=7.6
CUDA_DIR_OPT=/opt/cuda/$CUDA_version
if [ -d "$CUDA_DIR_OPT" ] ; then
  CUDA_DIR=$CUDA_DIR_OPT
  export CUDA_HOME=$CUDA_DIR
  export THEANO_FLAGS="cuda.root=$CUDA_HOME,device=gpu,floatX=float32"
  export PATH=$PATH:$CUDA_DIR/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_DIR/cudnn/$CUDNN_version/lib64:$CUDA_DIR/lib64
  export CPATH=$CUDA_DIR/cudnn/$CUDNN_version/include:$CPATH
fi


source /ha/home/limisiewicz/.virtualenvs/MultilingualTransformerProbing/bin/activate
cd /lnet/spec/work/people/limisiewicz/MultilingualTransformerProbing/src/ || exit

#FIRST_L=${1:-0}
D="../exp/experiments_average_layers/subsampled_trainset"
T="../resources/tf_data"
LANG=$1
LANG="en es sl id zh" 
python3 probe.py ${D} ${T} --languages $LANG --model "bert-base-multilingual-cased"  --layer-index -1 --tasks "dep_depth" "dep_distance" --clip-norm 1.5 --learning-rate 0.02 --batch-size 12 --ortho 0.05 --subsample-train 4000 

