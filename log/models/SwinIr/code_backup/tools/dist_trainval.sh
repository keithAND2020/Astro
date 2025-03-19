set -x
CONFIGS=${@:1}

GPUS=${CUDA_VISIBLE_DEVICES//,/}
GPUS=${#GPUS}
PORT=${PORT:-$RANDOM}
torchrun --standalone --nproc_per_node=${GPUS} --master_port=$PORT tools/trainval.py --launcher pytorch ${CONFIGS} 
