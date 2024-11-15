### Trainig
Please use the script "dist_train.sh" to start model training. The sample is configured for a node with 8 GPUs for training the combination of hamburger and dsan-t on ADE20K dataset. 
```bash
CONFIG=configs/ham/ham_dsan_t_ade20k_160k.py
GPUS=8
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3} \
```

### Eval

If you want to evaluate our models, please use the script "dist_test.sh". If you don't know how to do it, please check the user manual or documents of [mmsegmentation-documents](https://mmsegmentation.readthedocs.io/en/latest/).
