### Trainig
Please use the script "train.sh" to start model training. The sample is configured for a node with 8 GPUs, numbered 0,1,2,3,4,5,6,7. 
```bash
    MODEL=dsan_t #dsan_t, dsan_s
    DROP_PATH=0.1
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash distributed_train.sh 8 /path/to/imagenet \
    	  --model $MODEL -b 64 --lr 1e-3 --drop-path $DROP_PATH --workers=16
```

### Eval

If you want to evaluate our models, please use the script "eval.sh". If you don't know how to do it, please check the user manual or documents of [mmcv-documents](https://mmcv.readthedocs.io/en/v1.6.0/).