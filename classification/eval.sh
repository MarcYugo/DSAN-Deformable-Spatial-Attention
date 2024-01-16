MODEL=dsan_t # dsan_t,dsan_s
python3 validate.py /path/to/imagenet --model $MODEL \
  --checkpoint /path/to/model -b 64