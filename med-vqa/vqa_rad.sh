CUDA_VISIBLE_DEVICES=0 \
python main.py \
    --dataset rad \
    --attention BAN\
    --autoencoder \
    --maml \
    --qcr\
    --tcr\
    --glimpse 2\
    --glimpse_open 2\
    --glimpse_close 1\