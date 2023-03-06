#!/bin/bash
python -m geoclip.geoclip --data_path=/home/a.dhakal/active/datasets/YFCC100m/webdataset \
    --train_batch_size=512 \
    --val_batch_size=512 \
    --wandb_mode=online \
    --ckpt_path=none \
    --max_epochs=100 \
    --accelerator=gpu \
    --devices=1 \
    --num_workers=5 \
    --project_name=GeoClip \
    --mode=train \
    --train_epoch_length=2000 \
    --val_epoch_length=25 \
    --val_check_interval=100 \
    --project_name=GeoClip \
    --run_name=geoclip_13_224size_fixlr \
    --vit='32' \
    --ckpt_path='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/y8yd24yj/checkpoints/epoch=0-step=1100-top_k_score=0.736.ckpt' \
    --ckpt_mode='soft' \
    --learning_rate=0.00004

    #/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/6zbm4f59/checkpoints/step=4500-top_k_score=0.875.ckpt
    # --val_check_interval=5 \

#Important notes
# For 32-patch-base: resize=480, crop=448, batch_size=128
#For 14L:resize=420, crop=392 ### did not work. Cuda OOM 
#For 16-base:resize=336 crop=320 batchsize=64