#!/bin/bash
python -m geoclip.fit --data_path=/home/a.dhakal/active/datasets/YFCC100m/webdataset \
    --train_batch_size=150 \
    --val_batch_size=150 \
    --wandb_mode=online \
    --wandb_resume=false \
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
    --run_name=geomoco_32_4480_v2 \
    --vit='32' \
    --ckpt_path='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/0j3bfqje/checkpoints/step=8900-val_loss=8.029.ckpt' \
    --ckpt_mode='soft' \
    --learning_rate=0.0001 \
    --strategy='ddp' \
    --queue_size=4500 \
    --dim_size=512 \
    --warmup_its=9000 \
    --moco


    #/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/6zbm4f59/checkpoints/step=4500-top_k_score=0.875.ckpt
    # --val_check_interval=5 \

#Important notes
# For 32-patch-base: resize=480, crop=448, batch_size=128
#For 14L:resize=420, crop=392 ### did not work. Cuda OOM 
#For 16-base:resize=336 crop=320 batchsize=64