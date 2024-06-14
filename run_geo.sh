#!/bin/bash
export TORCH_DISTRIBUTED_DEBUG=INFO
echo $CUDA_VISIBLE_DEVICES



###fresh#####
python -m geoclip.fit --data_path=/path/to/webdataset \
    --train_batch_size=100 \
    --val_batch_size=100 \
    --wandb_mode=online \
    --wandb_resume=none \
    --max_epochs=10000 \
    --max_steps=1000000 \
    --accelerator=gpu \
    --devices=1 \
    --num_workers=8 \
    --project_name=GeoClip \
    --mode=train \
    --train_epoch_length=2000 \
    --val_epoch_length=50 \
    --val_check_interval=350 \
    --project_name=Sat2Cap \
    --run_name='test_run' \
    --vit='32' \
    --ckpt_path='none' \
    --ckpt_mode='soft' \
    --learning_rate=0.00005 \
    --temperature=0.07 \
    --top_k=10 \
    --strategy='ddp_find_unused_parameters_false' \
    --queue_size=9000 \
    --dim_size=512 \
    --warmup_its=3000 \
    --precision='medium' \
    --moco \
    --geo_encode \
    --dropout_rate=0.3 \




##retrace####
# python -m geoclip.fit --data_path=/home/a.dhakal/active/datasets/YFCC100m/webdataset \
#     --train_batch_size=100 \
#     --val_batch_size=100 \
#     --wandb_mode=online \
#     --wandb_resume=false \
#     --max_epochs=100 \
#     --accelerator=gpu \
#     --devices=1 \
#     --num_workers=4 \
#     --project_name=GeoClip \
#     --mode=train \
#     --train_epoch_length=2000 \
#     --val_epoch_length=300 \
#     --val_check_interval=250 \
#     --project_name=GeoClip \
#     --wandb_resume='none' \
#     --run_name='geomoco_retrace' \
#     --vit='32' \
#     --ckpt_path='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/temp_models/s212e5he/checkpoints/epoch=3-step=29500-top_k_score=0.920.ckpt' \
#     --ckpt_mode='hard' \
#     --learning_rate=0.00005 \
#     --temperature=0.07 \
#     --strategy='ddp_find_unused_parameters_false' \
#     --queue_size=9000 \
#     --dim_size=512 \
#     --warmup_its=3000 \
#     --precision='medium' \
#     --moco \
#     --geo_encode



# python -m geoclip.fit --data_path=/home/a.dhakal/active/datasets/YFCC100m/webdataset \
#     --train_batch_size=100 \
#     --val_batch_size=100 \
#     --wandb_mode=online \
#     --wandb_resume=false \
#     --max_epochs=100 \
#     --accelerator=gpu \
#     --devices=1 \
#     --num_workers=3 \
#     --project_name=GeoClip \
#     --mode=train \
#     --train_epoch_length=2000 \
#     --val_epoch_length=300 \
#     --val_check_interval=100 \
#     --project_name=GeoClip \
#     --run_name='geo_embed_geomoco_32_9k_noencode' \
#     --vit='32' \
#     --ckpt_path='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/u3oyk5ft/checkpoints/step=8600-val_loss=5.672.ckpt' \
#     --ckpt_mode='soft' \
#     --learning_rate=0.00005 \
#     --temperature=0.07 \
#     --strategy='ddp_find_unused_parameters_false' \
#     --queue_size=9000 \
#     --dim_size=512 \
#     --warmup_its=3000 \
#     --precision='medium' \
#     --moco \
#     --geo_encode
   
   
# --ckpt_path='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/u3oyk5ft/checkpoints/step=8600-val_loss=5.672.ckpt' \


    #/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/0j3bfqje/checkpoints/step=8900-val_loss=8.029.ckpt
    #/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/6zbm4f59/checkpoints/step=4500-top_k_score=0.875.ckpt
    # --val_check_interval=5 \

#Important notes
# For 32-patch-base: resize=480, crop=448, batch_size=128
#For 14L:resize=420, crop=392 ### did not work. Cuda OOM 
#For 16-base:resize=336 crop=320 batchsize=64