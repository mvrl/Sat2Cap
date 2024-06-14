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


