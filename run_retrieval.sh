#!/bin/bash

#  python -m geoclip.evaluations.test_retrieval \
# --batch_size=20000 \
#  --save_topk \
#  --k=5 \
#  --geo_encode \
#  --date_time='2012-05-01 08:00:00.0' \
#  --save_path=/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/wacv/retrieval_images/retrieval_images_morning

#  python -m geoclip.evaluations.test_retrieval \
# --batch_size=20000 \
#  --save_topk \
#  --k=5 \
#  --geo_encode \
#  --date_time='2012-05-01 23:00:00.0' \
#  --save_path=/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/wacv/retrieval_images/retrieval_images_night



#   python -m geoclip.evaluations.test_retrieval \
# --batch_size=20000 \
#  --save_topk \
#  --k=5 \
#  --geo_encode \
#  --date_time='2012-12-20 23:00:00.0' \
#  --save_path=/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/wacv/retrieval_images/retrieval_images_december_night


################### Trained With Dropout #################################
python -m geoclip.evaluations.test_retrieval \
--batch_size=10000 \
 --run_topk \
 --k=5 \
 --geo_encode \
 --ckpt_path=/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/f1dtv48z/checkpoints/step=86750-val_loss=4.100.ckpt

 python -m geoclip.evaluations.test_retrieval \
--batch_size=10000 \
 --run_topk \
 --k=1 \
 --geo_encode \
 --ckpt_path=/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/f1dtv48z/checkpoints/step=86750-val_loss=4.100.ckpt

 python -m geoclip.evaluations.test_retrieval \
--batch_size=10000 \
 --run_topk \
 --k=10 \
 --geo_encode \
 --ckpt_path=/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/f1dtv48z/checkpoints/step=86750-val_loss=4.100.ckpt

 python -m geoclip.evaluations.test_retrieval \
--batch_size=10000 \
 --run_topk \
 --k=5  \
 --ckpt_path=/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/f1dtv48z/checkpoints/step=86750-val_loss=4.100.ckpt

 python -m geoclip.evaluations.test_retrieval \
--batch_size=10000 \
 --run_topk \
 --k=1 \
 --ckpt_path=/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/f1dtv48z/checkpoints/step=86750-val_loss=4.100.ckpt

 python -m geoclip.evaluations.test_retrieval \
--batch_size=10000 \
 --run_topk \
 --k=10 \
 --ckpt_path=/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/f1dtv48z/checkpoints/step=86750-val_loss=4.100.ckpt


################################################### NO DROPOUT##############################################################
 python -m geoclip.evaluations.test_retrieval \
--batch_size=10000 \
 --run_topk \
 --k=5 \
 --geo_encode 

 python -m geoclip.evaluations.test_retrieval \
--batch_size=10000 \
 --run_topk \
 --k=1 \
 --geo_encode 

 python -m geoclip.evaluations.test_retrieval \
--batch_size=10000 \
 --run_topk \
 --k=10 \
 --geo_encode 

 python -m geoclip.evaluations.test_retrieval \
--batch_size=10000 \
 --run_topk \
 --k=5  

 python -m geoclip.evaluations.test_retrieval \
--batch_size=10000 \
 --run_topk \
 --k=1 
 

 python -m geoclip.evaluations.test_retrieval \
--batch_size=10000 \
 --run_topk \
 --k=10 

 ################################## CLIP ############################################################# python -m geoclip.evaluations.test_retrieval \
python -m geoclip.evaluations.test_retrieval \
--batch_size=10000 \
 --run_topk \
 --k=5  \
 --clip

 python -m geoclip.evaluations.test_retrieval \
--batch_size=10000 \
 --run_topk \
 --k=1 \
 --clip 
 

 python -m geoclip.evaluations.test_retrieval \
--batch_size=10000 \
 --run_topk \
 --k=10 \
 --clip



 ####file to model map
 ##Trained without dropout: /home/a.dhakal/active/user_a.dhakal/geoclip/logs/temp_models/s212e5he/checkpoints/step=38000-val_loss=4.957.ckpt
 ##Trained with dropout: /home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/f1dtv48z/checkpoints/step=86750-val_loss=4.100.ckpt
 ##Trained without metadata: /home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/u3oyk5ft/checkpoints/step=8600-val_loss=5.672.ckpt