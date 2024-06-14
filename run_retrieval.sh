#!/bin/bash


################### Trained With Dropout #################################
python -m geoclip.evaluations.test_retrieval \
--batch_size=10000 \
 --run_topk \
 --k=5 \
 --geo_encode \
 --ckpt_path=/path/to/model
