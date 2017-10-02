#!/bin/bash

PIPELINE_CONFIG=./models/faster_rcnn_multi/faster_rcnn_resnet101_light_multi.config
MODEL_DIR=./models/faster_rcnn_multi
TRAIN_DIR=./models/faster_rcnn_multi/train
CHECKPOINT_NUMBER=11293

CHECKPOINT_PREFIX=${TRAIN_DIR}/model.ckpt-${CHECKPOINT_NUMBER}


echo Exporting ...

python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG} \
    --trained_checkpoint_prefix ${CHECKPOINT_PREFIX} \
    --output_directory ${MODEL_DIR}/frozen_${CHECKPOINT_NUMBER}
