#!/bin/bash

# PIPELINE_CONFIG=./models/faster_rcnn_multi_filtered/faster_rcnn_resnet101_light_multi_filtered.config
# MODEL_DIR=./models/faster_rcnn_multi_filtered
# TRAIN_DIR=./models/faster_rcnn_multi_filtered/train
# CHECKPOINT_NUMBER=20000


# PIPELINE_CONFIG=./models/ssd_filtered/ssd_inception_v2_lights_filtered.config
# MODEL_DIR=./models/ssd_filtered
# TRAIN_DIR=./models/ssd_filtered/train
# CHECKPOINT_NUMBER=9598

PIPELINE_CONFIG=./models/faster_rcnn_multi_site/faster_rcnn_resnet101_light_multi_site.config
MODEL_DIR=./models/faster_rcnn_multi_site
TRAIN_DIR=./models/faster_rcnn_multi_site/train
CHECKPOINT_NUMBER=20000


CHECKPOINT_PREFIX=${TRAIN_DIR}/model.ckpt-${CHECKPOINT_NUMBER}


echo Exporting ...

python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG} \
    --trained_checkpoint_prefix ${CHECKPOINT_PREFIX} \
    --output_directory ${MODEL_DIR}/frozen_${CHECKPOINT_NUMBER}
