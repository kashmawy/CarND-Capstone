#!/bin/bash

# PIPELINE_CONFIG=./ssd_inception_v2_lights.config

# PIPELINE_CONFIG=./faster_rcnn_resnet101_light.config
# TRAIN_DIR=./models/faster_rcnn_other/train
# EVAL_DIR=./models/faster_rcnn_other/eval

# PIPELINE_CONFIG=./models/faster_rcnn_otherd/faster_rcnn_resnet101_light.config
# TRAIN_DIR=./models/faster_rcnn_otherd/train
# EVAL_DIR=./models/faster_rcnn_otherd/eval

# PIPELINE_CONFIG=./models/ssd_my1/ssd_inception_v2_lights.config
# TRAIN_DIR=./models/ssd_my1/train
# EVAL_DIR=./models/ssd_my1/eval


# PIPELINE_CONFIG=./models/faster_rcnn_sim/faster_rcnn_resnet101_light_sim.config
# TRAIN_DIR=./models/faster_rcnn_sim/train
# EVAL_DIR=./models/faster_rcnn_sim/eval

# PIPELINE_CONFIG=./models/faster_rcnn_multi/faster_rcnn_resnet101_light_multi.config
# TRAIN_DIR=./models/faster_rcnn_multi/train
# EVAL_DIR=./models/faster_rcnn_multi/eval


# PIPELINE_CONFIG=./models/faster_rcnn_multi_filtered/faster_rcnn_resnet101_light_multi_filtered.config
# TRAIN_DIR=./models/faster_rcnn_multi_filtered/train
# EVAL_DIR=./models/faster_rcnn_multi_filtered/eval


# PIPELINE_CONFIG=./models/ssd_filtered/ssd_inception_v2_lights_filtered.config
# TRAIN_DIR=./models/ssd_filtered/train
# EVAL_DIR=./models/ssd_filtered/eval

PIPELINE_CONFIG=./models/faster_rcnn_multi_site/faster_rcnn_resnet101_light_multi_site.config
TRAIN_DIR=./models/faster_rcnn_multi_site/train
EVAL_DIR=./models/faster_rcnn_multi_site/eval


echo PIPELINE_CONFIG = ${PIPELINE_CONFIG}

echo Training ...

python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PIPELINE_CONFIG} \
    --train_dir=${TRAIN_DIR}
