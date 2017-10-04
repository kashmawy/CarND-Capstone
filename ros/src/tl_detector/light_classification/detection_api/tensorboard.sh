#!/bin/bash


# MODEL_DIR=./models/faster_rcnn_other

# MODEL_DIR=./models/faster_rcnn_otherd

# MODEL_DIR=./models/ssd_my1

# MODEL_DIR=./models/faster_rcnn_sim

# MODEL_DIR=./models/faster_rcnn_multi

# MODEL_DIR=./models/faster_rcnn_multi_filtered


# MODEL_DIR=./models/ssd_filtered


MODEL_DIR=./models/faster_rcnn_multi_site



echo Running tensorboard ...

tensorboard --logdir=${MODEL_DIR}
