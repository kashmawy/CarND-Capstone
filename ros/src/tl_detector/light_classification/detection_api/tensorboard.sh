#!/bin/bash


# MODEL_DIR=./models/faster_rcnn_other

# MODEL_DIR=./models/faster_rcnn_otherd

# MODEL_DIR=./models/ssd_my1

MODEL_DIR=./models/faster_rcnn_sim


echo Running tensorboard ...

tensorboard --logdir=${MODEL_DIR}
