#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -m trainer.train --config_path "/home/jingqi/DeepLearningWorkshop/recipes/train_config.yaml"