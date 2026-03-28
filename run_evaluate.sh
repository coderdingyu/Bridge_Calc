#!/bin/bash
cd /home/dingyu/Bridge_Calc
source /data/miniconda3/bin/activate maskrcnn_env
python evaluate_map.py models/maskrcnn_bridge_best.pth 2>&1 | tee map_output.log