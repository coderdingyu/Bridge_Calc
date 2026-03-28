#!/usr/bin/env python3
import subprocess
import sys
import os

os.chdir('/home/dingyu/Bridge_Calc')

cmd = [
    '/data/miniconda3/envs/maskrcnn_env/bin/python',
    'evaluate_map.py',
    'models/maskrcnn_bridge_best.pth'
]

env = os.environ.copy()
env['PATH'] = '/data/miniconda3/envs/maskrcnn_env/bin:' + env['PATH']

with open('mAP_eval.log', 'w') as f:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in process.stdout:
        print(line, end='')
        f.write(line)
        f.flush()

print("\nmAP评估完成!")