#!/usr/bin/env python3
import os
import sys
import subprocess

os.chdir('/home/dingyu/Bridge_Calc')

env = os.environ.copy()
env['PATH'] = '/data/miniconda3/envs/maskrcnn_env/bin:' + env.get('PATH', '')

python_bin = '/data/miniconda3/envs/maskrcnn_env/bin/python'

proc = subprocess.Popen(
    [python_bin, 'evaluate_map.py', 'models/maskrcnn_bridge_best.pth'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    env=env,
    text=True,
    bufsize=1
)

with open('mAP_realtime.log', 'w') as f:
    for line in proc.stdout:
        print(line, end='')
        f.write(line)
        f.flush()

print("\n=== mAP评估完成 ===")
print("结果已保存到 mAP_realtime.log")