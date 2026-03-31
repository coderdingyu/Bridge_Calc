#!/usr/bin/env python3
import os, sys, subprocess

os.chdir('/home/dingyu/Bridge_Calc')
env = os.environ.copy()
env['PATH'] = '/data/miniconda3/envs/maskrcnn_env/bin:' + env.get('PATH', '')

log_file = open('per_class_result.log', 'w')

proc = subprocess.Popen(
    ['/data/miniconda3/envs/maskrcnn_env/bin/python', 'evaluate_map.py', 'models/maskrcnn_bridge_best.pth'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
)

for line in proc.stdout:
    print(line, end='')
    log_file.write(line)
    log_file.flush()

proc.wait()
log_file.close()
print("\n结果已保存到 per_class_result.log")