#!/usr/bin/env python3
"""
训练完成后自动执行推理和评估的脚本
用法: python auto_pipeline.py
"""
import os
import subprocess
import time
from datetime import datetime

WORK_DIR = "/home/dingyu/Bridge_Calc"
LOG_FILE = os.path.join(WORK_DIR, "training_new_final.log")
MODELS_DIR = os.path.join(WORK_DIR, "models")
TEST_RESULTS_DIR = os.path.join(WORK_DIR, "test_results")
PIPELINE_LOG = os.path.join(WORK_DIR, "pipeline.log")

os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

def log_message(msg, level="INFO"):
    """输出日志信息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [{level}] {msg}"
    print(log_msg)
    with open(PIPELINE_LOG, 'a') as f:
        f.write(log_msg + "\n")

def activate_conda():
    """返回激活conda环境的命令前缀"""
    return "source /data/miniconda3/etc/profile.d/conda.sh && conda activate maskrcnn_env && "

def wait_for_training_complete():
    """等待训练完成"""
    log_message("⏳ 监控训练过程...", "MONITOR")
    
    last_check_count = 0
    consecutive_no_change = 0
    
    while True:
        try:
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # 检查是否完成
                if "训练完成!!!" in content:
                    log_message("✅ 检测到训练完成信号！", "SUCCESS")
                    return True
                
                # 每30秒检查一次日志大小
                lines = len(content.split('\n'))
                if lines != last_check_count:
                    log_message(f"📝 日志更新 - 已有 {lines} 行", "DEBUG")
                    last_check_count = lines
                    consecutive_no_change = 0
                else:
                    consecutive_no_change += 1
                    if consecutive_no_change > 20:  # 10分钟无更新
                        log_message("⚠️ 日志已有10分钟无更新，可能训练卡住", "WARN")
            
            # 检查模型文件数
            if os.path.exists(MODELS_DIR):
                models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
                log_message(f"💾 已保存 {len(models)} 个模型文件", "DEBUG")
                
                # 如果有10个模型 + best，说明训练完成
                if len(models) >= 11:
                    log_message("✅ 检测到所有模型已保存！", "SUCCESS")
                    return True
            
            time.sleep(30)  # 每30秒检查一次
            
        except KeyboardInterrupt:
            log_message("训练监控已中断", "WARN")
            return False
        except Exception as e:
            log_message(f"❌ 监控错误: {e}", "ERROR")
            time.sleep(30)

def run_inference():
    """运行推理测试"""
    log_message("🎯 开始推理测试...", "PHASE")
    
    best_model = os.path.join(MODELS_DIR, "maskrcnn_bridge_best.pth")
    test_dir = "/data1/dingyu/datasets/labelme/test"
    
    if not os.path.exists(best_model):
        log_message(f"❌ 最佳模型不存在: {best_model}", "ERROR")
        return False
    
    cmd = f"{activate_conda()}cd {WORK_DIR} && python inference_final.py {best_model} {test_dir}"
    
    log_message(f"执行: {cmd}", "INFO")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
        
        if result.returncode == 0:
            log_message("✅ 推理完成！", "SUCCESS")
            log_message(f"推理输出:\n{result.stdout}", "OUTPUT")
            return True
        else:
            log_message(f"❌ 推理失败: {result.stderr}", "ERROR")
            return False
    except subprocess.TimeoutExpired:
        log_message("❌ 推理超时", "ERROR")
        return False

def calculate_map():
    """计算COCO mAP"""
    log_message("📊 计算COCO mAP指标...", "PHASE")
    
    best_model = os.path.join(MODELS_DIR, "maskrcnn_bridge_best.pth")
    
    cmd = f"{activate_conda()}cd {WORK_DIR} && python evaluate_map.py {best_model}"
    
    log_message(f"执行: {cmd}", "INFO")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            log_message("✅ mAP计算完成！", "SUCCESS")
            log_message(f"评估输出:\n{result.stdout}", "OUTPUT")
            return True
        else:
            log_message(f"⚠️ mAP计算可能出现问题:\n{result.stderr}", "WARN")
            return True  # 仍然继续
    except subprocess.TimeoutExpired:
        log_message("❌ mAP评估超时", "ERROR")
        return False

def main():
    """主流程"""
    print("\n" + "=" * 70)
    print("  🚀 Mask R-CNN 完整自动化流程")
    print("=" * 70 + "\n")
    
    log_message("🟢 流程启动", "START")
    
    # 1. 等待训练完成
    log_message("📌 第1阶段: 等待训练完成", "PHASE")
    if not wait_for_training_complete():
        log_message("❌ 训练监控失败，流程中止", "ERROR")
        return False
    
    # 2. 运行推理
    log_message("📌 第2阶段: 运行推理测试", "PHASE")
    if not run_inference():
        log_message("❌ 推理失败，流程中止", "ERROR")
        return False
    
    # 3. 计算mAP
    log_message("📌 第3阶段: 计算COCO mAP", "PHASE")
    calculate_map()
    
    log_message("🎉 完整流程已完成！", "SUCCESS")
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        log_message(f"❌ 流程异常: {e}", "ERROR")
        exit(1)
