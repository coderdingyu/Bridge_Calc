#!/bin/bash
# 实时监控训练进度脚本

echo "🔍 Mask R-CNN 训练实时监控系统"
echo "========================================"
echo ""

log_file="/home/dingyu/Bridge_Calc/training_new_final.log"

while [ -f "$log_file" ]; do
    lines=$(wc -l < "$log_file")
    
    echo "📊 当前日志行数: $lines"
    echo ""
    
    # 显示最新进度
    echo "⏳ 最新进度信息："
    tail -n 2 "$log_file" | grep -E "Epoch|Loss|✅|💾" || echo "  获取中..."
    
    echo ""
    echo "---"
    sleep 30
    clear
done
