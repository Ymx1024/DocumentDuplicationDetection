#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动脚本 - 直接使用TF-IDF模式，避免模型加载卡住
"""

import os

# 设置强制TF-IDF模式，跳过所有SentenceTransformer模型加载
os.environ['FORCE_TFIDF'] = '1'
os.environ['FAST_START'] = '1'

print("🚀 快速启动模式已启用")
print("📋 配置说明：")
print("   - 启动类型：强制TF-IDF模式")
print("   - 模型来源：TF-IDF后备模型")  
print("   - 跳过SentenceTransformer模型下载/加载")
print("   - 直接启动Web服务")
print("   - 计算精度：标准（适合大多数使用场景）")
print("   - 启动速度：极快（无模型加载时间）")

# 导入启动脚本
if __name__ == "__main__":
    import start_app 