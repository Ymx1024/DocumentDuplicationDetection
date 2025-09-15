#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动脚本 - 确保多核优化环境变量在应用启动前设置
"""

import os
import multiprocessing
import sys

# 设置多进程启动方法为spawn，解决CUDA fork问题
multiprocessing.set_start_method('spawn', force=True)

def setup_multicore_environment():
    """设置多核优化环境变量"""
    cpu_cores = multiprocessing.cpu_count()
    worker_threads = min(cpu_cores, 16)
    
    # 设置所有相关的线程数环境变量
    env_vars = {
        'OMP_NUM_THREADS': str(worker_threads),
        'MKL_NUM_THREADS': str(worker_threads),
        'NUMEXPR_NUM_THREADS': str(worker_threads),
        'OPENBLAS_NUM_THREADS': str(worker_threads),
        'VECLIB_MAXIMUM_THREADS': str(worker_threads),
        'NUMBA_NUM_THREADS': str(worker_threads),
        'BLIS_NUM_THREADS': str(worker_threads),
        'MKL_DYNAMIC': 'FALSE',  # 禁用动态线程数
        'OMP_DYNAMIC': 'FALSE',  # 禁用动态线程数
        'TOKENIZERS_PARALLELISM': 'false',  # 禁用tokenizers并行，避免fork警告
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"设置环境变量: {key}={value}")
    
    print(f"检测到CPU核心数: {cpu_cores}")
    print(f"使用工作线程数: {worker_threads}")
    print("多核环境配置完成！")

if __name__ == "__main__":
    # 清理可能存在的日志抑制环境变量
    os.environ.pop('SUPPRESS_INIT_LOGS', None)
    
    # 设置多核环境
    setup_multicore_environment()
    
    # 简化导入应用 - 所有复杂逻辑都已移到 utils.py 中
    print("📦 开始导入应用模块...")
    try:
        from app import app
        print("✅ 应用模块导入成功")
    except Exception as e:
        print(f"❌ 应用导入失败: {e}")
        print("💡 请检查依赖是否完整安装")
        sys.exit(1)
    
    # 显示计算设备信息和启动类型
    try:
        import torch
        from utils import get_model_info, device, _model, _use_tfidf_fallback
        
        print("\n" + "🚀" + "="*58 + "🚀")
        print("📱 文档重复检测系统启动")
        print("="*60)
        
        # 获取模型信息
        model_info = get_model_info()
        
        # 显示启动模式信息
        print("🔧 启动模式信息:")
        startup_mode_names = {
            'normal': '标准启动',
            'fast_start': '快速启动模式',
        }
        print(f"   └─ 启动类型: {startup_mode_names.get(model_info['startup_mode'], model_info['startup_mode'])}")
        
        # 显示模型来源信息
        model_source_names = {
            'online_download': '🌐 联网下载模型',
            'local_cache': '💾 本地缓存模型',
            'local_path': '📁 指定路径模型', 
            'tfidf_fallback': '📊 TF-IDF后备模型'
        }
        print(f"   └─ 模型来源: {model_source_names.get(model_info['model_source'], model_info['model_source'])}")
        
        if model_info['load_time_seconds'] > 0:
            print(f"   └─ 加载用时: {model_info['load_time_seconds']}秒")
        
        print()
        
        # 检查实际使用的计算模式
        actual_device_mode = "CPU"
        actual_model_type = "TF-IDF"
        
        if _model is not None:
            model_type = type(_model).__name__
            if 'SentenceTransformer' in model_type:
                actual_model_type = "SentenceTransformer"
                # 检查模型实际运行设备
                model_device = getattr(_model, 'device', 'cpu')
                if str(model_device).startswith('cuda'):
                    actual_device_mode = "GPU"
            else:
                actual_model_type = "TF-IDF"
                actual_device_mode = "CPU"  # TF-IDF总是在CPU上运行
        
        # 根据实际使用情况显示信息
        print("💻 计算环境信息:")
        if actual_device_mode == "GPU":
            print("   └─ 🎮 GPU加速模式 - 高性能计算")
            print(f"      ├─ GPU: {torch.cuda.get_device_name(0)}")
            print(f"      └─ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("   └─ 🖥️  CPU计算模式 - 多线程处理")
            print(f"      ├─ 处理器: {multiprocessing.cpu_count()}核心")
            if torch.cuda.is_available():
                print(f"      └─ 注意: GPU可用但未使用 (当前模型: {actual_model_type})")
        
        print()
        print("🧠 AI模型信息:")
        if actual_model_type == "SentenceTransformer":
            print("   └─ SentenceTransformer (高精度语义理解)")
            if model_info['model_source'] == 'online_download':
                print("      ├─ 首次联网下载完成，后续启动将使用缓存")
            elif model_info['model_source'] == 'local_cache':
                print("      ├─ 使用本地缓存，无需重新下载")
            elif model_info['model_source'] == 'local_path':
                print("      ├─ 使用自定义路径模型")
            print("      └─ 提供最佳的文档相似度计算精度")
        else:
            print("   └─ TF-IDF模型 (标准精度文本分析)")
            if model_info['startup_mode'] == 'fast_start':
                print("      ├─ 快速启动模式自动选择")
            else:
                print("      ├─ 自动回退到TF-IDF方案")
            print("      └─ 无需下载，启动快速，适合大多数场景")
        
        print("🌐 Web服务: http://0.0.0.0:5000")
        print("🚀" + "="*58 + "🚀")
        
    except Exception as e:
        print(f"设备信息显示失败: {e}")
    
    print("\n" + "🌐" + "="*58 + "🌐")
    print("🚀 启动Flask Web服务...")
    print("📡 服务将在 http://0.0.0.0:5000 上运行")
    print("⏱️  启动过程中请稍候...")
    print("🌐" + "="*58 + "🌐")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
