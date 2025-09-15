"""
智能启动管理器
按照 硬件 → 本地缓存 → 网络 的顺序进行检测和决策
"""

import os
import torch
import multiprocessing
import time
from pathlib import Path
from multiprocessing import Process, Queue


def _model_loading_worker(q, model_name_to_load, device):
    """在子进程中运行的加载函数（顶层函数，可以被pickle）"""
    try:
        # 在子进程中，日志是不可见的，但加载仍在进行
        from sentence_transformers import SentenceTransformer
        import torch
        import os
        
        # 设置环境变量优化加载
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        # 强制使用CPU，避免多进程中的CUDA问题
        device = 'cpu'
        
        # 优化加载参数
        load_kwargs = {
            'device': device,
            'trust_remote_code': True,  # 允许远程代码
        }
        
        # 如果是本地路径，添加额外优化
        if os.path.exists(model_name_to_load):
            load_kwargs['use_auth_token'] = False  # 本地模型不需要认证
        
        # 加载模型到指定设备
        model = SentenceTransformer(model_name_to_load, **load_kwargs)
        
        # 返回模型和实际使用的设备
        q.put((model, device))
        
    except Exception as e:
        # 将异常信息放入队列，以便主进程可以捕获
        q.put(e)


def load_model_with_timeout(model_name, timeout=30):
    """
    在独立的进程中加载模型，并设置超时保护。
    这是解决模型加载卡死的关键。
    """
    # 获取设备信息 - 简化版本，只检测CUDA
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    q = Queue()
    p = Process(target=_model_loading_worker, args=(q, model_name, device))
    p.daemon = True
    
    print(f"🔄 开始加载SentenceTransformer模型: '{model_name}' (超时: {timeout}秒)...")
    start_time = time.time()
    
    p.start()
    p.join(timeout)
    
    elapsed_time = time.time() - start_time

    if p.is_alive():
        print(f"❌ 模型加载超时！(超过 {timeout} 秒)")
        p.terminate()  # 强制终止子进程
        p.join()
        return None, "timeout", elapsed_time

    if not q.empty():
        result = q.get()
        if isinstance(result, Exception):
            print(f"❌ 模型加载时发生错误: {result}")
            return None, str(result), elapsed_time
        elif isinstance(result, tuple) and len(result) == 2:
            # 新格式：(model, actual_device)
            model, actual_device = result
            
            # 显示实际使用的设备
            if actual_device.startswith('cuda'):
                device_info = f"GPU ({actual_device})"
            else:
                device_info = "CPU"
            
            print(f"✅ 模型加载成功! (耗时: {elapsed_time:.2f}秒, 运行于: {device_info})")
            return model, "success", elapsed_time
        else:
            # 兼容旧格式（只返回model）
            model = result
            model_device = "CPU"
            if torch.cuda.is_available():
                try:
                    # 尝试将一个测试张量放到模型设备上，验证可用性
                    test_tensor = torch.tensor([1]).to(model.device)
                    model_device = f"GPU ({model.device})"
                except Exception:
                    model_device = "CPU (GPU验证失败)"
            
            print(f"✅ 模型加载成功! (耗时: {elapsed_time:.2f}秒, 运行于: {model_device})")
            return model, "success", elapsed_time
    
    return None, "unknown_error", elapsed_time


def _cpu_model_worker(q, model_name_to_load):
    """CPU模式模型加载工作进程"""
    try:
        from sentence_transformers import SentenceTransformer
        import os
        
        # 设置环境变量优化CPU加载
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        # 优化加载参数
        load_kwargs = {
            'device': 'cpu',
            'trust_remote_code': True,
        }
        
        # 如果是本地路径，添加额外优化
        if os.path.exists(model_name_to_load):
            load_kwargs['use_auth_token'] = False
        
        # 强制使用CPU
        model = SentenceTransformer(model_name_to_load, **load_kwargs)
        q.put((model, 'cpu'))
    except Exception as e:
        q.put(e)


def _load_model_cpu_only(model_name, timeout=30):
    """强制在CPU上加载模型"""
    q = Queue()
    p = Process(target=_cpu_model_worker, args=(q, model_name))
    p.daemon = True
    
    print(f"🔄 强制CPU模式加载: '{model_name}' (超时: {timeout}秒)...")
    start_time = time.time()
    
    p.start()
    p.join(timeout)
    
    elapsed_time = time.time() - start_time

    if p.is_alive():
        print(f"❌ CPU模式加载也超时！(超过 {timeout} 秒)")
        p.terminate()
        p.join()
        return None, "timeout", elapsed_time

    if not q.empty():
        result = q.get()
        if isinstance(result, Exception):
            print(f"❌ CPU模式加载时发生错误: {result}")
            return None, str(result), elapsed_time
        elif isinstance(result, tuple):
            model, device = result
            print(f"✅ CPU模式加载成功! (耗时: {elapsed_time:.2f}秒)")
            return model, "success", elapsed_time
    
    return None, "unknown_error", elapsed_time


def _load_model_simple(model_name, device='cpu', timeout=20):
    """简化的模型加载方法，减少超时时间"""
    try:
        from sentence_transformers import SentenceTransformer
        import os
        
        # 设置环境变量优化加载
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        print(f"🔄 简化模式加载: '{model_name}' (设备: {device}, 超时: {timeout}秒)...")
        start_time = time.time()
        
        # 直接加载，不使用多进程
        model = SentenceTransformer(model_name, device=device)
        
        elapsed_time = time.time() - start_time
        print(f"✅ 简化模式加载成功! (耗时: {elapsed_time:.2f}秒)")
        return model, "success", elapsed_time
        
    except Exception as e:
        elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
        print(f"❌ 简化模式加载失败: {e}")
        return None, str(e), elapsed_time


def detect_hardware_capabilities():
    """
    检测硬件能力
    返回: dict with 'device', 'gpu_available', 'gpu_info'
    """
    result = {
        'device': 'cpu',
        'gpu_available': False,
        'gpu_info': None
    }
    
    print("🔍 步骤1: 硬件能力检测")
    
    # 检测CUDA GPU
    if torch.cuda.is_available():
        try:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            result.update({
                'device': 'cuda:0',
                'gpu_available': True,
                'gpu_info': {
                    'name': gpu_name,
                    'memory_gb': round(gpu_memory, 1),
                    'count': gpu_count
                }
            })
            
            print(f"  ✅ GPU可用: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"  🎯 推荐设备: GPU (cuda:0)")
            
        except Exception as e:
            print(f"  ⚠️ GPU检测异常: {e}")
            print(f"  🔄 回退到CPU模式")
    else:
        print(f"  ❌ GPU不可用")
        print(f"  🖥️ 使用CPU模式")
    
    # CPU信息
    cpu_cores = multiprocessing.cpu_count()
    print(f"  💻 CPU核心数: {cpu_cores}")
    
    return result


def search_local_model_cache(model_keywords=None):
    """
    全目录搜索模型缓存，支持多模型检测
    返回: dict with 'found', 'models' (优先级列表), 'best_model'
    """
    if model_keywords is None:
        # 按优先级排序：高精度模型优先
        model_keywords = [
            'all-mpnet-base-v2',           # 最高精度
            'all-MiniLM-L6-v2', 
            'paraphrase-MiniLM-L6-v2',
            'paraphrase-multilingual-MiniLM-L12-v2',  # 多语言支持
            'sentence-transformers'        # 通用匹配
        ]
    
    print("🔍 步骤2: 本地模型缓存搜索")
    
    # 获取当前用户的主目录
    home_dir = Path.home()
    
    # 搜索路径列表
    search_paths = [
        # 用户指定路径
        os.environ.get('ST_MODEL_PATH', 'all-mpnet-base-v2'),
        
        # 标准Hugging Face缓存路径
        home_dir / '.cache' / 'huggingface' / 'hub',
        home_dir / '.cache' / 'huggingface' / 'transformers',
        
        # 可能的系统路径
        Path('/data/doc_similarity_env/cache/huggingface/hub'),
        Path('/root/.cache/huggingface/hub'),
        Path('/home/user/.cache/huggingface/hub'),
        
        # 项目本地缓存
        Path.cwd() / '.cache' / 'huggingface' / 'hub',
        Path.cwd() / 'models',
        
        # Windows常见路径
        home_dir / 'AppData' / 'Local' / 'huggingface' / 'hub' if os.name == 'nt' else None,
    ]
    
    # 移除None值并转换为字符串路径
    search_paths = [str(p) for p in search_paths if p is not None]
    
    # 检查用户指定路径是否是目录
    user_specified = os.environ.get('ST_MODEL_PATH', 'all-mpnet-base-v2')
    if os.path.isdir(user_specified):
        search_paths.insert(0, user_specified)
    
    print(f"  📁 搜索 {len(search_paths)} 个可能的路径...")
    
    # 存储找到的所有模型
    found_models = []
    
    for i, search_path in enumerate(search_paths, 1):
        print(f"  📂 [{i}/{len(search_paths)}] 检查: {search_path}")
        
        if not os.path.exists(search_path):
            print(f"     ❌ 路径不存在")
            continue
            
        try:
            # 递归搜索包含关键词的目录
            for root, dirs, files in os.walk(search_path):
                for keyword in model_keywords:
                    # 检查关键词是否在路径中
                    keyword_variants = [
                        keyword,
                        keyword.replace('/', '--'),  # Hugging Face命名规范
                        keyword.replace('-', '_')     # 可能的下划线变体
                    ]
                    
                    # 检查是否有任何变体在路径中
                    if any(variant in root for variant in keyword_variants):
                        # 跳过子目录（如 1_Pooling, 2_Dense 等）
                        if any(subdir in root for subdir in ['/1_Pooling', '/2_Dense', '/3_Dense', '/0_Transformer']):
                            continue
                        
                        # 检查是否包含模型文件
                        model_files = ['config.json', 'pytorch_model.bin', 'model.safetensors']
                        found_files = [f for f in model_files if os.path.isfile(os.path.join(root, f))]
                        
                        # 确保是完整的模型目录（必须包含config.json和至少一个模型文件）
                        if 'config.json' in found_files and len(found_files) >= 2:
                            # 避免重复添加相同模型
                            if not any(model['path'] == root for model in found_models):
                                model_info = {
                                    'path': root,
                                    'model_type': keyword,
                                    'files': found_files,
                                    'priority': model_keywords.index(keyword)  # 优先级（越小越优先）
                                }
                                found_models.append(model_info)
                                
                                print(f"     ✅ 发现模型: {keyword}")
                                print(f"     📍 路径: {root}")
                                print(f"     📄 文件: {', '.join(found_files)}")
                                
                                # 如果找到的是完整路径模型，尝试设置环境变量以便下次快速找到
                                if keyword != 'sentence-transformers':
                                    try:
                                        # 为下次启动设置环境变量提示
                                        print(f"     💡 提示: 可设置环境变量 ST_MODEL_PATH={root} 以快速启动")
                                    except Exception:
                                        pass
            
            if not found_models:
                print(f"     ❌ 未找到模型文件")
            
        except PermissionError:
            print(f"     ⚠️ 权限不足，跳过")
        except Exception as e:
            print(f"     ❌ 搜索失败: {e}")
    
    # 按优先级排序找到的模型
    found_models.sort(key=lambda x: x['priority'])
    
    if found_models:
        print(f"  🎉 共找到 {len(found_models)} 个模型:")
        for i, model in enumerate(found_models, 1):
            print(f"    [{i}] {model['model_type']} - {model['path']}")
        
        return {
            'found': True,
            'models': found_models,
            'best_model': found_models[0],  # 最高优先级的模型
            'count': len(found_models)
        }
    else:
        print("  💔 未找到任何本地模型缓存")
        return {
            'found': False, 
            'models': [], 
            'best_model': None,
            'count': 0
        }


def validate_model_integrity(model_path):
    """
    验证模型完整性
    返回: dict with 'valid', 'issues', 'repairable'
    """
    print(f"🔍 验证模型完整性: {model_path}")
    
    issues = []
    repairable = True
    
    try:
        # 检查基本文件是否存在
        required_files = ['config.json']
        model_files = ['pytorch_model.bin', 'model.safetensors']
        
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                issues.append(f"缺少必需文件: {file}")
                repairable = False
        
        # 检查至少有一个模型文件
        has_model_file = False
        for file in model_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                has_model_file = True
                break
        
        if not has_model_file:
            issues.append("缺少模型权重文件")
            repairable = False
        
        # 检查config.json内容
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                if 'model_type' not in config:
                    issues.append("config.json缺少model_type字段")
                    repairable = True  # 可以修复
                
                if 'architectures' not in config:
                    issues.append("config.json缺少architectures字段")
                    repairable = True  # 可以修复
                    
            except Exception as e:
                issues.append(f"config.json解析失败: {e}")
                repairable = False
        
        # 检查文件大小（防止空文件）
        for file in ['config.json'] + model_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                if os.path.getsize(file_path) == 0:
                    issues.append(f"文件为空: {file}")
                    repairable = False
        
        if not issues:
            print("  ✅ 模型完整性验证通过")
            return {'valid': True, 'issues': [], 'repairable': True}
        else:
            print(f"  ⚠️ 发现 {len(issues)} 个问题:")
            for issue in issues:
                print(f"    - {issue}")
            return {'valid': False, 'issues': issues, 'repairable': repairable}
            
    except Exception as e:
        print(f"  ❌ 模型完整性验证失败: {e}")
        return {'valid': False, 'issues': [f"验证过程出错: {e}"], 'repairable': False}


def test_model_availability(model_path, device='cpu', quick_test=True):
    """
    测试模型是否可用
    返回: dict with 'available', 'error'
    """
    print(f"🔍 测试模型可用性: {model_path}")
    
    try:
        from sentence_transformers import SentenceTransformer
        import os
        
        # 设置环境变量优化测试
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['OMP_NUM_THREADS'] = '1'
        
        # 快速测试加载（不在GPU上测试，避免长时间占用）
        print("  ⏳ 快速测试加载...")
        
        # 优化加载参数
        load_kwargs = {
            'device': 'cpu',
            'trust_remote_code': True,
        }
        
        # 如果是本地路径，添加额外优化
        if os.path.exists(model_path):
            load_kwargs['use_auth_token'] = False
        
        model = SentenceTransformer(model_path, **load_kwargs)
        
        if quick_test:
            # 快速测试：只检查模型是否能正常初始化
            print("  ⏳ 快速验证模型结构...")
            if hasattr(model, 'encode') and hasattr(model, '_modules'):
                print("  ✅ 模型结构验证通过")
                del model  # 释放内存
                return {'available': True, 'error': None}
            else:
                print("  ❌ 模型结构验证失败")
                return {'available': False, 'error': '模型结构不完整'}
        else:
            # 完整测试：测试编码功能
            print("  ⏳ 测试编码功能...")
            test_text = "This is a test sentence."
            embedding = model.encode(test_text, convert_to_tensor=True)
            
            if embedding is not None and len(embedding) > 0:
                print("  ✅ 模型测试通过")
                del model  # 释放内存
                return {'available': True, 'error': None}
            else:
                print("  ❌ 模型编码测试失败")
                return {'available': False, 'error': '编码结果无效'}
            
    except Exception as e:
        print(f"  ❌ 模型测试失败: {e}")
        return {'available': False, 'error': str(e)}


def check_network_connectivity():
    """
    检测网络连通性
    返回: dict with 'huggingface_available', 'general_internet'
    """
    print("🔍 步骤3: 网络连通性检测")
    
    def test_url(url, timeout=3):
        try:
            import urllib.request
            urllib.request.urlopen(url, timeout=timeout)
            return True
        except Exception:
            return False
    
    # 测试Hugging Face连通性
    hf_available = test_url("https://huggingface.co", timeout=5)
    if hf_available:
        print("  ✅ Hugging Face 可访问")
    else:
        print("  ❌ Hugging Face 不可访问")
    
    # 测试一般网络连通性
    general_internet = test_url("https://www.google.com", timeout=3) or test_url("https://www.baidu.com", timeout=3)
    if general_internet:
        print("  ✅ 互联网连接正常")
    else:
        print("  ❌ 互联网连接异常")
    
    return {
        'huggingface_available': hf_available,
        'general_internet': general_internet
    }


def repair_model(model_path, model_type):
    """
    修复损坏的模型
    返回: dict with 'success', 'new_path', 'error'
    """
    print(f"🔧 开始修复模型: {model_type}")
    print(f"📍 原路径: {model_path}")
    
    try:
        # 获取模型名称映射
        model_name_mapping = {
            'all-mpnet-base-v2': 'sentence-transformers/all-mpnet-base-v2',
            'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
            'paraphrase-MiniLM-L6-v2': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
            'paraphrase-multilingual-MiniLM-L12-v2': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        }
        
        if model_type not in model_name_mapping:
            return {'success': False, 'new_path': None, 'error': f'未知模型类型: {model_type}'}
        
        model_name = model_name_mapping[model_type]
        
        # 创建备份目录
        backup_path = model_path + '.backup'
        if os.path.exists(model_path):
            print(f"📦 创建备份: {backup_path}")
            import shutil
            shutil.move(model_path, backup_path)
        
        # 重新下载模型
        print(f"⬇️ 重新下载模型: {model_name}")
        from sentence_transformers import SentenceTransformer
        
        # 设置环境变量优化下载
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['OMP_NUM_THREADS'] = '1'
        
        # 下载模型到指定路径
        model = SentenceTransformer(model_name, cache_folder=os.path.dirname(model_path))
        
        # 获取实际下载路径
        new_path = model._modules['0'].auto_model.config.name_or_path
        if not new_path or not os.path.exists(new_path):
            # 如果无法获取路径，使用默认路径
            new_path = model_path
        
        print(f"✅ 模型修复成功!")
        print(f"📍 新路径: {new_path}")
        
        # 清理备份（如果新模型工作正常）
        if os.path.exists(backup_path):
            print(f"🗑️ 清理备份目录: {backup_path}")
            shutil.rmtree(backup_path)
        
        return {'success': True, 'new_path': new_path, 'error': None}
        
    except Exception as e:
        print(f"❌ 模型修复失败: {e}")
        
        # 恢复备份
        if os.path.exists(backup_path):
            print(f"🔄 恢复备份: {backup_path}")
            import shutil
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            shutil.move(backup_path, model_path)
        
        return {'success': False, 'new_path': None, 'error': str(e)}


def auto_repair_models(local_models, network_status):
    """
    自动修复有问题的模型
    返回: dict with 'repaired_models', 'failed_models'
    """
    if not network_status['huggingface_available']:
        print("⚠️ Hugging Face不可访问，跳过模型修复")
        return {'repaired_models': [], 'failed_models': []}
    
    print("🔧 开始自动模型修复检查")
    repaired_models = []
    failed_models = []
    
    for model in local_models['models']:
        model_path = model['path']
        model_type = model['model_type']
        
        print(f"\n🔍 检查模型: {model_type}")
        
        # 验证模型完整性
        integrity_result = validate_model_integrity(model_path)
        
        if integrity_result['valid']:
            print(f"✅ 模型 {model_type} 完整性正常")
            continue
        
        if not integrity_result['repairable']:
            print(f"❌ 模型 {model_type} 无法修复")
            failed_models.append({
                'model_type': model_type,
                'path': model_path,
                'issues': integrity_result['issues']
            })
            continue
        
        # 尝试修复模型
        print(f"🔧 尝试修复模型: {model_type}")
        repair_result = repair_model(model_path, model_type)
        
        if repair_result['success']:
            print(f"✅ 模型 {model_type} 修复成功")
            repaired_models.append({
                'model_type': model_type,
                'old_path': model_path,
                'new_path': repair_result['new_path']
            })
        else:
            print(f"❌ 模型 {model_type} 修复失败: {repair_result['error']}")
            failed_models.append({
                'model_type': model_type,
                'path': model_path,
                'issues': integrity_result['issues'],
                'repair_error': repair_result['error']
            })
    
    print(f"\n📊 修复结果:")
    print(f"  ✅ 成功修复: {len(repaired_models)} 个模型")
    print(f"  ❌ 修复失败: {len(failed_models)} 个模型")
    
    return {'repaired_models': repaired_models, 'failed_models': failed_models}


def intelligent_startup_strategy():
    """
    智能启动策略决策器
    按照 硬件 → 本地缓存 → 网络 的顺序进行检测和决策
    返回: dict with 'strategy', 'config', 'reason'
    """
    print("\n🤖 智能启动策略决策器")
    print("="*60)
    
    # 检查用户强制设置
    if os.getenv('FORCE_TFIDF', '0') == '1':
        return {
            'strategy': 'force_tfidf',
            'config': {'device': 'cpu', 'model_type': 'tfidf'},
            'reason': '用户强制指定TF-IDF模式'
        }
    
    if os.getenv('FAST_START', '0') == '1':
        return {
            'strategy': 'fast_start', 
            'config': {'device': 'cpu', 'model_type': 'tfidf'},
            'reason': '用户指定快速启动模式'
        }
    
    # 步骤1: 硬件检测
    hardware = detect_hardware_capabilities()
    
    # 步骤2: 本地模型搜索
    local_model = search_local_model_cache()
    
    # 步骤3: 网络检测
    network = check_network_connectivity()
    
    # 步骤4: 模型修复检查（如果网络可用且本地模型有问题）
    if network['huggingface_available'] and local_model['found']:
        print("\n🔧 步骤4: 模型修复检查")
        repair_result = auto_repair_models(local_model, network)
        
        # 如果有模型被修复，重新搜索本地模型
        if repair_result['repaired_models']:
            print("🔄 检测到模型修复，重新搜索本地模型...")
            local_model = search_local_model_cache()
    
    print("\n🎯 策略决策分析:")
    
    # 决策逻辑 - 多模型支持
    if local_model['found'] and local_model['count'] > 0:
        print(f"  📋 发现 {local_model['count']} 个本地模型，按优先级测试...")
        
        # 按优先级测试每个模型
        for i, model in enumerate(local_model['models']):
            print(f"  🔍 测试模型 {i+1}/{local_model['count']}: {model['model_type']}")
            model_test = test_model_availability(model['path'])
            
            if model_test['available']:
                # 找到可用的模型，选择最佳设备
                device = hardware['device']
                
                print(f"  ✅ 模型 {model['model_type']} 测试通过，将使用此模型")
                
                return {
                    'strategy': 'local_model',
                    'config': {
                        'device': device,
                        'model_path': model['path'],
                        'model_type': model['model_type'],
                        'gpu_available': hardware['gpu_available'],
                        'model_priority': i + 1,  # 在列表中的位置
                        'total_models': local_model['count']
                    },
                    'reason': f"本地模型 {model['model_type']} 可用 + {device.upper()}计算"
                }
            else:
                print(f"  ❌ 模型 {model['model_type']} 测试失败: {model_test['error']}")
                continue
        
        print(f"  ⚠️ 所有 {local_model['count']} 个本地模型都不可用")
        print(f"  🔄 继续检查网络下载选项...")
    
    # 本地模型不可用或不存在，检查网络下载
    if network['huggingface_available']:
        device = hardware['device']
        model_name = os.environ.get('ST_MODEL_PATH', 'all-mpnet-base-v2')
        
        return {
            'strategy': 'download_model',
            'config': {
                'device': device,
                'model_name': model_name,
                'gpu_available': hardware['gpu_available']
            },
            'reason': f"联网下载模型 + {device.upper()}计算"
        }
    
    # 网络不可用，检查TF-IDF可用性
    print("  ⚠️ 网络不可用，检查TF-IDF后备方案...")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        return {
            'strategy': 'tfidf_fallback',
            'config': {
                'device': 'cpu',
                'model_type': 'tfidf'
            },
            'reason': 'TF-IDF后备方案 (无网络连接)'
        }
    except ImportError:
        print("  ❌ TF-IDF依赖不可用")
        
        # 最后的CPU fallback
        return {
            'strategy': 'minimal_cpu',
            'config': {
                'device': 'cpu',
                'model_type': 'basic'
            },
            'reason': 'CPU基础模式 (最小启动需求)'
        }


def execute_startup_strategy(strategy_result):
    """
    执行启动策略
    返回: dict with 'success', 'model', 'actual_config'
    """
    strategy = strategy_result['strategy']
    config = strategy_result['config']
    
    print(f"\n🚀 执行启动策略: {strategy}")
    print(f"📋 配置: {config}")
    print(f"💡 原因: {strategy_result['reason']}")
    print("-" * 40)
    
    try:
        if strategy == 'local_model':
            return _execute_local_model_strategy(config)
        elif strategy == 'download_model':
            return _execute_download_strategy(config)
        elif strategy == 'tfidf_fallback':
            return _execute_tfidf_strategy(config)
        elif strategy == 'force_tfidf' or strategy == 'fast_start':
            return _execute_tfidf_strategy(config)
        elif strategy == 'minimal_cpu':
            return _execute_minimal_strategy(config)
        else:
            raise ValueError(f"未知策略: {strategy}")
            
    except Exception as e:
        print(f"❌ 策略执行失败: {e}")
        # 最后的后备方案
        print("🔄 执行最小化后备策略...")
        return _execute_minimal_strategy({'device': 'cpu', 'model_type': 'basic'})


def _execute_local_model_strategy(config):
    """执行本地模型策略，支持多模型回退"""
    model_path = config['model_path']
    device = config['device']
    model_type = config.get('model_type', 'unknown')
    total_models = config.get('total_models', 1)
    model_priority = config.get('model_priority', 1)
    
    print(f"📂 加载本地模型: {model_path}")
    print(f"🎯 目标设备: {device}")
    print(f"📊 模型信息: {model_type} (优先级 {model_priority}/{total_models})")
    
    # 优先级1: 简化加载（最快最稳定）
    print("🔄 尝试简化加载...")
    model, status, load_time = _load_model_simple(model_path, device='cpu', timeout=20)
    
    if status == "success" and model is not None:
        return {
            'success': True,
            'model': model,
            'actual_config': {
                'strategy': 'local_model_simple',
                'device': 'cpu',
                'model_path': model_path,
                'model_type': model_type,
                'load_time': load_time,
                'model_priority': model_priority
            }
        }
    
    # 优先级2: 多进程GPU加载
    if device.startswith('cuda'):
        print("🔄 简化加载失败，尝试多进程GPU加载...")
        model, status, load_time = load_model_with_timeout(model_path, timeout=30)
        
        if status == "success" and model is not None:
            return {
                'success': True,
                'model': model,
                'actual_config': {
                    'strategy': 'local_model_gpu',
                    'device': device,
                    'model_path': model_path,
                    'model_type': model_type,
                    'load_time': load_time,
                    'model_priority': model_priority
                }
            }
    
    # 优先级3: 多进程CPU加载
    print("🔄 GPU加载失败，尝试多进程CPU加载...")
    model, status, load_time = _load_model_cpu_only(model_path, timeout=30)
    
    if status == "success" and model is not None:
        return {
            'success': True,
            'model': model,
            'actual_config': {
                'strategy': 'local_model_cpu',
                'device': 'cpu',
                'model_path': model_path,
                'model_type': model_type,
                'load_time': load_time,
                'model_priority': model_priority
            }
        }
    
    # 如果所有加载方式都失败，尝试其他可用模型
    if total_models > 1:
        print(f"🔄 所有加载方式都失败，尝试其他可用模型...")
        return _try_alternative_models(config, device)
    
    raise Exception(f"本地模型加载失败: 简化加载、GPU加载、CPU加载都失败")


def _try_alternative_models(config, device):
    """尝试其他可用的本地模型"""
    # 重新搜索所有可用模型
    local_models = search_local_model_cache()
    
    if not local_models['found'] or local_models['count'] == 0:
        raise Exception("没有其他可用模型")
    
    current_model_path = config['model_path']
    
    # 尝试其他模型
    for model in local_models['models']:
        if model['path'] == current_model_path:
            continue  # 跳过已经失败的模型
            
        print(f"🔄 尝试备用模型: {model['model_type']}")
        print(f"📍 路径: {model['path']}")
        
        # 测试模型可用性
        test_result = test_model_availability(model['path'])
        if not test_result['available']:
            print(f"❌ 备用模型 {model['model_type']} 不可用: {test_result['error']}")
            continue
        
        # 优先级1: 简化加载（最快最稳定）
        print(f"🔄 尝试简化加载备用模型 {model['model_type']}...")
        model_obj, status, load_time = _load_model_simple(model['path'], device='cpu', timeout=20)
        
        if status == "success" and model_obj is not None:
            print(f"✅ 备用模型 {model['model_type']} 简化加载成功!")
            return {
                'success': True,
                'model': model_obj,
                'actual_config': {
                    'strategy': 'local_model_fallback_simple',
                    'device': 'cpu',
                    'model_path': model['path'],
                    'model_type': model['model_type'],
                    'load_time': load_time,
                    'fallback_reason': f"原模型 {config.get('model_type', 'unknown')} 加载失败"
                }
            }
        
        # 优先级2: 多进程GPU加载
        if device.startswith('cuda'):
            print(f"🔄 简化加载失败，尝试多进程GPU加载备用模型 {model['model_type']}...")
            model_obj, status, load_time = load_model_with_timeout(model['path'], timeout=30)
            
            if status == "success" and model_obj is not None:
                print(f"✅ 备用模型 {model['model_type']} GPU加载成功!")
                return {
                    'success': True,
                    'model': model_obj,
                    'actual_config': {
                        'strategy': 'local_model_fallback_gpu',
                        'device': device,
                        'model_path': model['path'],
                        'model_type': model['model_type'],
                        'load_time': load_time,
                        'fallback_reason': f"原模型 {config.get('model_type', 'unknown')} 加载失败"
                    }
                }
        
        # 优先级3: 多进程CPU加载
        print(f"🔄 GPU加载失败，尝试多进程CPU加载备用模型 {model['model_type']}...")
        model_obj, status, load_time = _load_model_cpu_only(model['path'], timeout=30)
        
        if status == "success" and model_obj is not None:
            print(f"✅ 备用模型 {model['model_type']} CPU加载成功!")
            return {
                'success': True,
                'model': model_obj,
                'actual_config': {
                    'strategy': 'local_model_fallback_cpu',
                    'device': 'cpu',
                    'model_path': model['path'],
                    'model_type': model['model_type'],
                    'load_time': load_time,
                    'fallback_reason': f"原模型 {config.get('model_type', 'unknown')} 加载失败"
                }
            }
        else:
            print(f"❌ 备用模型 {model['model_type']} 所有加载方式都失败: {status}")
            continue
    
    raise Exception("所有本地模型都加载失败")


def _execute_download_strategy(config):
    """执行下载模型策略"""
    model_name = config['model_name']
    device = config['device']
    
    print(f"🌐 下载并加载模型: {model_name}")
    print(f"🎯 目标设备: {device}")
    
    model, status, load_time = load_model_with_timeout(model_name, timeout=60)
    
    if status == "success" and model is not None:
        return {
            'success': True,
            'model': model,
            'actual_config': {
                'strategy': 'download_model',
                'device': device,
                'model_name': model_name,
                'load_time': load_time
            }
        }
    else:
        raise Exception(f"模型下载失败: {status}")


def _execute_tfidf_strategy(config):
    """执行TF-IDF策略"""
    print("📊 初始化TF-IDF模型...")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as _sk_cos_sim
        
        tfidf_model = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        
        return {
            'success': True,
            'model': None,  # TF-IDF在别处处理
            'actual_config': {
                'strategy': 'tfidf',
                'device': 'cpu',
                'model_type': 'tfidf'
            }
        }
    except ImportError as e:
        raise Exception(f"TF-IDF依赖不可用: {e}")


def _execute_minimal_strategy(config):
    """执行最小化策略"""
    print("⚠️ 最小化启动模式...")
    print("💡 仅提供基础文本处理功能")
    
    return {
        'success': True,
        'model': None,
        'actual_config': {
            'strategy': 'minimal',
            'device': 'cpu',
            'model_type': 'basic'
        }
    }


def run_intelligent_startup():
    """
    运行完整的智能启动流程
    返回: dict with startup results
    """
    try:
        # 1. 获取启动策略
        strategy_result = intelligent_startup_strategy()
        
        # 2. 执行策略
        execution_result = execute_startup_strategy(strategy_result)
        
        return {
            'success': execution_result['success'],
            'strategy_result': strategy_result,
            'execution_result': execution_result
        }
        
    except Exception as e:
        print(f"❌ 智能启动流程失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'strategy_result': None,
            'execution_result': None
        } 