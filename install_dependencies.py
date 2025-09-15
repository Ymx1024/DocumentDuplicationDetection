#!/usr/bin/env python3
"""
依赖安装脚本
解决文本提取和GPU识别的依赖问题
"""

import subprocess
import sys
import os

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n{description}...")
    print(f"执行命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ 成功: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def detect_cuda_tag():
    """检测CUDA版本，返回适配的tag: 'cu121' / 'cu118' / None(CPU)"""
    # 优先用 nvidia-smi
    try:
        out = subprocess.check_output('nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits', shell=True, text=True).strip()
        if out:
            ver = out.split('\n')[0].strip()
            if ver.startswith('12.1'):
                return 'cu121'
            if ver.startswith('11.8'):
                return 'cu118'
            # 常见兼容：12.2/12.3 也可用 cu121 轮子
            if ver.startswith('12.'):
                return 'cu121'
    except Exception:
        pass
    # 其次用 nvcc
    try:
        out = subprocess.check_output('nvcc --version', shell=True, text=True).strip()
        if 'release 12.1' in out:
            return 'cu121'
        if 'release 11.8' in out:
            return 'cu118'
        if 'release 12.' in out:
            return 'cu121'
    except Exception:
        pass
    return None

def install_pytorch():
    """安装匹配CUDA的PyTorch；若无CUDA则安装CPU版。并固定到有CUDA轮子的稳定版本，避免2.8 CPU-only。"""
    cuda_tag = detect_cuda_tag()
    # 选定稳定版本（有对应CUDA轮子）
    torch_ver = '2.3.1'
    tv_ver = '0.18.1'
    ta_ver = '2.3.1'
    if cuda_tag == 'cu121':
        index = 'https://download.pytorch.org/whl/cu121'
        cmd = f"pip install torch=={torch_ver} torchvision=={tv_ver} torchaudio=={ta_ver} --index-url {index}"
        return run_command(cmd, f"安装PyTorch CUDA({cuda_tag}) 版本 {torch_ver}")
    elif cuda_tag == 'cu118':
        index = 'https://download.pytorch.org/whl/cu118'
        cmd = f"pip install torch=={torch_ver} torchvision=={tv_ver} torchaudio=={ta_ver} --index-url {index}"
        return run_command(cmd, f"安装PyTorch CUDA({cuda_tag}) 版本 {torch_ver}")
    else:
        # 无CUDA，安装CPU版（避免2.8，固定到2.3.1）
        cmd = f"pip install torch=={torch_ver} torchvision=={tv_ver} torchaudio=={ta_ver}"
        return run_command(cmd, "安装PyTorch CPU 版本 2.3.1")

def main():
    print("🚀 开始安装和更新依赖...")
    
    # 升级pip
    run_command("python -m pip install --upgrade pip", "升级pip")
    
    # 卸载有问题的包
    print("\n🗑️ 卸载有问题的包...")
    run_command("pip uninstall -y xlrd textract torch torchvision torchaudio", "卸载旧版本xlrd/textract/torch系列")
    
    # 安装/升级核心依赖
    print("\n📦 安装核心依赖...")
    install_pytorch()
    run_command("pip install --upgrade sentence-transformers", "安装sentence-transformers")
    run_command("pip install --upgrade pandas openpyxl", "安装pandas和openpyxl")
    run_command("pip install --upgrade xlrd>=2.0.1", "安装新版本xlrd")
    run_command("pip install --upgrade python-docx", "安装python-docx")
    run_command("pip install --upgrade pdfminer.six", "安装pdfminer.six")
    run_command("pip install --upgrade pyyaml", "安装pyyaml")
    run_command("pip install --upgrade nltk", "安装nltk")
    run_command("pip install --upgrade flask", "安装flask")
    run_command("pip install --upgrade scikit-learn", "安装scikit-learn（TF-IDF支持）")
    
    # 安装文本提取工具
    print("\n📄 安装文本提取工具...")
    print("注意：如需DOC旧格式兜底解析，可选安装 textract 1.6.3")
    run_command("pip install textract==1.6.3", "安装textract 1.6.3版本（可选）")
    print("安装替代文本提取库...")
    run_command("pip install PyPDF2", "安装PyPDF2")
    run_command("pip install pdfplumber", "安装pdfplumber")
    run_command("pip install docx2txt", "安装docx2txt")
    run_command("pip install python-pptx", "安装python-pptx")
    run_command("pip install chardet", "安装字符编码检测")
    run_command("pip install xlwings", "安装xlwings（Excel处理备用方案）")
    run_command("pip install pyexcel", "安装pyexcel（跨平台Excel处理）")
    run_command("pip install pyexcel-xls", "安装pyexcel-xls（xls文件支持）")
    run_command("pip install xlutils", "安装xlutils（Excel工具集）")
    run_command("pip install psutil", "安装psutil（系统监控）")
    
    # Linux(Ubuntu) 系统依赖（处理.doc/.ppt等老格式）
    if os.name != 'nt':
        print("\n🐧 安装Linux系统依赖（antiword/catdoc(包含catppt)）...")
        run_command("sudo apt-get update -y", "更新apt索引")
        run_command("sudo apt-get install -y antiword catdoc", "安装antiword/catdoc/catppt")
        run_command("sudo apt-get install -y libreoffice", "安装LibreOffice（Excel文件处理）")
    else:
        print("\n🪟 Windows系统依赖...")
        print("注意：在Windows上，某些文本提取功能可能需要额外的系统工具")
        print("建议安装：")
        print("1. Microsoft Office 或 LibreOffice (用于处理DOC/DOCX文件)")
        print("2. Adobe Reader (用于处理PDF文件)")
    
    # 验证安装
    print("\n🔍 验证安装...")
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA版本: {torch.version.cuda}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
    
    try:
        import sentence_transformers
        print(f"✅ sentence-transformers版本: {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"❌ sentence-transformers导入失败: {e}")
    
    try:
        import pandas as pd
        print(f"✅ pandas版本: {pd.__version__}")
    except ImportError as e:
        print(f"❌ pandas导入失败: {e}")
    
    print("\n🎉 依赖安装完成！")
    print("\n📝 下一步：")
    print("1. 重启Python环境")
    print("2. 运行应用测试GPU识别")
    print("3. 如果仍有问题，请检查错误信息")

if __name__ == "__main__":
    main()
