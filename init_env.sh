#!/bin/bash
# 设置代理并安装依赖

# 设置代理
export http_proxy="http://192.168.10.100:10808"
export https_proxy="http://192.168.10.100:10808"
echo "HTTP 和 HTTPS 代理已设置为 192.168.10.100:10808。"

# 安装依赖
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "依赖已根据 requirements.txt 安装完成。"
else
    echo "未找到 requirements.txt 文件，跳过安装依赖。"
fi

echo "初始化完成，虚拟环境已就绪。"
