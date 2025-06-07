#!/bin/bash

echo "启动垃圾分类Web应用..."
echo "正在检查依赖项..."

# 检查Python是否安装
if ! command -v python &> /dev/null; then
    echo "错误：Python未安装。请先安装Python。"
    exit 1
fi

# 检查pip是否安装
if ! command -v pip &> /dev/null; then
    echo "错误：pip未安装。请先安装pip。"
    exit 1
fi

# 检查streamlit是否安装
if ! python -c "import streamlit" &> /dev/null; then
    echo "正在安装依赖项..."
    pip install -r requirements.txt
fi

echo "启动Web应用..."
echo "应用将在浏览器中自动打开："
echo "本地地址：http://localhost:8501"
echo ""
echo "使用说明："
echo "1. 在浏览器中打开上述地址"
echo "2. 上传垃圾图片（支持JPG、PNG、JPEG格式）"
echo "3. 查看分类结果和置信度"
echo ""
echo "按 Ctrl+C 停止应用"
echo "================================"

# 启动streamlit应用
streamlit run web_app.py 