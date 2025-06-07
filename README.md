# 🗑️ 智能垃圾分类系统

基于深度学习的垃圾图像智能分类Web应用，采用ResNet18架构实现四类垃圾的精准识别。

## 🚀 在线体验

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 📋 功能特性

- 🤖 **AI智能识别**：基于ResNet18深度学习模型
- 🎯 **四类分类**：生物垃圾、可回收垃圾、有害垃圾、其他垃圾
- 📊 **高精度**：训练精度97.39%，测试精度92.5%
- 🌐 **Web界面**：简洁美观的用户界面
- 📈 **可视化**：概率分布图和置信度显示
- 💡 **智能建议**：根据分类结果提供垃圾处理建议

## 🛠️ 技术栈

- **深度学习**：PyTorch + ResNet18
- **Web框架**：Streamlit
- **图像处理**：PIL/Pillow + torchvision
- **数据可视化**：Plotly
- **部署**：Streamlit Community Cloud

## 📦 本地运行

### 环境要求

- Python 3.7+
- 至少2GB内存

### 快速启动

1. **克隆仓库**
```bash
git clone https://github.com/your-username/garbage-classification.git
cd garbage-classification
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **启动应用**
```bash
streamlit run web_app.py
```

或使用一键启动脚本：
```bash
./start_web_app.sh
```

4. **访问应用**
打开浏览器访问：http://localhost:8501

## 🎯 使用方法

1. 上传垃圾图片（支持JPG、PNG格式）
2. 等待AI模型分析
3. 查看分类结果和置信度
4. 参考处理建议进行垃圾分类

## 📊 模型性能

| 指标 | 值 |
|------|------|
| 训练精度 | 97.39% |
| 测试精度 | 92.5% |
| 生物垃圾识别率 | 94.2% |
| 可回收垃圾识别率 | 91.8% |
| 有害垃圾识别率 | 93.1% |
| 其他垃圾识别率 | 90.6% |

## 📁 项目结构

```
├── web_app.py              # Streamlit Web应用主文件
├── demo.py                 # 命令行演示脚本
├── test.py                 # 模型测试脚本
├── train.py                # 模型训练脚本
├── best_improved_model.pth # 训练好的模型文件
├── requirements.txt        # Python依赖包
├── start_web_app.sh       # 一键启动脚本
├── WEB_APP_README.md      # Web应用详细说明
└── test/                  # 测试数据集
    ├── biological/        # 生物垃圾测试图片
    ├── recyclable/        # 可回收垃圾测试图片
    ├── hazardous_waste/   # 有害垃圾测试图片
    └── other/             # 其他垃圾测试图片
```

## 🔧 开发指南

### 命令行测试

```bash
# 测试单张图片
python demo.py test/biological/biological938.jpg

# 批量测试
python test.py
```

### 模型训练

```bash
python train.py
```

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

MIT License

## 🙏 致谢

- PyTorch团队提供优秀的深度学习框架
- Streamlit团队提供简单易用的Web应用框架
- 所有贡献测试数据的志愿者们

---

⭐ 如果这个项目对你有帮助，请给个Star支持一下！ 