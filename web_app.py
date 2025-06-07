#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
垃圾分类Web应用
基于Streamlit的图片上传和分类检测系统
"""

import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# 页面配置
st.set_page_config(
    page_title="智能垃圾分类系统",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #28a745;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class ImprovedGarbageClassificationCNN(nn.Module):
    """垃圾分类CNN模型"""
    
    def __init__(self, num_classes=4, use_pretrained=True, model_name='resnet18'):
        super(ImprovedGarbageClassificationCNN, self).__init__()
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            if model_name == 'resnet18':
                # 创建ResNet18模型结构
                self.backbone = models.resnet18(pretrained=False)
                # 加载本地ResNet18权重
                if os.path.exists('resnet18-f37072fd.pth'):
                    state_dict = torch.load('resnet18-f37072fd.pth', map_location='cpu')
                    self.backbone.load_state_dict(state_dict)
                # 冻结大部分层
                for param in list(self.backbone.parameters())[:-10]:
                    param.requires_grad = False
                num_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            # 自定义分类头
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(num_features),
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


@st.cache_resource
def load_model():
    """加载垃圾分类模型（使用缓存避免重复加载）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = ImprovedGarbageClassificationCNN(
            num_classes=4,
            use_pretrained=True,
            model_name='resnet18'
        )
        
        # 检查模型文件
        model_path = 'best_improved_model.pth'
        if not os.path.exists(model_path):
            st.error(f"❌ 模型文件不存在: {model_path}")
            return None, None
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        return model, device
        
    except Exception as e:
        st.error(f"❌ 加载模型失败: {e}")
        return None, None


def preprocess_image(image):
    """图像预处理"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 确保图像是RGB格式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)


def predict_image(model, device, image):
    """预测图像类别"""
    class_names = ['biological', 'hazardous_waste', 'others', 'recyclable']
    class_names_chinese = ['生物垃圾', '有害垃圾', '其他垃圾', '可回收垃圾']
    class_descriptions = [
        '🥬 食物残渣、果皮等有机垃圾',
        '🔋 电池、化学品等危险废物', 
        '🗑️不可回收的一般垃圾',
        '♻️ 塑料、玻璃、纸张、金属等可回收物品'
    ]
    
    # 置信度阈值
    CONFIDENCE_THRESHOLD = 0.95
    
    try:
        # 预处理图像
        input_tensor = preprocess_image(image).to(device)
        
        # 模型预测
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        # 判断是否为垃圾（置信度阈值判断）
        if confidence < CONFIDENCE_THRESHOLD:
            # 置信度低于阈值，判断为非垃圾
            result = {
                'is_garbage': False,
                'predicted_class': -1,
                'class_name': 'not_garbage',
                'class_name_chinese': '不是垃圾',
                'class_description': '🚫 此图片不是垃圾，请上传垃圾图片进行分类',
                'confidence': confidence,
                'max_confidence': confidence,
                'probabilities': probabilities.cpu().numpy(),
                'all_classes': class_names_chinese,
                'all_descriptions': class_descriptions,
                'threshold': CONFIDENCE_THRESHOLD
            }
        else:
            # 置信度足够高，判断为垃圾
            result = {
                'is_garbage': True,
                'predicted_class': predicted_class,
                'class_name': class_names[predicted_class],
                'class_name_chinese': class_names_chinese[predicted_class],
                'class_description': class_descriptions[predicted_class],
                'confidence': confidence,
                'max_confidence': confidence,
                'probabilities': probabilities.cpu().numpy(),
                'all_classes': class_names_chinese,
                'all_descriptions': class_descriptions,
                'threshold': CONFIDENCE_THRESHOLD
            }
        
        return result
        
    except Exception as e:
        st.error(f"❌ 预测失败: {e}")
        return None


def create_probability_chart(result):
    """创建概率分布图表"""
    try:
        if not result['is_garbage']:
            # 如果不是垃圾，显示所有类别的低概率
            class_names_chinese = result['all_classes']
            probabilities = result['probabilities']
            colors = ['#FFB6C1', '#FFB6C1', '#FFB6C1', '#FFB6C1']  # 浅色表示低置信度
            title_text = f"各类别概率分布 (最高置信度: {result['max_confidence']:.2%}, 阈值: {result['threshold']:.0%})"
        else:
            # 如果是垃圾，正常显示
            class_names_chinese = result['all_classes']
            probabilities = result['probabilities']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            title_text = f"各类别概率分布 (置信度: {result['confidence']:.2%})"
        
        fig = go.Figure(data=[
            go.Bar(
                y=class_names_chinese,
                x=probabilities,
                orientation='h',
                marker=dict(color=colors),
                text=[f'{p:.2%}' for p in probabilities],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=title_text,
            xaxis_title="概率",
            yaxis_title="垃圾类别",
            height=300,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            yaxis=dict(showgrid=False)
        )
        
        return fig
        
    except Exception as e:
        print(f"创建图表时出错: {e}")
        return None


def main():
    # 页面标题
    st.markdown('<h1 class="main-header">🗑️ 智能垃圾分类系统</h1>', unsafe_allow_html=True)
    
    # 侧边栏信息
    with st.sidebar:
        st.markdown("### 📋 系统信息")
        st.markdown("""
        - **模型**: ResNet18 + 自定义分类头
        - **准确率**: 97.39%
        - **分类类别**: 4类
        - **支持格式**: JPG, PNG, JPEG
        - **置信度阈值**: 95% (低于此值判断为非垃圾)
        """)
        
        st.markdown("### 🎯 分类类别")
        st.markdown("""
        - 🥬 **生物垃圾**: 食物残渣、果皮
        - 🔋 **有害垃圾**: 电池、化学品  
        - 🗑️ **其他垃圾**: 不可回收垃圾
        - ♻️ **可回收垃圾**: 塑料、玻璃、纸张
        """)
        
        st.markdown("### 💡 使用说明")
        st.markdown("""
        1. 上传图片文件
        2. 等待AI分析
        3. 查看分类结果
        4. 参考置信度判断
        """)
    
    # 主要内容区域
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 上传图片")
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "选择要分类的垃圾图片",
            type=['jpg', 'jpeg', 'png'],
            help="支持JPG、PNG格式，建议图片清晰且垃圾物品居中"
        )
        
        if uploaded_file is not None:
            # 显示上传的图片
            image = Image.open(uploaded_file)
            st.image(image, caption="上传的图片", use_container_width=True)
            
            # 图片信息
            st.markdown(f"""
            <div class="info-box">
            📊 <strong>图片信息:</strong><br>
            • 文件名: {uploaded_file.name}<br>
            • 尺寸: {image.size[0]} × {image.size[1]} 像素<br>
            • 格式: {image.format}<br>
            • 模式: {image.mode}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🤖 AI分析结果")
        
        if uploaded_file is not None:
            # 加载模型
            with st.spinner("🔄 正在加载AI模型..."):
                model, device = load_model()
            
            if model is not None:
                # 进行预测
                with st.spinner("🧠 AI正在分析图片..."):
                    result = predict_image(model, device, image)
                
                if result is not None:
                    # 显示预测结果
                    confidence = result['confidence']
                    
                    if not result['is_garbage']:
                        # 不是垃圾的情况
                        st.markdown(f"""
                        <div class="result-box">
                        <h3>🚫 识别结果</h3>
                        <h2 style="color: #dc3545; margin: 0.5rem 0;">
                            {result['class_description']}
                        </h2>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                            <strong>判断:</strong> {result['class_name_chinese']}
                        </p>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                            <strong>最高置信度:</strong> 
                            <span class="confidence-low">
                                🔴 {confidence:.2%} (低于阈值 {result['threshold']:.0%})
                            </span>
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 显示概率分布图
                        st.plotly_chart(
                            create_probability_chart(result), 
                            use_container_width=True
                        )
                        
                        # 提示信息
                        st.info("💡 **提示**: 请上传清晰的垃圾图片进行分类检测。本系统专门识别生物垃圾、可回收垃圾、有害垃圾和其他垃圾。")
                        
                    else:
                        # 是垃圾的情况（原有逻辑）
                        # 根据置信度设置颜色
                        if confidence >= 0.98:
                            confidence_class = "confidence-high"
                            confidence_icon = "🟢"
                        elif confidence >= 0.96:
                            confidence_class = "confidence-medium" 
                            confidence_icon = "🟡"
                        else:
                            confidence_class = "confidence-low"
                            confidence_icon = "🔴"
                        
                        st.markdown(f"""
                        <div class="result-box">
                        <h3>🎯 预测结果</h3>
                        <h2 style="color: #2E8B57; margin: 0.5rem 0;">
                            {result['class_description']}
                        </h2>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                            <strong>类别:</strong> {result['class_name_chinese']} ({result['class_name']})
                        </p>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                            <strong>置信度:</strong> 
                            <span class="{confidence_class}">
                                {confidence_icon} {confidence:.2%}
                            </span>
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 显示概率分布图
                        st.plotly_chart(
                            create_probability_chart(result), 
                            use_container_width=True
                        )
                        
                        # 处理建议
                        st.markdown("### 💡 处理建议")
                        if result['predicted_class'] == 0:  # 生物垃圾
                            st.success("🥬 请投入**绿色**垃圾桶（生物垃圾），可用于堆肥处理。")
                        elif result['predicted_class'] == 1:  # 有害垃圾
                            st.error("🔋 请投入**红色**垃圾桶（有害垃圾），需要专门处理！")
                        elif result['predicted_class'] == 2:  # 其他垃圾
                            st.info("🗑️ 请投入**灰色**垃圾桶（其他垃圾），进行焚烧处理。")
                        else:  # 可回收垃圾
                            st.success("♻️ 请投入**蓝色**垃圾桶（可回收垃圾），可循环利用！")
                        
                        # 置信度解释
                        if confidence < 0.96:
                            st.warning("⚠️ 置信度相对较低，建议人工确认分类结果。")
                        elif confidence < 0.98:
                            st.info("ℹ️ 置信度良好，结果基本可信。")
                        else:
                            st.success("✅ 置信度极高，结果非常可信！")
        
        else:
            st.info("👆 请在左侧上传图片开始分析")
    
    # 页面底部信息
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📈 模型性能")
        st.metric("准确率", "97.39%", delta="优秀")
    
    with col2:
        st.markdown("### 🎯 测试结果") 
        st.metric("实际测试", "92.5%", delta="良好")
    
    with col3:
        st.markdown("### 📊 数据规模")
        st.metric("测试图片", "765张", delta="充足")
    
    # 使用提示
    with st.expander("📖 详细使用说明"):
        st.markdown("""
        #### 🔍 如何获得最佳分类效果:
        
        1. **图片质量**: 选择清晰、光线良好的图片
        2. **物品居中**: 确保垃圾物品在图片中央，占据主要位置
        3. **背景简洁**: 避免复杂背景干扰
        4. **单一物品**: 一次只分类一个主要物品
        5. **合适角度**: 选择能清楚看到物品特征的角度
        
        #### 🎯 分类标准:
        
        - **生物垃圾**: 易腐烂的有机物，如剩菜剩饭、瓜果皮核
        - **有害垃圾**: 对环境有害的物品，如电池、灯管、药品
        - **可回收垃圾**: 可循环利用的物品，如塑料瓶、纸张、金属
        - **其他垃圾**: 除上述三类外的其他废弃物
        
        #### ⚠️ 注意事项:
        
        - AI预测仅供参考，最终分类请结合实际情况
        - 如果置信度较低，建议咨询相关部门
        - 本系统基于训练数据，可能对新型垃圾识别效果有限
        """)


if __name__ == "__main__":
    main() 