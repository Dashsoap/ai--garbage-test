#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
垃圾分类模型演示脚本
快速测试单张图片的分类结果
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse

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


class GarbageClassifier:
    """垃圾分类器"""
    
    def __init__(self, model_path='best_improved_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['biological', 'hazardous_waste', 'others', 'recyclable']
        self.class_names_chinese = ['生物垃圾', '有害垃圾', '其他垃圾', '可回收垃圾']
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 加载模型
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """加载训练好的模型"""
        try:
            model = ImprovedGarbageClassificationCNN(
                num_classes=len(self.class_names),
                use_pretrained=True,
                model_name='resnet18'
            )
            
            # 检查模型文件
            if not os.path.exists(model_path):
                print(f"模型文件不存在: {model_path}")
                return None
            
            # 加载模型权重
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            print(f"✅ 成功加载模型: {model_path}")
            print(f"🖥️  使用设备: {self.device}")
            return model
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return None
    
    def predict(self, image_path):
        """预测单张图片"""
        if self.model is None:
            print("❌ 模型未成功加载")
            return None
        
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 模型预测
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()
            
            # 返回结果
            result = {
                'predicted_class': predicted_class,
                'class_name': self.class_names[predicted_class],
                'class_name_chinese': self.class_names_chinese[predicted_class],
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy()
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None
    
    def print_result(self, result, image_path):
        """打印预测结果"""
        print(f"\n📸 图片: {image_path}")
        print("=" * 50)
        print(f"🎯 预测类别: {result['class_name_chinese']} ({result['class_name']})")
        print(f"🔥 置信度: {result['confidence']:.2%}")
        print("\n📊 各类别概率:")
        
        for i, (name_en, name_cn) in enumerate(zip(self.class_names, self.class_names_chinese)):
            prob = result['probabilities'][i]
            bar = "█" * int(prob * 20)  # 简单的进度条
            print(f"   {name_cn:6} | {prob:.2%} {bar}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='垃圾分类模型演示')
    parser.add_argument('image_path', help='要分类的图片路径')
    parser.add_argument('--model', default='best_improved_model.pth', 
                       help='模型文件路径 (默认: best_improved_model.pth)')
    
    args = parser.parse_args()
    
    print("🗑️  垃圾分类模型演示")
    print("=" * 50)
    
    # 检查图片文件是否存在
    if not os.path.exists(args.image_path):
        print(f"❌ 图片文件不存在: {args.image_path}")
        return
    
    # 创建分类器
    classifier = GarbageClassifier(args.model)
    
    # 进行预测
    result = classifier.predict(args.image_path)
    
    if result:
        classifier.print_result(result, args.image_path)
    else:
        print("❌ 预测失败")


if __name__ == "__main__":
    main() 