import os
# 禁用OpenMP以避免初始化错误
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 添加字体警告过滤
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
warnings.filterwarnings('ignore', message='.*does not have a glyph.*')

# 设置PyTorch使用单线程，避免OpenMP冲突
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

class ImprovedGarbageClassificationCNN(nn.Module):
    """改进版垃圾分类CNN模型 - 与训练时保持一致"""
    
    def __init__(self, num_classes=4, use_pretrained=True, model_name='resnet18'):
        super(ImprovedGarbageClassificationCNN, self).__init__()
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            if model_name == 'resnet18':
                # 创建ResNet18模型结构
                self.backbone = models.resnet18(pretrained=False)
                # 加载本地ResNet18权重
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


class GarbageClassificationTester:
    """
    垃圾分类模型测试类 - PyTorch版本
    
    用于加载训练好的PyTorch模型，随机选择测试图像进行分类预测，
    并可视化预测结果与真实标签的对比
    """
    
    def __init__(self, model_path='improved_garbage_classification_pytorch.pth', test_dir='test'):
        """
        初始化测试器
        
        Args:
            model_path (str): 训练好的模型文件路径
            test_dir (str): 测试数据目录
        """
        self.model_path = model_path
        self.test_dir = test_dir
        self.img_size = (224, 224)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 垃圾分类类别（需要与训练时保持一致）
        self.class_names = ['biological', 'hazardous_waste', 'others', 'recyclable']
        self.class_names_chinese = ['生物垃圾', '有害垃圾', '其他垃圾', '可回收垃圾']
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 加载模型
        self.load_model()
        
        # 获取测试图像路径
        self.get_test_images()
    
    def load_model(self):
        """加载训练好的PyTorch模型"""
        try:
            # 创建模型实例
            self.model = ImprovedGarbageClassificationCNN(
                num_classes=len(self.class_names),
                use_pretrained=True,
                model_name='resnet18'
            )
            
            # 检查模型文件路径
            model_files = [self.model_path, 'best_improved_model.pth']
            loaded_model_path = None
            
            for model_file in model_files:
                if os.path.exists(model_file):
                    loaded_model_path = model_file
                    break
            
            if loaded_model_path is None:
                print(f"模型文件不存在: {self.model_path}")
                print(f"备选模型文件也不存在: best_improved_model.pth")
                print("请确保运行过训练程序生成模型文件")
                return False
            
            # 加载模型权重
            checkpoint = torch.load(loaded_model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"成功加载PyTorch模型: {loaded_model_path} (从checkpoint)")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"成功加载PyTorch模型: {loaded_model_path} (直接权重)")
            
            # 移动模型到设备并设置为评估模式
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"使用设备: {self.device}")
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("请确保模型文件存在且路径正确")
            return False
        return True
    
    def get_test_images(self):
        """获取所有测试图像的路径和真实标签"""
        self.test_images = []
        
        if not os.path.exists(self.test_dir):
            print(f"测试目录不存在: {self.test_dir}")
            return
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.test_dir, class_name)
            if os.path.exists(class_dir):
                # 递归搜索所有子目录中的图像文件
                self._find_images_recursive(class_dir, class_idx, class_name)
        
        print(f"找到 {len(self.test_images)} 张测试图像")
        for i, class_name in enumerate(self.class_names_chinese):
            count = sum(1 for img in self.test_images if img['true_class'] == i)
            print(f"   {class_name}: {count} 张")
    
    def _find_images_recursive(self, directory, class_idx, class_name):
        """
        递归搜索目录中的图像文件
        
        Args:
            directory (str): 要搜索的目录
            class_idx (int): 类别索引
            class_name (str): 类别名称
        """
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            
            if os.path.isfile(item_path):
                # 如果是文件，检查是否为图像文件
                if item.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.test_images.append({
                        'path': item_path,
                        'true_class': class_idx,
                        'true_class_name': class_name,
                        'true_class_chinese': self.class_names_chinese[class_idx]
                    })
            elif os.path.isdir(item_path):
                # 如果是目录，递归搜索
                self._find_images_recursive(item_path, class_idx, class_name)
    
    def preprocess_image(self, img_path):
        """
        预处理单张图像 - PyTorch版本
        
        Args:
            img_path (str): 图像文件路径
            
        Returns:
            torch.Tensor: 预处理后的图像张量
        """
        try:
            # 加载图像并转换为RGB
            img = Image.open(img_path).convert('RGB')
            # 应用预处理变换
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            return img_tensor
        except Exception as e:
            print(f"图像预处理失败 {img_path}: {e}")
            return None
    
    def predict_single_image(self, img_path):
        """
        对单张图像进行预测 - PyTorch版本
        
        Args:
            img_path (str): 图像文件路径
            
        Returns:
            tuple: (预测类别索引, 预测概率数组)
        """
        img_tensor = self.preprocess_image(img_path)
        if img_tensor is None:
            return None, None
        
        # 进行预测
        with torch.no_grad():
            predictions = self.model(img_tensor)
            probabilities = F.softmax(predictions, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
        
        return predicted_class, probabilities
    
    def test_random_images(self, num_images=12):
        """
        随机选择图像进行测试
        
        Args:
            num_images (int): 要测试的图像数量
        """
        if len(self.test_images) == 0:
            print("没有找到测试图像")
            return
        
        # 随机选择图像
        selected_images = random.sample(self.test_images, min(num_images, len(self.test_images)))
        
        print(f"开始测试 {len(selected_images)} 张随机图像...")
        
        # 存储结果
        results = []
        correct_predictions = 0
        
        for i, img_info in enumerate(selected_images):
            # 进行预测
            predicted_class, probabilities = self.predict_single_image(img_info['path'])
            
            if predicted_class is not None:
                is_correct = predicted_class == img_info['true_class']
                if is_correct:
                    correct_predictions += 1
                
                results.append({
                    'img_info': img_info,
                    'predicted_class': predicted_class,
                    'predicted_class_name': self.class_names[predicted_class],
                    'predicted_class_chinese': self.class_names_chinese[predicted_class],
                    'probabilities': probabilities,
                    'confidence': probabilities[predicted_class],
                    'is_correct': is_correct
                })
        
        # 计算准确率
        accuracy = correct_predictions / len(results) if results else 0
        print(f"\n测试结果统计:")
        print(f"   总测试图像: {len(results)}")
        print(f"   正确预测: {correct_predictions}")
        print(f"   错误预测: {len(results) - correct_predictions}")
        print(f"   准确率: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # 可视化结果
        self.visualize_results(results)
        
        return results
    
    def visualize_results(self, results):
        """
        可视化预测结果
        
        Args:
            results (list): 预测结果列表
        """
        if not results:
            print("没有结果可以可视化")
            return
        
        # 计算子图布局
        num_images = len(results)
        cols = 4
        rows = (num_images + cols - 1) // cols
        
        # 创建图形
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, result in enumerate(results):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # 加载并显示图像
            img = Image.open(result['img_info']['path']).convert('RGB')
            ax.imshow(img)
            
            # 设置标题
            true_label = result['img_info']['true_class_chinese']
            pred_label = result['predicted_class_chinese']
            confidence = result['confidence']
            
            if result['is_correct']:
                title_color = 'green'
                status = '[正确]'
            else:
                title_color = 'red'
                status = '[错误]'
            
            title = f"{status} 真实: {true_label}\n预测: {pred_label}\n置信度: {confidence:.3f}"
            ax.set_title(title, fontsize=12, color=title_color, fontweight='bold')
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(num_images, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 绘制置信度分布图
        self.plot_confidence_distribution(results)
    
    def plot_confidence_distribution(self, results):
        """
        绘制预测置信度分布图
        
        Args:
            results (list): 预测结果列表
        """
        correct_confidences = [r['confidence'] for r in results if r['is_correct']]
        incorrect_confidences = [r['confidence'] for r in results if not r['is_correct']]
        
        plt.figure(figsize=(12, 6))
        
        # 子图1: 置信度直方图
        plt.subplot(1, 2, 1)
        if correct_confidences:
            plt.hist(correct_confidences, bins=10, alpha=0.7, label=f'正确预测 ({len(correct_confidences)})', color='green')
        if incorrect_confidences:
            plt.hist(incorrect_confidences, bins=10, alpha=0.7, label=f'错误预测 ({len(incorrect_confidences)})', color='red')
        plt.xlabel('预测置信度')
        plt.ylabel('频次')
        plt.title('预测置信度分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 各类别预测准确率
        plt.subplot(1, 2, 2)
        class_accuracies = []
        class_counts = []
        
        for i, class_name in enumerate(self.class_names_chinese):
            class_results = [r for r in results if r['img_info']['true_class'] == i]
            if class_results:
                correct_count = sum(1 for r in class_results if r['is_correct'])
                accuracy = correct_count / len(class_results)
                class_accuracies.append(accuracy)
                class_counts.append(len(class_results))
            else:
                class_accuracies.append(0)
                class_counts.append(0)
        
        bars = plt.bar(self.class_names_chinese, class_accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.ylabel('准确率')
        plt.title('各类别预测准确率')
        plt.xticks(rotation=45)
        
        # 在柱状图上添加数值标签
        for i, (bar, acc, count) in enumerate(zip(bars, class_accuracies, class_counts)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.2f}\n({count}张)', ha='center', va='bottom', fontsize=10)
        
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def test_specific_class(self, class_name, num_images=8):
        """
        测试特定类别的图像
        
        Args:
            class_name (str): 类别名称（英文）
            num_images (int): 要测试的图像数量
        """
        if class_name not in self.class_names:
            print(f"无效的类别名称: {class_name}")
            print(f"可用类别: {self.class_names}")
            return
        
        # 筛选特定类别的图像
        class_images = [img for img in self.test_images if img['true_class_name'] == class_name]
        
        if not class_images:
            print(f"没有找到类别 '{class_name}' 的测试图像")
            return
        
        # 随机选择图像
        selected_images = random.sample(class_images, min(num_images, len(class_images)))
        
        print(f"测试类别 '{self.class_names_chinese[self.class_names.index(class_name)]}' 的 {len(selected_images)} 张图像...")
        
        # 进行测试
        results = []
        correct_predictions = 0
        
        for img_info in selected_images:
            predicted_class, probabilities = self.predict_single_image(img_info['path'])
            if predicted_class is not None:
                is_correct = predicted_class == img_info['true_class']
                if is_correct:
                    correct_predictions += 1
                    
                results.append({
                    'img_info': img_info,
                    'predicted_class': predicted_class,
                    'predicted_class_name': self.class_names[predicted_class],
                    'predicted_class_chinese': self.class_names_chinese[predicted_class],
                    'probabilities': probabilities,
                    'confidence': probabilities[predicted_class],
                    'is_correct': is_correct
                })
        
        # 计算并显示该类别的准确率
        accuracy = correct_predictions / len(results) if results else 0
        print(f"   准确率: {accuracy:.3f} ({accuracy*100:.1f}%) - 正确: {correct_predictions}/{len(results)}")
        
        return results


def main():
    """
    主函数：执行改进版垃圾分类PyTorch模型测试
    """
    print("=" * 60)
    print("改进版垃圾分类CNN模型测试程序 - PyTorch版本")
    print("=" * 60)
    
    # 创建测试器实例
    tester = GarbageClassificationTester(
        model_path='improved_garbage_classification_pytorch.pth',
        test_dir='test'
    )
    
    # 检查模型和数据是否加载成功
    if not hasattr(tester, 'model') or len(tester.test_images) == 0:
        print("初始化失败，请检查模型文件和测试数据目录")
        print("模型文件应该是: improved_garbage_classification_pytorch.pth")
        print("备选模型文件: best_improved_model.pth")
        return
    
    print("\n" + "="*50)
    print("开始随机图像测试...")
    print("="*50)
    
    # 测试随机图像
    results = tester.test_random_images(num_images=20)
    
    print("\n" + "="*50)
    print("各类别测试结果...")
    print("="*50)
    
    # 测试每个类别并统计
    all_class_results = []
    total_correct = 0
    total_tested = 0
    
    for class_name in tester.class_names:
        class_results = tester.test_specific_class(class_name, num_images=10)
        if class_results:
            all_class_results.extend(class_results)
            class_correct = sum(1 for r in class_results if r['is_correct'])
            total_correct += class_correct
            total_tested += len(class_results)
    
    # 显示总体统计
    print("\n" + "="*50)
    print("改进版PyTorch模型测试统计")
    print("="*50)
    
    if total_tested > 0:
        overall_accuracy = total_correct / total_tested
        print(f"总体测试结果:")
        print(f"   总测试图像: {total_tested}")
        print(f"   正确预测: {total_correct}")
        print(f"   错误预测: {total_tested - total_correct}")
        print(f"   测试准确率: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
        
        # 各类别详细统计
        print(f"\n各类别测试准确率详情:")
        for i, class_name in enumerate(tester.class_names_chinese):
            class_results = [r for r in all_class_results if r['img_info']['true_class'] == i]
            if class_results:
                correct_count = sum(1 for r in class_results if r['is_correct'])
                accuracy = correct_count / len(class_results)
                print(f"   {class_name}: {accuracy:.3f} ({accuracy*100:.1f}%) - {correct_count}/{len(class_results)}")
            else:
                print(f"   {class_name}: 无测试数据")
    
    print("\nPyTorch模型测试完成！生成的文件:")
    print("- test_results.png: 随机图像测试结果")
    print("- confidence_analysis.png: 置信度分析图")



if __name__ == "__main__":
    main() 