import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, datasets, models
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ImprovedGarbageClassificationCNN(nn.Module):
    """
    改进版垃圾分类CNN模型 - PyTorch版本
    
    主要改进：
    1. 使用预训练模型（ResNet/EfficientNet）进行迁移学习
    2. 处理类别不平衡问题
    3. 更强的数据增强
    4. 更好的正则化策略
    """
    
    def __init__(self, num_classes=4, use_pretrained=True, model_name='resnet50'):
        """
        初始化改进版垃圾分类CNN模型
        
        Args:
            num_classes (int): 分类类别数，默认4
            use_pretrained (bool): 是否使用预训练模型
            model_name (str): 预训练模型名称
        """
        super(ImprovedGarbageClassificationCNN, self).__init__()
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            if model_name == 'resnet50':
                self.backbone = models.resnet50(pretrained=True)
                # 冻结前面的层
                for param in list(self.backbone.parameters())[:-20]:
                    param.requires_grad = False
                # 替换最后的分类层
                num_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()  # 移除原始分类层
                
            elif model_name == 'efficientnet_b0':
                self.backbone = models.efficientnet_b0(pretrained=True)
                # 冻结前面的层
                for param in list(self.backbone.parameters())[:-20]:
                    param.requires_grad = False
                # 替换最后的分类层
                num_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Identity()
                
            else:  # 默认使用resnet18
                self.backbone = models.resnet18(pretrained=True)
                for param in list(self.backbone.parameters())[:-10]:
                    param.requires_grad = False
                num_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            
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
            
        else:
            # 自定义CNN架构
            self.backbone = nn.Sequential(
                # 第一个卷积块
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
                
                # 第二个卷积块
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
                
                # 第三个卷积块
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
                
                # 第四个卷积块
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
                
                # 全局平均池化
                nn.AdaptiveAvgPool2d((1, 1))
            )
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        if self.use_pretrained:
            # 预训练模型的特征已经是展平的
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


class ImprovedGarbageClassificationTrainer:
    """改进版垃圾分类训练器"""
    
    def __init__(self, train_dir='train', test_dir='test', img_size=(224, 224), batch_size=32):
        """
        初始化训练器
        
        Args:
            train_dir (str): 训练数据目录
            test_dir (str): 测试数据目录
            img_size (tuple): 图像尺寸，默认(224, 224)
            batch_size (int): 批次大小，默认32
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 垃圾分类类别
        self.class_names = ['biological', 'hazardous_waste', 'others', 'recyclable']
        self.num_classes = len(self.class_names)
        
        print(f"初始化改进版垃圾分类CNN模型 - PyTorch版本")
        print(f"使用设备: {self.device}")
        print(f"图像尺寸: {self.img_size}")
        print(f"批次大小: {self.batch_size}")
        print(f"分类类别: {self.class_names}")
        
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_weights = None
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
    
    def prepare_data(self):
        """准备训练和测试数据，包含更强的数据增强"""
        print("正在准备数据...")
        
        # 更强的训练数据增强
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1)
        ])
        
        # 验证/测试数据变换
        val_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 加载训练数据集
        full_train_dataset = datasets.ImageFolder(
            root=self.train_dir,
            transform=train_transform
        )
        
        # 分割训练集和验证集
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size]
        )
        
        # 为验证集设置不同的变换
        val_dataset.dataset = datasets.ImageFolder(
            root=self.train_dir,
            transform=val_transform
        )
        
        # 加载测试数据集
        test_dataset = datasets.ImageFolder(
            root=self.test_dir,
            transform=val_transform
        )
        
        # 计算类别权重
        self.calculate_class_weights(full_train_dataset)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"训练样本数: {len(train_dataset)}")
        print(f"验证样本数: {len(val_dataset)}")
        print(f"测试样本数: {len(test_dataset)}")
        print(f"类别映射: {full_train_dataset.class_to_idx}")
    
    def calculate_class_weights(self, dataset):
        """计算类别权重以处理数据不平衡问题"""
        print("计算类别权重...")
        
        # 统计各类别样本数量
        class_counts = np.zeros(self.num_classes)
        for _, label in dataset:
            class_counts[label] += 1
        
        # 计算类别权重
        total_samples = len(dataset)
        self.class_weights = torch.FloatTensor([
            total_samples / (self.num_classes * count) if count > 0 else 1.0
            for count in class_counts
        ]).to(self.device)
        
        # 使用平方根来减少极端权重
        self.class_weights = torch.sqrt(self.class_weights)
        
        print("类别权重:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}: {self.class_weights[i]:.3f} (样本数: {int(class_counts[i])})")
    
    def build_model(self, use_pretrained=True, model_name='resnet50'):
        """
        构建改进的CNN模型
        
        Args:
            use_pretrained (bool): 是否使用预训练模型
            model_name (str): 预训练模型名称
        """
        print("正在构建改进版CNN模型...")
        
        self.model = ImprovedGarbageClassificationCNN(
            num_classes=self.num_classes,
            use_pretrained=use_pretrained,
            model_name=model_name
        )
        self.model = self.model.to(self.device)
        
        if use_pretrained:
            print(f"使用{model_name}预训练模型")
        else:
            print("使用改进的自定义CNN架构")
        
        # 打印模型结构
        print("模型结构:")
        print(self.model)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n模型总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
    
    def train_model(self, epochs=100, learning_rate=0.00001):
        """
        训练改进的模型
        
        Args:
            epochs (int): 训练轮数，默认100
            learning_rate (float): 学习率，默认0.00001
        """
        print(f"开始训练改进模型，训练轮数: {epochs}")
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5, verbose=True, min_lr=1e-8
        )
        
        best_val_acc = 0.0
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                if batch_idx % 50 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(self.train_loader)}, '
                          f'Loss: {loss.item():.4f}')
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            # 计算平均损失和准确率
            train_loss = train_loss / len(self.train_loader)
            val_loss = val_loss / len(self.val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # 记录训练历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(optimizer.param_groups[0]['lr'])
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
            print(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
            print(f'  学习率: {optimizer.param_groups[0]["lr"]:.2e}')
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_improved_model.pth')
                print(f'  新的最佳验证准确率: {best_val_acc:.2f}%，模型已保存')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'验证准确率连续{patience}轮未改善，提前停止训练')
                    break
            
            print('-' * 60)
        
        print("模型训练完成!")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
    
    def evaluate_model(self):
        """评估改进模型性能"""
        print("正在评估改进模型...")
        
        # 加载最佳模型
        if os.path.exists('best_improved_model.pth'):
            self.model.load_state_dict(torch.load('best_improved_model.pth'))
            print("已加载最佳模型权重")
        
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                probabilities = F.softmax(output, dim=1)
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        test_loss = test_loss / len(self.test_loader)
        test_accuracy = 100. * test_correct / test_total
        
        # 计算Top-2准确率
        all_probabilities = np.array(all_probabilities)
        top2_predictions = np.argsort(all_probabilities, axis=1)[:, -2:]
        top2_correct = sum([target in pred for target, pred in zip(all_targets, top2_predictions)])
        top2_accuracy = 100. * top2_correct / test_total
        
        print(f"测试集损失: {test_loss:.4f}")
        print(f"测试集准确率: {test_accuracy:.2f}%")
        print(f"测试集Top-2准确率: {top2_accuracy:.2f}%")
        
        # 生成分类报告
        report = classification_report(
            all_targets, 
            all_predictions, 
            target_names=self.class_names,
            digits=4
        )
        print("\n分类报告:")
        print(report)
        
        # 生成混淆矩阵
        cm = confusion_matrix(all_targets, all_predictions)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('改进模型混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.tight_layout()
        plt.savefig('improved_pytorch_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return test_accuracy / 100.0  # 返回0-1范围的准确率
    
    def plot_training_history(self):
        """绘制训练历史图表"""
        if not self.train_losses:
            print("没有训练历史数据，请先训练模型")
            return
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 绘制损失曲线
        ax1.plot(epochs, self.train_losses, label='训练损失', linewidth=2)
        ax1.plot(epochs, self.val_losses, label='验证损失', linewidth=2)
        ax1.set_title('模型损失变化', fontsize=14, fontweight='bold')
        ax1.set_xlabel('训练轮数')
        ax1.set_ylabel('损失值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制准确率曲线
        ax2.plot(epochs, self.train_accuracies, label='训练准确率', linewidth=2)
        ax2.plot(epochs, self.val_accuracies, label='验证准确率', linewidth=2)
        ax2.set_title('模型准确率变化', fontsize=14, fontweight='bold')
        ax2.set_xlabel('训练轮数')
        ax2.set_ylabel('准确率 (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 绘制学习率变化
        ax3.plot(epochs, self.learning_rates, label='学习率', linewidth=2, color='red')
        ax3.set_title('学习率变化', fontsize=14, fontweight='bold')
        ax3.set_xlabel('训练轮数')
        ax3.set_ylabel('学习率')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 绘制验证准确率的移动平均
        if len(self.val_accuracies) > 5:
            window_size = min(5, len(self.val_accuracies))
            moving_avg = np.convolve(self.val_accuracies, np.ones(window_size)/window_size, mode='valid')
            ax4.plot(epochs[window_size-1:], moving_avg, label=f'验证准确率移动平均({window_size})', linewidth=2, color='green')
            ax4.plot(epochs, self.val_accuracies, label='验证准确率', linewidth=1, alpha=0.7, color='blue')
            ax4.set_title('验证准确率平滑曲线', fontsize=14, fontweight='bold')
            ax4.set_xlabel('训练轮数')
            ax4.set_ylabel('准确率 (%)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig('improved_pytorch_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印最佳性能
        if self.val_accuracies:
            best_val_acc = max(self.val_accuracies)
            best_epoch = self.val_accuracies.index(best_val_acc) + 1
            print(f"最佳验证准确率: {best_val_acc:.2f}% (第{best_epoch}轮)")
    
    def save_model(self, filepath='improved_garbage_classification_pytorch.pth'):
        """保存训练好的模型"""
        if self.model is None:
            print("没有可保存的模型，请先训练模型")
            return
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'img_size': self.img_size,
            'class_weights': self.class_weights
        }, filepath)
        print(f"改进模型已保存到: {filepath}")


def main():
    """
    主函数：执行改进版垃圾分类模型训练和评估流程
    """
    print("=" * 60)
    print("改进版垃圾分类CNN模型训练系统 - PyTorch版本")
    print("=" * 60)
    
    # 创建改进训练器实例
    trainer = ImprovedGarbageClassificationTrainer(
        train_dir='train',
        test_dir='test',
        img_size=(224, 224),
        batch_size=32
    )
    
    # 准备数据
    trainer.prepare_data()
    
    # 构建模型（使用预训练模型）
    trainer.build_model(use_pretrained=True, model_name='resnet18')
    
    # 训练模型
    trainer.train_model(epochs=100, learning_rate=0.00001)
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    # 评估模型
    test_accuracy = trainer.evaluate_model()
    
    # 保存模型
    trainer.save_model()
    
    # 检查是否达到要求
    if test_accuracy >= 0.70:
        print(f"\n✅ 改进模型达到要求！测试集准确率: {test_accuracy*100:.2f}% (≥70%)")
    else:
        print(f"\n❌ 改进模型未达到要求。测试集准确率: {test_accuracy*100:.2f}% (<70%)")
        print("建议：进一步调整超参数或收集更多数据")
    
    print("\n训练完成！生成的文件:")
    print("- improved_garbage_classification_pytorch.pth: 改进的训练模型")
    print("- best_improved_model.pth: 最佳模型检查点")
    print("- improved_pytorch_training_history.png: 训练历史图表")
    print("- improved_pytorch_confusion_matrix.png: 混淆矩阵")


if __name__ == "__main__":
    main() 