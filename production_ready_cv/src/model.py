import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardCNN(nn.Module):
    """
    一个生产级的、结构化的 CNN 模型定义。
    相比于实验脚本，我们使用了模块化和清晰的结构。
    """
    def __init__(self, num_classes=10, dropout=0.5):
        super(StandardCNN, self).__init__()
        
        # 定义特征提取器 (Feature Extractor)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 定义分类器 (Classifier)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(num_classes=10, dropout=0.5):
    """模型工厂函数，方便以后替换模型"""
    return StandardCNN(num_classes, dropout)
