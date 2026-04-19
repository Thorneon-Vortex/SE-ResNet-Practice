import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

"""
进阶任务：基于迁移学习 (Fine-tuning) 的自定义图像分类器

场景假设：我们要识别 2 种中草药 (人参 Ginseng 和 当归 Angelica)。
核心思路：
1. 加载一个在 ImageNet 上预训练好的模型 (ResNet18)。
2. "冻结" (Freeze) 模型的前面所有层，使它们的权重在训练中不再更新。
3. "替换" (Replace) 最后的分类层 (Fully Connected Layer)，以匹配我们的 2 个类别。
4. 仅训练最后的分类层。
"""

def fine_tune_demo():
    # 1. 设置超参数
    num_classes = 2  # 我们假设只有 2 种中草药
    batch_size = 4
    learning_rate = 0.001
    num_epochs = 5
    
    # 2. 加载预训练模型 (ResNet18)
    # weights=models.ResNet18_Weights.DEFAULT 会自动加载预训练权重
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 3. 【核心知识点：冻结参数】
    # 遍历模型的所有参数，将 requires_grad 设为 False。
    # 这样在反向传播时，这些层的梯度不会被计算，权重也就不会更新。
    for param in model.parameters():
        param.requires_grad = False
    
    # 4. 【核心知识点：替换最后的分类层】
    # 在 ResNet 中，最后一层叫 'fc' (Fully Connected)。
    # 我们先获取 fc 层的输入维度 (通常是 512)
    num_ftrs = model.fc.in_features
    
    # 替换成一个新的 Linear 层。
    # 注意：新创建的层，其 requires_grad 默认就是 True，所以这层会被训练。
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 【核心知识点：仅优化需要更新的参数】
    # 我们只把 model.fc.parameters() 传给优化器，而不是整个 model.parameters()。
    # 这样做更安全，也能节省计算资源。
    optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=0.9)
    
    # 6. 数据准备 (这里仅展示逻辑，不实际运行训练)
    # 工业界标准：使用 ImageFolder。
    # 你的文件夹结构应该是：
    # data/
    #   train/
    #     Ginseng/ (放人参图片)
    #     Angelica/ (放当归图片)
    #   val/
    #     Ginseng/
    #     Angelica/
    
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("模型配置完成！")
    print(f"原 FC 层输入维度: {num_ftrs}")
    print(f"现 FC 层结构: {model.fc}")
    
    # 验证一下：哪些参数在更新？
    print("\n--- 待更新的参数 (requires_grad=True) ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"层名称: {name}, 参数形状: {param.shape}")

if __name__ == "__main__":
    fine_tune_demo()
    
    print("\n--- 进阶学习笔记 ---")
    print("1. 为什么冻结？")
    print("   预训练模型已经学到了通用的视觉特征（边缘、形状、纹理）。对于小数据集，重新训练这些特征会导致过拟合。")
    print("2. 什么时候不冻结？")
    print("   如果你的数据量非常大，或者你的图片和 ImageNet 差别极大（如医学 X 光片），可以尝试 '解冻' 更多层。")
    print("3. 学习率设置：")
    print("   微调时的学习率通常比从零训练小 10 倍左右，因为我们不想剧烈破坏已经学好的特征。")
