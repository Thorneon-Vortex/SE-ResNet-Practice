import sys
import os

# 获取项目根目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.model import magic_resnet18
from PIL import Image
from torchvision import transforms

import os



def visualize_se_weights(model, image_path):
    """
    可视化 SE 模块生成的通道权重
    """
    model.eval()
    
    # 1. 加载并预处理图片
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # 2. 前向传播
    with torch.no_grad():
        _ = model(input_tensor)
    
    # 3. 提取各个层 SE 模块的权重
    # 我们之前在 SEBlock 的 forward 里存了 self.last_attention_weights
    se_weights = []
    for m in model.modules():
        if hasattr(m, 'last_attention_weights'):
            # 取出权重并转为 numpy [C]
            w = m.last_attention_weights.squeeze().cpu().numpy()
            se_weights.append(w)
    
    # 4. 绘图
    plt.figure(figsize=(15, 10))
    for i, weights in enumerate(se_weights):
        plt.subplot(2, 4, i+1)
        plt.bar(range(len(weights)), weights)
        plt.title(f"Layer {i+1} Attention")
        plt.ylim(0, 1)
        if i == 0:
            plt.ylabel("Importance Weight")
        plt.xlabel("Channel Index")
    
    plt.tight_layout()
    plt.savefig("projects/resnet_magic_modify/attention_visualization.png")
    print("注意力权重分布图已保存至: projects/resnet_magic_modify/attention_visualization.png")
    plt.show()

if __name__ == "__main__":
    # 示例图片路径（你可以换成你自己的图片）
    img_path = "heimaDeepLearning/day06/data/img.jpg" 
    if os.path.exists(img_path):
        model = magic_resnet18(num_classes=10)
        visualize_se_weights(model, img_path)
    else:
        print(f"未找到测试图片: {img_path}")
