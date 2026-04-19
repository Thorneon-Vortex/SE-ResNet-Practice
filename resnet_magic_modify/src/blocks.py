import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block 
    论文：Squeeze-and-Excitation Networks (CVPR 2018)
    
    核心思想：
    1. Squeeze: 通过全局平均池化将每个通道的特征压缩为一个实数（全局感受野）。
    2. Excitation: 通过两层全连接层（中间有压缩）学习通道间的相关性。
    3. Scale: 将学习到的权重乘回原特征图，实现“重点通道加强，无用通道抑制”。
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # 全局平均池化: [B, C, H, W] -> [B, C, 1, 1]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 激励网络: 包含两个全连接层（用 1x1 卷积实现效果相同且更方便）
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        
        # 保存当前的权重，用于可视化
        self.last_attention_weights = y.detach()
        
        # Scale: 将权重 y 乘回到输入 x 上
        return x * y.expand_as(x)

class BasicBlockSE(nn.Module):
    """
    集成了 SE 模块的 ResNet 基础残差块
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlockSE, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # --- 魔改点 1: 加入 SE 注意力机制 ---
        self.se = SEBlock(planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # --- 魔改点 1: 在残差相加前应用注意力 ---
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # --- 魔改点 2: 修改残差连接 (示例：引入一个可学习的缩放因子) ---
        # 原版是 out += identity
        # 这里我们可以引入一个简单的权重，虽然 ResNet 强调 Identity 的纯粹性，
        # 但在某些魔改版本（如 ReZero）中，这被证明能加速极深网络的收敛。
        out += identity
        out = self.relu(out)

        return out
