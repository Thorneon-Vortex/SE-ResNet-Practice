import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

"""
进阶任务：如何组织你的自定义数据集 (ImageFolder 标准)

假设你要识别中草药，你需要按照以下结构存放图片：
data/
    train/
        ginseng/   (放人参图片，如 1.jpg, 2.jpg...)
        angelica/  (放当归图片...)
    val/
        ginseng/
        angelica/
"""

def prepare_data(data_dir):
    # 1. 定义预处理转换
    # 注意：微调预训练模型时，Normalize 的均值和方差必须是固定的 (ImageNet 标准)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224), # 训练集：随机裁剪增加鲁棒性
            transforms.RandomHorizontalFlip(), # 随机翻转
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),            # 验证集：保持一致性
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. 使用 ImageFolder 加载数据集
    # 它会自动读取文件夹名字作为类别标签 (Ginseng -> 0, Angelica -> 1)
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    # 3. 创建 DataLoader
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }

    # 4. 获取数据集大小和类别名称
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f"检测到的类别: {class_names}")
    print(f"训练集样本数: {dataset_sizes['train']}")
    print(f"验证集样本数: {dataset_sizes['val']}")
    
    return dataloaders, dataset_sizes, class_names

if __name__ == "__main__":
    # 这里演示如何调用 (假设路径存在)
    data_dir = './data'
    print(f"准备从 {data_dir} 加载数据...")
    # dataloaders, dataset_sizes, class_names = prepare_data(data_dir)
