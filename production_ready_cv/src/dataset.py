import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from PIL import Image
import logging

class SafeCIFAR10(datasets.CIFAR10):
    """
    健壮的 CIFAR10 数据集类。
    处理脏数据策略：如果图片损坏，记录日志并返回一个全零张量，防止程序崩溃。
    """
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as e:
            logging.error(f"无法加载索引为 {index} 的图片: {e}")
            # 返回一个全零图片和无效标签，或者根据需求返回 None
            # 注意：如果返回 None，需要在 DataLoader 的 collate_fn 中过滤
            img = torch.zeros(3, 32, 32)
            target = -1 
            return img, target

def get_data_loaders(config):
    """
    根据配置加载数据集。
    工程化思路：将预处理 (Transform) 与数据加载分离，并支持多种数据集。
    """
    
    # 1. 定义预处理转换
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # CIFAR10 的标准均值方差
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 2. 加载数据集
    dataset_name = config['data']['dataset_name']
    root = config['data']['root']
    download = config['data']['download']
    
    if dataset_name == "CIFAR10":
        train_dataset = SafeCIFAR10(root=root, train=True, transform=train_transform, download=download)
        val_dataset = SafeCIFAR10(root=root, train=False, transform=val_transform, download=download)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # 3. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=True, 
        num_workers=config['train']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=False, 
        num_workers=config['train']['num_workers']
    )
    
    return train_loader, val_loader
