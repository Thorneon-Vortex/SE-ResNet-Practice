import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import argparse
import time
from datetime import datetime

# 导入自定义模块
from projects.production_ready_cv.src.model import get_model
from projects.production_ready_cv.src.dataset import get_data_loaders

def train(config):
    """
    生产级训练循环。
    特点：
    1. 配置化：所有参数从 config 读取。
    2. 监控：集成 TensorBoard。
    3. 健壮性：自动选择设备 (CUDA/MPS/CPU)。
    4. 可追溯性：实验结果按时间戳保存。
    """
    
    # 1. 环境准备
    # 获取绝对路径，防止因为运行脚本的目录不同导致找不到路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    def get_abs_path(path):
        if os.path.isabs(path):
            return path
        # 如果是相对路径，相对于项目根目录
        return os.path.join(project_root, os.path.basename(path))

    log_base_dir = os.path.join(project_root, "logs")
    checkpoint_base_dir = os.path.join(project_root, "checkpoints")
    
    # 环境准备
    device_name = config['train']['device']
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{config['data']['dataset_name']}_{timestamp}"
    log_dir = os.path.join(log_base_dir, exp_name)
    checkpoint_dir = os.path.join(checkpoint_base_dir, exp_name)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    print(f"实验启动！设备: {device}")
    print(f"日志保存至: {os.path.abspath(log_dir)}")
    print(f"模型保存至: {os.path.abspath(checkpoint_dir)}")

    # 2. 数据加载
    train_loader, val_loader = get_data_loaders(config)
    
    # 3. 模型、损失函数、优化器
    model = get_model(
        num_classes=config['model']['num_classes'], 
        dropout=config['model']['dropout']
    ).to(device)
    
    # 【快速验证可视化】写入一张图片到 TensorBoard，确保链路通畅
    try:
        sample_images, _ = next(iter(train_loader))
        writer.add_images('InputSamples', sample_images[:8], 0)
        print("已向 TensorBoard 写入样例图片，请刷新浏览器查看。")
    except Exception as e:
        print(f"写入样例图片失败: {e}")

    # --- 优化策略 1: 使用带 Label Smoothing 的损失函数 ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # --- 优化策略 2: 增加 Weight Decay (L2 正则化) ---
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config['train']['learning_rate'],
        momentum=0.9,
        weight_decay=5e-4  # 典型的正则化系数
    )
    
    # --- 优化策略 3: 引入余弦退火学习率调度器 ---
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])
    
    # 4. 训练循环
    epochs = config['train']['epochs']
    best_acc = 0.0
    
    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('LR/current', current_lr, epoch)
            
            start_time = time.time()
            for batch_idx, (images, labels) in enumerate(train_loader):
                try:
                    images, labels = images.to(device), labels.to(device)
                    
                    # 前向传播
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # 统计
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    if (batch_idx + 1) % 100 == 0:
                        current_loss = train_loss / (batch_idx + 1)
                        current_acc = 100. * correct / total
                        print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                              f"Loss: {current_loss:.4f} Acc: {current_acc:.2f}%")
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"| WARNING: 显存溢出 (OOM)! 正在清理缓存并尝试跳过当前 Batch...")
                        if hasattr(torch, 'cuda'): torch.cuda.empty_cache()
                        if hasattr(torch, 'mps'): torch.mps.empty_cache()
                        continue
                    else:
                        raise e
            
            # 记录训练日志到 TensorBoard
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100. * correct / total
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            
            # 5. 验证
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step() # 调整学习率
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            duration = time.time() - start_time
            print(f"==> Epoch {epoch+1} Done. Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% Time: {duration:.2f}s")
            
            # 6. 保存 Checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"发现更好的模型，已保存至: {checkpoint_path}")
            
            if (epoch + 1) % config['logging']['save_freq'] == 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth"))

    except KeyboardInterrupt:
        print("训练被用户手动停止。")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
    finally:
        writer.close()
        print("训练结束。")

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return val_loss / len(val_loader), 100. * correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production-Ready Training Script")
    # 修改默认路径，使其更符合当前的目录结构
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    
    # 确定配置文件的绝对路径
    if args.config is None:
        # 如果未指定，默认查找脚本同级目录下的 config/config.yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "config.yaml")
    else:
        config_path = args.config
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)
