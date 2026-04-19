import sys
import os

# 获取项目根目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# 核心修复：只将项目根目录加入 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import yaml
import torchvision.models as models
from projects.production_ready_cv.train import train as original_train

def main():
    config_path = "projects/resnet_magic_modify/config/config_orig.yaml"
    if not os.path.exists(config_path):
        from setup_comparison import create_configs
        create_configs()
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 这里的关键是：我们需要确保 train 函数使用的是原始 resnet18
    import projects.production_ready_cv.train as train_module
    def get_orig_model(num_classes, dropout=0.0):
        return models.resnet18(num_classes=num_classes)
    
    train_module.get_model = get_orig_model
    
    # 运行训练
    original_train(config)

if __name__ == "__main__":
    main()
