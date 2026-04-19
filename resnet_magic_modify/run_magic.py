import sys
import os

# 获取项目根目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# 核心修复：只将项目根目录加入 sys.path
# 然后使用绝对路径导入所有模块，彻底避免 src 命名冲突
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import yaml
# 使用绝对导入
from projects.resnet_magic_modify.src.model import magic_resnet18
from projects.production_ready_cv.train import train as original_train, get_model

# 这是一个包装器，用来加载魔改模型并运行 production_ready_cv 里的训练逻辑
def main():
    config_path = "projects/resnet_magic_modify/config/config_magic.yaml"
    if not os.path.exists(config_path):
        from setup_comparison import create_configs
        create_configs()
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 这里的关键是：我们需要确保 train 函数使用的是我们的 magic_resnet18
    # 我们通过“猴子补丁”临时替换掉 production_ready_cv.train 模块里的 get_model
    import projects.production_ready_cv.train as train_module
    
    def get_magic_model(num_classes, dropout):
        return magic_resnet18(num_classes=num_classes)
    
    # 替换
    train_module.get_model = get_magic_model
    
    # 运行训练
    original_train(config)

if __name__ == "__main__":
    main()
