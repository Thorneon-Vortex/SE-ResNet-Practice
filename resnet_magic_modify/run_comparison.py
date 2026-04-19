import torch
import torchvision.models as models
from src.model import magic_resnet18
from projects.production_ready_cv.train import train
import yaml
import os
import argparse

def run_experiment(model_type):
    # 加载对应的配置
    config_path = f"projects/resnet_magic_modify/config/config_{model_type}.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 动态注入模型加载逻辑
    # 我们需要稍微修改一下 train 函数或者在 train.py 里做适配
    # 这里我们采用一种“注入”的方式：修改 train.py 使其能根据配置加载不同模型
    
    print(f"\n>>> 启动实验: {model_type.upper()}")
    
    # 这里我们直接手动调用训练逻辑，但模型需要根据 model_type 切换
    # 为了简单起见，我们直接在 train 内部根据 config 切换即可
    
    # 注意：我们需要确保 train.py 的 get_model 能识别我们的魔改模型
    # 这里我们通过 monkey patch 或者传参的方式处理
    
    # 实际上，最好的工程实践是修改 train.py 里的 get_model 函数
    pass

if __name__ == "__main__":
    # 实际运行建议直接通过命令行分别运行两个 train.py 任务
    print("建议运行以下两条命令进行对比：")
    print("1. 魔改版: python projects/resnet_magic_modify/run_magic.py")
    print("2. 原始版: python projects/resnet_magic_modify/run_orig.py")
