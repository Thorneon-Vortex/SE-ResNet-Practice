import yaml
import os

def create_configs():
    base_config = {
        'train': {
            'batch_size': 128,
            'epochs': 10,  # 对比实验可以用短一点的 epoch 观察趋势
            'learning_rate': 0.001,
            'num_workers': 4,
            'device': 'auto'
        },
        'model': {
            'num_classes': 10,
            'dropout': 0.5
        },
        'data': {
            'dataset_name': 'CIFAR10',
            'root': './data',
            'download': True
        },
        'logging': {
            'log_dir': 'projects/resnet_magic_modify/logs',
            'checkpoint_dir': 'projects/resnet_magic_modify/checkpoints',
            'save_freq': 5
        }
    }

    # 创建魔改版配置
    magic_config = base_config.copy()
    magic_config['logging']['log_dir'] = 'projects/resnet_magic_modify/logs/magic'
    
    # 创建原始版配置
    orig_config = base_config.copy()
    orig_config['logging']['log_dir'] = 'projects/resnet_magic_modify/logs/original'

    os.makedirs('projects/resnet_magic_modify/config', exist_ok=True)
    
    with open('projects/resnet_magic_modify/config/config_magic.yaml', 'w') as f:
        yaml.dump(magic_config, f)
    
    with open('projects/resnet_magic_modify/config/config_orig.yaml', 'w') as f:
        yaml.dump(orig_config, f)
    
    print("对比实验配置文件已生成。")

if __name__ == "__main__":
    create_configs()
