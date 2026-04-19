import torch
from torchsummary import summary
import torchvision.models as models
from src.model import magic_resnet18

def compare_models():
    """
    对比原始 ResNet18 和 魔改版 ResNet18。
    1. 结构验证：确保 SE 模块已集成。
    2. 参数量验证：注意力机制增加了多少参数？
    3. 运行验证：前向传播是否正常。
    """
    
    print("--- 原始 ResNet18 (标准版) ---")
    orig_model = models.resnet18(num_classes=10)
    # 打印部分结构
    print(f"原始 Block1 结构: \n{orig_model.layer1[0]}")
    
    print("\n--- 魔改版 ResNet18 (SE + Custom Link) ---")
    magic_model = magic_resnet18(num_classes=10)
    # 打印部分结构
    print(f"魔改版 Block1 结构: \n{magic_model.layer1[0]}")
    
    # 使用 torchsummary 进行可视化对比 (如果环境中有的话)
    try:
        from torchsummary import summary
        print("\n[魔改版摘要]")
        summary(magic_model, (3, 32, 32), device="cpu")
    except ImportError:
        print("\n请安装 torchsummary (pip install torchsummary) 以查看详细摘要。")
    
    # 简单的运行测试
    test_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = magic_model(test_input)
    print(f"\n前向传播测试成功！输出维度: {output.shape}")

if __name__ == "__main__":
    compare_models()
