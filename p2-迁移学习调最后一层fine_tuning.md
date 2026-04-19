

### **核心代码实现**
我编写了两个核心脚本来演示这个过程：

1.  **[fine_tune_tutorial.py](file:///Users/geniuswan/PycharmProjects/ai_develop/projects/fine_tuning_tutorial/fine_tune_tutorial.py)**：
    *   **冻结 (Freezing)**：通过 `param.requires_grad = False` 锁定预训练层的权重，不让它们在训练中改变。
    *   **替换 (Replacing)**：通过 `model.fc = nn.Linear(num_ftrs, num_classes)` 将 ResNet18 原本识别 1000 类的“头”切掉，换成我们识别 2 类（如人参、当归）的“新头”。
    *   **局部优化**：告诉优化器只更新 `model.fc.parameters()`，这样计算效率极高。

2.  **[dataset_loader.py](file:///Users/geniuswan/PycharmProjects/ai_develop/projects/fine_tuning_tutorial/dataset_loader.py)**：
    *   演示了如何使用 PyTorch 的标准 `ImageFolder` 来组织你的中草药图片数据。只需按文件夹分类存放，PyTorch 就能自动识别类别。

### **进阶知识点总结**

- **为什么要冻结？**
  预训练模型（如 ResNet）的前几层已经学会了识别线条、圆圈、纹理等通用特征。如果你的数据集很小（比如只有几百张图），直接全量训练会导致严重的**过拟合**。冻结这些层能保留这些通用的“视觉常识”。

- **什么时候“解冻”？**
  - **Case 1：数据量很大**。你可以尝试解冻最后几个卷积块（Fine-tune more layers），让模型更深地适应你的数据。
  - **Case 2：领域跨度大**。如果你要识别的是显微镜下的细胞图片或 X 光片（这和 ImageNet 的自然物体完全不同），你可能需要更早地解冻层。

- **微调的技巧：**
  *   **学习率 (LR)**：通常设得非常小（如 `1e-4` 或 `1e-5`），因为我们不想“剧烈破坏”已经学好的特征。
  *   **归一化 (Normalization)**：必须使用 ImageNet 的标准均值 `[0.485, 0.456, 0.406]` 和方差 `[0.229, 0.224, 0.225]`，因为模型是在这个分布下训练出来的。

### **你可以尝试的操作：**
你可以运行 `python projects/fine_tuning_tutorial/fine_tune_tutorial.py`，它会打印出哪些参数现在是“可训练”的。你会发现只有最后的 `fc.weight` 和 `fc.bias` 在列表中，这证明了你的“手术”非常成功！

**如果你有实际的图片数据，我可以继续教你如何写出完整的训练循环代码。**