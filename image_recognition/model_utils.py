import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import json
import requests

class ImageClassifier:
    def __init__(self):
        # 使用最新的 Weights API 加载预训练模型
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.model.eval()
        
        # 获取预处理转换
        self.preprocess = self.weights.transforms()
        
        # 获取类别标签
        self.categories = self.weights.meta["categories"]

    def predict(self, image: Image.Image, top_k=5):
        """
        对输入的 PIL 图像进行分类
        """
        # 确保图像是 RGB 模式
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # 预处理并添加 batch 维度
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        # 检查是否有 GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(device)
        input_batch = input_batch.to(device)

        with torch.no_grad():
            output = self.model(input_batch)

        # 计算概率
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # 获取 top-k 结果
        top_prob, top_catid = torch.topk(probabilities, top_k)
        
        results = []
        for i in range(top_prob.size(0)):
            results.append({
                "category": self.categories[top_catid[i]],
                "probability": top_prob[i].item()
            })
            
        return results

if __name__ == "__main__":
    # 测试代码
    classifier = ImageClassifier()
    print("模型加载成功，共有类别数:", len(classifier.categories))
