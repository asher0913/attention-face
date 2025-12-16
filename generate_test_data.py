#!/usr/bin/env python3
"""
生成测试数据文件，解决 FileNotFoundError: test_cifar10_image.pt
"""

import torch
import numpy as np
from torchvision import datasets, transforms

def generate_test_data():
    """生成测试数据文件"""
    
    print("🔧 生成测试数据文件...")
    
    # CIFAR-10 数据预处理（与训练时保持一致）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    # 加载CIFAR-10测试集
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # 提取测试图像和标签
    test_images = []
    test_labels = []
    
    # 只取前100个样本作为测试（避免文件过大）
    for i in range(min(100, len(testset))):
        image, label = testset[i]
        test_images.append(image)
        test_labels.append(label)
    
    # 转换为tensor
    test_images_tensor = torch.stack(test_images)
    test_labels_tensor = torch.tensor(test_labels)
    
    # 保存测试数据
    torch.save(test_images_tensor, "test_cifar10_image.pt")
    torch.save(test_labels_tensor, "test_cifar10_label.pt")
    
    print(f"✅ 测试数据已生成:")
    print(f"   - test_cifar10_image.pt: {test_images_tensor.shape}")
    print(f"   - test_cifar10_label.pt: {test_labels_tensor.shape}")

if __name__ == "__main__":
    generate_test_data()
