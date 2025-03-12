# 创建新文件 models/few_shot.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback  # 自动添加的导入

class PrototypicalNetwork(nn.Module):
    """原型网络用于少样本学习"""
    def __init__(self, feature_dim):
        super(PrototypicalNetwork, self).__init__()
        self.feature_dim = feature_dim
        
    def forward(self, support_features, support_labels, query_features):
        """计算查询样本到类原型的距离"""
        # 计算每个类的原型
        unique_classes = torch.unique(support_labels)
        prototypes = []
        
        for c in unique_classes:
            # 提取该类的所有特征
            class_features = support_features[support_labels == c]
            # 计算原型(均值)
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
            
        prototypes = torch.stack(prototypes)
        
        # 计算查询样本到每个原型的距离
        dists = torch.cdist(query_features, prototypes)
        
        # 返回负距离作为相似度分数(较小的距离表示较高的相似度)
        return -dists