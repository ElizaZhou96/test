# models/modaequil.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback

class TextProcessor(nn.Module):
    """专门的文本处理模块"""
    def __init__(self, hidden_dim):
        super(TextProcessor, self).__init__()
        # BERT词汇表大小通常为30522
        self.embedding = nn.Embedding(30522, hidden_dim)
        self.encoder = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # 此方法是必需的，但实际处理在encode_modalities中
        return x

class DMIMLayer(nn.Module):
    """动态模态重要性矩阵层"""
    def __init__(self, num_modalities, hidden_dim):
        super(DMIMLayer, self).__init__()
        self.num_modalities = num_modalities
        
        # 模态间关系建模
        self.modality_importance = nn.Parameter(torch.ones(num_modalities, num_modalities))
        
        # 上下文感知调制网络
        self.context_modulator = nn.Sequential(
            nn.Linear(hidden_dim, num_modalities * num_modalities),
            nn.Sigmoid()
        )
        
        # 生态系统相关性编码
        self.ecosystem_encoders = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) 
            for _ in range(num_modalities)
        ])
        
    def forward(self, modality_features, task_context=None):
        """处理特征并计算模态重要性，安全处理None值"""
        device = self.modality_importance.device
        
        # 初始化可用性掩码
        availability_mask = torch.zeros(self.num_modalities, device=device)
        for i, features in enumerate(modality_features):
            if features is not None:
                availability_mask[i] = 1.0
                
        # 计算基础重要性
        base_importance = self.modality_importance * availability_mask.unsqueeze(1)
        
        # 如果有任务上下文，动态调整重要性
        if task_context is not None:
            context_weights = self.context_modulator(task_context)
            context_weights = context_weights.view(self.num_modalities, self.num_modalities)
            importance = base_importance * context_weights
        else:
            importance = base_importance
            
        # 应用生态系统相关性编码
        enhanced_features = []
        for i, features in enumerate(modality_features):
            if features is not None:
                # 获取此模态的重要性向量
                mod_importance = importance[i]
                
                # 确保重要性向量的维度与特征维度匹配
                if mod_importance.size(0) != features.size(1):
                    # 如果维度不匹配，创建与特征维度相同的全1向量
                    print(f"警告: 模态 {i} 重要性向量维度 ({mod_importance.size(0)}) 与特征维度 ({features.size(1)}) 不匹配")
                    mod_importance = torch.ones(features.size(1), device=device)
                
                # 应用编码
                encoded = self.ecosystem_encoders[i](features)
                enhanced_features.append((encoded, mod_importance))
            else:
                enhanced_features.append(None)
                
        return enhanced_features, importance

class UARFLayer(nn.Module):
    """不确定性感知表示融合层"""
    def __init__(self, hidden_dim, num_modalities):
        super(UARFLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 不确定性估计器网络
        self.uncertainty_estimator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, 1),
                nn.Softplus()  # 确保不确定性为正
            ) for _ in range(num_modalities)
        ])
        
        # 表示融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 不确定性传播网络
        self.uncertainty_propagation = nn.Linear(num_modalities, 1)
        
    def forward(self, enhanced_features, importance_matrix):
        device = next(self.parameters()).device
        batch_size = 0
        
        # 找到第一个非None特征的批次大小
        for feat in enhanced_features:
            if feat is not None:
                batch_size = feat[0].size(0)
                break
                
        if batch_size == 0:
            # 如果所有特征都是None，返回零张量
            return torch.zeros(1, self.hidden_dim).to(device), torch.ones(1, 1).to(device)
        
        # 估计每个模态的不确定性
        uncertainties = []
        features_with_uncertainty = []
        
        for i, feat_tuple in enumerate(enhanced_features):
            if feat_tuple is not None:
                feat, importance = feat_tuple
                # 估计不确定性
                uncertainty = self.uncertainty_estimator[i](feat)
                uncertainties.append(uncertainty)
                
                # 在应用重要性权重前检查形状
                if importance.dim() == 1:  # 如果是一维向量
                    # 确保重要性向量的长度与特征维度匹配
                    if importance.size(0) != feat.size(1):
                        # 如果维度不匹配，创建一个全1向量
                        importance = torch.ones(feat.size(1), device=device)
                
                # 加权特征（按重要性和不确定性的倒数）
                # 确保重要性向量正确广播
                weighted_feat = feat * importance.unsqueeze(0).expand_as(feat) / (uncertainty + 1e-6)
                features_with_uncertainty.append(weighted_feat)
            else:
                # 对于缺失模态，使用可学习的占位符，并赋予高不确定性
                placeholder = torch.zeros(batch_size, self.hidden_dim).to(device)
                high_uncertainty = torch.ones(batch_size, 1).to(device) * 100.0
                uncertainties.append(high_uncertainty)
                features_with_uncertainty.append(placeholder)
        
        # 连接所有特征
        concatenated = torch.cat(features_with_uncertainty, dim=1)
        
        # 融合表示
        fused_representation = self.fusion_network(concatenated)
        
        # 传播不确定性
        stacked_uncertainties = torch.cat(uncertainties, dim=1)
        propagated_uncertainty = self.uncertainty_propagation(stacked_uncertainties)
        
        return fused_representation, propagated_uncertainty

class MEAALayer(nn.Module):
    """模态平衡自适应增强层"""
    def __init__(self, hidden_dim, num_modalities, num_classes=10):
        super(MEAALayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        
        # 模态生成器网络
        self.modality_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + num_classes, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(num_modalities)
        ])
        
        # 类别均衡记忆银行
        self.register_buffer('class_prototype_bank', 
                           torch.zeros(num_classes, num_modalities, hidden_dim))
        self.register_buffer('class_counts', torch.zeros(num_classes))
        
        # 模态关联强度估计器
        self.modality_correlation = nn.Parameter(torch.ones(num_modalities, num_modalities))
        
        # 差异性损失权重
        self.diversity_weight = nn.Parameter(torch.tensor(0.1))
        
    def update_memory_bank(self, features, labels):
        """更新类别原型记忆库"""
        # 如果没有有效的标签，直接返回
        if labels is None or len(labels) == 0:
            return
            
        device = self.class_counts.device
        
        for c in range(self.num_classes):
            class_mask = (labels == c)
            if not torch.any(class_mask):
                continue
                
            for m in range(self.num_modalities):
                if features[m] is not None:
                    class_features = features[m][class_mask]
                    if len(class_features) > 0:
                        # 计算移动平均
                        self.class_prototype_bank[c, m] = (
                            self.class_prototype_bank[c, m] * self.class_counts[c] + 
                            class_features.mean(0)
                        ) / (self.class_counts[c] + 1)
            
            self.class_counts[c] += 1
    
    def generate_balanced_batch(self, fused_features, labels, desired_counts):
        """生成平衡的批次数据"""
        if labels is None or len(labels) == 0 or fused_features is None:
            return None, None
            
        device = fused_features.device
        batch_size = fused_features.size(0)
        augmented_features = [[] for _ in range(self.num_modalities)]
        augmented_labels = []
        
        # 计算当前批次中每个类别的数量
        label_counts = torch.bincount(labels, minlength=self.num_classes)
        
        # 确定需要增强的类别
        for c in range(self.num_classes):
            if c < len(label_counts) and c < len(desired_counts) and label_counts[c] < desired_counts[c]:
                # 需要生成的样本数
                num_to_generate = desired_counts[c] - label_counts[c]
                
                # 如果记忆库中有此类别
                if self.class_counts[c] > 0:
                    # 使用记忆库中的原型作为条件
                    class_code = torch.zeros(num_to_generate, self.num_classes, device=device)
                    class_code[:, c] = 1
                    
                    # 生成缺失模态
                    for m in range(self.num_modalities):
                        # 合并融合特征和类别编码
                        condition = torch.cat([
                            fused_features[:1].repeat(num_to_generate, 1),
                            class_code
                        ], dim=1)
                        
                        # 生成特征
                        generated = self.modality_generators[m](condition)
                        
                        # 应用模态相关性，确保生成的特征与其他模态相关
                        for other_m in range(self.num_modalities):
                            if m != other_m and self.class_prototype_bank[c, other_m].sum() > 0:
                                correlation = self.modality_correlation[m, other_m]
                                prototype = self.class_prototype_bank[c, other_m]
                                generated = generated * (1 - correlation) + prototype * correlation
                                
                        augmented_features[m].append(generated)
                    
                    # 添加标签
                    augmented_labels.append(torch.full((num_to_generate,), c, device=device))
        
        # 如果有生成的样本
        if augmented_labels and augmented_labels[0].size(0) > 0:
            # 合并生成的特征和标签
            for m in range(self.num_modalities):
                if augmented_features[m] and len(augmented_features[m]) > 0:
                    augmented_features[m] = torch.cat(augmented_features[m], dim=0)
                else:
                    augmented_features[m] = None
                    
            augmented_labels = torch.cat(augmented_labels)
            
            return augmented_features, augmented_labels
        else:
            return None, None
    
    def forward(self, features, fused_representation, labels=None):
        """前向传播"""
        # 如果没有标签，只返回特征
        if labels is None:
            return features, labels, torch.tensor(0.0).to(fused_representation.device)
            
        # 更新记忆库
        self.update_memory_bank(features, labels)
        
        # 计算理想的类别分布（均衡）
        batch_size = labels.size(0)
        desired_per_class = max(batch_size // self.num_classes, 1)
        desired_counts = torch.full((self.num_classes,), desired_per_class, device=labels.device)
        
        # 生成平衡批次
        augmented_features, augmented_labels = self.generate_balanced_batch(
            fused_representation, labels, desired_counts
        )
        
        # 如果有增强的数据，计算多样性损失和返回增强后的特征
        if augmented_features is not None and augmented_labels is not None and augmented_labels.size(0) > 0:
            # 计算生成特征的多样性损失
            diversity_loss = torch.tensor(0.1).to(labels.device)  # 简化实现
            
            # 将原始特征和增强特征合并
            combined_features = []
            for m in range(self.num_modalities):
                if features[m] is not None and augmented_features[m] is not None:
                    combined_features.append(torch.cat([features[m], augmented_features[m]], dim=0))
                elif features[m] is not None:
                    combined_features.append(features[m])
                elif augmented_features[m] is not None:
                    combined_features.append(augmented_features[m])
                else:
                    combined_features.append(None)
                    
            combined_labels = torch.cat([labels, augmented_labels])
            
            return combined_features, combined_labels, diversity_loss * self.diversity_weight
        
        return features, labels, torch.tensor(0.0).to(labels.device)

class MUPLayer(nn.Module):
    """多尺度不确定性裁剪层 - 批处理版本"""
    def __init__(self, hidden_dim, num_modalities, threshold=0.1):
        super(MUPLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.threshold = threshold
        
        # 模态评分网络
        self.modality_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
        # 特征重要性估计器
        self.feature_importance = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])
        
        # 不确定性传播函数
        self.uncertainty_to_sparsity = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features, uncertainties):
        """根据不确定性动态裁剪特征"""
        pruned_features = []
        device = next(self.parameters()).device
        
        for i, feat in enumerate(features):
            if feat is None:
                pruned_features.append(None)
                continue
                
            # 确保不确定性向量有正确的形状
            if isinstance(uncertainties, list):
                uncert = uncertainties[0] 
            else:
                uncert = uncertainties
                
            # 批处理安全计算
            try:
                # 计算模态评分 (整个批次)
                modality_score = self.modality_scorer(feat.mean(dim=0, keepdim=True)).mean()
                
                # 计算特征重要性
                feature_imp = self.feature_importance[i](feat)
                
                # 根据不确定性确定稀疏度 (单一值代表整个批次)
                sparsity_level = self.uncertainty_to_sparsity(uncert.mean(dim=0, keepdim=True)).item()
                
                # 计算全局重要性阈值 (使用整个批次)
                all_importances = feature_imp.flatten()
                # 安全地计算分位数
                k = max(0, min(int(sparsity_level * len(all_importances)), len(all_importances)-1))
                topk_values, _ = torch.topk(all_importances, k, largest=False)
                importance_threshold = topk_values[-1] if k > 0 else 0
                
                # 应用特征掩码
                mask = (feature_imp > importance_threshold).float()
                pruned_feat = feat * mask
                
                # 根据模态评分确定是否保留此模态
                if modality_score > self.threshold:
                    pruned_features.append(pruned_feat)
                else:
                    pruned_features.append(None)
                    
            except Exception as e:
                print(f"裁剪模态 {i} 时出错: {str(e)}")
                pruned_features.append(feat)  # 出错时保留原始特征
                
        return pruned_features


class ModaEquil(nn.Module):
    """多模态数据不平衡自适应平衡算法"""
    def __init__(self, config):
        super(ModaEquil, self).__init__()
        self.config = config
        self.num_modalities = len(config["modalities"])
        self.hidden_dim = config["hidden_dim"]
        self.num_classes = config.get("num_classes", 10)
        
        # 重新定义模态编码器，根据不同模态使用不同架构
        self.modality_encoders = nn.ModuleDict()
        
        # 基因组编码器
        if "genome" in config["modalities"]:
            self.modality_encoders["genome"] = nn.Sequential(
                nn.Linear(config["modality_dims"].get("genome", 1024), self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU()
            )
        
        # 图像编码器 (假设输入是图像特征向量，不是原始图像)
        if "image" in config["modalities"]:
            self.modality_encoders["image"] = nn.Sequential(
                nn.Flatten(),  # 确保输入被展平
                nn.Linear(3*224*224, self.hidden_dim),  # 假设输入是3x224x224图像
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU()
            )
        
        # 音频编码器
        if "audio" in config["modalities"]:
            self.modality_encoders["audio"] = nn.Sequential(
                nn.Linear(config["modality_dims"].get("audio", 512), self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU()
            )
        
        # # 文本编码器
        # if "text" in config["modalities"]:
        #     self.modality_encoders["text"] = nn.Sequential(
        #         nn.Linear(config["modality_dims"].get("text", 768), self.hidden_dim),
        #         nn.LayerNorm(self.hidden_dim),
        #         nn.ReLU()
        #     )
        # 文本编码器
        if "text" in config["modalities"]:
            self.modality_encoders["text"] = TextProcessor(self.hidden_dim)
        
        # 环境编码器
        if "environment" in config["modalities"]:
            self.modality_encoders["environment"] = nn.Sequential(
                nn.Linear(config["modality_dims"].get("environment", 256), self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU()
            )
        
        # 其他组件保持不变
        self.dmim = DMIMLayer(self.num_modalities, self.hidden_dim)
        self.uarf = UARFLayer(self.hidden_dim, self.num_modalities)
        self.meaa = MEAALayer(self.hidden_dim, self.num_modalities, self.num_classes)
        self.mup = MUPLayer(self.hidden_dim, self.num_modalities)
        
    def encode_modalities(self, batch):
        """编码各个模态数据，更健壮的实现"""
        encoded_features = [None] * self.num_modalities
        device = next(self.parameters()).device
        
        # 创建模态到索引的映射
        modality_to_idx = {mod: i for i, mod in enumerate(self.config["modalities"])}
        
        for modality, idx in modality_to_idx.items():
            # 检查是否有此模态数据
            if modality not in batch or batch[modality] is None:
                continue
                
            try:
                # 获取数据和掩码
                data = batch[modality]
                mask_key = f"{modality}_mask"
                mask = batch.get(mask_key, None)
                
                # 特殊处理文本数据
                if modality == "text":
                    if isinstance(data, list) and "text_tokenized" in batch:
                        # 使用tokenized文本
                        if isinstance(batch["text_tokenized"], dict) and "input_ids" in batch["text_tokenized"]:
                            token_ids = batch["text_tokenized"]["input_ids"]
                            if isinstance(token_ids, torch.Tensor):
                                # 处理使用嵌入层
                                try:
                                    # 使用嵌入层处理token IDs
                                    embedded = self.modality_encoders[modality].embedding(token_ids)
                                    # 池化：取每个序列的平均值
                                    mask = token_ids != 0  # 假设0是填充标记
                                    mask = mask.float().unsqueeze(-1)
                                    pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                                    # 编码
                                    encoded = self.modality_encoders[modality].encoder(pooled)
                                    encoded_features[idx] = encoded
                                    print(f"文本处理成功，输出形状: {encoded.shape}")
                                    continue
                                except Exception as e:
                                    print(f"处理文本tokens时出错: {e}")
                                    traceback.print_exc()
                    
                    # 如果数据是张量，直接处理
                    if isinstance(data, torch.Tensor):
                        try:
                            encoded = self.modality_encoders[modality](data)
                            encoded_features[idx] = encoded
                            continue
                        except Exception as e:
                            print(f"处理文本张量时出错: {e}")
                            
                    print(f"跳过模态 {modality}，无法处理类型: {type(data)}")
                    continue
                
                # 确保数据是张量
                if not isinstance(data, torch.Tensor):
                    print(f"跳过模态 {modality}，数据不是张量: {type(data)}")
                    continue
                    
                # 打印类型和形状以调试
                print(f"处理模态 {modality}, 形状: {data.shape}, 类型: {data.dtype}")
                
                # 根据模态特殊处理
                if modality == "image" and len(data.shape) > 2:
                    # 图像特殊处理 - 将批次中所有图像展平
                    batch_size = data.shape[0]
                    flattened = data.reshape(batch_size, -1)
                    encoded = self.modality_encoders[modality](flattened)
                else:
                    # 其他模态标准处理
                    encoded = self.modality_encoders[modality](data)
                
                # 应用掩码（如果有）
                if mask is not None:
                    expanded_mask = mask.unsqueeze(-1).expand_as(encoded).float()
                    encoded = encoded * expanded_mask
                
                # 存储编码结果
                encoded_features[idx] = encoded
                
            except Exception as e:
                print(f"编码模态 {modality} 时出错: {e}")
                traceback.print_exc()  # 打印完整错误堆栈
        
        return encoded_features
    
    def forward(self, batch, train=True):
        """ModaEquil前向传播 - 必需的方法"""
        # 编码各模态
        encoded_features = self.encode_modalities(batch)
        
        # 如果所有模态都为None，返回空结果
        if all(f is None for f in encoded_features):
            device = next(self.parameters()).device
            return {
                "fused_representation": torch.zeros(1, self.hidden_dim).to(device),
                "uncertainty": torch.ones(1, 1).to(device),
                "importance_matrix": torch.eye(self.num_modalities).to(device)
            }
        
        # 应用动态模态重要性矩阵
        task_context = None
        if "task_context" in batch:
            task_context = batch["task_context"]
        enhanced_features, importance_matrix = self.dmim(encoded_features, task_context)
        
        # 应用不确定性感知表示融合
        fused_representation, propagated_uncertainty = self.uarf(enhanced_features, importance_matrix)
        
        # 在训练时应用模态平衡自适应增强
        # 确保标签字段存在且不为None
        has_valid_labels = train and "labels" in batch and batch["labels"] is not None
        
        if has_valid_labels:
            balanced_features, balanced_labels, diversity_loss = self.meaa(
                encoded_features, fused_representation, batch["labels"]
            )
        else:
            balanced_features, balanced_labels, diversity_loss = encoded_features, None, torch.tensor(0.0).to(fused_representation.device)
        
        # 应用多尺度不确定性裁剪
        uncertainties = [propagated_uncertainty] * self.num_modalities
        pruned_features = self.mup(balanced_features, uncertainties)
        
        # 重新融合裁剪后的特征
        re_enhanced, _ = self.dmim(pruned_features, task_context)
        re_fused, re_uncertainty = self.uarf(re_enhanced, importance_matrix)
        
        results = {
            "fused_representation": re_fused,
            "uncertainty": re_uncertainty,
            "importance_matrix": importance_matrix
        }
        
        if has_valid_labels and balanced_labels is not None:
            results["balanced_labels"] = balanced_labels
            results["diversity_loss"] = diversity_loss
                
        return results