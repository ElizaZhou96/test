# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import traceback  # 添加此行导入traceback模块
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


from config import CONFIG
from models.efrn import EFRN
from models.sedf import SEDF
from models.efp import EFP
from models.esat import ESAT
from utils.data_loader import get_data_loaders

from models.modaequil import ModaEquil


class BioHarmony(nn.Module):
    """BioHarmony完整模型"""
    def __init__(self, config):
        super(BioHarmony, self).__init__()
        self.config = config
        
        # 初始化各个组件
        self.efrn = EFRN(config)
        self.sedf = SEDF(config)
        self.efp = EFP(config)
        self.esat = ESAT(config)
        
        # 添加ModaEquil组件
        self.use_modaequil = config.get("use_modaequil", False)
        if self.use_modaequil:
            self.modaequil = ModaEquil(config)
            print("\n[INFO] ModaEquil algorithm activated! Using advanced multi-modal balancing.\n")
        else:
            print("\n[INFO] Using standard multi-modal processing.\n")

        
        # 修改分类器，使其接收融合表示
        self.classifier = nn.Sequential(
            nn.Linear(config["hidden_dim"], config["hidden_dim"]*2),
            nn.LayerNorm(config["hidden_dim"]*2),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_dim"]*2, config.get("num_classes", 10))
        )
        
    def forward(self, batch, train=True):
        """前向传播，整合各个组件"""
        results = {}
        
        # 使用ModaEquil处理多模态数据
        if self.use_modaequil:
            # 清理batch中的掩码，只保留数据
            clean_batch = {k: v for k, v in batch.items() if not k.endswith('_mask')}
            
            # 调用ModaEquil获取平衡的融合表示
            try:
                modaequil_outputs = self.modaequil(clean_batch, train)
                
                # 使用ModaEquil的融合表示
                results["fused_representation"] = modaequil_outputs["fused_representation"]
                results["uncertainty"] = modaequil_outputs["uncertainty"]
                results["importance_matrix"] = modaequil_outputs["importance_matrix"]
                
                # 如果有平衡标签，添加到结果
                if "balanced_labels" in modaequil_outputs:
                    results["balanced_labels"] = modaequil_outputs["balanced_labels"]
                    
                # 添加多样性损失
                if "diversity_loss" in modaequil_outputs:
                    results["diversity_loss"] = modaequil_outputs["diversity_loss"]
                    
                # 生成分类预测
                if results["fused_representation"] is not None:
                    # 使用可用的标签
                    if "labels" in batch and batch["labels"] is not None:
                        results["logits"] = self.classifier(results["fused_representation"])
                        
                return results
            except Exception as e:
                print(f"ModaEquil处理失败，回退到标准处理: {str(e)}")
                traceback.print_exc()
        
        # 原始处理流程（如果未启用ModaEquil或ModaEquil失败）
        try:
            # 准备EFRN输入 - 确保所有输入都是张量
            efrn_inputs = {}
            for modality in self.config["modalities"]:
                if modality in batch and batch[modality] is not None:
                    # 特殊处理文本模态
                    if modality == "text" and isinstance(batch[modality], list):
                        try:
                            # 使用SEDF对文本进行编码
                            text_embedding = self.sedf.encode_text(batch[modality])
                            efrn_inputs[modality] = text_embedding
                            print(f"文本模态编码成功，形状: {text_embedding.shape}")
                            continue  # 跳过后续检查
                        except Exception as e:
                            print(f"文本编码失败: {str(e)}")
                            traceback.print_exc()
                    
                    # 其他模态的原有处理逻辑
                    if isinstance(batch[modality], torch.Tensor):
                        efrn_inputs[modality] = batch[modality]
                    elif isinstance(batch[modality], list) and all(isinstance(x, torch.Tensor) for x in batch[modality] if x is not None):
                        # 如果是张量列表，尝试堆叠
                        valid_tensors = [x for x in batch[modality] if x is not None]
                        if valid_tensors:
                            try:
                                stacked = torch.stack(valid_tensors)
                                efrn_inputs[modality] = stacked
                            except:
                                print(f"无法堆叠模态 {modality} 的张量列表")
                    else:
                        print(f"跳过模态 {modality}，不是有效张量或张量列表")
                    
            # 只有在有输入的情况下才尝试运行EFRN
            if efrn_inputs:
                # 打印EFRN输入以调试
                for modality, data in efrn_inputs.items():
                    print(f"EFRN输入 - 模态: {modality}, 形状: {data.shape}")
                    
                try:
                    efrn_outputs = self.efrn(efrn_inputs)
                    results["ecological_fingerprint"] = efrn_outputs["mean_fingerprint"]
                    results["fingerprint_samples"] = efrn_outputs["fingerprints"]
                except Exception as e:
                    print(f"EFRN处理失败: {str(e)}")
                    traceback.print_exc()
                    # 创建一个默认的生态指纹
                    device = next(self.parameters()).device
                    results["ecological_fingerprint"] = torch.zeros(batch_size, self.config["ecological_features"]).to(device)
                    
                # 后续处理...只有在前一步成功时才执行
                if "ecological_fingerprint" in results:
                    # 使用SEDF进行生态-语义转换
                    sedf_inputs = {
                        "ecological_data": results["ecological_fingerprint"]
                    }
                    
                    if "text" in batch and isinstance(batch["text"], torch.Tensor):
                        sedf_inputs["text_data"] = batch["text"]
                        
                    if "traditional_knowledge" in batch and isinstance(batch["traditional_knowledge"], torch.Tensor):
                        sedf_inputs["traditional_knowledge"] = batch["traditional_knowledge"]
                        
                    sedf_outputs = self.sedf(sedf_inputs)
                    results.update(sedf_outputs)
                    
                    # 使用EFP进行生态功能预测
                    if "environment" in batch and isinstance(batch["environment"], torch.Tensor):
                        try:
                            efp_inputs = {
                                "input_data": results["ecological_fingerprint"],
                                "environment_change": batch.get("environment_change")
                            }
                            
                            print(f"调用EFP模型，输入形状: {results['ecological_fingerprint'].shape}")
                            efp_outputs = self.efp(efp_inputs)
                            print(f"EFP输出成功，键: {list(efp_outputs.keys())}")
                            results.update(efp_outputs)
                        except Exception as e:
                            print(f"EFP处理失败: {str(e)}")
                            traceback.print_exc()
                        
                    # 使用ESAT进行生态语义对齐
                    if "semantic_features" in sedf_outputs:
                        esat_inputs = {
                            "semantic_features": sedf_outputs["semantic_features"],
                            "env_context": batch.get("environment") if isinstance(batch.get("environment"), torch.Tensor) else None
                        }
                        
                        esat_outputs = self.esat(esat_inputs)
                        results["semantic_alignment"] = esat_outputs
        
        # 在前向传播过程中发生错误时
        except Exception as e:
            print(f"前向传播过程中发生错误: {str(e)}")
            traceback.print_exc()
            # 确保至少返回一个空结果
            if not results:
                device = next(self.parameters()).device
                # 更安全地获取批次大小
                batch_size = 1  # 默认值
                for v in batch.values():
                    if isinstance(v, torch.Tensor):
                        batch_size = v.size(0)
                        break
                results["fused_representation"] = torch.zeros(batch_size, self.config["hidden_dim"]).to(device)

        return results

# 在train函数中添加早停
def train(model, train_loader, val_loader, config):
    """训练BioHarmony模型"""
    device = torch.device(config["device"])
    model = model.to(device)
    
    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # 调试模式开始 - 只处理少量批次
    debug_mode = False  # 设置为True启用调试
    
    if debug_mode:
        print("\n=== 运行调试模式 ===")
        # 处理少量批次
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 2:  # 只测试2个批次
                break
                
            print(f"\n--- 处理批次 {batch_idx+1}/2 ---")
            print(f"批次包含的键: {list(batch.keys())}")
            
            # 检查每个模态的数据类型和形状
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    print(f"模态 '{key}': 张量, 形状 {batch[key].shape}, 类型 {batch[key].dtype}")
                elif isinstance(batch[key], list):
                    print(f"模态 '{key}': 列表, 长度 {len(batch[key])}")
                    if batch[key] and isinstance(batch[key][0], torch.Tensor):
                        print(f"  - 第一个元素: 张量, 形状 {batch[key][0].shape}")
                else:
                    print(f"模态 '{key}': {type(batch[key])}")
            
            # 将数据移至设备
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # 前向传播
            try:
                print("\n尝试前向传播...")
                outputs = model(batch)
                print("前向传播成功!")
                
                # 打印输出
                print(f"模型输出键: {list(outputs.keys())}")
                
                # 计算简单损失
                loss = torch.tensor(0.0, device=device)
                if "fused_representation" in outputs:
                    repr_loss = torch.mean(outputs["fused_representation"]**2) * 0.001
                    loss += repr_loss
                    print(f"表示损失: {repr_loss.item()}")
                    
                if "ecological_fingerprint" in outputs:
                    eco_loss = torch.mean(outputs["ecological_fingerprint"]**2) * 0.001
                    loss += eco_loss
                    print(f"生态指纹损失: {eco_loss.item()}")
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print(f"批次 {batch_idx+1} 训练完成，总损失: {loss.item()}")
            except Exception as e:
                print(f"批次处理失败: {str(e)}")
                traceback.print_exc()
        
        print("\n=== 调试模式完成 ===")
        return
    # 调试模式结束 - 只处理少量批次

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config["num_epochs"]
    )
    
    # 定义损失函数
    mse_loss = nn.MSELoss()
    
    # 训练循环
    best_val_loss = float('inf')


    for epoch in range(config["num_epochs"]):
        model.train()
        train_losses = []
        
        # 训练一个轮次
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            # 将数据移到设备
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # 前向传播
            outputs = model(batch)
            
            # 计算损失
            loss = 0.0
            
            if config.get("use_modaequil", False):
                # 分类损失
                if "logits" in outputs and "labels" in batch:
                    # 如果有平衡标签，使用平衡标签
                    target_labels = outputs.get("balanced_labels", batch["labels"])
                    if target_labels is not None and "logits" in outputs:
                        classification_loss = F.cross_entropy(outputs["logits"], target_labels)
                        loss += classification_loss
                
                # 多样性损失
                if "diversity_loss" in outputs:
                    loss += outputs["diversity_loss"]
                    
                # 不确定性正则化损失
                if "uncertainty" in outputs:
                    # 鼓励合理的不确定性估计（既不太高也不太低）
                    uncertainty_loss = torch.mean(outputs["uncertainty"])
                    loss += 0.01 * uncertainty_loss  # 小权重
            # else:
            
            # 生态指纹重建损失
            if "ecological_fingerprint" in outputs and "ecological_fingerprint" in batch:
                fingerprint_loss = mse_loss(
                    outputs["ecological_fingerprint"],
                    batch["ecological_fingerprint"]
                )
                loss += fingerprint_loss
            
            # 语义一致性损失
            if "text_consistency" in outputs:
                loss += (1.0 - outputs["text_consistency"])
            
            # 生态-语义转换一致性损失
            if "eco_reconstructed" in outputs and "ecological_fingerprint" in batch:
                eco_consistency_loss = mse_loss(
                    outputs["eco_reconstructed"],
                    batch["ecological_fingerprint"]
                )
                loss += eco_consistency_loss
            
            # 量子状态表示KL散度损失
            if "mu" in outputs and "log_var" in outputs:
                kl_loss = -0.5 * torch.sum(
                    1 + outputs["log_var"] - outputs["mu"].pow(2) - outputs["log_var"].exp()
                )
                loss += 0.001 * kl_loss  # 使用小权重防止KL散度支配损失
                
            # 环境适应性预测损失
            if "adapted_fingerprint" in outputs and "target_adaptation" in batch:
                adaptation_loss = mse_loss(
                    outputs["adapted_fingerprint"],
                    batch["target_adaptation"]
                )
                loss += adaptation_loss
                
            # 概念具象化一致性损失
            if "semantic_alignment" in outputs and "concretization" in outputs["semantic_alignment"]:
                if isinstance(outputs["semantic_alignment"]["concretization"], dict) and "consistency" in outputs["semantic_alignment"]["concretization"]:
                    loss += outputs["semantic_alignment"]["concretization"]["consistency"]
            
            
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均训练损失
        avg_train_loss = np.mean(train_losses)
        
        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # 将数据移到设备
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                
                # 前向传播
                outputs = model(batch)
                
                # 计算损失（与训练相同）
                loss = 0.0
                
                if "ecological_fingerprint" in outputs and "ecological_fingerprint" in batch:
                    fingerprint_loss = mse_loss(
                        outputs["ecological_fingerprint"],
                        batch["ecological_fingerprint"]
                    )
                    loss += fingerprint_loss
                
                if "text_consistency" in outputs:
                    loss += (1.0 - outputs["text_consistency"])
                
                if "eco_reconstructed" in outputs and "ecological_fingerprint" in batch:
                    eco_consistency_loss = mse_loss(
                        outputs["eco_reconstructed"],
                        batch["ecological_fingerprint"]
                    )
                    loss += eco_consistency_loss
                
                val_losses.append(loss.item())
        
        # 计算平均验证损失
        avg_val_loss = np.mean(val_losses)
        
        # 打印进度
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if not os.path.exists(config["checkpoint_dir"]):
                os.makedirs(config["checkpoint_dir"])
            torch.save(model.state_dict(), os.path.join(config["checkpoint_dir"], "best_model.pth"))
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")

# 在main.py中添加自监督预训练选项
def pretrain_self_supervised(model, data_loader, config):
    """使用对比学习进行自监督预训练"""
    device = torch.device(config["device"])
    model = model.to(device)
    
    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"] * 0.1,  # 较低的学习率用于预训练
        weight_decay=config["weight_decay"]
    )
    
    # 对比损失
    contrastive_loss = nn.CrossEntropyLoss()
    
    # 温度参数
    temperature = 0.07
    
    for epoch in range(10):  # 预训练轮次
        model.train()
        train_losses = []
        
        for batch in tqdm(data_loader, desc=f"Self-supervised Pretraining {epoch+1}/10"):
            # 将数据移到设备
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # 获取不同模态的表示
            modalities = []
            for modality in ["genome", "image", "audio", "text"]:
                if modality in batch and batch[modality] is not None:
                    if modality == "text":
                        # 处理文本
                        modalities.append(model.sedf.encode_text(batch[modality]))
                    else:
                        # 为其他模态创建编码
                        encoder = getattr(model.efrn.encoders, modality, None)
                        if encoder is not None:
                            modalities.append(encoder(batch[modality]))
            
            if len(modalities) < 2:
                continue  # 需要至少两种模态
                
            # 计算模态间的相似度矩阵
            similarities = []
            labels = []
            
            for i, mod_i in enumerate(modalities):
                for j, mod_j in enumerate(modalities):
                    if i != j:
                        # 计算归一化的特征
                        mod_i_norm = F.normalize(mod_i, dim=1)
                        mod_j_norm = F.normalize(mod_j, dim=1)
                        
                        # 计算余弦相似度
                        sim = torch.matmul(mod_i_norm, mod_j_norm.t()) / temperature
                        similarities.append(sim)
                        
                        # 对角线上的元素是正例
                        labels.append(torch.arange(sim.size(0)).to(device))
            
            # 计算对比损失
            loss = 0
            for sim, lbl in zip(similarities, labels):
                loss += contrastive_loss(sim, lbl)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_loss = sum(train_losses) / len(train_losses)
        print(f"Pretraining Epoch {epoch+1} - Loss: {avg_loss:.4f}")
    
    return model

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BioHarmony Training Script")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--use_modaequil", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to use ModaEquil algorithm")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # 更新配置
    CONFIG["data_path"] = args.data_path
    CONFIG["batch_size"] = args.batch_size
    CONFIG["num_epochs"] = args.num_epochs
    CONFIG["learning_rate"] = args.lr
    CONFIG["device"] = args.device
    CONFIG["use_modaequil"] = args.use_modaequil

    # 创建数据加载器
    train_loader, val_loader = get_data_loaders(CONFIG)
    
    # 创建模型
    model = BioHarmony(CONFIG)
    
    # 训练模型
    train(model, train_loader, val_loader, CONFIG)
    
    print("Training completed!")

if __name__ == "__main__":
    main()