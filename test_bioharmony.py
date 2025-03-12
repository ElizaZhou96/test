# test_bioharmony.py
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import traceback

from config import CONFIG
from main import BioHarmony
from utils.data_loader import BioHarmonyDataset, safe_bioharmony_collate_fn

def load_model(config, checkpoint_path):
    """加载训练好的BioHarmony模型"""
    model = BioHarmony(config)
    device = torch.device(config["device"])
    
    # 加载权重
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"成功加载模型: {checkpoint_path}")
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        traceback.print_exc()
    
    model = model.to(device)
    model.eval()
    return model

def create_dataloader(config, data_path, batch_size=4):
    """创建测试数据加载器"""
    # 修改数据路径指向测试数据
    custom_config = config.copy()
    custom_config["data_path"] = data_path
    
    # 创建测试数据集
    test_dataset = BioHarmonyDataset(custom_config, split="")  # 空字符串使其直接使用data_path
    test_dataset.data_path = data_path  # 确保使用正确的路径
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=0,
        collate_fn=safe_bioharmony_collate_fn
    )
    
    return test_loader

def run_test(model, config, task, data_path):
    """执行测试任务"""
    print(f"\n===== 执行任务: {task} =====")
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 根据任务加载相应数据
    test_loader = create_dataloader(config, data_path)
    device = torch.device(config["device"])
    
    # 存储结果
    results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"执行{task}任务"):
            # 将数据移到设备
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # 根据任务修改批次
            if task == "environmental_change" and "environment" in batch:
                # 创建环境变化数据
                eco_ids = batch["ecosystem_id"]
                future_envs = []
                
                for eco_id in eco_ids:
                    future_path = os.path.join(data_path, "environments", f"{eco_id}_future.npy")
                    if os.path.exists(future_path):
                        future_env = np.load(future_path)
                        future_envs.append(future_env)
                    else:
                        # 如果没有特定的未来环境，创建一个简单的变化
                        current_env = batch["environment"][len(future_envs)].cpu().numpy()
                        future_env = current_env * 1.2  # 简单增加20%
                        future_envs.append(future_env)
                
                # 转换为张量并计算环境变化
                future_env_tensor = torch.tensor(np.array(future_envs), device=device)
                batch["environment_change"] = future_env_tensor - batch["environment"]
            
            # 前向传播
            try:
                outputs = model(batch)
                
                # 收集任务相关结果
                task_result = {
                    "species_ids": batch["species_id"],
                    "ecosystem_ids": batch["ecosystem_id"]
                }
                
                # 根据任务收集特定输出
                if task == "fingerprint" and "ecological_fingerprint" in outputs:
                    task_result["fingerprints"] = outputs["ecological_fingerprint"].cpu().numpy()
                
                elif task == "cross_modal":
                    # 检查是否有跨模态预测
                    if "predicted_audio" in outputs:
                        task_result["audio_predictions"] = outputs["predicted_audio"].cpu().numpy()
                    if "predicted_image" in outputs:
                        task_result["image_predictions"] = outputs["predicted_image"].cpu().numpy()
                
                elif task == "biodiversity" and "ecological_fingerprint" in outputs:
                    task_result["fingerprints"] = outputs["ecological_fingerprint"].cpu().numpy()
                
                elif task == "environmental_change" and "adapted_fingerprint" in outputs:
                    task_result["current_fingerprints"] = outputs["ecological_fingerprint"].cpu().numpy()
                    task_result["adapted_fingerprints"] = outputs["adapted_fingerprint"].cpu().numpy()
                
                elif task == "traditional" and "semantic_output" in outputs:
                    task_result["semantic_bridges"] = outputs["semantic_output"].cpu().numpy()
                    if "bridged_knowledge" in outputs:
                        task_result["bridged_knowledge"] = outputs["bridged_knowledge"].cpu().numpy()
                
                results.append(task_result)
            except Exception as e:
                print(f"处理批次时出错: {str(e)}")
                traceback.print_exc()
    
    return results

def visualize_results(results, task):
    """可视化测试结果"""
    if not results:
        print("没有结果可视化")
        return
    
    # 合并所有批次结果
    all_data = {}
    for key in results[0].keys():
        if key in ["species_ids", "ecosystem_ids"]:
            all_data[key] = []
            for result in results:
                all_data[key].extend(result[key])
        else:
            try:
                all_data[key] = np.vstack([result[key] for result in results if key in result])
            except:
                all_data[key] = [item for result in results if key in result for item in result[key]]
    
    # 根据任务创建可视化
    plt.figure(figsize=(12, 8))
    
    if task == "fingerprint" and "fingerprints" in all_data:
        # 可视化生态指纹
        fingerprints = all_data["fingerprints"]
        species_ids = all_data["species_ids"]
        
        for i in range(min(5, len(fingerprints))):
            plt.subplot(1, 5, i+1)
            plt.imshow(fingerprints[i].reshape(8, 16), cmap="viridis")
            plt.title(f"Species: {species_ids[i]}")
            plt.colorbar()
        
        plt.suptitle("生态指纹可视化")
        plt.tight_layout()
        plt.savefig("results/fingerprint_visualization.png")
        print(f"生态指纹可视化已保存到 results/fingerprint_visualization.png")
    
    elif task == "environmental_change" and "adapted_fingerprints" in all_data:
        # 可视化环境适应变化
        current = all_data["current_fingerprints"]
        adapted = all_data["adapted_fingerprints"]
        species_ids = all_data["species_ids"]
        
        # 计算变化幅度
        changes = np.linalg.norm(adapted - current, axis=1)
        indices = np.argsort(changes)[::-1][:5]  # 取变化最大的5个样本
        
        plt.bar(range(len(indices)), changes[indices])
        plt.xticks(range(len(indices)), [species_ids[i] for i in indices], rotation=45)
        plt.title("物种对环境变化的适应性")
        plt.ylabel("变化幅度")
        plt.tight_layout()
        plt.savefig("results/environmental_adaptation.png")
        print(f"环境适应性可视化已保存到 results/environmental_adaptation.png")
    
    elif task == "traditional" and "semantic_bridges" in all_data:
        # 可视化传统知识桥接
        from sklearn.manifold import TSNE
        
        bridges = all_data["semantic_bridges"]
        species_ids = all_data["species_ids"]
        
        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42)
        bridges_2d = tsne.fit_transform(bridges)
        
        plt.scatter(bridges_2d[:, 0], bridges_2d[:, 1], s=100)
        for i, species_id in enumerate(species_ids):
            plt.annotate(species_id, (bridges_2d[i, 0], bridges_2d[i, 1]))
        
        plt.title("传统知识与科学数据的语义桥接")
        plt.tight_layout()
        plt.savefig("results/traditional_knowledge_bridge.png")
        print(f"传统知识桥接可视化已保存到 results/traditional_knowledge_bridge.png")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="BioHarmony模型测试")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth", help="模型路径")
    parser.add_argument("--task", type=str, default="all", 
                        choices=["all", "fingerprint", "cross_modal", "biodiversity", "environmental_change", "traditional"],
                        help="要执行的任务")
    parser.add_argument("--data_dir", type=str, default="data", help="测试数据根目录")
    args = parser.parse_args()
    
    # 加载模型
    model = load_model(CONFIG, args.model)
    
    # 确定要执行的任务
    tasks = ["fingerprint", "cross_modal", "biodiversity", "environmental_change", "traditional"]
    if args.task != "all":
        tasks = [args.task]
    
    # 执行各个任务
    for task in tasks:
        data_path = os.path.join(args.data_dir, f"test_{task}")
        if not os.path.exists(data_path):
            print(f"警告: 测试数据路径 {data_path} 不存在，跳过任务 {task}")
            continue
            
        results = run_test(model, CONFIG, task, data_path)
        visualize_results(results, task)
    
    print("\n所有测试任务完成！")

if __name__ == "__main__":
    main()