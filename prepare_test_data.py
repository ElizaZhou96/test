# prepare_test_data.py
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import shutil

def create_test_directories(base_path):
    """创建测试数据目录结构"""
    subdirs = [
        "genomes", "images", "audio_features", 
        "descriptions", "environments", "traditional_knowledge"
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(base_path, subdir), exist_ok=True)

def create_test_metadata(base_path, num_samples=10, task_type=None):
    """创建测试元数据文件"""
    data = {
        "species_id": [f"test_sp_{i}" for i in range(num_samples)],
        "scientific_name": [f"Test Species genus_{i}" for i in range(num_samples)],
        "has_genome": [True for _ in range(num_samples)],
        "has_image": [i % 2 == 0 for i in range(num_samples)],  # 一半有图像
        "has_audio": [i % 3 == 0 for i in range(num_samples)],  # 三分之一有音频
        "has_text": [True for _ in range(num_samples)],
        "ecosystem_id": [f"test_eco_{i % 5}" for i in range(num_samples)],
        "class_id": [i % 10 for i in range(num_samples)]
    }
    
    # 为特定任务调整数据
    if task_type == "cross_modal":
        # 确保一些样本只有一种模态
        for i in range(num_samples):
            if i % 4 == 0:  # 只有图像，没有音频
                data["has_image"][i] = True
                data["has_audio"][i] = False
            elif i % 4 == 1:  # 只有音频，没有图像
                data["has_image"][i] = False
                data["has_audio"][i] = True
    
    elif task_type == "biodiversity":
        # 添加已知/未知生态系统标记
        data["known_ecosystem"] = [i < num_samples//2 for i in range(num_samples)]
    
    elif task_type == "environmental_change":
        # 添加环境变化标记
        data["has_environmental_change"] = [True for _ in range(num_samples)]
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(base_path, "metadata.csv"), index=False)
    return df

def generate_test_data(base_path, metadata, task_type=None):
    """生成测试数据文件"""
    for _, row in metadata.iterrows():
        species_id = row["species_id"]
        ecosystem_id = row["ecosystem_id"]
        
        # 基因组数据
        if row["has_genome"]:
            genome_data = np.random.randn(1024).astype(np.float32)
            np.save(os.path.join(base_path, "genomes", f"{species_id}.npy"), genome_data)
        
        # 图像数据
        if row["has_image"]:
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(base_path, "images", f"{species_id}.jpg"))
        
        # 音频特征
        if row["has_audio"]:
            audio_data = np.random.randn(512).astype(np.float32)
            np.save(os.path.join(base_path, "audio_features", f"{species_id}.npy"), audio_data)
        
        # 文本描述
        description = f"This is a test description for species {species_id}. It belongs to ecosystem {ecosystem_id}."
        with open(os.path.join(base_path, "descriptions", f"{species_id}.txt"), "w") as f:
            f.write(description)
        
        # 传统知识
        traditional = f"Traditional knowledge about {species_id}: This species is known to local communities as a bioindicator."
        with open(os.path.join(base_path, "traditional_knowledge", f"{species_id}.txt"), "w") as f:
            f.write(traditional)
    
    # 环境数据
    ecosystems = set(metadata["ecosystem_id"])
    for eco_id in ecosystems:
        env_data = np.random.randn(256).astype(np.float32)
        np.save(os.path.join(base_path, "environments", f"{eco_id}.npy"), env_data)
        
        # 为环境变化任务创建未来环境数据
        if task_type == "environmental_change":
            future_env = env_data * 1.2 + np.random.randn(256).astype(np.float32) * 0.1
            np.save(os.path.join(base_path, "environments", f"{eco_id}_future.npy"), future_env)

# 为每个任务创建测试数据
def prepare_all_test_data():
    # 生态指纹生成
    fingerprint_path = "data/test_fingerprint"
    create_test_directories(fingerprint_path)
    metadata = create_test_metadata(fingerprint_path, num_samples=10)
    generate_test_data(fingerprint_path, metadata)
    
    # 跨模态转换
    crossmodal_path = "data/test_crossmodal"
    create_test_directories(crossmodal_path)
    metadata = create_test_metadata(crossmodal_path, num_samples=10, task_type="cross_modal")
    generate_test_data(crossmodal_path, metadata)
    
    # 生物多样性预测
    biodiversity_path = "data/test_biodiversity"
    create_test_directories(biodiversity_path)
    metadata = create_test_metadata(biodiversity_path, num_samples=15, task_type="biodiversity")
    generate_test_data(biodiversity_path, metadata)
    
    # 环境变化应对
    environment_path = "data/test_environment"
    create_test_directories(environment_path)
    metadata = create_test_metadata(environment_path, num_samples=10, task_type="environmental_change")
    generate_test_data(environment_path, metadata, task_type="environmental_change")
    
    # 传统知识整合
    traditional_path = "data/test_traditional"
    create_test_directories(traditional_path)
    metadata = create_test_metadata(traditional_path, num_samples=10)
    generate_test_data(traditional_path, metadata)
    
    print("所有测试数据准备完成！")

if __name__ == "__main__":
    prepare_all_test_data()