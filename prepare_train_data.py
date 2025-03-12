# regenerate_data.py
import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import torch

from config import CONFIG

def create_directories(base_path):
    """创建所需的目录结构"""
    subdirs = [
        "genomes", "images", "audio_features", 
        "descriptions", "environments", "traditional_knowledge"
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(base_path, subdir), exist_ok=True)

def create_metadata(base_path, num_samples=100, num_classes=10):
    """创建metadata.csv文件，确保包含class_id"""
    data = {
        "species_id": [f"sp_{i}" for i in range(num_samples)],
        "scientific_name": [f"Species genus_{i}" for i in range(num_samples)],
        "has_genome": [True for i in range(num_samples)],  # 确保所有样本都有基因组数据
        "has_image": [i % 3 == 0 for i in range(num_samples)],
        "has_audio": [i % 4 == 0 for i in range(num_samples)],
        "has_text": [True for i in range(num_samples)],
        "ecosystem_id": [f"eco_{i % 5}" for i in range(num_samples)],
        "class_id": [i % num_classes for i in range(num_samples)]  # 添加类别ID
    }
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(base_path, "metadata.csv"), index=False)
    return df

def create_synthetic_data(base_path, metadata):
    """为每个物种创建合成数据"""
    for _, row in metadata.iterrows():
        species_id = row["species_id"]
        ecosystem_id = row["ecosystem_id"]
        
        # 生成基因组数据 (所有样本)
        genome_data = np.random.randn(1024).astype(np.float32)
        np.save(os.path.join(base_path, "genomes", f"{species_id}.npy"), genome_data)
        
        # 生成图像数据 (部分样本)
        if row["has_image"]:
            # 生成224x224x3的随机图像
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(base_path, "images", f"{species_id}.jpg"))
        
        # 生成音频特征 (部分样本)
        if row["has_audio"]:
            audio_data = np.random.randn(512).astype(np.float32)
            np.save(os.path.join(base_path, "audio_features", f"{species_id}.npy"), audio_data)
        
        # 生成文本描述 (所有样本)
        description = f"This is a synthetic description for species {species_id}. It belongs to ecosystem {ecosystem_id}."
        with open(os.path.join(base_path, "descriptions", f"{species_id}.txt"), "w") as f:
            f.write(description)
        
        # 生成传统知识
        traditional = f"Traditional ecological knowledge about {species_id} in ecosystem {ecosystem_id}."
        with open(os.path.join(base_path, "traditional_knowledge", f"{species_id}.txt"), "w") as f:
            f.write(traditional)
    
    # 为每个生态系统创建环境数据
    for eco_id in set(metadata["ecosystem_id"]):
        env_data = np.random.randn(256).astype(np.float32)
        np.save(os.path.join(base_path, "environments", f"{eco_id}.npy"), env_data)

def main():
    # 清理现有数据目录（可选）
    print("清理现有数据目录...")
    if os.path.exists("data"):
        shutil.rmtree("data")
    
    # 准备训练集
    train_path = "data/train"
    create_directories(train_path)
    num_classes = CONFIG.get("num_classes", 10)
    train_metadata = create_metadata(train_path, num_samples=100, num_classes=num_classes)
    create_synthetic_data(train_path, train_metadata)
    
    # 准备验证集
    val_path = "data/val"
    create_directories(val_path)
    val_metadata = create_metadata(val_path, num_samples=20, num_classes=num_classes)
    create_synthetic_data(val_path, val_metadata)
    
    print("合成数据准备完成! 包含类别标签。")

if __name__ == "__main__":
    main()