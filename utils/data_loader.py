# utils/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import json
from PIL import Image
import torchvision.transforms as transforms
from transformers import BertTokenizer
# 在utils/data_loader.py中增强数据增强
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter
from utils.collate import safe_bioharmony_collate_fn

class BioHarmonyDataset(Dataset):
    """多模态生物多样性数据集"""
    
    def __init__(self, config, split="train", transform=None):
        self.config = config
        self.data_path = os.path.join(config["data_path"], split)
        self.split = split
        # 为训练集使用更强的数据增强
        if split == "train":
            self.transform = transforms.Compose([
                RandomResizedCrop(224, scale=(0.8, 1.0)),
                RandomHorizontalFlip(),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        # 加载物种索引
        metadata_path = os.path.join(self.data_path, "metadata.csv")
        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
        else:
            # 如果没有真实数据，创建合成数据结构以进行测试
            print(f"Creating synthetic data structure for {split} set...")
            self.metadata = self._create_synthetic_metadata()
            
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
    def _create_synthetic_metadata(self):
        """创建合成数据用于测试"""
        num_samples = 100 if self.split == "train" else 20
        
        # 创建基本物种信息
        data = {
            "species_id": [f"sp_{i}" for i in range(num_samples)],
            "scientific_name": [f"Species genus_{i}" for i in range(num_samples)],
            "has_genome": [i % 2 == 0 for i in range(num_samples)],
            "has_image": [i % 3 == 0 for i in range(num_samples)],
            "has_audio": [i % 4 == 0 for i in range(num_samples)],
            "has_text": [True for i in range(num_samples)],
            "ecosystem_id": [f"eco_{i % 5}" for i in range(num_samples)],
            "class_id": [i % self.config.get("num_classes", 10) for i in range(num_samples)]  # 添加类别ID
        }
        
        return pd.DataFrame(data)
    
    def _load_genome_data(self, species_id):
        """加载或生成基因组数据"""
        filepath = os.path.join(self.data_path, "genomes", f"{species_id}.npy")
        if os.path.exists(filepath):
            return np.load(filepath)
        else:
            # 生成合成数据
            return np.random.randn(1024).astype(np.float32)
    
    def _load_image_data(self, species_id):
        """加载或生成图像数据"""
        filepath = os.path.join(self.data_path, "images", f"{species_id}.jpg")
        if os.path.exists(filepath):
            img = Image.open(filepath).convert('RGB')
            return self.transform(img)
        else:
            # 生成合成数据
            return torch.randn(3, 224, 224)
    
    def _load_audio_data(self, species_id):
        """加载或生成音频特征"""
        filepath = os.path.join(self.data_path, "audio_features", f"{species_id}.npy")
        if os.path.exists(filepath):
            return np.load(filepath)
        else:
            # 生成合成数据
            return np.random.randn(512).astype(np.float32)
    
    def _load_text_data(self, species_id):
        """加载或生成文本描述"""
        filepath = os.path.join(self.data_path, "descriptions", f"{species_id}.txt")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return f.read().strip()
        else:
            # 生成合成描述
            return f"This is a synthetic description for species {species_id}."
    
    def _load_environment_data(self, ecosystem_id):
        """加载或生成环境数据"""
        filepath = os.path.join(self.data_path, "environments", f"{ecosystem_id}.npy")
        if os.path.exists(filepath):
            return np.load(filepath)
        else:
            # 生成合成数据
            return np.random.randn(256).astype(np.float32)
    
    def _load_traditional_knowledge(self, species_id):
        """加载或生成传统生态知识"""
        filepath = os.path.join(self.data_path, "traditional_knowledge", f"{species_id}.txt")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return f.read().strip()
        else:
            # 生成合成数据
            return f"Traditional ecological knowledge about {species_id}."
            
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """获取多模态物种数据"""
        row = self.metadata.iloc[idx]
        species_id = row["species_id"]
        ecosystem_id = row["ecosystem_id"]
        
        data = {
            "species_id": species_id,
            "ecosystem_id": ecosystem_id
        }
        
        # 加载各种模态的数据
        if row["has_genome"]:
            data["genome"] = torch.tensor(self._load_genome_data(species_id))
        
        if row["has_image"]:
            data["image"] = self._load_image_data(species_id)
        
        if row["has_audio"]:
            data["audio"] = torch.tensor(self._load_audio_data(species_id))
        
        if row["has_text"]:
            text = self._load_text_data(species_id)
            data["text"] = text
            
            # 获取BERT输入
            tokenized = self.tokenizer(
                text, 
                padding="max_length", 
                truncation=True, 
                max_length=128,
                return_tensors="pt"
            )
            data["text_tokenized"] = {
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0]
            }
        
        # 添加类别标签（使用安全的访问方式）
        if "class_id" in row:
            data["labels"] = torch.tensor(row["class_id"], dtype=torch.long)
        else:
            # 如果没有class_id，使用物种ID的哈希作为伪标签
            # 将hash映射到[0, num_classes-1]范围内
            species_hash = hash(species_id) % self.config.get("num_classes", 10)
            data["labels"] = torch.tensor(species_hash, dtype=torch.long)

        # 加载环境数据
        data["environment"] = torch.tensor(self._load_environment_data(ecosystem_id))
        
        # 加载传统生态知识
        traditional_knowledge = self._load_traditional_knowledge(species_id)
        data["traditional_knowledge"] = traditional_knowledge
        
        # 创建生态指纹（在真实实现中应该来自数据集）
        # 这里生成合成数据
        data["ecological_fingerprint"] = torch.randn(self.config["ecological_features"])
        
        return data

def get_data_loaders(config):
    """创建数据加载器"""
    train_dataset = BioHarmonyDataset(config, split="train")
    val_dataset = BioHarmonyDataset(config, split="val")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"],
        shuffle=True, 
        num_workers=0,  # 使用0进行调试
        collate_fn=safe_bioharmony_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"],
        shuffle=False, 
        num_workers=0,  # 使用0进行调试
        collate_fn=safe_bioharmony_collate_fn
    )
    
    return train_loader, val_loader