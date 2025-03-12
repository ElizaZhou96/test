# config.py
import torch

CONFIG = {
    # 数据配置
    "data_path": "./data/",
    "modalities": ["genome", "image", "audio", "text", "environment"],
    "batch_size": 32,
    
    # 模型配置
    "hidden_dim": 768,
    "num_heads": 12,
    "num_layers": 6,
    "dropout": 0.1,
    "resonance_strength": 0.7,
    "quantum_samples": 5,
    "integration_strength": 0.5,
    
    # 训练配置
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "num_epochs": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # 路径配置
    "checkpoint_dir": "./checkpoints/",
    "log_dir": "./logs/",
    
    # 数据集特定参数
    "num_species": 1000,
    "num_ecosystems": 50,
    "ecological_features": 128,
    "semantic_features": 256,

    "use_modaequil": True,  # 是否使用ModaEquil
    "num_classes": 10,      # 默认类别数量
    "modality_dims": {      # 各模态输入维度
        "genome": 1024,
        "image": 3*224*224,
        "audio": 512,
        "text": 768,
        "environment": 256
    }
}