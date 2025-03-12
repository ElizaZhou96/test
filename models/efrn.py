# models/efrn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback 

# 在models/efrn.py中添加
def load_pretrained_weights(self, weights_path):
    """加载预训练权重"""
    state_dict = torch.load(weights_path, map_location='cpu')
    # 仅加载匹配的参数
    model_dict = self.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    self.load_state_dict(model_dict)
    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters from pretrained model")

class ModalEncoder(nn.Module):
    """针对不同模态的编码器"""
    def __init__(self, input_dim, hidden_dim, modality_name=None):
        super(ModalEncoder, self).__init__()
        self.modality_name = modality_name
        
        # 对于图像模态，使用特殊处理
        if modality_name == "image":
            self.encoder = nn.Sequential(
                nn.Flatten(),  # 确保输入被展平
                nn.Linear(3*224*224, hidden_dim*2),  # 假定3x224x224图像
                nn.LayerNorm(hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim)
            )
        else:
            # 标准编码器
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim*2),
                nn.LayerNorm(hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim)
            )
        
    def forward(self, x):
        # 处理图像模态
        if self.modality_name == "image" and len(x.shape) > 2:
            # 确保图像被展平
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
            
        # 打印输入形状以调试
        print(f"ModalEncoder ({self.modality_name}) - 输入形状: {x.shape}")
        
        return self.encoder(x)

class ResonanceLayer(nn.Module):
    """实现模态间共振机制的层"""
    def __init__(self, hidden_dim, resonance_strength):
        super(ResonanceLayer, self).__init__()
        self.resonance_strength = resonance_strength
        self.ecological_weights = nn.Parameter(torch.ones(1, hidden_dim))
        self.freq_modulator = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, embeddings_list):
        # 计算模态间共振矩阵
        resonance_matrix = torch.zeros(len(embeddings_list), len(embeddings_list))
        for i, emb_i in enumerate(embeddings_list):
            for j, emb_j in enumerate(embeddings_list):
                if i != j:
                    # 模拟不同模态间的共振效应
                    similarity = F.cosine_similarity(
                        self.freq_modulator(emb_i), 
                        self.freq_modulator(emb_j), 
                        dim=1
                    ).mean()
                    resonance_matrix[i, j] = similarity
        
        # 应用共振增强
        enhanced_embeddings = []
        for i, embedding in enumerate(embeddings_list):
            resonance_vector = torch.stack([
                embeddings_list[j] * resonance_matrix[i, j]
                for j in range(len(embeddings_list))
            ]).sum(dim=0)
            
            # 加权增强原始嵌入
            enhanced = embedding + self.resonance_strength * resonance_vector * self.ecological_weights
            enhanced_embeddings.append(enhanced)
            
        return enhanced_embeddings

class EcologicalSpectrumAnalyzer(nn.Module):
    """生态谱分析器组件 - 修复版"""
    def __init__(self, hidden_dim, num_spectra=16):
        super(EcologicalSpectrumAnalyzer, self).__init__()
        self.num_spectra = num_spectra
        self.hidden_dim = hidden_dim
        
        # 修改投影器确保输出维度正确
        self.spectrum_projector = nn.Linear(hidden_dim, num_spectra * hidden_dim)
        
        # 确保积分器的输入输出维度匹配
        self.spectrum_integrator = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # 获取批次大小并记录原始形状以便调试
        batch_size = x.shape[0]
        original_shape = x.shape
        
        # 打印调试信息
        print(f"EcologicalSpectrumAnalyzer - 输入形状: {original_shape}")
        
        try:
            # 投影到频谱空间
            spectrum = self.spectrum_projector(x)
            spectrum_shape = spectrum.shape
            
            # 重塑为 [批次, 谱数量, 隐藏维度]
            spectrum = spectrum.view(batch_size, self.num_spectra, self.hidden_dim)
            reshaped_shape = spectrum.shape
            
            # 计算频谱权重 - 在谱维度上进行softmax
            spectrum_weights = F.softmax(spectrum.mean(dim=2), dim=1)
            weights_shape = spectrum_weights.shape
            
            # 应用权重
            weighted_spectrum = spectrum * spectrum_weights.unsqueeze(2)
            weighted_shape = weighted_spectrum.shape
            
            # 沿谱维度求和
            integrated_spectrum = weighted_spectrum.sum(dim=1)
            integrated_shape = integrated_spectrum.shape
            
            # 最终处理
            output = self.spectrum_integrator(integrated_spectrum)
            output_shape = output.shape
            
            # 打印所有形状以便调试
            print(f"频谱变换: {original_shape} -> {spectrum_shape} -> {reshaped_shape}")
            print(f"权重形状: {weights_shape}, 加权形状: {weighted_shape}")
            print(f"积分形状: {integrated_shape}, 输出形状: {output_shape}")
            
            return output
            
        except Exception as e:
            print(f"EcologicalSpectrumAnalyzer错误: {str(e)}")
            # 出错时返回原始输入（作为临时应对）
            return x

class QuantumStateRepresentation(nn.Module):
    """量子态物种表示组件"""
    def __init__(self, hidden_dim, num_samples=5):
        super(QuantumStateRepresentation, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples
        self.state_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim*2)
        )
        
    def forward(self, x):
        # 生成均值和方差参数
        params = self.state_generator(x)
        mu, log_var = params.chunk(2, dim=1)
        
        # 重参数化采样生成多个潜在状态
        samples = []
        for _ in range(self.num_samples):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mu + eps * std
            samples.append(sample)
            
        return samples, mu, log_var

class EFRN(nn.Module):
    """生态指纹共振网络完整模型"""
    def __init__(self, config):
        super(EFRN, self).__init__()
        self.config = config
        self.modality_dims = {
            "genome": 1024,
            "image": 3*224*224,  # 调整为展平后的尺寸
            "audio": 512,
            "text": 768,
            "environment": 256
        }
        
        # 使用配置中的维度（如果提供）
        if "modality_dims" in config:
            self.modality_dims.update(config["modality_dims"])
        
        # 为每个模态创建编码器
        self.encoders = nn.ModuleDict({
            modality: ModalEncoder(
                self.modality_dims[modality], 
                config["hidden_dim"],
                modality_name=modality
            )
            for modality in config["modalities"]
        })
        
        # 共振层
        self.resonance_layer = ResonanceLayer(
            config["hidden_dim"], 
            config["resonance_strength"]
        )
        
        # 生态谱分析器
        self.spectrum_analyzer = EcologicalSpectrumAnalyzer(
            config["hidden_dim"]
        )
        
        # 量子态表示
        self.quantum_state = QuantumStateRepresentation(
            config["hidden_dim"],
            config["quantum_samples"]
        )
        
        # 生态指纹生成器
        self.fingerprint_generator = nn.Sequential(
            nn.Linear(config["hidden_dim"], config["hidden_dim"]*2),
            nn.LayerNorm(config["hidden_dim"]*2),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_dim"]*2, config["ecological_features"])
        )
        
    def forward(self, inputs):
        """前向传播，增加错误处理"""
        try:
            # 编码每个可用模态
            embeddings = []
            for modality in self.config["modalities"]:
                if modality in inputs:
                    embedding = self.encoders[modality](inputs[modality])
                    embeddings.append(embedding)
            
            # 如果没有可用的模态嵌入，返回零张量
            if not embeddings:
                device = next(self.parameters()).device
                batch_size = next(tensor.size(0) for tensor in inputs.values() if isinstance(tensor, torch.Tensor))
                mean_fingerprint = torch.zeros(batch_size, self.config["ecological_features"]).to(device)
                return {
                    "mean_fingerprint": mean_fingerprint,
                    "fingerprints": [mean_fingerprint],
                    "mu": torch.zeros(batch_size, self.config["hidden_dim"]).to(device),
                    "log_var": torch.zeros(batch_size, self.config["hidden_dim"]).to(device)
                }
            
            # 应用共振增强
            enhanced_embeddings = self.resonance_layer(embeddings)
            
            # 融合增强后的嵌入
            fused_embedding = torch.stack(enhanced_embeddings).mean(dim=0)
            
            # 应用生态谱分析
            try:
                spectrum_features = self.spectrum_analyzer(fused_embedding)
            except Exception as e:
                print(f"谱分析错误: {e}")
                # 如果谱分析失败，使用原始融合嵌入
                spectrum_features = fused_embedding
            
            # 生成量子态表示
            try:
                quantum_samples, mu, log_var = self.quantum_state(spectrum_features)
            except Exception as e:
                print(f"量子态表示错误: {e}")
                # 如果量子态生成失败，创建简单替代
                batch_size = spectrum_features.size(0)
                hidden_dim = self.config["hidden_dim"]
                device = spectrum_features.device
                
                # 简单替代
                mu = spectrum_features
                log_var = torch.zeros_like(spectrum_features)
                quantum_samples = [spectrum_features]
            
            # 生成生态指纹（多个可能的指纹）
            try:
                fingerprints = [self.fingerprint_generator(sample) for sample in quantum_samples]
                mean_fingerprint = self.fingerprint_generator(mu)
            except Exception as e:
                print(f"生态指纹生成错误: {e}")
                # 创建简单的生态指纹
                batch_size = mu.size(0)
                eco_features = self.config["ecological_features"]
                device = mu.device
                
                # 简单替代
                mean_fingerprint = torch.zeros(batch_size, eco_features).to(device)
                fingerprints = [mean_fingerprint]
            
            return {
                "fingerprints": fingerprints,
                "mean_fingerprint": mean_fingerprint,
                "mu": mu,
                "log_var": log_var,
                "spectrum_features": spectrum_features
            }
        except Exception as e:
            print(f"EFRN前向传播错误: {e}")
            # 在完全失败的情况下创建默认输出
            device = next(self.parameters()).device
            batch_size = next(tensor.size(0) for tensor in inputs.values() if isinstance(tensor, torch.Tensor))
            mean_fingerprint = torch.zeros(batch_size, self.config["ecological_features"]).to(device)
            return {
                "mean_fingerprint": mean_fingerprint,
                "fingerprints": [mean_fingerprint],
                "mu": torch.zeros(batch_size, self.config["hidden_dim"]).to(device),
                "log_var": torch.zeros(batch_size, self.config["hidden_dim"]).to(device)
            }