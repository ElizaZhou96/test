# models/efp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback 

class DiffusionModel(nn.Module):
    """功能扩散模型核心组件 - 修复版"""
    def __init__(self, feature_dim, hidden_dim, num_steps=50):
        super(DiffusionModel, self).__init__()
        self.feature_dim = feature_dim
        self.num_steps = num_steps
        
        # 噪声预测网络
        self.noise_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # 修改时间步嵌入，确保输出维度匹配feature_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim)  # 改为feature_dim而不是hidden_dim
        )
        
        # 时间条件调制层
        # self.time_modulation = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.SiLU(),
        #     nn.Linear(hidden_dim, hidden_dim)
        # )
        
        # 预定义一个beta调度
        self.betas = torch.linspace(1e-4, 0.02, num_steps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def forward(self, x, t):
        """预测添加到x的噪声 - 使用时间调制而非拼接"""
        # 处理时间嵌入
        t_emb = self.time_embed(t.unsqueeze(-1))
        
        # 使用时间调制而非拼接
        # time_features = self.time_modulation(t_emb)
        
        # 特征调制：乘法或加法调制而非拼接
        # modulated_x = x + time_features
        modulated_x = x + t_emb
        
        # 预测噪声
        return self.noise_predictor(modulated_x)
    
    def sample(self, shape, device, condition=None):
        """生成新样本 - 添加错误处理"""
        try:
            # 从标准正态分布采样
            x = torch.randn(shape).to(device)
            
            # 逐步去噪
            for i in range(self.num_steps - 1, 0, -1):
                t = torch.ones(shape[0], dtype=torch.long).to(device) * i
                t_float = t.float() / self.num_steps
                
                # 预测噪声
                if condition is not None:
                    # 如果有条件情况的特殊处理方式
                    condition_dim = condition.size(1)
                    
                    # 检查条件和特征维度是否匹配
                    if condition_dim != self.feature_dim:
                        print(f"警告: 条件维度 ({condition_dim}) 与特征维度 ({self.feature_dim}) 不匹配")
                        # 调整条件维度
                        if condition_dim > self.feature_dim:
                            condition = condition[:, :self.feature_dim]
                        else:
                            padding = torch.zeros(shape[0], self.feature_dim - condition_dim).to(device)
                            condition = torch.cat([condition, padding], dim=1)
                    
                    # 混合输入和条件 - 使用加法而非拼接
                    conditioned_x = x + 0.1 * condition  # 使用小权重保留原始信息
                    predicted_noise = self.forward(conditioned_x, t_float)
                else:
                    predicted_noise = self.forward(x, t_float)
                
                # 计算权重
                alpha = self.alphas[i]
                alpha_cumprod = self.alphas_cumprod[i]
                alpha_cumprod_prev = self.alphas_cumprod[i-1] if i > 0 else torch.tensor(1.0)
                
                # 计算均值方差
                beta = 1 - alpha
                variance = beta * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)
                mean_coef1 = torch.sqrt(alpha_cumprod_prev) * beta / (1. - alpha_cumprod)
                mean_coef2 = torch.sqrt(alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)
                
                # 去噪步骤
                mean = (x - mean_coef1 * predicted_noise) / mean_coef2
                
                if i > 1:
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(variance) * noise
                else:
                    x = mean
            
            return x
            
        except Exception as e:
            print(f"扩散采样错误: {str(e)}")
            traceback.print_exc()
            # 出错时返回零张量
            return torch.zeros(shape).to(device)

class EFP(nn.Module):
    """生态功能预测器完整模型"""
    def __init__(self, config):
        super(EFP, self).__init__()
        self.config = config
        
        # 条件编码器（用于从一种模态预测另一种模态）
        self.condition_encoder = nn.Sequential(
            nn.Linear(config["ecological_features"], config["hidden_dim"]),
            nn.LayerNorm(config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_dim"], config["ecological_features"])
        )
        
        # 功能扩散模型
        self.diffusion = DiffusionModel(
            config["ecological_features"],
            config["hidden_dim"],
            num_steps=50
        )
        
        # 适应性预测网络
        self.adaptation_network = nn.Sequential(
            nn.Linear(config["ecological_features"]*2, config["hidden_dim"]),
            nn.LayerNorm(config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_dim"], config["ecological_features"])
        )
        
    def predict_ecological_fingerprint(self, partial_data, modality_type=None):
        """从部分数据预测完整生态指纹 - 增强的错误处理"""
        try:
            # 打印输入形状以便调试
            print(f"EFP - 输入形状: {partial_data.shape}, 特征维度: {self.config['ecological_features']}")
            
            # 编码条件
            condition = self.condition_encoder(partial_data)
            print(f"EFP - 条件形状: {condition.shape}")
            
            # 使用扩散模型生成完整指纹
            batch_size = partial_data.shape[0]
            
            # 确认特征维度
            feature_dim = self.config["ecological_features"]
            if condition.size(1) != feature_dim:
                print(f"警告: 条件维度 ({condition.size(1)}) 与特征维度 ({feature_dim}) 不匹配")
            
            # 生成指纹
            full_fingerprint = self.diffusion.sample(
                (batch_size, feature_dim),
                partial_data.device,
                condition
            )
            
            return full_fingerprint
            
        except Exception as e:
            print(f"生态指纹预测错误: {str(e)}")
            traceback.print_exc()
            # 返回零张量作为备用
            return torch.zeros(partial_data.size(0), self.config["ecological_features"]).to(partial_data.device)
    
    def predict_adaptation(self, species_data, environment_change):
        """预测物种对环境变化的适应性"""
        # 组合物种数据和环境变化信息
        combined = torch.cat([species_data, environment_change], dim=1)
        
        # 预测适应后的生态特征
        adapted_features = self.adaptation_network(combined)
        
        return adapted_features
    
    def forward(self, inputs):
        input_data = inputs["input_data"]
        modality = inputs.get("modality")
        environment_change = inputs.get("environment_change")
        
        results = {}
        
        # 预测完整生态指纹
        full_fingerprint = self.predict_ecological_fingerprint(input_data, modality)
        results["full_fingerprint"] = full_fingerprint
        
        # 如果提供了环境变化信息，预测适应性
        if environment_change is not None:
            adaptation = self.predict_adaptation(full_fingerprint, environment_change)
            results["adapted_fingerprint"] = adaptation
            
        return results