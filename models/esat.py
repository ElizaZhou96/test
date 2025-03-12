# models/esat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback 

class EnvironmentalContextAttention(nn.Module):
    """环境敏感注意力机制"""
    def __init__(self, hidden_dim, num_heads=8):
        super(EnvironmentalContextAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 多头注意力组件
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 修改环境调制组件，使其能够处理不同维度的环境上下文
        # 首先使用投影将env_context从任意维度映射到hidden_dim
        self.env_projector = nn.Linear(256, hidden_dim)  # 256是env_context的维度
        
        # 然后使用调制器处理投影后的环境上下文
        self.env_modulator = nn.Sequential(
            nn.Linear(hidden_dim, num_heads),
            nn.Sigmoid()
        )
        
    def forward(self, x, env_context=None):
        batch_size = x.shape[0]
        
        # 生成Q, K, V投影
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 如果有环境上下文，调制注意力头
        if env_context is not None:
            # 先投影环境上下文到正确的维度
            projected_env = self.env_projector(env_context)
            
            # 为每个注意力头生成一个环境调制因子
            env_weights = self.env_modulator(projected_env).unsqueeze(1)  # [batch, 1, num_heads]
            
            # 应用环境调制
            scores = scores * env_weights.unsqueeze(-1)
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 计算加权值
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # 最终投影
        output = self.out_proj(context)
        
        return output

class MultiLevelInterpreter(nn.Module):
    """多层次生态解释器 - 生态系统层级感知版本"""
    def __init__(self, hidden_dim, num_levels=4):  # 默认4个层次：个体、种群、群落、生态系统
        super(MultiLevelInterpreter, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.level_size = hidden_dim // num_levels
        
        # 每个生态层次的专用解释器
        self.level_interpreters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.level_size)  # 每个层次提取特定维度的信息
            )
            for _ in range(num_levels)
        ])
        
        # 生态层次重要性评估器 - 决定在当前语境下哪个层次最相关
        self.level_importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_levels),
            nn.Softmax(dim=1)
        )
        
        # 层次整合网络 - 将所有层次的解释重新整合为完整表示
        self.level_integrator = nn.Linear(self.level_size * num_levels, hidden_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 评估每个生态层次的相关性
        importance = self.level_importance(x)  # [batch, num_levels]
        
        # 在每个层次上生成解释
        level_outputs = []
        for i, interpreter in enumerate(self.level_interpreters):
            # 生成此层次的解释
            level_output = interpreter(x)  # [batch, level_size]
            level_outputs.append(level_output)
            
        # 整合所有层次的解释
        stacked_outputs = torch.cat(level_outputs, dim=1)  # [batch, level_size * num_levels]
        
        # 应用权重并整合
        integrated_output = self.level_integrator(stacked_outputs)
        
        # 应用残差连接 - 保留原始信息
        final_output = integrated_output + x
        
        return final_output, importance

class ConceptConcretizer(nn.Module):
    """概念具象化网络"""
    def __init__(self, hidden_dim, num_concepts=16, num_indicators=8):
        super(ConceptConcretizer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_concepts = num_concepts
        self.num_indicators = num_indicators
        
        # 概念提取器
        self.concept_extractor = nn.Linear(hidden_dim, num_concepts * hidden_dim // 4)
        
        # 指标生成器
        self.indicator_generator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim // 4, hidden_dim // 8),
                nn.LayerNorm(hidden_dim // 8),
                nn.ReLU(),
                nn.Linear(hidden_dim // 8, num_indicators)
            )
            for _ in range(num_concepts)
        ])
        
        # 抽象化器（从指标到概念）
        self.abstractifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_indicators, hidden_dim // 8),
                nn.LayerNorm(hidden_dim // 8),
                nn.ReLU(),
                nn.Linear(hidden_dim // 8, hidden_dim // 4)
            )
            for _ in range(num_concepts)
        ])
        
        # 概念集成器
        self.concept_integrator = nn.Linear(num_concepts * hidden_dim // 4, hidden_dim)
        
    def concretize(self, x):
        """从抽象概念到具体指标"""
        batch_size = x.shape[0]
        
        # 提取概念表示
        concept_features = self.concept_extractor(x)
        concept_features = concept_features.view(batch_size, self.num_concepts, -1)
        
        # 为每个概念生成指标
        indicators = []
        for i, generator in enumerate(self.indicator_generator):
            concept = concept_features[:, i]
            indicator = generator(concept)
            indicators.append(indicator)
            
        # 堆叠所有指标
        all_indicators = torch.stack(indicators, dim=1)  # [batch, num_concepts, num_indicators]
        
        return concept_features, all_indicators
    
    def abstractify(self, indicators):
        """从具体指标到抽象概念"""
        batch_size = indicators.shape[0]
        
        # 为每个概念从指标重建抽象表示
        abstract_concepts = []
        for i, abstractifier in enumerate(self.abstractifier):
            concept_indicators = indicators[:, i]
            abstract_concept = abstractifier(concept_indicators)
            abstract_concepts.append(abstract_concept)
            
        # 堆叠所有概念
        all_concepts = torch.cat(abstract_concepts, dim=1)  # [batch, num_concepts * (hidden_dim//4)]
        
        # 整合为单一表示
        integrated = self.concept_integrator(all_concepts)
        
        return integrated
    
    def forward(self, x, mode="both"):
        if mode == "concretize" or mode == "both":
            concepts, indicators = self.concretize(x)
            if mode == "concretize":
                return concepts, indicators
                
        if mode == "abstractify" or mode == "both":
            if mode == "abstractify":
                # 假设输入直接是指标
                abstractified = self.abstractify(x)
                return abstractified
            else:
                # 使用从concretize生成的指标
                abstractified = self.abstractify(indicators)
                
        # 检查概念一致性
        consistency = F.mse_loss(
            x, abstractified
        )
                
        return {
            "concepts": concepts,
            "indicators": indicators,
            "abstractified": abstractified,
            "consistency": consistency
        }

class ESAT(nn.Module):
    """生态语义对齐变换器完整模型"""
    def __init__(self, config):
        super(ESAT, self).__init__()
        self.config = config
        
        # 环境敏感注意力
        self.env_attention = EnvironmentalContextAttention(
            config["hidden_dim"],
            num_heads=config["num_heads"]
        )
        
        # 多层次解释器
        self.multi_level = MultiLevelInterpreter(
            config["hidden_dim"]
        )
        
        # 概念具象化网络
        self.concretizer = ConceptConcretizer(
            config["hidden_dim"]
        )
        
        # 变换器层
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=config["hidden_dim"],
            nhead=config["num_heads"],
            dim_feedforward=config["hidden_dim"]*4,
            dropout=config["dropout"],
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=config["num_layers"]
        )
        
        # 输出映射
        self.output_mapping = nn.Sequential(
            nn.Linear(config["hidden_dim"], config["hidden_dim"]),
            nn.LayerNorm(config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_dim"], config["semantic_features"])
        )
        
    def forward(self, inputs):
        semantic_features = inputs["semantic_features"]
        env_context = inputs.get("env_context")
        
        # 应用环境敏感注意力
        if semantic_features.dim() == 2:
            # 添加序列维度以适应变换器
            semantic_features = semantic_features.unsqueeze(1)
            
        if env_context is not None:
            attended = self.env_attention(semantic_features, env_context)
        else:
            attended = semantic_features
            
        # 应用变换器编码器
        transformed = self.transformer_encoder(attended)
        
        # 取第一个位置的输出（如果是序列）
        if transformed.dim() == 3:
            transformed = transformed[:, 0]
        
        # 应用多层次解释
        multi_level_output, level_importance = self.multi_level(transformed)
        
        # 应用概念具象化
        concretization = self.concretizer(multi_level_output)
        
        # 生成最终输出
        output = self.output_mapping(multi_level_output)
        
        return {
            "output": output,
            "level_importance": level_importance,
            "concretization": concretization,
            "transformed": transformed
        }