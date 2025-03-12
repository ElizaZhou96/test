# models/sedf.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import traceback 

class NarrativeGraph(nn.Module):
    """叙事逻辑推理引擎的故事图实现"""
    def __init__(self, hidden_dim, num_relations=8):
        super(NarrativeGraph, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        
        # 关系编码器
        self.relation_encoders = nn.ModuleList([
            nn.Linear(hidden_dim*2, hidden_dim)
            for _ in range(num_relations)
        ])
        
        # 时间流编码器
        self.temporal_encoder = nn.GRU(
            hidden_dim, hidden_dim, batch_first=True
        )
        
        # 故事图推理层
        self.graph_reasoning = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        
    def forward(self, nodes, edges, temporal_sequence=None):
        # 编码边关系
        encoded_edges = []
        for i, (src, rel_type, dst) in enumerate(edges):
            # 连接源节点和目标节点的表示
            edge_input = torch.cat([nodes[src], nodes[dst]], dim=1)
            # 使用对应类型的关系编码器
            encoded_edge = self.relation_encoders[rel_type](edge_input)
            encoded_edges.append(encoded_edge)
            
        if len(encoded_edges) > 0:
            encoded_edges = torch.stack(encoded_edges)
        else:
            # 如果没有边，返回零张量
            return torch.zeros(nodes.shape[0], self.hidden_dim).to(nodes.device)
        
        # 应用时间流编码（如果提供）
        if temporal_sequence is not None:
            temporal_output, _ = self.temporal_encoder(temporal_sequence)
            # 融合边关系和时间信息
            graph_state = encoded_edges.mean(0) + temporal_output[:, -1]
        else:
            graph_state = encoded_edges.mean(0)
        
        # 应用图推理
        reasoning_output = self.graph_reasoning(graph_state)
        
        return reasoning_output

class CulturalScienceBridge(nn.Module):
    """文化-科学桥接层"""
    def __init__(self, hidden_dim):
        super(CulturalScienceBridge, self).__init__()
        self.scientific_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.cultural_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.alignment_network = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.integration_gate = nn.Sequential(
            nn.Linear(hidden_dim*2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, scientific_data, cultural_knowledge):
        # 编码科学数据和文化知识
        sci_encoded = self.scientific_encoder(scientific_data)
        cul_encoded = self.cultural_encoder(cultural_knowledge)
        
        # 计算整合程度
        combined = torch.cat([sci_encoded, cul_encoded], dim=1)
        integration_level = self.integration_gate(combined)
        
        # 生成对齐向量
        alignment_vector = self.alignment_network(combined)
        
        # 根据整合程度进行加权融合
        integrated = (integration_level * alignment_vector + 
                     (1 - integration_level) * sci_encoded)
        
        return integrated, integration_level

class SEDF(nn.Module):
    """语义-生态双重流动机制完整模型"""
    def __init__(self, config):
        super(SEDF, self).__init__()
        self.config = config
        
        # 加载预训练BERT模型作为语义编码基础
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # 冻结BERT参数
        for param in self.bert_model.parameters():
            param.requires_grad = False
            
        # 语义到生态的转换
        self.semantic_to_ecological = nn.Sequential(
            nn.Linear(768, config["hidden_dim"]),
            nn.LayerNorm(config["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["hidden_dim"], config["ecological_features"])
        )
        
        # 生态到语义的转换
        self.ecological_to_semantic = nn.Sequential(
            nn.Linear(config["ecological_features"], config["hidden_dim"]),
            nn.LayerNorm(config["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["hidden_dim"], 768)
        )
        
        # 叙事逻辑推理引擎
        self.narrative_reasoning = NarrativeGraph(config["hidden_dim"])
        
        # 文化-科学桥接层
        self.cultural_bridge = CulturalScienceBridge(config["hidden_dim"])
        
        # 输出映射层
        self.output_projector = nn.Linear(
            config["hidden_dim"], 
            config["semantic_features"]
        )
        
    def encode_text(self, text_list):
        # 使用BERT编码文本
        inputs = self.tokenizer(
            text_list, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(next(self.parameters()).device)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        return outputs.last_hidden_state[:, 0, :]  # 使用[CLS]标记的表示
    
    def to_semantic(self, ecological_data):
        """生态到语义的转换"""
        semantic_features = self.ecological_to_semantic(ecological_data)
        return semantic_features
    
    def to_ecological(self, semantic_data):
        """语义到生态的转换"""
        ecological_features = self.semantic_to_ecological(semantic_data)
        return ecological_features
        
    def forward(self, inputs):
        ecological_data = inputs.get("ecological_data")
        text_data = inputs.get("text_data")
        scientific_measures = inputs.get("scientific_measures")
        traditional_knowledge = inputs.get("traditional_knowledge")
        narrative_nodes = inputs.get("narrative_nodes")
        narrative_edges = inputs.get("narrative_edges")
        temporal_data = inputs.get("temporal_data")
        
        results = {}
        
        # 处理生态数据（如果有）
        if ecological_data is not None:
            semantic_from_eco = self.to_semantic(ecological_data)
            results["semantic_features"] = semantic_from_eco
            # 双向一致性检查
            eco_reconstructed = self.to_ecological(semantic_from_eco)
            results["eco_reconstructed"] = eco_reconstructed
            
        # 处理文本数据（如果有）
        if text_data is not None:
            text_embeddings = self.encode_text(text_data)
            ecological_from_text = self.to_ecological(text_embeddings)
            results["ecological_from_text"] = ecological_from_text
            # 双向一致性检查
            text_reconstructed = self.to_semantic(ecological_from_text)
            results["text_consistency"] = F.cosine_similarity(
                text_embeddings, text_reconstructed, dim=1
            ).mean()
            
        # 应用叙事逻辑推理（如果提供了故事图数据）
        if narrative_nodes is not None and narrative_edges is not None:
            narrative_output = self.narrative_reasoning(
                narrative_nodes, narrative_edges, temporal_data
            )
            results["narrative_reasoning"] = narrative_output
            
        # 应用文化-科学桥接（如果提供了相关数据）
        if scientific_measures is not None and traditional_knowledge is not None:
            bridged_knowledge, integration_level = self.cultural_bridge(
                scientific_measures, traditional_knowledge
            )
            results["bridged_knowledge"] = bridged_knowledge
            results["integration_level"] = integration_level
            
        # 生成最终输出
        if "bridged_knowledge" in results:
            final_features = results["bridged_knowledge"]
        elif "narrative_reasoning" in results:
            final_features = results["narrative_reasoning"]
        elif "semantic_features" in results:
            final_features = results["semantic_features"]
        elif "ecological_from_text" in results:
            final_features = self.ecological_to_semantic(results["ecological_from_text"])
        else:
            # 如果没有任何输入，返回零向量
            device = next(self.parameters()).device
            final_features = torch.zeros(1, self.config["hidden_dim"]).to(device)
        
        semantic_output = self.output_projector(final_features)
        results["semantic_output"] = semantic_output
        
        return results