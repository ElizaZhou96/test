# check_model.py
import torch
import torch.nn as nn
from models.efrn import EFRN, EcologicalSpectrumAnalyzer
from config import CONFIG

def check_spectrum_analyzer():
    """测试谱分析器组件"""
    print("\n--- 测试谱分析器 ---")
    
    # 创建谱分析器实例
    hidden_dim = CONFIG["hidden_dim"]
    analyzer = EcologicalSpectrumAnalyzer(hidden_dim)
    
    # 生成测试输入
    batch_size = 2
    test_input = torch.randn(batch_size, hidden_dim)
    
    print(f"测试输入形状: {test_input.shape}")
    
    # 运行前向传播
    try:
        output = analyzer(test_input)
        print(f"输出形状: {output.shape}")
        print("谱分析器测试成功！")
    except Exception as e:
        print(f"谱分析器测试失败: {e}")
        import traceback
        traceback.print_exc()

def check_efrn_model():
    """测试EFRN模型各组件"""
    print("\n--- 测试EFRN模型 ---")
    
    # 创建EFRN实例
    efrn = EFRN(CONFIG)
    
    # 准备测试输入
    batch_size = 2
    test_inputs = {
        "genome": torch.randn(batch_size, 1024),
        "image": torch.randn(batch_size, 3, 224, 224),
        "audio": torch.randn(batch_size, 512),
        "environment": torch.randn(batch_size, 256)
    }
    
    # 测试编码器
    print("\n测试编码器...")
    for modality, data in test_inputs.items():
        try:
            encoder = efrn.encoders[modality]
            output = encoder(data)
            print(f"模态 {modality}: 输入形状 {data.shape} -> 输出形状 {output.shape}")
        except Exception as e:
            print(f"编码器 {modality} 测试失败: {e}")
    
    # 测试完整前向传播
    print("\n测试完整前向传播...")
    try:
        outputs = efrn(test_inputs)
        print("EFRN前向传播成功!")
        print(f"输出键: {list(outputs.keys())}")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: 形状 {value.shape}")
            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                print(f"{key}: 列表，第一个元素形状 {value[0].shape}")
    except Exception as e:
        print(f"EFRN前向传播失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行测试
    check_spectrum_analyzer()
    check_efrn_model()