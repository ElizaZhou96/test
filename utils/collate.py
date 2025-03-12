# utils/collate.py
import torch
import numpy as np

def safe_bioharmony_collate_fn(batch):
    """增强的collate函数，确保数据一致性"""
    if not batch:
        return {}
        
    # 收集所有键
    all_keys = set()
    for sample in batch:
        all_keys.update(sample.keys())
        
    # 初始化结果
    result = {key: [] for key in all_keys}
    
    # 收集所有样本的值
    for sample in batch:
        for key in all_keys:
            if key in sample:
                result[key].append(sample[key])
            else:
                result[key].append(None)
    
    # 处理每种数据类型
    for key in list(result.keys()):
        # 跳过所有值都为None的键
        if all(x is None for x in result[key]):
            continue
            
        # 筛选出非None值
        valid_values = [x for x in result[key] if x is not None]
        if not valid_values:
            continue
            
        # 获取第一个有效值
        first_val = valid_values[0]
        
        # 处理张量
        if isinstance(first_val, torch.Tensor):
            # 尝试堆叠所有有效张量
            try:
                # 筛选出形状匹配的张量
                shapes = [v.shape for v in valid_values]
                if len(set(str(s) for s in shapes)) == 1:  # 所有形状相同
                    # 创建零张量替代None值
                    padded_values = []
                    for v in result[key]:
                        if v is not None:
                            padded_values.append(v)
                        else:
                            padded_values.append(torch.zeros_like(first_val))
                            
                    # 堆叠
                    result[key] = torch.stack(padded_values)
                else:
                    # 形状不同，保持为列表
                    print(f"警告: 键 '{key}' 的张量形状不一致，保持为列表")
            except Exception as e:
                print(f"堆叠 '{key}' 时出错: {str(e)}")
                # 保持为列表
                pass
                
        # 处理字典数据
        elif isinstance(first_val, dict):
            # 收集所有子键
            all_subkeys = set()
            for d in valid_values:
                all_subkeys.update(d.keys())
                
            # 为每个子键准备一个列表
            processed_dict = {subkey: [] for subkey in all_subkeys}
            mask_dict = {}  # 创建单独的掩码字典，避免在迭代时修改字典
            
            # 收集值
            for sample_dict in result[key]:
                for subkey in all_subkeys:
                    if sample_dict is not None and subkey in sample_dict:
                        processed_dict[subkey].append(sample_dict[subkey])
                    else:
                        processed_dict[subkey].append(None)
            
            # 处理每个子键
            for subkey in list(processed_dict.keys()):  # 使用list()创建键的副本
                valid_subvals = [x for x in processed_dict[subkey] if x is not None]
                if valid_subvals and isinstance(valid_subvals[0], torch.Tensor):
                    # 准备填充值
                    padded_subvals = []
                    valid_indices = [i for i, x in enumerate(processed_dict[subkey]) if x is not None]
                    
                    for i, val in enumerate(processed_dict[subkey]):
                        if i in valid_indices:
                            padded_subvals.append(val)
                        else:
                            padded_subvals.append(torch.zeros_like(valid_subvals[0]))
                    
                    # 尝试堆叠
                    try:
                        processed_dict[subkey] = torch.stack(padded_subvals)
                        # 将掩码添加到单独的字典中
                        mask_dict[f"{subkey}_mask"] = torch.tensor(
                            [1 if i in valid_indices else 0 for i in range(len(padded_subvals))],
                            dtype=torch.bool
                        )
                    except:
                        pass
            
            # 将掩码合并到处理后的字典中
            processed_dict.update(mask_dict)
            result[key] = processed_dict
    
    return result