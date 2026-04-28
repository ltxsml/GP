import json
import os
from collections import Counter

def get_sample_category(spo_list):
    if len(spo_list) <= 1:
        return "Normal"
    
    entity_map = {}
    pair_map = {}
    is_seo, is_epo = False, False
    
    for spo in spo_list:
        # 使用原文本中的起止位置元组来唯一标识实体
        s_ent = (spo.get("sub_start"), spo.get("sub_end"))
        o_ent = (spo.get("obj_start"), spo.get("obj_end"))
        
        # 统计实体对 (判断 EPO: 同一对实体有多个关系)
        pair = tuple(sorted((s_ent, o_ent)))
        if pair:
            pair_map[pair] = pair_map.get(pair, 0) + 1
            if pair_map[pair] > 1: is_epo = True
        
        # 统计单个实体 (判断 SEO: 一个实体参与多个三元组)
        for ent in [s_ent, o_ent]:
            if ent:
                entity_map[ent] = entity_map.get(ent, 0) + 1
                if entity_map[ent] > 1: is_seo = True
            
    if is_epo: return "EPO"
    if is_seo: return "SEO"
    return "Normal"

def analyze_overlap(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    categories = [get_sample_category(sample.get('spo_list', [])) for sample in data]
            
    total = len(categories)
    counter = Counter(categories)
    
    print(f"=== 关系重叠类型统计报告 ({os.path.basename(file_path)}) ===")
    print(f"样本总数: {total}")
    for cat in ["Normal", "SEO", "EPO"]:
        count = counter.get(cat, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{cat:7s} : {count:5d} 个 \t(占比 {percentage:5.2f}%)")

if __name__ == "__main__":
    # 默认统计训练集，你也可以修改为 dev_joint.json
    target_file = "datasets/CMeIE/train_joint.json"
    analyze_overlap(target_file)