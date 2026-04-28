import json
import os
from collections import Counter

def analyze_entity_lengths(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    lengths = []
    for sample in data:
        for ent in sample.get('entity_list', []):
            # 实体长度 = end - start + 1
            ent_len = ent.get('end', 0) - ent.get('start', 0) + 1
            lengths.append(ent_len)
            
    if not lengths:
        print("数据集中未找到实体！")
        return
        
    print(f"=== 实体长度统计报告 ({os.path.basename(file_path)}) ===")
    print(f"实体总数: {len(lengths)}")
    print(f"最短实体: {min(lengths)} 字符")
    print(f"最长实体: {max(lengths)} 字符")
    print(f"平均长度: {sum(lengths)/len(lengths):.2f} 字符\n")
    
    print("=== 长度分布详情 (按长度从小到大) ===")
    counter = Counter(lengths)
    # 按实体长度从小到大排序打印
    for length, count in sorted(counter.items()):
        percentage = (count / len(lengths)) * 100
        print(f"长度 {length:2d} : {count:5d} 个 \t(占比 {percentage:5.2f}%)")

if __name__ == "__main__":
    # 默认统计验证集，如果你想看训练集可以把 dev_joint 换成 train_joint
    target_file = "datasets/CMeIE/dev_joint.json"
    analyze_entity_lengths(target_file)
