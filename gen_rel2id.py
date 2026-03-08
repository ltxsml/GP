import json

def generate_rel2id(input_path, output_path):
    # 存储所有发现的关系类型
    rel_set = set()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        # CMeIE 原始格式通常是每行一个 JSON
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            spo_list = data.get('spo_list', [])
            for spo in spo_list:
                predicate = spo.get('predicate')
                if predicate:
                    rel_set.add(predicate) # 实际应为 rel_set.add(predicate)

    # 排序以保证每次生成的 ID 顺序一致
    sorted_rels = sorted(list(rel_set))
    
    # 构建映射字典 {关系名: ID}
    rel2id = {rel: i for i, rel in enumerate(sorted_rels)}
    
    # 保存为 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rel2id, f, ensure_ascii=False, indent=4)
        
    print(f"成功提取 {len(rel2id)} 种关系，映射文件已保存至: {output_path}")
    print("关系列表样例:", list(rel2id.keys())[:5])

# 使用示例
generate_rel2id('datasets/CMeIE/CMeIE_train.json', 'datasets/CMeIE/rel2id.json')