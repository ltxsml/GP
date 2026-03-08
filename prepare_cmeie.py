# 文件位置：prepare_cmeie.py
import json
import os

def process_cmeie_data(input_path, output_path):
    """
    将 CMeIE 原始格式转换为联合抽取模型需要的标准 JSON 格式。
    保留负样本（空 spo_list 的数据）和测试集数据。
    """
    processed_data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            text = data['text']
            spo_list = data.get('spo_list', [])
            
            entity_dict = {}
            standard_spo_list = []
            
            # 如果存在关系标注，则提取实体和关系的坐标
            for spo in spo_list:
                subject = spo.get('subject', '')
                subject_type = spo.get('subject_type', '')
                predicate = spo.get('predicate', '')
                
                # 兼容不同格式的 object 字段 (有些是字典，有些直接是字符串)
                object_data = spo.get('object', {})
                object_ = object_data.get('@value', '') if isinstance(object_data, dict) else object_data
                
                object_type_data = spo.get('object_type', {})
                object_type = object_type_data.get('@value', '') if isinstance(object_type_data, dict) else object_type_data
                
                # 查找实体在 text 中的起止位置
                sub_start = text.find(subject)
                obj_start = text.find(object_)
                
                if sub_start != -1 and obj_start != -1:
                    sub_end = sub_start + len(subject) - 1
                    obj_end = obj_start + len(object_) - 1
                    
                    # 记录实体 (去重)
                    ent_sub_key = f"{subject}_{sub_start}_{sub_end}"
                    if ent_sub_key not in entity_dict:
                        entity_dict[ent_sub_key] = {"ent": subject, "type": subject_type, "start": sub_start, "end": sub_end}
                        
                    ent_obj_key = f"{object_}_{obj_start}_{obj_end}"
                    if ent_obj_key not in entity_dict:
                        entity_dict[ent_obj_key] = {"ent": object_, "type": object_type, "start": obj_start, "end": obj_end}
                    
                    # 记录关系
                    standard_spo_list.append({
                        "predicate": predicate,
                        "sub_start": sub_start, "sub_end": sub_end,
                        "obj_start": obj_start, "obj_end": obj_end
                    })
                    
            # 【核心修改】：无论 entity_dict 和 standard_spo_list 是否为空，都把这条数据加进去！
            processed_data.append({
                "text": text,
                "entity_list": list(entity_dict.values()),
                "spo_list": standard_spo_list
            })

    # 保存为标准 JSON 数组格式
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
    print(f"处理完成，保存至 {output_path}，共 {len(processed_data)} 条数据。")

# --- 请在这里调用你的路径 ---
if __name__ == '__main__':
    # 示例路径，请根据你实际的文件位置修改
    # 处理训练集
    # process_cmeie_data('datasets/CMeIE/CMeIE_train.json', 'datasets/CMeIE/train_joint.json')
    # 处理验证集
    # process_cmeie_data('datasets/CMeIE/CMeIE_dev.json', 'datasets/CMeIE/dev_joint.json')
    
    # 测试集 (此处填入你刚刚报错的文件路径)
    process_cmeie_data('datasets/CMeIE/CMeIE_dev.json', 'datasets/CMeIE/dev_joint.json')