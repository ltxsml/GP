import json
import random
import os
import sys

def bio_to_entities(text, labels):
    """将 BIO 标签序列转换为 (start, end, type) 格式"""
    entities = []
    start = -1
    label_type = None
    
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if start != -1:
                entities.append((start, i - 1, label_type))
            start = i
            label_type = label.split("-")[1]
        elif label.startswith("I-"):
            if start != -1 and label.split("-")[1] == label_type:
                continue
            else: # 容错处理：非法 I 标签
                if start != -1:
                    entities.append((start, i - 1, label_type))
                start = -1
        else: # "O" 标签
            if start != -1:
                entities.append((start, i - 1, label_type))
                start = -1
    if start != -1:
        entities.append((start, len(labels) - 1, label_type))
    return entities

def split_and_save(input_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
    
    # 格式转换
    processed_data = []
    all_labels = set()
    for item in lines:
        entities = bio_to_entities(item['text'], item['labels'])
        processed_data.append({
            "text": item['text'],
            "entity_list": entities
        })
        for _, _, label in entities:
            all_labels.add(label)
    
    # 随机打乱并切分
    random.seed(2333)
    random.shuffle(processed_data)
    
    n = len(processed_data)
    train_end = int(n * 0.8)
    dev_end = int(n * 0.9)
    
    train_data = processed_data[:train_end]
    dev_data = processed_data[train_end:dev_end]
    test_data = processed_data[dev_end:]
    
    # 保存文件
    with open(os.path.join(output_dir, "train.json"), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "dev.json"), 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "test.json"), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
        
    # 生成 ent2id.json
    ent2id = {label: i for i, label in enumerate(sorted(list(all_labels)))}
    with open(os.path.join(output_dir, "ent2id.json"), 'w', encoding='utf-8') as f:
        json.dump(ent2id, f, ensure_ascii=False, indent=2)
        
    print(f"处理完成！\n训练集: {len(train_data)}\n验证集: {len(dev_data)}\n测试集: {len(test_data)}")
    print(f"实体类别: {ent2id}")

if __name__ == "__main__":
    # 请确保 cmeee.json 在当前目录下
    split_and_save("cmeee.json", "./datasets/CMeEE")
    input_file = "cmeee.json"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    if os.path.exists(input_file):
        split_and_save(input_file, "./datasets/CMeEE")
    else:
        print(f"错误：找不到文件 '{input_file}'。请确保文件在当前目录下，或通过命令行参数指定路径。")