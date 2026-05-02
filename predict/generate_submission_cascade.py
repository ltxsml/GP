import os
import torch
import json
import zipfile
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel
import config
from models.JointGlobalPointer import JointCascadeGlobalPointer

# 1. 基础配置
conf = config.train_config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_dict(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 加载字典
ent2id_path = os.path.join(conf["data_home"], conf.get("exp_name", "Default"), "ent2id.json")
rel2id_path = os.path.join(conf["data_home"], conf.get("exp_name", "Default"), "rel2id.json")
ent2id = load_dict(ent2id_path)
rel2id = load_dict(rel2id_path)

# 构建反向映射字典 (ID -> Name)
id2ent = {int(v): k for k, v in ent2id.items()}
id2rel = {int(v): k for k, v in rel2id.items()}

ent_type_size = len(ent2id)
rel_type_size = len(rel2id)

tokenizer = BertTokenizerFast.from_pretrained(conf["bert_path"], do_lower_case=False)

def extract_triples_for_submission(text, model, threshold=0.2):
    """带严格格式组装的解码函数"""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            text, max_length=conf["hyper_parameters"]["max_seq_len"],
            truncation=True, return_offsets_mapping=True, return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        token_type_ids = inputs["token_type_ids"].to(device)
        offset_mapping = inputs["offset_mapping"][0].cpu().numpy()

        with torch.cuda.amp.autocast():
            ent_logits, rel_logits = model(input_ids, attention_mask, token_type_ids)
            
        ent_logits = ent_logits[0].cpu().numpy() 
        rel_logits = rel_logits[0].cpu().numpy() 

        # 解码实体，带上 type 信息
        entities_by_start = {}
        for ent_type_id, start, end in zip(*np.where(ent_logits > threshold)):
            char_start = offset_mapping[start][0]
            char_end = offset_mapping[end][1]
            if char_start == char_end == 0: continue
                
            ent_text = text[char_start:char_end]
            ent_type = id2ent[ent_type_id]
            
            if start not in entities_by_start:
                entities_by_start[start] = []
            entities_by_start[start].append({"text": ent_text, "type": ent_type})

        # 组装关系，严格遵循 CMeIE 格式
        triples = []
        for rel_type_id, sub_start, obj_start in zip(*np.where(rel_logits > threshold)):
            if sub_start in entities_by_start and obj_start in entities_by_start:
                rel_type = id2rel[rel_type_id]
                for sub in entities_by_start[sub_start]:
                    for obj in entities_by_start[obj_start]:
                        triples.append({
                            "predicate": rel_type,
                            "subject": sub["text"],
                            "subject_type": sub["type"],
                            "object": { "@value": obj["text"] },
                            "object_type": { "@value": obj["type"] }
                        })
        return triples

if __name__ == "__main__":
    # 1. 初始化模型并加载权重
    encoder = BertModel.from_pretrained(conf["bert_path"])
    model = JointCascadeGlobalPointer(encoder, ent_type_size, rel_type_size, inner_dim=64, use_boundary_attn=True, use_dynamic_gate=True)
                                      
    # 👇 【请将这里替换为您 F1最高 的那个 .pt 权重文件路径】
    model_path = r"outputs\\CMeIE_joint\\2026-05-02_18.48.38\\Cascade_best_rel_f1_0.6247.pt" 
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    print("✅ 模型加载成功！开始批量预测...")
    
    # 2. 读取测试集并预测
    test_file = os.path.join(conf["data_home"], conf.get("exp_name", "Default"), "test_joint.json")
    output_file = "CMeIE_test.json"  # 官方评测系统严格要求的文件名
    
    test_data = load_dict(test_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(test_data, desc="Generating Submission"):
            text = sample["text"]
            # 使用确定的最佳阈值 0
            pred_spo_list = extract_triples_for_submission(text, model, threshold=0)
            
            # 按照官方要求：一行一个 JSON
            output_line = {
                "text": text,
                "spo_list": pred_spo_list
            }
            f.write(json.dumps(output_line, ensure_ascii=False) + "\n")
            
    # 3. 自动打包成 zip 文件用于直接提交
    zip_filename = "CMeIE_submission.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_file, arcname=output_file)
        
    print(f"\n🎉 预测完毕！比赛提交文件已打包至当前目录的: {zip_filename}，请直接前往官网提交此包！")