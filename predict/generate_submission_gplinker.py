import os
import torch
import json
import zipfile
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel
import config
from models.JointGlobalPointer import GPLinker

# 1. 基础配置
conf = config.train_config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_dict(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        return json.load(f)

ent2id_path = os.path.join(conf["data_home"], conf.get("exp_name", "Default"), "ent2id.json")
rel2id_path = os.path.join(conf["data_home"], conf.get("exp_name", "Default"), "rel2id.json")
ent2id = load_dict(ent2id_path)
rel2id = load_dict(rel2id_path)

id2ent = {int(v): k for k, v in ent2id.items()}
id2rel = {int(v): k for k, v in rel2id.items()}

ent_type_size = len(ent2id)
rel_type_size = len(rel2id)
tokenizer = BertTokenizerFast.from_pretrained(conf["bert_path"], do_lower_case=False)

def extract_triples_gplinker(text, model, threshold=0.0):
    """GPLinker 专用的严谨双重矩阵解码函数"""
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
            ent_logits, hh_logits, tt_logits = model(input_ids, attention_mask, token_type_ids)
            
        ent_logits = ent_logits[0].cpu().numpy() 
        hh_logits = hh_logits[0].cpu().numpy() 
        tt_logits = tt_logits[0].cpu().numpy() 

        # 1. 解码实体，按照起始位置(start)进行分组并携带尾部信息(end)
        entities_by_start = {}
        for ent_type_id, start, end in zip(*np.where(ent_logits > threshold)):
            char_start = offset_mapping[start][0]
            char_end = offset_mapping[end][1]
            if char_start == char_end == 0: continue
                
            ent_text = text[char_start:char_end]
            ent_type = id2ent[ent_type_id]
            
            if start not in entities_by_start:
                entities_by_start[start] = []
            entities_by_start[start].append({
                "end": end, 
                "text": ent_text, 
                "type": ent_type
            })

        # 2. 组装关系：必须满足 hh 和 tt 双重验证
        triples = []
        for rel_type_id, sub_start, obj_start in zip(*np.where(hh_logits > threshold)):
            if sub_start in entities_by_start and obj_start in entities_by_start:
                rel_type = id2rel[rel_type_id]
                # 遍历候选的主语和宾语
                for sub in entities_by_start[sub_start]:
                    for obj in entities_by_start[obj_start]:
                        # 【核心防守】：不仅首-首要匹配，尾-尾链接也必须大于阈值！
                        if tt_logits[rel_type_id, sub["end"], obj["end"]] > threshold:
                            triples.append({
                                "predicate": rel_type,
                                "subject": sub["text"],
                                "subject_type": sub["type"],
                                "object": { "@value": obj["text"] },
                                "object_type": { "@value": obj["type"] }
                            })
        return triples

if __name__ == "__main__":
    encoder = BertModel.from_pretrained(conf["bert_path"])
    # 注意：此处实例化的是 GPLinker
    model = GPLinker(encoder, ent_type_size, rel_type_size, inner_dim=64, use_boundary_attn=True)
                                      
    # 👇 【请将这里替换为最佳的 GPLinker 权重路径】
    model_path = r"outputs\CMeIE_joint\xxxx\GPLinker_best_rel_f1_xxxx.pt" 
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    print("✅ GPLinker 模型加载成功！开始批量预测...")
    
    test_file = os.path.join(conf["data_home"], conf.get("exp_name", "Default"), "test_joint.json")
    output_file = "CMeIE_test.json" 
    
    test_data = load_dict(test_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(test_data, desc="Generating Submission"):
            # 👇【请将这里的 threshold 修改为第一步中搜索出的最佳阈值】
            pred_spo_list = extract_triples_gplinker(sample["text"], model, threshold=0.0)
            f.write(json.dumps({"text": sample["text"], "spo_list": pred_spo_list}, ensure_ascii=False) + "\n")
            
    # 自动打包
    zip_filename = "CMeIE_submission_GPLinker.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_file, arcname=output_file)
        
    print(f"\n🎉 预测完毕！GPLinker 比赛提交文件已打包至: {zip_filename}")