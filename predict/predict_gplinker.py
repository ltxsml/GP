import os
import torch
import json
import numpy as np
from transformers import BertTokenizerFast, BertModel
import config
from models.JointGlobalPointer import GPLinker

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
id2ent = {int(v): k for k, v in ent2id.items()}
id2rel = {int(v): k for k, v in rel2id.items()}

ent_type_size = len(ent2id)
rel_type_size = len(rel2id)
tokenizer = BertTokenizerFast.from_pretrained(conf["bert_path"], do_lower_case=False)

def extract_triples_gplinker_visual(text, model, threshold=0.0):
    """用于可视化观看的 GPLinker 解码函数"""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, max_length=conf["hyper_parameters"]["max_seq_len"], truncation=True, return_offsets_mapping=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        token_type_ids = inputs["token_type_ids"].to(device)
        offset_mapping = inputs["offset_mapping"][0].cpu().numpy()

        with torch.cuda.amp.autocast():
            ent_logits, hh_logits, tt_logits = model(input_ids, attention_mask, token_type_ids)
            
        ent_logits = ent_logits[0].cpu().numpy() 
        hh_logits = hh_logits[0].cpu().numpy() 
        tt_logits = tt_logits[0].cpu().numpy() 

        # 解码实体
        entities_by_start = {}
        for ent_type_id, start, end in zip(*np.where(ent_logits > threshold)):
            char_start, char_end = offset_mapping[start][0], offset_mapping[end][1]
            if char_start == char_end == 0: continue
            if start not in entities_by_start: entities_by_start[start] = []
            entities_by_start[start].append({"end": end, "text": text[char_start:char_end], "type": id2ent[ent_type_id]})

        # 解码关系
        triples = []
        for rel_type_id, sub_start, obj_start in zip(*np.where(hh_logits > threshold)):
            if sub_start in entities_by_start and obj_start in entities_by_start:
                rel_type = id2rel[rel_type_id]
                for sub in entities_by_start[sub_start]:
                    for obj in entities_by_start[obj_start]:
                        # GPLinker 的严谨之处：尾部也要匹配
                        if tt_logits[rel_type_id, sub["end"], obj["end"]] > threshold:
                            triples.append({"subject": sub["text"], "predicate": rel_type, "object": obj["text"]})
        return triples

if __name__ == "__main__":
    encoder = BertModel.from_pretrained(conf["bert_path"])
    model = GPLinker(encoder, ent_type_size, rel_type_size, inner_dim=64, use_boundary_attn=True)
                                      
    # 👇【步骤1：替换为您跑出的最佳 GPLinker 权重路径】
    model_path = r"outputs\CMeIE_joint\xxxx\GPLinker_best_rel_f1_xxxx.pt" 
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)
        
        # 👇【步骤2：输入您想测试的任意一句话，并修改为您搜出的最佳阈值】
        test_text = "患者出现恶心、呕吐，并伴随剧烈腹痛，高度怀疑为急性胰腺炎，建议立即使用奥美拉唑进行抑酸治疗。"
        triples = extract_triples_gplinker_visual(test_text, model, threshold=0.0) 
        print(f"\n📝 输入文本: {test_text}")
        print(f"✨ 抽取结果: {json.dumps(triples, ensure_ascii=False, indent=2)}")