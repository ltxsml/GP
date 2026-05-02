import os
import torch
import json
import numpy as np
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

def extract_triples_cascade(text, model, threshold=0.0):
    """
    级联模型 (Cascade) 核心解码预测函数
    """
    model.eval()
    with torch.no_grad():
        # 1. 文本编码 (获取 offset_mapping 用于精准截取字符串)
        inputs = tokenizer(
            text,
            max_length=conf["hyper_parameters"]["max_seq_len"],
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        token_type_ids = inputs["token_type_ids"].to(device)
        offset_mapping = inputs["offset_mapping"][0].cpu().numpy()

        # 2. 模型推理 (自动混合精度加速)
        with torch.cuda.amp.autocast():
            ent_logits, rel_logits = model(input_ids, attention_mask, token_type_ids)
            
        ent_logits = ent_logits[0].cpu().numpy() # shape: [ent_type_size, seq_len, seq_len]
        rel_logits = rel_logits[0].cpu().numpy() # shape: [rel_type_size, seq_len, seq_len]

        # ==========================================
        # 3. 步骤一：解码实体，并按“Start坐标”进行哈希分组
        # ==========================================
        entities_by_start = {}
        for ent_type_id, start, end in zip(*np.where(ent_logits > threshold)):
            char_start = offset_mapping[start][0]
            char_end = offset_mapping[end][1]
            if char_start == char_end == 0:  # 排除 [CLS], [SEP] 等特殊 Token
                continue
                
            ent_text = text[char_start:char_end]
            ent_type = id2ent[ent_type_id]
            
            # 将该起点下的所有可能实体都记录下来（支持嵌套同首字实体的解析）
            if start not in entities_by_start:
                entities_by_start[start] = []
            entities_by_start[start].append({"text": ent_text, "type": ent_type})

        # ==========================================
        # 4. 步骤二：解码关系，通过“Start坐标”牵线搭桥
        # ==========================================
        triples = []
        for rel_type_id, sub_start, obj_start in zip(*np.where(rel_logits > threshold)):
            # 如果预测出的主语起点和宾语起点，恰好在我们刚刚找到的实体列表里
            if sub_start in entities_by_start and obj_start in entities_by_start:
                rel_type = id2rel[rel_type_id]
                
                # 遍历所有可能的 (主语实体, 宾语实体) 组合
                for sub in entities_by_start[sub_start]:
                    for obj in entities_by_start[obj_start]:
                        triples.append({
                            "subject": sub["text"],
                            "predicate": rel_type,
                            "object": obj["text"]
                        })
                        
        return triples

if __name__ == "__main__":
    # 1. 初始化结构与权重
    encoder = BertModel.from_pretrained(conf["bert_path"])
    model = JointCascadeGlobalPointer(encoder, ent_type_size, rel_type_size, inner_dim=64,
                                      use_boundary_attn=True, use_dynamic_gate=True,
                                      use_mlp_rel=False) # <--- 加载老模型时设为 False
                                      
    # 【请修改这里】填入您训练出的最好的 Cascade 模型权重路径
    model_path = r"D:\\GlobalPointer_pytorch-main\\outputs\\CMeIE_joint\\2026-05-02_13.34.24\\Cascade_best_rel_f1_0.6168.pt" 
    if os.path.exists(model_path):
        # 增加 weights_only=True 消除 PyTorch 安全性警告
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)
        print("✅ 模型加载成功！开始预测...")
        
        test_text = "胃炎患者如果服用阿司匹林，容易引发胃溃疡和消化道出血。"
        triples = extract_triples_cascade(test_text, model, threshold=0.0)
        
        print(f"\n输入文本: {test_text}")
        print(f"抽取结果: {json.dumps(triples, ensure_ascii=False, indent=2)}")
    else:
        print(f"❌ 找不到模型权重: {model_path}，请修改路径后重试。")