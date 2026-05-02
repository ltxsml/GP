import os
import torch
import numpy as np
from tqdm import tqdm
import config
from transformers import BertModel
from models.JointGlobalPointer import GPLinker

# 引入训练脚本中的基础配置
from train_joint import data_generator, ent_type_size, rel_type_size, device, use_boundary_attn_bool

def search_best_threshold_gplinker():
    # 1. 准备数据和模型 (一定要指定 model_type="GPLinker" 才会解包 7 个参数)
    _, valid_dataloader = data_generator(model_type="GPLinker")
    encoder = BertModel.from_pretrained(config.train_config["bert_path"])
    
    # 初始化 GPLinker 模型
    model = GPLinker(encoder, ent_type_size, rel_type_size, inner_dim=64, use_boundary_attn=use_boundary_attn_bool)
    
    # 👇【请将这里替换为您跑出的最佳 GPLinker 权重路径】
    model_path = r"outputs\CMeIE_joint\xxxx\GPLinker_best_rel_f1_xxxx.pt"
    
    print(f"加载模型: {model_path}")
    if not os.path.exists(model_path):
        print(f"⚠️ 找不到文件: {model_path}，请修改路径！")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    print("\n=> 正在提取验证集特征并同步搜索最佳阈值，请稍候...")
    thresholds = np.arange(-2.0, 2.1, 0.1)
    stats = {th: {"X": 0.0, "Y": 0.0, "Z": 0.0} for th in thresholds}
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for batch_data in tqdm(valid_dataloader):
                (_, batch_input_ids, batch_attention_mask, batch_token_type_ids, _, batch_hh_labels, batch_tt_labels) = batch_data
                _, hh_logits, tt_logits = model(batch_input_ids.to(device), batch_attention_mask.to(device), batch_token_type_ids.to(device))
                
                # 以首首链接 (hh) 为基准搜索 (因为关系主要由主语和宾语的起始字决定)
                rel_logits = hh_logits.cpu()
                batch_rel_labels = batch_hh_labels.cpu()
                Z_batch = torch.sum(batch_rel_labels).item()
                pos_logits = rel_logits[batch_rel_labels == 1]
                
                for th in thresholds:
                    stats[th]["Y"] += torch.sum(rel_logits > th).item()   
                    stats[th]["X"] += torch.sum(pos_logits > th).item()   
                    stats[th]["Z"] += Z_batch                             

    print("\n=> 搜索完毕！各阈值结果如下：")
    best_f1, best_th = 0.0, 0.0
    for th in thresholds:
        X, Y, Z = stats[th]["X"], stats[th]["Y"], stats[th]["Z"]
        f1 = 2 * X / (Y + Z) if (Y + Z) > 0 else 0
        if f1 > best_f1: best_f1 = f1; best_th = th
        print(f"阈值: {th:5.1f} | P: {X/(Y+1e-8):.4f} | R: {X/(Z+1e-8):.4f} | F1: {f1:.4f}")
        
    print(f"\n🏆 最佳阈值: {best_th:.1f}, 对应最高 F1 可以达到: {best_f1:.4f}")

if __name__ == "__main__":
    search_best_threshold_gplinker()