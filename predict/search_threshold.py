import os
import torch
import numpy as np
from tqdm import tqdm
import config
from transformers import BertModel
from models.JointGlobalPointer import JointCascadeGlobalPointer

# 引入训练脚本中的基础配置
from train_joint import data_generator, ent_type_size, rel_type_size, device

def search_best_threshold():
    # 1. 准备数据和模型
    _, valid_dataloader = data_generator(model_type="Cascade")
    encoder = BertModel.from_pretrained(config.train_config["bert_path"])
    
    # 初始化 Cascade 模型
    model = JointCascadeGlobalPointer(
        encoder, ent_type_size, rel_type_size, inner_dim=64,
        use_boundary_attn=True, use_dynamic_gate=True
    )
    
    # 【请修改这里】替换为您跑出的 F1=0.6162 的模型权重路径
    model_path = r"outputs\\CMeIE_joint\\2026-05-02_18.48.38\\Cascade_best_rel_f1_0.6195.pt"
    
    print(f"加载模型: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    print("\n=> 正在提取验证集特征并同步搜索最佳阈值，请稍候...")
    
    # 遍历从 -2.0 到 2.0，步长 0.1 的阈值
    thresholds = np.arange(-2.0, 2.1, 0.1)
    stats = {th: {"X": 0.0, "Y": 0.0, "Z": 0.0} for th in thresholds}
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for batch_data in tqdm(valid_dataloader):
                (_, batch_input_ids, batch_attention_mask, batch_token_type_ids, _, batch_rel_labels) = batch_data
                _, rel_logits = model(batch_input_ids.to(device), batch_attention_mask.to(device), batch_token_type_ids.to(device))
                
                rel_logits = rel_logits.cpu()
                batch_rel_labels = batch_rel_labels.cpu()
                
                # Z是真实的标签总数，每个batch都是固定的
                Z_batch = torch.sum(batch_rel_labels).item()
                # 提前提取出真实标签为1的位置的预测概率
                pos_logits = rel_logits[batch_rel_labels == 1]
                
                for th in thresholds:
                    stats[th]["Y"] += torch.sum(rel_logits > th).item()   # 预测为1的总数
                    stats[th]["X"] += torch.sum(pos_logits > th).item()   # 预测正确数
                    stats[th]["Z"] += Z_batch                             # 真实总数

    print("\n=> 搜索完毕！各阈值结果如下：")
    best_f1, best_th = 0.0, 0.0
    
    for th in thresholds:
        X = stats[th]["X"]
        Y = stats[th]["Y"]
        Z = stats[th]["Z"]
        
        f1 = 2 * X / (Y + Z) if (Y + Z) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1; best_th = th
        print(f"阈值: {th:5.1f} | P: {X/(Y+1e-8):.4f} | R: {X/(Z+1e-8):.4f} | F1: {f1:.4f}")
        
    print(f"\n🏆 最佳阈值: {best_th:.1f}, 对应最高 F1 可以达到: {best_f1:.4f}")

if __name__ == "__main__":
    search_best_threshold()