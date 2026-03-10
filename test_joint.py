import os
import config
import torch
import json
from transformers import BertTokenizerFast, BertModel
from tqdm import tqdm

# 导入联合抽取的组件
from models.GlobalPointer import MyDataset, MetricsCalculator
from models.JointGlobalPointer import JointCascadeGlobalPointer, DataMakerJoint

conf = config.train_config
hyper_parameters = conf["hyper_parameters"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. 配置测试集路径和词典
test_data_path = os.path.join(conf["data_home"], conf["exp_name"], "test_joint.json") # 确保你的测试集叫这个名字
ent2id_path = os.path.join(conf["data_home"], conf["exp_name"], "ent2id.json")
rel2id_path = os.path.join(conf["data_home"], conf["exp_name"], "rel2id.json")

def load_data(data_path):
    with open(data_path, 'r', encoding="utf-8") as f:
        return json.load(f)

ent2id = load_data(ent2id_path)
rel2id = load_data(rel2id_path)
ent_type_size = len(ent2id)
rel_type_size = len(rel2id)

tokenizer = BertTokenizerFast.from_pretrained(conf["bert_path"], do_lower_case=False)
data_maker = DataMakerJoint(tokenizer)
metrics = MetricsCalculator()

def get_test_dataloader():
    test_data = load_data(test_data_path)
    max_seq_len = hyper_parameters["max_seq_len"]
    
    test_dataloader = torch.utils.data.DataLoader(
        MyDataset(test_data),
        batch_size=hyper_parameters["batch_size"],
        shuffle=False, # 测试集不需要打乱
        collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id, rel2id)
    )
    return test_dataloader

def evaluate_on_test(model, dataloader):
    model.eval()
    ent_total_X, ent_total_Y, ent_total_Z = 0., 0., 0.
    rel_total_X, rel_total_Y, rel_total_Z = 0., 0., 0.
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Testing"):
            (_, batch_input_ids, batch_attention_mask, batch_token_type_ids, 
             batch_ent_labels, batch_rel_labels, _) = batch_data
             
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)
            batch_ent_labels = batch_ent_labels.to(device)
            batch_rel_labels = batch_rel_labels.to(device)

            ent_logits, rel_logits, _ = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            
            eX, eY, eZ = metrics.get_evaluate_fpr(ent_logits, batch_ent_labels)
            ent_total_X += eX; ent_total_Y += eY; ent_total_Z += eZ
            
            rX, rY, rZ = metrics.get_evaluate_fpr(rel_logits, batch_rel_labels)
            rel_total_X += rX; rel_total_Y += rY; rel_total_Z += rZ

    ent_f1 = 2 * ent_total_X / (ent_total_Y + ent_total_Z) if (ent_total_Y + ent_total_Z) > 0 else 0
    ent_p = ent_total_X / ent_total_Y if ent_total_Y > 0 else 0
    ent_r = ent_total_X / ent_total_Z if ent_total_Z > 0 else 0
    
    rel_f1 = 2 * rel_total_X / (rel_total_Y + rel_total_Z) if (rel_total_Y + rel_total_Z) > 0 else 0
    rel_p = rel_total_X / rel_total_Y if rel_total_Y > 0 else 0
    rel_r = rel_total_X / rel_total_Z if rel_total_Z > 0 else 0

    print("\n" + "★"*40)
    print("测试集最终评估结果 (Test Set Results):")
    print(f"[Entity]   Precision: {ent_p:.4f}, Recall: {ent_r:.4f}, F1: {ent_f1:.4f}")
    print(f"[Relation] Precision: {rel_p:.4f}, Recall: {rel_r:.4f}, F1: {rel_f1:.4f}")
    print("★"*40 + "\n")

if __name__ == '__main__':
    # 2. 初始化模型 (确保与训练时的结构完全一致)
    encoder = BertModel.from_pretrained(conf["bert_path"])
    # 注意：这里的 use_boundary_attn 必须和你训练最好模型时使用的开关一致
    model = JointCascadeGlobalPointer(encoder, ent_type_size, rel_type_size, inner_dim=64, use_boundary_attn=True)
    
    # 3. 指定你训练出来的最好的模型权重路径
    # 请将下面这个路径替换为 outputs 文件夹下那个 F1 最高的 .pt 文件路径
    best_model_path = "outputs/Group_A_Full_joint/xxxx-xx-xx_xx.xx.xx/joint_model_x_f1_0.xxxx.pt" 
    
    print(f"Loading best model from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = model.to(device)
    
    test_dataloader = get_test_dataloader()
    evaluate_on_test(model, test_dataloader)