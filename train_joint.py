"""
Date: 2024-05-20
Description: 层叠式实体-关系联合抽取模型训练脚本 (面向医疗领域复杂关系)
"""

import os
import config
import sys
import torch
import json
import time
import glob
from transformers import BertTokenizerFast, BertModel
from tqdm import tqdm
import wandb

# 导入你原有的基础组件
from models.GlobalPointer import MyDataset, MetricsCalculator
# 导入我们新建的联合抽取组件 (假设你把上一轮的代码放在了这个文件里)
from models.JointGlobalPointer import JointCascadeGlobalPointer, DataMakerJoint, JointExtractionLoss


import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 新增命令行参数解析 ---
parser = argparse.ArgumentParser(description="联合抽取消融实验")
# parser.add_argument('--exp_name', type=str, default='Group_A_Full')
parser.add_argument('--use_boundary_attn', type=str, default='True')
parser.add_argument('--lambda_scl', type=float, default=0.1)
args = parser.parse_args()

# --- 读取配置并覆盖 ---
conf = config.train_config
hyper_parameters = conf["hyper_parameters"]

# 用命令行参数覆盖默认配置
# conf["exp_name"] = args.exp_name
use_boundary_attn_bool = True if args.use_boundary_attn == 'True' else False

# 读取配置
# conf = config.train_config
hyper_parameters = conf["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
conf["num_workers"] = 6 if sys.platform.startswith("linux") else 0

# 固定随机种子保证复现性
torch.manual_seed(hyper_parameters["seed"])
torch.backends.cudnn.deterministic = True

# 初始化 Wandb (推荐用于监控联合抽取的多个 Loss)
if conf["logger"] == "wandb" and conf["run_type"] == "train":
    wandb.init(project="Joint_GlobalPointer_" + conf["exp_name"], config=hyper_parameters)
    wandb.run.name = conf["run_name"] + "_joint_" + wandb.run.id
    model_state_dict_dir = wandb.run.dir
    logger = wandb
elif conf["run_type"] == "train":
    model_state_dict_dir = os.path.join(conf["path_to_save_model"], conf["exp_name"] + "_joint",
                                        time.strftime("%Y-%m-%d_%H.%M.%S", time.gmtime()))
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)

tokenizer = BertTokenizerFast.from_pretrained(conf["bert_path"], do_lower_case=False)

def load_data(data_path):
    with open(data_path, 'r', encoding="utf-8") as f:
        return json.load(f)

# 加载实体类别字典
ent2id_path = os.path.join(conf["data_home"], conf["exp_name"], "ent2id.json") # 请确保路径正确
ent2id = load_data(ent2id_path)
ent_type_size = len(ent2id)

# 【新增】加载关系类别字典
rel2id_path = os.path.join(conf["data_home"], conf["exp_name"], "rel2id.json") # 你需要手动生成一个关系到id的映射
rel2id = load_data(rel2id_path)
rel_type_size = len(rel2id)

def data_generator():
    """生成联合抽取 DataLoader"""
    # 这里假设你把通过 prepare_cmeie.py 处理好的数据命名为 train_joint.json
    train_data_path = os.path.join(conf["data_home"], conf["exp_name"], "train_joint.json")
    valid_data_path = os.path.join(conf["data_home"], conf["exp_name"], "dev_joint.json")
    
    train_data = load_data(train_data_path)
    valid_data = load_data(valid_data_path)

    max_seq_len = hyper_parameters["max_seq_len"]
    data_maker = DataMakerJoint(tokenizer) # 使用支持实体+关系双标签的 DataMaker

    train_dataloader = torch.utils.data.DataLoader(
        MyDataset(train_data),
        batch_size=hyper_parameters["batch_size"],
        shuffle=True,
        num_workers=conf["num_workers"],
        collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id, rel2id)
    )
    
    valid_dataloader = torch.utils.data.DataLoader(
        MyDataset(valid_data),
        batch_size=hyper_parameters["batch_size"],
        shuffle=False,
        num_workers=conf["num_workers"],
        collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id, rel2id)
    )
    return train_dataloader, valid_dataloader


# 初始化模型与损失函数
encoder = BertModel.from_pretrained(conf["bert_path"])
# 参数: encoder, 实体类别数, 关系类别数, inner_dim
model = JointCascadeGlobalPointer(encoder, ent_type_size, rel_type_size, inner_dim=64,use_boundary_attn=use_boundary_attn_bool)
model = model.to(device)

criterion = JointExtractionLoss(lambda_scl=args.lambda_scl) # 联合损失函数，λ可调
metrics = MetricsCalculator()

def train_step(batch_data, model, optimizer, criterion):
    # 解包联合数据
    (batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, 
     batch_ent_labels, batch_rel_labels, batch_token_labels) = batch_data
     
    batch_input_ids = batch_input_ids.to(device)
    batch_attention_mask = batch_attention_mask.to(device)
    batch_token_type_ids = batch_token_type_ids.to(device)
    batch_ent_labels = batch_ent_labels.to(device)
    batch_rel_labels = batch_rel_labels.to(device)
    batch_token_labels = batch_token_labels.to(device)

    optimizer.zero_grad()
    
    # 前向传播：返回实体得分、关系得分、增强特征
    ent_logits, rel_logits, enhanced_state = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
    
    # 计算多任务损失
    total_loss, loss_ent, loss_rel, loss_scl = criterion(
        ent_logits, batch_ent_labels, 
        rel_logits, batch_rel_labels, 
        enhanced_state, batch_token_labels
    )
    
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), loss_ent.item(), loss_rel.item(), loss_scl.item()

def train(model, dataloader, epoch, optimizer):
    model.train()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    
    total_loss_sum = 0.
    for batch_ind, batch_data in pbar:
        loss, l_ent, l_rel, l_scl = train_step(batch_data, model, optimizer, criterion)
        total_loss_sum += loss
        avg_loss = total_loss_sum / (batch_ind + 1)
        
        pbar.set_description(f'Epoch: {epoch + 1}/{hyper_parameters["epochs"]}, Step: {batch_ind + 1}/{len(dataloader)}')
        pbar.set_postfix(loss=avg_loss, l_ent=l_ent, l_rel=l_rel, l_scl=l_scl)

        if conf["logger"] == "wandb" and batch_ind % conf["log_interval"] == 0:
            logger.log({
                "epoch": epoch,
                "train_total_loss": avg_loss,
                "loss_entity": l_ent,
                "loss_relation": l_rel,
                "loss_contrastive": l_scl,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
    return avg_loss
def valid(model, dataloader):
    model.eval()
    
    # 分别统计实体和关系的 X, Y, Z (预测正确数，预测总数，真实总数)
    ent_total_X, ent_total_Y, ent_total_Z = 0., 0., 0.
    rel_total_X, rel_total_Y, rel_total_Z = 0., 0., 0.
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating"):
            (_, batch_input_ids, batch_attention_mask, batch_token_type_ids, 
             batch_ent_labels, batch_rel_labels, _) = batch_data
             
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)
            batch_ent_labels = batch_ent_labels.to(device)
            batch_rel_labels = batch_rel_labels.to(device)

            ent_logits, rel_logits, _ = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            
            # 评估实体
            eX, eY, eZ = metrics.get_evaluate_fpr(ent_logits, batch_ent_labels)
            ent_total_X += eX; ent_total_Y += eY; ent_total_Z += eZ
            
            # 评估关系
            rX, rY, rZ = metrics.get_evaluate_fpr(rel_logits, batch_rel_labels)
            rel_total_X += rX; rel_total_Y += rY; rel_total_Z += rZ

    # 计算 Entity F1
    ent_f1 = 2 * ent_total_X / (ent_total_Y + ent_total_Z) if (ent_total_Y + ent_total_Z) > 0 else 0
    ent_p = ent_total_X / ent_total_Y if ent_total_Y > 0 else 0
    ent_r = ent_total_X / ent_total_Z if ent_total_Z > 0 else 0
    
    # 计算 Relation F1
    rel_f1 = 2 * rel_total_X / (rel_total_Y + rel_total_Z) if (rel_total_Y + rel_total_Z) > 0 else 0
    rel_p = rel_total_X / rel_total_Y if rel_total_Y > 0 else 0
    rel_r = rel_total_X / rel_total_Z if rel_total_Z > 0 else 0

    print("\n" + "="*40)
    print(f"[Entity]   P: {ent_p:.4f}, R: {ent_r:.4f}, F1: {ent_f1:.4f}")
    print(f"[Relation] P: {rel_p:.4f}, R: {rel_r:.4f}, F1: {rel_f1:.4f}")
    print("="*40 + "\n")
    
    if conf["logger"] == "wandb":
        logger.log({
            "valid_ent_f1": ent_f1, "valid_ent_p": ent_p, "valid_ent_r": ent_r,
            "valid_rel_f1": rel_f1, "valid_rel_p": rel_p, "valid_rel_r": rel_r
        })
        
    # 
    return ent_f1, rel_f1


import matplotlib.pyplot as plt

def plot_simplified_metrics(history, save_path="training_results.png"):
    """
    只绘制总 Loss 和两个 F1 分数的简化版图表
    """
    epochs = range(1, len(history['total_loss']) + 1)
    plt.figure(figsize=(12, 5)) # 调整为适合并排显示两张图的比例

    # 左图：总 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['total_loss'], 'b-o', label='Total Loss', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # 右图：实体与关系的 F1 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['ent_f1'], 'r-s', label='Entity F1')
    plt.plot(epochs, history['rel_f1'], 'g-o', label='Relation F1')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.0) # F1 分数固定在 0 到 1 之间
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"图表已成功生成并保存至: {save_path}")

if __name__ == '__main__':
    if conf["run_type"] == "train":
        train_dataloader, valid_dataloader = data_generator()

        history = {
        'total_loss': [], 
        'ent_f1': [], 
        'rel_f1': []
    }
        optimizer = torch.optim.Adam(model.parameters(), lr=float(hyper_parameters["lr"]))
        max_rel_f1 = 0.
        
        for epoch in range(hyper_parameters["epochs"]):
            avg_loss=train(model, train_dataloader, epoch, optimizer)
            ent_f1, current_rel_f1 = valid(model, valid_dataloader)
            history['total_loss'].append(avg_loss)
            history['ent_f1'].append(ent_f1)
            history['rel_f1'].append(current_rel_f1)
            # 根据联合抽取中最重要的关系抽取 F1 值来保存模型
            if current_rel_f1 > max_rel_f1:
                max_rel_f1 = current_rel_f1
                if current_rel_f1 > conf.get("f1_2_save", 0.1): 
                    model_state_num = len(glob.glob(model_state_dict_dir + "/model_state_dict_*.pt"))
                    save_path = os.path.join(model_state_dict_dir, f"joint_model_{model_state_num}_f1_{max_rel_f1:.4f}.pt")
                    torch.save(model.state_dict(), save_path)
                    print(f"--> Saved best model to {save_path}")
            
            print(f"Best Relation F1 so far: {max_rel_f1:.4f}")
        plot_simplified_metrics(history)


