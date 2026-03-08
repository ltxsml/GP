import os
import config
import sys
import torch
import json
from transformers import BertTokenizerFast, BertModel
from common.utils import Preprocessor, multilabel_categorical_crossentropy
# 注意：此处改为从你刚才创建的 modelplus 导入新组件
from modelplus import DataMaker, MyDataset, ImprovedGlobalPointer, MetricsCalculator, plot_training_results
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import glob
import wandb
import time

# 加载配置
config = config.train_config
hyper_parameters = config["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(hyper_parameters["seed"])
torch.backends.cudnn.deterministic = True

# 初始化 WandB 或 本地目录
if config["logger"] == "wandb" and config["run_type"] == "train":
    wandb.init(project="Improved_GP_" + config["exp_name"], config=hyper_parameters)
    wandb.run.name = config["run_name"] + "_Gated_" + wandb.run.id
    model_state_dict_dir = wandb.run.dir
    logger = wandb
else:
    model_state_dict_dir = os.path.join(config["path_to_save_model"], config["exp_name"], "improved_" + time.strftime("%Y-%m-%d_%H.%M.%S"))
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)

tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], do_lower_case=False)
metrics = MetricsCalculator()

# --- 数据加载函数 (保持原有逻辑) ---
def load_data(data_path, data_type="train"):
    """读取数据集
    由于 prepare_cmeee.py 生成的是标准 JSON 数组格式，
    我们需要直接使用 json.load(f) 一次性读入。
    """
    if data_type == "train" or data_type == "valid":
        # 核心修改点：直接 load 整个列表
        with open(data_path, 'r', encoding="utf-8") as f:
            datas = json.load(f) 
        return datas
    else:
        # ent2id 等其他文件的读取保持不变
        return json.load(open(data_path, encoding="utf-8"))

# --- 修改点 1: Data Generator 适配 lex_features ---
def data_generator(data_type="train"):
    ent2id_path = os.path.join(config["data_home"], config["exp_name"], config["ent2id"])
    ent2id = load_data(ent2id_path, "ent2id")
    
    # 获取路径
    train_path = os.path.join(config["data_home"], config["exp_name"], config["train_data"])
    valid_path = os.path.join(config["data_home"], config["exp_name"], config["valid_data"])
    
    train_data = load_data(train_path, "train") if data_type == "train" else []
    valid_data = load_data(valid_path, "valid")
    
    max_seq_len = hyper_parameters["max_seq_len"]
    # 使用 modelplus 中支持词表的 DataMaker
    data_maker = DataMaker(tokenizer, dict_path="THUOCL_medical.txt") 

    train_loader = DataLoader(MyDataset(train_data), batch_size=hyper_parameters["batch_size"],
                              shuffle=True, num_workers=config["num_workers"],
                              collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id))
    
    valid_loader = DataLoader(MyDataset(valid_data), batch_size=hyper_parameters["batch_size"],
                              shuffle=False, num_workers=config["num_workers"],
                              collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id))
    
    return train_loader, valid_loader, ent2id

# --- 修改点 2: 训练步适配门控输入 ---
def train_step(batch_train, model, optimizer, criterion):
    # 解构 6 个返回值：多了最后的 batch_lex_features
    _, ids, mask, tids, labels, lex_feat = batch_train
    
    ids, mask, tids, labels, lex_feat = (ids.to(device), mask.to(device), 
                                         tids.to(device), labels.to(device), lex_feat.to(device))

    # 喂入模型：增加 lex_features 参数
    logits = model(ids, mask, tids, lex_features=lex_feat)
    loss = criterion(labels, logits)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# --- 修改点 3: 验证步适配 ---
def valid_step(batch_valid, model):
    _, ids, mask, tids, labels, lex_feat = batch_valid
    ids, mask, tids, labels, lex_feat = (ids.to(device), mask.to(device), 
                                         tids.to(device), labels.to(device), lex_feat.to(device))
    with torch.no_grad():
        logits = model(ids, mask, tids, lex_features=lex_feat)
    X, Y, Z = metrics.get_evaluate_fpr(logits, labels)
    return X, Y, Z

def valid(model, dataloader):
    model.eval()
    total_X, total_Y, total_Z = 0., 0., 0.
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating"):
            X, Y, Z = valid_step(batch_data, model)
            total_X, total_Y, total_Z = total_X + X, total_Y + Y, total_Z + Z

    # 核心计算逻辑
    f1, precision, recall = 0., 0., 0.
    if total_Y + total_Z > 0:
        f1 = 2 * total_X / (total_Y + total_Z + 1e-12)
    if total_Y > 0:
        precision = total_X / (total_Y + 1e-12)
    if total_Z > 0:
        recall = total_X / (total_Z + 1e-12)

    print("******************************************")
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    print("******************************************")
    
    if config["logger"] == "wandb":
        logger.log({
            "valid_precision": precision, 
            "valid_recall": recall, 
            "valid_f1": f1
        })
    return f1
# --- 主训练流程 ---
if __name__ == '__main__':
    train_dataloader, valid_dataloader, ent2id = data_generator()
    
    # 1. 实例化改进模型
    encoder = BertModel.from_pretrained(config["bert_path"])
    model = ImprovedGlobalPointer(encoder, len(ent2id), 64).to(device)
    
    if config["logger"] == "wandb": 
        wandb.watch(model)
    
    # 2. 优化器：建议改用 AdamW (BERT 训练标准)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(hyper_parameters["lr"]))
    
    # 3. 调度器逻辑 (从普通版迁移并增强)
    if hyper_parameters.get("scheduler") == "CAWR":
        T_mult = hyper_parameters.get("T_mult", 1)
        rewarm_epoch_num = hyper_parameters.get("rewarm_epoch_num", 2)
        # T_0 为重启周期：这里设置为 rewarm_epoch_num 个 Epoch 的总步数
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=len(train_dataloader) * rewarm_epoch_num,
            T_mult=T_mult,
            eta_min=1e-7
        )
        print(f"--- 激活 CAWR 调度器: rewarm_epoch_num={rewarm_epoch_num}, T_mult={T_mult} ---")
    else:
        scheduler = None

    # Loss 封装
    def loss_fun(y_true, y_pred):
        y_true = y_true.reshape(y_true.shape[0] * len(ent2id), -1)
        y_pred = y_pred.reshape(y_pred.shape[0] * len(ent2id), -1)
        return multilabel_categorical_crossentropy(y_true, y_pred)

    max_f1 = 0.
    for epoch in range(hyper_parameters["epochs"]):
        model.train()
        total_loss = 0.
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        
        for batch_ind, batch_data in pbar:
            loss = train_step(batch_data, model, optimizer, loss_fun)
            
            # --- 关键：执行学习率更新 ---
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss
            avg_loss = total_loss / (batch_ind + 1)
            pbar.set_description(f'Epoch: {epoch + 1}/{hyper_parameters["epochs"]}')
            pbar.set_postfix(loss=avg_loss, lr=optimizer.param_groups[0]["lr"])

        # 验证逻辑保持不变
        valid_f1 = valid(model, valid_dataloader)
        
        if valid_f1 > max_f1:
            max_f1 = valid_f1
            if valid_f1 > config["f1_2_save"]:
                save_path = os.path.join(model_state_dict_dir, f"best_model_f1_{max_f1:.4f}.pt")
                torch.save(model.state_dict(), save_path)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | F1: {valid_f1:.4f} | Best: {max_f1:.4f}")