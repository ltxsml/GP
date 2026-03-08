import ahocorasick
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from common.utils import Preprocessor

# ==========================================
# 模块一：医学词典处理器 (基于 AC 自动机)
# ==========================================
class LexiconProcessor:
    def __init__(self, dict_path, embedding_dim=200):
        self.embedding_dim = embedding_dim
        self.actree = ahocorasick.Automaton()
        
        # 加载 THUOCL 医学词库
        print(f"Loading lexicon from {dict_path}...")
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    word = parts[0]
                    # 将词汇作为 value 存入，方便后续提取
                    self.actree.add_word(word, word)
        self.actree.make_automaton()

    # def get_lexicon_features(self, text, max_seq_len):
    #     # 初始化全零矩阵 [max_len, dim]
    #     features = np.zeros((max_seq_len, self.embedding_dim))
        
    #     # AC 自动机全文扫描，复杂度为 O(n)
    #     for end_index, word in self.actree.iter(text):
    #         start_index = end_index - len(word) + 1
    #         # 仅在文本范围内记录特征
    #         if start_index < max_seq_len and end_index < max_seq_len:
    #             # 论文逻辑：为匹配到词表的 Token 注入先验知识权重
    #             # 此处演示为“存在即赋值”，如果有预训练词向量效果更佳
    #             features[start_index : end_index + 1] += 1.0
                
    #     return torch.tensor(features, dtype=torch.float)
    # 优化后的 get_lexicon_features 核心逻辑
    def get_lexicon_features(self, text, max_seq_len):
        features = np.zeros((max_seq_len, self.embedding_dim), dtype=np.float32)
        # 预先过滤长度，减少循环次数
        for end_index, word in self.actree.iter(text):
            start_index = end_index - len(word) + 1
            if start_index < max_seq_len:
                end_pos = min(end_index + 1, max_seq_len)
                # 批量赋值优化
                features[start_index : end_pos] = 1.0 
        return torch.from_numpy(features) # 比 torch.tensor 更快

# ==========================================
# 模块二：门控融合网络 (Gated Fusion)
# ==========================================
class GatedFusion(nn.Module):
    def __init__(self, hidden_size, lex_dim):
        super().__init__()
        # 门控权重计算：拼接 BERT 隐层和词表特征
        self.gate = nn.Linear(hidden_size + lex_dim, hidden_size)
        # 空间投影：对齐不同来源的向量空间
        self.lex_proj = nn.Linear(lex_dim, hidden_size)

    def forward(self, h_bert, l_lex):
        # 门控激活：g = sigmoid(Wg * [h; l])
        g = torch.sigmoid(self.gate(torch.cat([h_bert, l_lex], dim=-1)))
        # 词汇投影：hat{l} = tanh(Wl * l)
        l_hat = torch.tanh(self.lex_proj(l_lex))
        # 动态融合：x = g * l_hat + (1 - g) * h_bert
        return g * l_hat + (1 - g) * h_bert

# ==========================================
# 模块三：改进后的 GlobalPointer 模型
# ==========================================
class ImprovedGlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, lex_dim=200, RoPE=True):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.RoPE = RoPE

        # 核心创新点：注入门控融合模块
        self.fusion = GatedFusion(self.hidden_size, lex_dim)
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        # 1. 生成位置序列 [seq_len, 1]
        position_ids = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(-1)
        
        # 2. 生成频率项 [output_dim // 2]
        indices = torch.arange(0, output_dim // 2, dtype=torch.float, device=device)
        indices = torch.pow(10000, -2 * indices / output_dim)
        
        # 3. 计算嵌入 [seq_len, output_dim // 2]
        embeddings = position_ids * indices
        
        # 4. 拼接正弦和余弦 [seq_len, output_dim // 2, 2]
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        # 5. 重塑为 [seq_len, output_dim]
        embeddings = embeddings.reshape(seq_len, output_dim)
        
        # 6. 关键步骤：扩展到 batch_size 维度 [batch_size, seq_len, output_dim]
        # 使用 repeat 或 expand 使其匹配输入形状
        return embeddings.unsqueeze(0).expand(batch_size, seq_len, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids, lex_features):
        device = input_ids.device
        # 确保词汇特征与模型在同一设备
        lex_features = lex_features.to(device)

        # 1. 基础 BERT 编码
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs[0] 

        # 2. 门控词汇融合 (创新点所在)
        fused_state = self.fusion(last_hidden_state, lex_features)

        # 3. 生成 Query 和 Key
        outputs = self.dense(fused_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        # 4. 旋转位置编码 RoPE
        if self.RoPE:
            batch_size, seq_len = last_hidden_state.shape[:2]
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim, device)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1).reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1).reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # 5. 得分矩阵计算
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand_as(logits)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5

# ==========================================
# 模块四：数据加载与 Batch 封装
# ==========================================
class DataMaker(object):
    def __init__(self, tokenizer, dict_path="THUOCL_medical.txt", add_special_tokens=True):
        self.tokenizer = tokenizer
        self.preprocessor = Preprocessor(tokenizer, add_special_tokens)
        self.lexicon_processor = LexiconProcessor(dict_path)

    def generate_batch(self, batch_data, max_seq_len, ent2id, data_type="train"):
        sample_list, input_ids_list, attention_mask_list, token_type_ids_list = [], [], [], []
        labels_list, lex_feat_list = [], []

        for sample in batch_data:
            # BERT Tokenize
            inputs = self.tokenizer(sample["text"], max_length=max_seq_len, 
                                    truncation=True, padding='max_length')
            # 词汇匹配特征
            lex_feat = self.lexicon_processor.get_lexicon_features(sample["text"], max_seq_len)
            
            # 标签构建
            labels = np.zeros((len(ent2id), max_seq_len, max_seq_len))
            if data_type != "predict":
                spans = self.preprocessor.get_ent2token_spans(sample["text"], sample["entity_list"])
                for s, e, l in spans:
                    if s < max_seq_len and e < max_seq_len:
                        labels[ent2id[l], s, e] = 1

            sample_list.append(sample)
            input_ids_list.append(torch.tensor(inputs["input_ids"]).long())
            attention_mask_list.append(torch.tensor(inputs["attention_mask"]).long())
            token_type_ids_list.append(torch.tensor(inputs["token_type_ids"]).long())
            lex_feat_list.append(lex_feat)
            if data_type != "predict": labels_list.append(torch.tensor(labels).long())

        return (sample_list, 
                torch.stack(input_ids_list), 
                torch.stack(attention_mask_list), 
                torch.stack(token_type_ids_list), 
                torch.stack(labels_list) if labels_list else None, 
                torch.stack(lex_feat_list))
    


    # ==========================================
# 模块五：Dataset 封装 (用于 DataLoader)
# ==========================================
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# ==========================================
# 模块六：指标计算 (Metrics)
# ==========================================
class MetricsCalculator:
    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.data.cpu().numpy()
        y_true = y_true.data.cpu().numpy()
        # 阈值过滤，GlobalPointer 输出大于 0 视为存在实体
        pred = np.where(y_pred > 0, 1, 0)
        
        X = np.sum(pred * y_true) # 预测正确的数量
        Y = np.sum(pred)          # 预测出的实体总数
        Z = np.sum(y_true)        # 样本中的真实体总数
        return X, Y, Z
    
def plot_training_results(losses, f1_scores):
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, 'b-o', label='Training Loss')
    plt.title('Convergence Analysis')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, f1_scores, 'r-s', label='Validation F1')
    plt.title('Performance Growth')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('train_metrics.png', dpi=300)
    plt.show()