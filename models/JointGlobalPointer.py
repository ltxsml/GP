import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BoundaryAwareAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        # 多头注意力用于边界特征交互
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        # 边界门控网络
        self.boundary_gate = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_mask):
        # 将 attention_mask 转换为 MultiheadAttention 需要的 key_padding_mask (True 表示需要被 mask 掉)
        key_padding_mask = (attention_mask == 0)
        
        # 1. 全局边界特征交互
        attn_output, _ = self.attention(
            query=hidden_states, 
            key=hidden_states, 
            value=hidden_states, 
            key_padding_mask=key_padding_mask
        )
        
        # 2. 边界门控机制：学习哪些 Token 更可能是边界
        gate = torch.sigmoid(self.boundary_gate(attn_output))
        
        # 3. 增强边界特征并残差连接
        enhanced_states = self.layer_norm(hidden_states + gate * attn_output)
        return enhanced_states
    

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [batch_size, seq_len, hidden_size]
        labels: 对应的实体或关系类别标签 (简化版，用于Token级别的特征对比)
        """
        # 将特征展平并进行 L2 归一化
        features = F.normalize(features, p=2, dim=-1)
        batch_size, seq_len, hidden_dim = features.shape
        flat_features = features.view(-1, hidden_dim) # [batch * seq_len, hidden_dim]
        flat_labels = labels.view(-1)                 # [batch * seq_len]

        # 仅对有实体/关系的 Token 计算对比损失 (过滤掉背景 O 标签)
        mask = (flat_labels > 0)
        valid_features = flat_features[mask]
        valid_labels = flat_labels[mask]

        if len(valid_labels) < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # 计算余弦相似度矩阵
        similarity_matrix = torch.matmul(valid_features, valid_features.T) / self.temperature

        # 构建正样本掩码 (类别相同为正样本)
        labels_matrix = valid_labels.unsqueeze(0) == valid_labels.unsqueeze(1)
        labels_mask = labels_matrix.float().to(features.device)

        # 排除自身与自身的对比
        eye_mask = torch.eye(labels_mask.shape[0], device=features.device)
        labels_mask = labels_mask - eye_mask

        # 计算 InfoNCE 损失
        exp_sim = torch.exp(similarity_matrix) * (1 - eye_mask)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # 求和求平均
        scl_loss = - (labels_mask * log_prob).sum(dim=1) / (labels_mask.sum(dim=1) + 1e-8)
        return scl_loss.mean()
    


class JointExtractionLoss(nn.Module):
    def __init__(self, lambda_scl=0.1):
        super().__init__()
        self.lambda_scl = lambda_scl
        self.scl_loss_fn = SupervisedContrastiveLoss()

    def multilabel_categorical_crossentropy(self, y_pred, y_true):
        """标准的 Global Pointer 稀疏多标签交叉熵损失"""
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()

    def forward(self, ent_pred, ent_true, rel_pred, rel_true, features, token_labels):
        # 1. 实体分类损失 L_ent
        loss_ent = self.multilabel_categorical_crossentropy(ent_pred, ent_true)
        
        # 2. 关系分类损失 L_rel
        loss_rel = self.multilabel_categorical_crossentropy(rel_pred, rel_true)
        
        # 3. 监督对比学习损失 L_scl
        loss_scl = self.scl_loss_fn(features, token_labels)

        # 总损失 = 实体损失 + 关系损失 + λ * 对比损失
        total_loss = loss_ent + loss_rel + self.lambda_scl * loss_scl
        
        return total_loss, loss_ent, loss_rel, loss_scl
    

    
class JointCascadeGlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, rel_type_size, inner_dim, RoPE=True):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.rel_type_size = rel_type_size # 新增：关系类别数量
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.RoPE = RoPE

        # 创新点1：加入边界感知注意力
        self.boundary_attention = BoundaryAwareAttention(self.hidden_size)

        # 第一级：实体抽取 Global Pointer
        self.ent_dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        
        # 第二级：关系抽取 Global Pointer (输入为 hidden_size + 实体先验特征)
        # 创新点2：将实体的隐藏状态作为关系的先验知识注入
        self.rel_prior_proj = nn.Linear(self.ent_type_size, self.hidden_size)
        self.rel_dense = nn.Linear(self.hidden_size * 2, self.rel_type_size * self.inner_dim * 2)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        # (复用你原来的 RoPE 代码逻辑)
        position_ids = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float, device=device)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape(seq_len, output_dim)
        return embeddings.unsqueeze(0).expand(batch_size, seq_len, output_dim)

    def compute_gp_matrix(self, hidden_states, dense_layer, type_size, attention_mask):
        """复用的 Global Pointer 矩阵计算核心逻辑"""
        batch_size, seq_len = hidden_states.shape[:2]
        device = hidden_states.device
        
        outputs = dense_layer(hidden_states)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        if self.RoPE:
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim, device)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1).reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1).reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand_as(logits)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        return logits / self.inner_dim ** 0.5

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 1. 基础编码
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs[0] 

        # 2. 边界感知注意力增强 (取代原本直接传入 dense 层)
        enhanced_state = self.boundary_attention(last_hidden_state, attention_mask)

        # 3. 第一级抽取：实体识别 (Entity Extraction)
        ent_logits = self.compute_gp_matrix(enhanced_state, self.ent_dense, self.ent_type_size, attention_mask)

        # 4. 特征融合：将实体预测的分布特征映射回隐藏空间作为关系抽取的先验
        # ent_logits 的形状为 [batch, ent_type, seq_len, seq_len]
        # 我们对其在尾部维度进行 pooling 或 max，提取每个 token 的实体倾向性
        ent_prior, _ = torch.max(ent_logits, dim=-1) # [batch, ent_type, seq_len]
        ent_prior = ent_prior.transpose(1, 2)        # [batch, seq_len, ent_type]
        ent_prior_features = torch.relu(self.rel_prior_proj(ent_prior))

        # 级联拼接：原始语义特征 + 实体边界增强特征 + 实体先验特征
        rel_hidden_state = torch.cat([enhanced_state, ent_prior_features], dim=-1)

        # 5. 第二级抽取：关系识别 (Relation Extraction)
        # 关系矩阵：(batch, rel_type_size, seq_len(头实体), seq_len(尾实体))
        rel_logits = self.compute_gp_matrix(rel_hidden_state, self.rel_dense, self.rel_type_size, attention_mask)

        # 返回实体 logits，关系 logits，以及用于对比学习的增强特征
        return ent_logits, rel_logits, enhanced_state
    
class DataMakerJoint(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate_inputs(self, datas, max_seq_len, ent2id, rel2id, data_type="train"):
        all_inputs = []
        for sample in datas:
            text = sample["text"]
            # 使用 return_offsets_mapping 能够精准匹配字符级与 Token 级的位置！非常重要！
            inputs = self.tokenizer(
                text,
                max_length=max_seq_len,
                truncation=True,
                padding='max_length',
                return_offsets_mapping=True 
            )
            
            offset_mapping = inputs["offset_mapping"]
            
            ent_labels = None
            rel_labels = None
            token_labels = None # 用于对比学习的 token 级标签
            
            if data_type != "predict":
                # [ent_type_size, seq_len, seq_len]
                ent_labels = np.zeros((len(ent2id), max_seq_len, max_seq_len))
                # [rel_type_size, seq_len, seq_len]
                rel_labels = np.zeros((len(rel2id), max_seq_len, max_seq_len))
                # [seq_len] 用于给对比学习标明每个 token 属于哪个实体类别（取首个类别）
                token_labels = np.zeros(max_seq_len) 

                # 构建字符索引到 token 索引的映射
                char2token = {}
                for idx, (start_char, end_char) in enumerate(offset_mapping):
                    if start_char == end_char == 0:
                        continue
                    for char_idx in range(start_char, end_char):
                        char2token[char_idx] = idx

                # 1. 构建实体矩阵
                for ent in sample.get("entity_list", []):
                    s_char, e_char = ent["start"], ent["end"]
                    if s_char in char2token and e_char in char2token:
                        s_tok, e_tok = char2token[s_char], char2token[e_char]
                        ent_type_id = ent2id.get(ent["type"], 0)
                        ent_labels[ent_type_id, s_tok, e_tok] = 1
                        
                        # 记录 token 级标签，供 SCL(对比学习) 损失函数使用
                        token_labels[s_tok:e_tok+1] = ent_type_id + 1 

                # 2. 构建关系矩阵
                for spo in sample.get("spo_list", []):
                    sub_s_char, obj_s_char = spo["sub_start"], spo["obj_start"]
                    # 我们用主语的 Head token 和 宾语的 Head token 来表示它们的关系
                    if sub_s_char in char2token and obj_s_char in char2token:
                        sub_head_tok = char2token[sub_s_char]
                        obj_head_tok = char2token[obj_s_char]
                        rel_type_id = rel2id.get(spo["predicate"], 0)
                        rel_labels[rel_type_id, sub_head_tok, obj_head_tok] = 1

            # 转换为 Tensor
            input_ids = torch.tensor(inputs["input_ids"]).long()
            attention_mask = torch.tensor(inputs["attention_mask"]).long()
            token_type_ids = torch.tensor(inputs["token_type_ids"]).long()
            
            if data_type != "predict":
                ent_labels = torch.tensor(ent_labels).float()
                rel_labels = torch.tensor(rel_labels).float()
                token_labels = torch.tensor(token_labels).long()
                
            all_inputs.append((sample, input_ids, attention_mask, token_type_ids, ent_labels, rel_labels, token_labels))
            
        return all_inputs

    def generate_batch(self, batch_data, max_seq_len, ent2id, rel2id, data_type="train"):
        batch_inputs = self.generate_inputs(batch_data, max_seq_len, ent2id, rel2id, data_type)
        
        sample_list = [item[0] for item in batch_inputs]
        batch_input_ids = torch.stack([item[1] for item in batch_inputs], dim=0)
        batch_attention_mask = torch.stack([item[2] for item in batch_inputs], dim=0)
        batch_token_type_ids = torch.stack([item[3] for item in batch_inputs], dim=0)
        
        if data_type != "predict":
            batch_ent_labels = torch.stack([item[4] for item in batch_inputs], dim=0)
            batch_rel_labels = torch.stack([item[5] for item in batch_inputs], dim=0)
            batch_token_labels = torch.stack([item[6] for item in batch_inputs], dim=0)
            return sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_ent_labels, batch_rel_labels, batch_token_labels
        else:
            return sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, None, None, None
