
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
    

class JointExtractionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_scales = nn.Parameter(torch.tensor([0.0, 0.0]))

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

    def forward(self, ent_pred, ent_true, rel_pred, rel_true):
        # 1. 实体分类损失 L_ent
        loss_ent = self.multilabel_categorical_crossentropy(ent_pred, ent_true)
        
        # 2. 关系分类损失 L_rel
        loss_rel = self.multilabel_categorical_crossentropy(rel_pred, rel_true)
        precision_ent = torch.exp(-self.loss_scales[0])
        precision_rel = torch.exp(-self.loss_scales[1])

        # 总损失 = 实体损失 + 关系损失
        # total_loss = loss_ent + loss_rel
        total_loss = (precision_ent * loss_ent + self.loss_scales[0]) + \
                     (precision_rel * loss_rel + self.loss_scales[1])
        
        return total_loss, loss_ent, loss_rel
    

# class JointCascadeGlobalPointer(nn.Module):
#     def __init__(self, encoder, ent_type_size, rel_type_size, inner_dim, 
#                  use_boundary_attn=True,
#                  use_dynamic_gate=True, # ====== 新增：动态门控开关 ======
#                  RoPE=True):
#         super().__init__()
#         self.encoder = encoder
#         self.ent_type_size = ent_type_size
#         self.rel_type_size = rel_type_size 
#         self.inner_dim = inner_dim
#         self.hidden_size = encoder.config.hidden_size
#         self.use_dynamic_gate = use_dynamic_gate
#         self.RoPE = RoPE
#         self.use_boundary_attn = use_boundary_attn
        
#         # 模块1：边界感知注意力
#         self.boundary_attention = BoundaryAwareAttention(self.hidden_size)

#         # 第一级：实体抽取 Global Pointer
#         self.ent_dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        
#         # 特征映射层
#         self.rel_prior_proj = nn.Linear(self.ent_type_size, self.hidden_size)
        
       
#        # 模块2：动态门控特征融合网络 (仅在开启时初始化)
#         if self.use_dynamic_gate:
#             self.gate_network = nn.Sequential(
#                 nn.Linear(self.hidden_size * 2, self.hidden_size),
#                 nn.Sigmoid() 
#             )

#         # 第二级：关系抽取 Global Pointer
#         self.rel_dense = nn.Linear(self.hidden_size * 2, self.rel_type_size * self.inner_dim * 2)

#     def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
#         position_ids = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(-1)
#         indices = torch.arange(0, output_dim // 2, dtype=torch.float, device=device)
#         indices = torch.pow(10000, -2 * indices / output_dim)
#         embeddings = position_ids * indices
#         embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
#         embeddings = embeddings.reshape(seq_len, output_dim)
#         return embeddings.unsqueeze(0).expand(batch_size, seq_len, output_dim)

#     def compute_gp_matrix(self, hidden_states, dense_layer, type_size, attention_mask, mask_tril=True):
#         batch_size, seq_len = hidden_states.shape[:2]
#         device = hidden_states.device
        
#         outputs = dense_layer(hidden_states)
#         outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
#         outputs = torch.stack(outputs, dim=-2)
#         qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

#         if self.RoPE:
#             pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim, device)
#             cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
#             sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
#             qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1).reshape(qw.shape)
#             qw = qw * cos_pos + qw2 * sin_pos
#             kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1).reshape(kw.shape)
#             kw = kw * cos_pos + kw2 * sin_pos

#         logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
#         pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand_as(logits)
#         logits = logits * pad_mask - (1 - pad_mask) * 1e12
#         if mask_tril:
#             mask = torch.tril(torch.ones_like(logits), -1)
#             logits = logits - mask * 1e12
#         return logits / self.inner_dim ** 0.5

#     def forward(self, input_ids, attention_mask, token_type_ids):
#         # 1. 基础编码
#         context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
#         last_hidden_state = context_outputs[0] 

#         # 2. 边界感知增强
#         if self.use_boundary_attn:
#             enhanced_state = self.boundary_attention(last_hidden_state, attention_mask)
#         else:
#             enhanced_state = last_hidden_state

#         # 3. 第一级抽取：实体识别
#         ent_logits = self.compute_gp_matrix(enhanced_state, self.ent_dense, self.ent_type_size, attention_mask, mask_tril=True)
#         ent_prob = torch.sigmoid(ent_logits)

#         # 4. 获取实体先验特征
#         ent_prior, _ = torch.max(ent_prob, dim=-1) # [batch, ent_type, seq_len]
#         ent_prior = ent_prior.transpose(1, 2)        # [batch, seq_len, ent_type]
#         ent_prior_features = torch.relu(self.rel_prior_proj(ent_prior))

#         # ==========================================
#         # 5. 门控提纯与特征融合
#         # ==========================================
#         if self.use_dynamic_gate:
#             # 开启门控：计算动态权重，过滤错误倾向
#             gate_input = torch.cat([enhanced_state, ent_prior_features], dim=-1)
#             gate_value = self.gate_network(gate_input) 
#             gated_ent_features = gate_value * ent_prior_features 
#             rel_hidden_state = torch.cat([enhanced_state, gated_ent_features], dim=-1)
#         else:
#             # 关闭门控：退化为基础的暴力拼接 (包含未经提纯的先验噪声)
#             rel_hidden_state = torch.cat([enhanced_state, ent_prior_features], dim=-1)
        


#         # 6. 第二级抽取：关系识别
#         rel_logits = self.compute_gp_matrix(rel_hidden_state, self.rel_dense, self.rel_type_size, attention_mask, mask_tril=False)

#         # 移除掉用于 SCL 的 enhanced_state 返回项
#         return ent_logits, rel_logits

class JointCascadeGlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, rel_type_size, inner_dim, 
                 use_boundary_attn=True, use_dynamic_gate=True, RoPE=True):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.rel_type_size = rel_type_size 
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.use_dynamic_gate = use_dynamic_gate
        self.RoPE = RoPE
        self.use_boundary_attn = use_boundary_attn
        
        # 1. 边界感知组件
        self.boundary_attention = BoundaryAwareAttention(self.hidden_size)

        # 2. 实体抽取层
        self.ent_dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        
        # 3. 关系先验投影层
        self.rel_prior_proj = nn.Linear(self.ent_type_size, self.hidden_size)

        # 4. 动态门控网络
        if self.use_dynamic_gate:
            self.gate_network = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.Sigmoid() 
            )

        # 5. 关系抽取层：输入维度固定为 hidden_size * 3 (原始 + 先验 + Span语义)
        self.rel_dense = nn.Linear(self.hidden_size * 3, self.rel_type_size * self.inner_dim * 2)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        position_ids = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float, device=device)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape(seq_len, output_dim)
        return embeddings.unsqueeze(0).expand(batch_size, seq_len, output_dim)

    def compute_gp_matrix(self, hidden_states, dense_layer, type_size, attention_mask, mask_tril=True):
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
        if mask_tril:
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * 1e12
        return logits / self.inner_dim ** 0.5

    def forward(self, input_ids, attention_mask, token_type_ids):
        # A. 基础编码
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs[0] 

        # B. 边界增强
        enhanced_state = self.boundary_attention(last_hidden_state, attention_mask) if self.use_boundary_attn else last_hidden_state

        # C. 第一级：实体识别 (必须先计算以获取概率矩阵)
        ent_logits = self.compute_gp_matrix(enhanced_state, self.ent_dense, self.ent_type_size, attention_mask, mask_tril=True)
        ent_prob = torch.sigmoid(ent_logits) # [batch, ent_type, seq, seq]

        # D. Span 语义注入 (基于实体预测的动态聚合)
        span_prob_matrix = torch.max(ent_prob, dim=1)[0] # [batch, seq, seq]
        span_semantics = torch.matmul(span_prob_matrix, enhanced_state) # [batch, seq, hidden_size]

        # E. 获取实体先验特征
        ent_prior, _ = torch.max(ent_prob, dim=-1) 
        ent_prior = ent_prior.transpose(1, 2) # [batch, seq, ent_type]
        ent_prior_features = torch.relu(self.rel_prior_proj(ent_prior))

        # F. 动态门控提纯与特征融合
        if self.use_dynamic_gate:
            gate_input = torch.cat([enhanced_state, ent_prior_features], dim=-1)
            gate_value = self.gate_network(gate_input) 
            final_ent_features = gate_value * ent_prior_features 
        else:
            final_ent_features = ent_prior_features

        # 拼接 3 部分特征：原始语义 + 实体先验/门控 + Span 结构语义
        rel_hidden_state = torch.cat([enhanced_state, final_ent_features, span_semantics], dim=-1)

        # G. 第二级：关系识别
        rel_logits = self.compute_gp_matrix(rel_hidden_state, self.rel_dense, self.rel_type_size, attention_mask, mask_tril=False)

        return ent_logits, rel_logits
    
class DataMakerJoint(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate_inputs(self, datas, max_seq_len, ent2id, rel2id, data_type="train"):
        all_inputs = []
        for sample in datas:
            text = sample["text"]
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
            
            if data_type != "predict":
                # 修改点：加入 dtype=np.int8，立省 87% 的 DRAM 内存占用！
                ent_labels = np.zeros((len(ent2id), max_seq_len, max_seq_len), dtype=np.int8)
                rel_labels = np.zeros((len(rel2id), max_seq_len, max_seq_len), dtype=np.int8)

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

                # 2. 构建关系矩阵
                for spo in sample.get("spo_list", []):
                    sub_s_char, obj_s_char = spo["sub_start"], spo["obj_start"]
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
                # 移除了 token_labels 的组装
                all_inputs.append((sample, input_ids, attention_mask, token_type_ids, ent_labels, rel_labels))
            
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
            # 移除了 token_labels 的返回
            return sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_ent_labels, batch_rel_labels
        else:
            return sample_list