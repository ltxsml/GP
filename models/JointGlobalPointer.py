
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class DecoupledBoundaryGate(nn.Module):
    """
    解耦边界门控 (Decoupled Boundary Gate)
    独立预测每个 Token 作为"起点(Start)"和"终点(End)"的概率，
    并将其直接作用于 GlobalPointer 的 Query 和 Key 上进行特征缩放，硬性过滤非边界噪声。
    """
    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()
        
        # 预测作为起点(Start)的概率
        self.start_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(), # 新增一层激活函数
            nn.Dropout(dropout_prob), # 新增 Dropout 防止过拟合
            nn.Linear(hidden_size // 2, hidden_size // 4), # 新增一层线性层
            nn.GELU(),
            nn.Dropout(dropout_prob), # 新增 Dropout 防止过拟合
            nn.Linear(hidden_size // 4, 1), # 调整输入维度
            nn.Sigmoid()
        )
        
        # 预测作为终点(End)的概率
        self.end_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(), # 新增一层激活函数
            nn.Dropout(dropout_prob), # 新增 Dropout 防止过拟合
            nn.Linear(hidden_size // 2, hidden_size // 4), # 新增一层线性层
            nn.GELU(),
            nn.Dropout(dropout_prob), # 新增 Dropout 防止过拟合
            nn.Linear(hidden_size // 4, 1), # 调整输入维度
            nn.Sigmoid()
        )

    def forward(self, hidden_states, attention_mask):
        # 输出形状均为 [batch_size, seq_len, 1]
        start_prob = self.start_gate(hidden_states)
        end_prob = self.end_gate(hidden_states)
        return start_prob, end_prob
    

class JointExtractionLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.loss_scales = nn.Parameter(torch.tensor([0.0, 0.0]))
        self.alpha = alpha

    def multilabel_categorical_crossentropy(self, y_pred, y_true):
        """标准的 Global Pointer 稀疏多标签交叉熵损失"""
        # 【修正】将 4D 矩阵展平为 2D，否则 logsumexp 会计算错误并加剧梯度爆炸
        batch_size, type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * type_size, -1)
        y_pred = y_pred.reshape(batch_size * type_size, -1)
        
        y_true = y_true.to(y_pred.dtype)
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e4
        y_pred_pos = y_pred - (1 - y_true) * 1e4
        
        y_pred_neg = y_pred_neg.to(y_pred.dtype)
        y_pred_pos = y_pred_pos.to(y_pred.dtype)
        
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

        # 基础的多任务自动平衡损失
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
                 use_boundary_attn=True, use_dynamic_gate=True, RoPE=True, 
                 use_mlp_rel=True): # ====== 新增：关系抽取层结构兼容开关 ======
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.rel_type_size = rel_type_size 
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.use_dynamic_gate = use_dynamic_gate
        self.RoPE = RoPE
        self.use_boundary_attn = use_boundary_attn
        self.use_mlp_rel = use_mlp_rel
        
        # 1. 解耦边界门控组件
        self.boundary_gate = DecoupledBoundaryGate(self.hidden_size)

        # 2. 实体抽取层
        self.ent_dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        
        # 3. 关系先验投影层
        self.rel_prior_proj = nn.Linear(self.ent_type_size, self.hidden_size)
        # 【新增】尾部类型投影层：为尾部 Token 注入类型特征
        self.tail_type_proj = nn.Linear(self.ent_type_size, self.hidden_size)
        # 【新增】尾部上下文投影层：用于整合实体的尾部跨度信息
        self.tail_context_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # 4. 动态门控网络
        if self.use_dynamic_gate:
            self.gate_network = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.Sigmoid() 
            )

        # 5. 关系抽取层：输入维度固定为 hidden_size * 2 (原始 + 先验)
        # 根据开关动态选择结构，以兼容加载老版本训练的模型权重
        if self.use_mlp_rel:
            self.rel_dense = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size * 2), # 修复 1：保持 1536 维，不制造信息瓶颈
                # 修复 2：移除 LayerNorm，保护 GlobalPointer 点乘所需的特征绝对幅值
                nn.GELU(),
                nn.Dropout(0.1),                                       # 修复 3：引入 Dropout 防止多层网络的过拟合
                nn.Linear(self.hidden_size * 2, self.rel_type_size * self.inner_dim * 2)
            )
        else:
            self.rel_dense = nn.Linear(self.hidden_size * 2, self.rel_type_size * self.inner_dim * 2)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        position_ids = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float, device=device)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape(seq_len, output_dim)
        return embeddings.unsqueeze(0).expand(batch_size, seq_len, output_dim)

    def compute_gp_matrix(self, hidden_states, dense_layer, type_size, attention_mask, mask_tril=True, q_gate=None, k_gate=None):
        batch_size, seq_len = hidden_states.shape[:2]
        device = hidden_states.device
        
        outputs = dense_layer(hidden_states)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        # 【核心注入】使用门控概率对 Query 和 Key 进行硬性特征缩放
        # 改为“软门控 (Soft Gate)”：不彻底抹杀特征，保留 20% 的底噪，
        # 给予下游矩阵纠错空间，有效挽救因门控过于自信而丢失的短实体。
        base_ratio = 0.1 # 稍微收紧门控，尝试提升精确率
        if q_gate is not None: qw = qw * (q_gate.unsqueeze(-1) * (1 - base_ratio) + base_ratio)
        if k_gate is not None: kw = kw * (k_gate.unsqueeze(-1) * (1 - base_ratio) + base_ratio)

        if self.RoPE:
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim, device)
            pos_emb = pos_emb.to(qw.dtype) # 对齐半精度，防止隐式提升回 FP32
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1).reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1).reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        
        # 【深度修正】原版掩码仅能遮蔽 Tail 端的 PAD。当 mask_tril=False 时，会导致 Head 端预测出 PAD。
        # 使用真实的二维十字掩码矩阵：同时遮蔽无效的 Head 和 Tail
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(2) * attention_mask.unsqueeze(1).unsqueeze(3)
        pad_mask = pad_mask.to(logits.dtype)
        logits = logits * pad_mask - (1 - pad_mask) * 1e4
        
        if mask_tril:
            mask = torch.tril(torch.ones_like(logits), -1)
            mask = mask.to(logits.dtype)
            logits = logits - mask * 1e4
        logits = logits / (self.inner_dim ** 0.5)
        return logits.to(hidden_states.dtype)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # A. 基础编码
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs[0] 

        # B. 边界门控预测：获取每个 Token 作为 Start 和 End 的概率
        if self.use_boundary_attn:
            start_prob, end_prob = self.boundary_gate(last_hidden_state, attention_mask)
        else:
            start_prob, end_prob = None, None

        # C. 第一级：实体识别 (实体由 Start 连向 End)
        ent_logits = self.compute_gp_matrix(last_hidden_state, self.ent_dense, self.ent_type_size, attention_mask, mask_tril=True, q_gate=start_prob, k_gate=end_prob)
        ent_prob = torch.sigmoid(ent_logits) # [batch, ent_type, seq, seq]

        # E. 获取实体先验特征
        # 1. 基础头部先验：改回 max，将概率严格约束在 [0, 1]，防止长序列累加导致特征爆炸
        ent_head_prior, _ = torch.max(ent_prob, dim=-1) # [batch, ent_type, seq_head]
        ent_head_prior = ent_head_prior.transpose(1, 2) # [batch, seq_head, ent_type]
        base_prior_features = torch.relu(self.rel_prior_proj(ent_head_prior))
        
        # 计算尾部类型先验，同样使用 max 约束
        ent_tail_prior, _ = torch.max(ent_prob, dim=2) # [batch, ent_type, seq_tail]
        ent_tail_prior = ent_tail_prior.transpose(1, 2) # [batch, seq_tail, ent_type]
        tail_type_features = torch.relu(self.tail_type_proj(ent_tail_prior))
        
        # 【新增】将尾部类型特征注入到语义特征中
        tail_aware_state = last_hidden_state + tail_type_features

        # 2. 科学的 Span-to-Head 跨度注意力机制 (必须进行加权归一化)
        # 恢复 sum：保留实体类型维度的量子叠加态（解决一词多义重叠问题）
        span_attn = torch.sum(ent_prob, dim=1) # [batch, seq_head, seq_tail] 
        # 关键修正：防止 FP16 下 1e-6 下溢导致 0/0=NaN 的梯度异常
        span_attn_weights = span_attn / (torch.sum(span_attn, dim=-1, keepdim=True) + 1e-4)
        
        # 利用矩阵乘法，获取加权平均后的尾部上下文
        tail_context = torch.matmul(span_attn_weights, tail_aware_state) 
        tail_context_features = torch.relu(self.tail_context_proj(tail_context))

        # F. 动态门控提纯与特征融合
        if self.use_dynamic_gate:
            # 3. 融合：既包含头部的类型概率先验，又包含了完整的尾部跨度语义
            ent_prior_features = base_prior_features + tail_context_features
            gate_input = torch.cat([last_hidden_state, ent_prior_features], dim=-1)
            gate_value = self.gate_network(gate_input) 
            final_ent_features = gate_value * ent_prior_features 
        else:
            final_ent_features = ent_prior_features

        # 拼接 2 部分特征：原始语义 + 实体先验/门控
        rel_hidden_state = torch.cat([last_hidden_state, final_ent_features], dim=-1)

        # G. 第二级：关系识别 (Cascade 模型中关系是主语起点连向宾语起点，因此全是 Start)
        rel_logits = self.compute_gp_matrix(rel_hidden_state, self.rel_dense, self.rel_type_size, attention_mask, mask_tril=False, q_gate=start_prob, k_gate=start_prob)

        return ent_logits, rel_logits
    
class DataMakerJoint(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate_inputs(self, datas, max_seq_len, ent2id, rel2id, data_type="train", model_type="Cascade"):
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
            
            # 1. 初始化标签
            # 实体标签始终存在 [ent_type, seq, seq]
            ent_labels = np.zeros((len(ent2id), max_seq_len, max_seq_len), dtype=np.int8)
            
            if model_type == "GPLinker":
                # GPLinker: 首首链接和尾尾链接 [rel_type, seq, seq]
                hh_labels = np.zeros((len(rel2id), max_seq_len, max_seq_len), dtype=np.int8)
                tt_labels = np.zeros((len(rel2id), max_seq_len, max_seq_len), dtype=np.int8)
                rel_labels = None
            else:
                # Cascade: 级联关系标签
                rel_labels = np.zeros((len(rel2id), max_seq_len, max_seq_len), dtype=np.int8)
                hh_labels, tt_labels = None, None

            if data_type != "predict":
                char2token = {}
                for idx, (start_char, end_char) in enumerate(offset_mapping):
                    if start_char == end_char == 0:
                        continue
                    for char_idx in range(start_char, end_char):
                        char2token[char_idx] = idx

                # 2. 构建实体矩阵 (通用)
                for ent in sample.get("entity_list", []):
                    s_char, e_char = ent["start"], ent["end"]
                    if s_char in char2token and e_char in char2token:
                        s_tok, e_tok = char2token[s_char], char2token[e_char]
                        ent_type_id = ent2id.get(ent["type"], 0)
                        ent_labels[ent_type_id, s_tok, e_tok] = 1

                # 3. 构建关系矩阵 (区分模型类型)
                for spo in sample.get("spo_list", []):
                    # 获取首尾字符索引
                    s_s_c, s_e_c = spo["sub_start"], spo["sub_end"]
                    o_s_c, o_e_c = spo["obj_start"], spo["obj_end"]
                    
                    # 确保所有边界都能映射到 Token
                    if all(c in char2token for c in [s_s_c, s_e_c, o_s_c, o_e_c]):
                        s_s, s_e = char2token[s_s_c], char2token[s_e_c]
                        o_s, o_e = char2token[o_s_c], char2token[o_e_c]
                        rel_id = rel2id.get(spo["predicate"], 0)

                        if model_type == "GPLinker":
                            # GPLinker: 标注 HH (首首) 和 TT (尾尾)
                            hh_labels[rel_id, s_s, o_s] = 1
                            tt_labels[rel_id, s_e, o_e] = 1
                        else:
                            # Cascade: 标注原有级联标签 (主宾首部对齐)
                            rel_labels[rel_id, s_s, o_s] = 1
                            
                        # 注入基于 Chunk 的局部 Token 坐标，供验证集 decode 使用
                        spo["s_tok_start"] = s_s
                        spo["o_tok_start"] = o_s

            # 4. 转换为 Tensor 附加到 Batch 中
            input_ids = torch.tensor(inputs["input_ids"]).long()
            attention_mask = torch.tensor(inputs["attention_mask"]).long()
            token_type_ids = torch.tensor(inputs["token_type_ids"]).long()
            # 改用 int8 疏通数据加载瓶颈，到 GPU 后会无损转换
            ent_labels_ts = torch.tensor(ent_labels, dtype=torch.int8)
            
            if model_type == "GPLinker":
                hh_labels_ts = torch.tensor(hh_labels, dtype=torch.int8)
                tt_labels_ts = torch.tensor(tt_labels, dtype=torch.int8)
                all_inputs.append((sample, input_ids, attention_mask, token_type_ids, ent_labels_ts, hh_labels_ts, tt_labels_ts))
            else:
                rel_labels_ts = torch.tensor(rel_labels, dtype=torch.int8)
                all_inputs.append((sample, input_ids, attention_mask, token_type_ids, ent_labels_ts, rel_labels_ts))
            
        return all_inputs

    def generate_batch(self, batch_data, max_seq_len, ent2id, rel2id, data_type="train", model_type="Cascade"):
        # 增加 model_type 参数并透传给 generate_inputs
        batch_inputs = self.generate_inputs(batch_data, max_seq_len, ent2id, rel2id, data_type, model_type)
        
        sample_list = [item[0] for item in batch_inputs]
        batch_input_ids = torch.stack([item[1] for item in batch_inputs], dim=0)
        batch_attention_mask = torch.stack([item[2] for item in batch_inputs], dim=0)
        batch_token_type_ids = torch.stack([item[3] for item in batch_inputs], dim=0)
        batch_ent_labels = torch.stack([item[4] for item in batch_inputs], dim=0)
        
        if model_type == "GPLinker":
            # 返回 7 个值 (GPLinker 模式)
            batch_hh_labels = torch.stack([item[5] for item in batch_inputs], dim=0)
            batch_tt_labels = torch.stack([item[6] for item in batch_inputs], dim=0)
            return sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_ent_labels, batch_hh_labels, batch_tt_labels
        else:
            # 返回 6 个值 (Cascade 模式)
            batch_rel_labels = torch.stack([item[5] for item in batch_inputs], dim=0)
            return sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_ent_labels, batch_rel_labels
        

# models/JointGlobalPointer.py

# --- 在文件末尾添加 GPLinker 模型 ---
class GPLinker(nn.Module):
    def __init__(self, encoder, ent_type_size, rel_type_size, inner_dim, RoPE=True, use_boundary_attn=True):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = encoder.config.hidden_size
        self.ent_type_size = ent_type_size
        self.rel_type_size = rel_type_size
        self.inner_dim = inner_dim
        self.RoPE = RoPE
        self.use_boundary_attn = use_boundary_attn

        # 1. 实体识别矩阵 (Entity)
        self.ent_dense = nn.Linear(self.hidden_size, ent_type_size * inner_dim * 2)
        
        # 2. 关系首-首链接矩阵 (Head-Head)
        self.hh_dense = nn.Linear(self.hidden_size, rel_type_size * inner_dim * 2)
        
        # 3. 关系尾-尾链接矩阵 (Tail-Tail)
        self.tt_dense = nn.Linear(self.hidden_size, rel_type_size * inner_dim * 2)

        # 边界门控
        self.boundary_gate = DecoupledBoundaryGate(self.hidden_size)

    # 复用之前定义的 compute_gp_matrix 逻辑（由于在同一个文件，可以直接调用或拷贝）
    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        """为 RoPE 生成旋转位置嵌入"""
        position_ids = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float, device=device)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape(seq_len, output_dim)
        return embeddings.unsqueeze(0).expand(batch_size, seq_len, output_dim)
    def compute_gp_matrix(self, hidden_states, dense_layer, type_size, attention_mask, mask_tril=True, q_gate=None, k_gate=None):
        batch_size, seq_len = hidden_states.shape[:2]
        device = hidden_states.device
        
        outputs = dense_layer(hidden_states)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        # 特征缩放
        # 同样在 GPLinker 中应用软门控
        base_ratio = 0.2
        if q_gate is not None: qw = qw * (q_gate.unsqueeze(-1) * (1 - base_ratio) + base_ratio)
        if k_gate is not None: kw = kw * (k_gate.unsqueeze(-1) * (1 - base_ratio) + base_ratio)

        if self.RoPE:
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim, device)
            pos_emb = pos_emb.to(qw.dtype)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1).reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1).reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        
        # 同样修正 GPLinker 中的 PAD 掩码逻辑
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(2) * attention_mask.unsqueeze(1).unsqueeze(3)
        pad_mask = pad_mask.to(logits.dtype)
        logits = logits * pad_mask - (1 - pad_mask) * 1e4
        
        if mask_tril:
            mask = torch.tril(torch.ones_like(logits), -1)
            mask = mask.to(logits.dtype)
            logits = logits - mask * 1e4
        logits = logits / (self.inner_dim ** 0.5)
        return logits.to(hidden_states.dtype)

    def forward(self, input_ids, attention_mask, token_type_ids):
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs[0] 

        if self.use_boundary_attn:
            start_prob, end_prob = self.boundary_gate(last_hidden_state, attention_mask)
        else:
            start_prob, end_prob = None, None

        # 并行计算三个矩阵
        # 1. 实体：Start 连向 End
        ent_logits = self.compute_gp_matrix(last_hidden_state, self.ent_dense, self.ent_type_size, attention_mask, mask_tril=True, q_gate=start_prob, k_gate=end_prob)
        # 2. 关系首首 (Head-Head)：主语 Start 连向宾语 Start，因此全是 Start 门控
        hh_logits = self.compute_gp_matrix(last_hidden_state, self.hh_dense, self.rel_type_size, attention_mask, mask_tril=False, q_gate=start_prob, k_gate=start_prob)
        # 3. 关系尾尾 (Tail-Tail)：主语 End 连向宾语 End，因此全是 End 门控
        tt_logits = self.compute_gp_matrix(last_hidden_state, self.tt_dense, self.rel_type_size, attention_mask, mask_tril=False, q_gate=end_prob, k_gate=end_prob)

        return ent_logits, hh_logits, tt_logits

# --- 添加 GPLinker 专用损失函数 ---
class GPLinkerLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 同样可以集成自动损失平衡参数 s1, s2, s3
        self.loss_scales = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))

    def multilabel_categorical_crossentropy(self, y_pred, y_true):
        # 复用你代码中原有的多标签交叉熵逻辑
        batch_size, type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * type_size, -1)
        y_pred = y_pred.reshape(batch_size * type_size, -1)

        y_true = y_true.to(y_pred.dtype)
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e4
        y_pred_pos = y_pred - (1 - y_true) * 1e4
        
        y_pred_neg = y_pred_neg.to(y_pred.dtype)
        y_pred_pos = y_pred_pos.to(y_pred.dtype)
        
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()

    def forward(self, ent_logits, ent_labels, hh_logits, hh_labels, tt_logits, tt_labels):
        l_ent = self.multilabel_categorical_crossentropy(ent_logits, ent_labels)
        l_hh = self.multilabel_categorical_crossentropy(hh_logits, hh_labels)
        l_tt = self.multilabel_categorical_crossentropy(tt_logits, tt_labels)

        # 自动权重计算
        w1, w2, w3 = torch.exp(-self.loss_scales[0]), torch.exp(-self.loss_scales[1]), torch.exp(-self.loss_scales[2])
        
        total_loss = (w1 * l_ent + self.loss_scales[0]) + \
                     (w2 * l_hh + self.loss_scales[1]) + \
                     (w3 * l_tt + self.loss_scales[2])
        
        return total_loss, l_ent, l_hh, l_tt