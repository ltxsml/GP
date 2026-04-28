# import torch
# import torch.nn as nn

# class GPLinkerLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 如果你想在 GPLinker 实验中也使用自动平衡，保留这个
#         self.loss_scales = nn.Parameter(torch.tensor([0.0, 0.0, 0.0])) 

#     def cross_entropy(self, y_pred, y_true):
#         # 保持你原有的多标签交叉熵逻辑
#         pass

#     def forward(self, ent_logits, ent_labels, hh_logits, hh_labels, tt_logits, tt_labels):
#         l_ent = self.cross_entropy(ent_logits, ent_labels)
#         l_hh = self.cross_entropy(hh_logits, hh_labels)
#         l_tt = self.cross_entropy(tt_logits, tt_labels)

#         # 自动平衡：w = exp(-s)
#         w1, w2, w3 = torch.exp(-self.loss_scales[0]), torch.exp(-self.loss_scales[1]), torch.exp(-self.loss_scales[2])
        
#         total_loss = (w1 * l_ent + self.loss_scales[0]) + \
#                      (w2 * l_hh + self.loss_scales[1]) + \
#                      (w3 * l_tt + self.loss_scales[2])
        
#         return total_loss, l_ent, l_hh, l_tt

# class GPLinker(nn.Module):
#     def __init__(self, encoder, ent_type_size, rel_type_size, inner_dim=64, RoPE=True):
#         super().__init__()
#         self.encoder = encoder
#         self.hidden_size = encoder.config.hidden_size
#         self.ent_type_size = ent_type_size
#         self.rel_type_size = rel_type_size
#         self.inner_dim = inner_dim
#         self.RoPE = RoPE

#         # 1. 实体识别矩阵 (Entity)
#         self.ent_dense = nn.Linear(self.hidden_size, ent_type_size * inner_dim * 2)
        
#         # 2. 关系首-首链接矩阵 (Head-Head)
#         self.hh_dense = nn.Linear(self.hidden_size, rel_type_size * inner_dim * 2)
        
#         # 3. 关系尾-尾链接矩阵 (Tail-Tail)
#         self.tt_dense = nn.Linear(self.hidden_size, rel_type_size * inner_dim * 2)

#     # 复用你之前的 sinusoidal_position_embedding 和 compute_gp_matrix 逻辑
#     # 为了简洁，这里假设你已经将这些基础函数提取到了工具类中，或者直接拷贝过来
#     def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
#         """为 RoPE 生成旋转位置嵌入"""
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
#         context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
#         last_hidden_state = context_outputs[0] 

#         # 并行计算三个矩阵
#         ent_logits = self.compute_gp_matrix(last_hidden_state, self.ent_dense, self.ent_type_size, attention_mask, mask_tril=True)
#         hh_logits = self.compute_gp_matrix(last_hidden_state, self.hh_dense, self.rel_type_size, attention_mask, mask_tril=False)
#         tt_logits = self.compute_gp_matrix(last_hidden_state, self.tt_dense, self.rel_type_size, attention_mask, mask_tril=False)

#         return ent_logits, hh_logits, tt_logits
    


    