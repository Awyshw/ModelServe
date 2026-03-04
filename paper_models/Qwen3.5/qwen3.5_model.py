import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------- 设备配置：GPU > MPS > CPU ---------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# -------------- 门控混合专家(MoE) ----------------
class MoEExpert(nn.Module):
    """单个专家网络(FFN层)"""
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()  # GELU激活函数，主要用途：避免梯度爆炸
    
    def forward(self, x):
        # FFN计算：x -> 升维 -> 激活 -> 降维 -> dropout
        return self.dropout(self.w2(self.act(self.w1(x))))
    
class MoELayer(nn.Module):
    """"稀疏 MoE 层：门控选择专家， 仅激活部分专家"""
    def __init__(self, d_model, d_ffn, n_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        # 初始化所有专家(总参数量 = n_experts * (d_model * d_ffn + d_ffn * d_model))
        self.experts = nn.ModuleList(
            [
                MoEExpert(d_model, d_ffn, dropout)
                for _ in range(n_experts)
            ]
        )

        # 门控网络：输出每个 token 对每个专家的权重
        self.router = nn.Linear(d_model, n_experts, bias=False)
    
    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        return: [batch_size, seq_len, d_model] (仅激活部分top_k专家)
        """
        batch_size, seq_len, d_model = x.shape

        # 1. 计算每个 token 对每个专家的权重
        router_logits = self.router(x)  # [batch_size, seq_len, n_experts]
        # 2. 选择 top_k 专家，并计算其权重
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # [batch_size, seq_len, top_k]
        topk_weights = F.softmax(top_k_logits, dim=-1)  # [batch_size, seq_len, top_k]
        
        # 3. 初始化输出
        output = torch.zeros_like(x, device=device)
        # 4. 遍历每个选中的专家，计算并加权输出(仅激活top_k专家)
        for k in range(self.top_k):
            # 获取第 k 个专家的索引和权重
            expert_indices = top_k_indices[..., k]  # [batch_size, seq_len]
            expert_weights = topk_weights[..., k:k+1]  # [batch_size, seq_len, 1]

            # 遍历所有专家，处理分配给该专家的 token
            for expert_idx in range(self.n_experts):
                # 找到当前 token 是否选择该专家
                mask = (expert_indices == expert_idx)  # [batch_size, seq_len]
                if mask.sum() == 0:
                    continue  # 无 token 分配给该专家，跳过（不激活）

                # 提取该专家专处理的 token
                x_masked = x[mask]  # [n_masked_tokens, d_model]
                # 专家前向计算
                expert_output = self.experts[expert_idx](x_masked)  # [n_masked_tokens, d_model]
                # 加权求和
                output[mask] += expert_output * expert_weights[mask]
        return output

# --------------- Gated DeltaNet + 门控注意力 -------------------
class GatedDeltaNet(nn.Module):
    """混合注意力：密集注意力 + 线性 Delta 注意力 + 门控融合"""
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        # 密集注意力参数(Scaled Dot-Product Attention)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # 线性 Delta 注意力参数
        self.w_q_delta = nn.Linear(d_model, d_model, bias=False)
        self.w_k_delta = nn.Linear(d_model, d_model, bias=False)
        self.w_v_delta = nn.Linear(d_model, d_model, bias=False)
        self.w_o_delta = nn.Linear(d_model, d_model, bias=False)

        # 门控权重：平衡密集注意力和 Delta 注意力
        self.gate = nn.Linear(d_model, 2, bias=False)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """传统密集注意力"""
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, device=device))
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        return torch.matmul(attn_probs, v)
    
    def linear_delta_attention(self, q, k, v):
        """DeltaNet 线性注意力：O(n)复杂度，适配长上下文"""
        # 简化版 Delta 注意力：基于核函数的线性计算
        batch_size, n_head, seq_len, d_k = q.shape
        #  线性注意力核心（O(n)复杂度)
        # 1. 核函数激活
        q_linear = F.relu(q)
        k_linear = F.relu(k)

        # 2. 计算全局上下文
        context = torch.matmul(k.transpose(-2, -1), v)

        # 3. 线性注意力输出
        attn_output = torch.matmul(q_linear, context)

        # 4. 归一化（可选，保证数值稳定)
        z = torch.sum(q_linear * torch.sum(k_linear, dim=-2, keepdim=True), dim=-1, keepdim=True) + 1e-6
        attn_output = attn_output / z

        return attn_output
        
        return attn_output
    
    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, d_model]
        return: 融合后的注意力输出([batch_size, seq_len, d_model])
        """
        batch_size, seq_len, d_model = x.shape

        # 1. 拆分多头
        def split_heads(x):
            return x.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        
        # 2. 计算密集注意力
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        q_head, k_head, v_head = split_heads(q), split_heads(k), split_heads(v)
        dense_attn = self.scaled_dot_product_attention(q_head, k_head, v_head, mask)
        dense_attn = dense_attn.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        dense_attn = self.w_o(dense_attn)

        # 3. 计算 Delta 线性注意力
        q_delta, k_delta, v_delta = self.w_q_delta(x), self.w_k_delta(x), self.w_v_delta(x)
        q_delta_head, k_delta_head, v_delta_head = split_heads(q_delta), split_heads(k_delta), split_heads(v_delta)
        delta_attn = self.linear_delta_attention(q_delta_head, k_delta_head, v_delta_head)
        delta_attn = delta_attn.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        delta_attn = self.w_o_delta(delta_attn)

        # 4. 门控融合：动态平衡两种注意力
        gate_weights = F.softmax(self.gate(x), dim=-1)  # [batch_size, seq_len, 2]
        output = gate_weights[..., 0:1] * dense_attn + gate_weights[..., 1:2] * delta_attn

        return self.dropout(output)
    
class Qwen35Layer(nn.Module):
    """Qwen3.5 层：混合注意力 + MoE FFN"""
    def __init__(self, d_model, n_head, d_ff, n_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = GatedDeltaNet(d_model, n_head, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe_ffn = MoELayer(d_model, d_ff, n_experts, top_k, dropout)

    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.moe_ffn(self.norm2(x))
        return x
    
# --------------- 测试代码 -------------------
if __name__ == '__main__':
    # 超参数
    d_model = 512
    n_head = 8
    d_ff = 2048
    n_experts = 8
    top_k = 2

    # 初始化模型
    qwen_layer = Qwen35Layer(d_model, n_head, d_ff, n_experts, top_k).to(device)

    # 输入数据
    x = torch.randn(2, 32, d_model).to(device)
    mask = torch.ones(2, 32, 32).to(device)

    # 前向传播
    output = qwen_layer(x, mask)
    print(f"输入形状：{x.shape}")
    print(f"输出形状：{output.shape}")
    print(f"单个推理激活专家数: {top_k}/{n_experts} (激活率 {top_k/n_experts*100:.1f}%)")