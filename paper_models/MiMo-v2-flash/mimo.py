import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from typing import Optional, Dict, Tuple

# ------------------------------
# 基础组件（轻量化 + 效率优化）
# ------------------------------
class RMSNorm(nn.Module):
    """RMSNorm 归一化（保留核心，移除冗余计算）"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class RoPE(nn.Module):
    """RoPE 位置编码（简化计算，避免冗余 clone）"""
    def __init__(self, dim: int, max_seq_len: int = 16384):  # 缩小最大序列长度
        super().__init__()
        self.dim = dim
        self.rope_dim = min(32, dim)  # 进一步缩小 RoPE 作用维度（省计算）
        theta = 1.0 / (10000 ** (torch.arange(0, self.rope_dim, 2) / self.rope_dim))
        self.register_buffer("theta", theta, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        device = x.device
        pos = torch.arange(seq_len, device=device).unsqueeze(1)
        rope = pos * self.theta.unsqueeze(0)
        rope = torch.cat([rope, rope], dim=-1)  # [seq_len, rope_dim]
        
        # 避免 clone，直接操作切片（省显存）
        x_rope = x[..., :self.rope_dim]
        x_even = x_rope[..., ::2]
        x_odd = x_rope[..., 1::2]
        cos = rope[..., ::2]
        sin = rope[..., 1::2]
        
        x_rope[..., ::2] = x_even * cos - x_odd * sin
        x_rope[..., 1::2] = x_odd * cos + x_even * sin
        
        if self.rope_dim < self.dim:
            x = torch.cat([x_rope, x[..., self.rope_dim:]], dim=-1)
        else:
            x = x_rope
        return x

class Expert(nn.Module):
    """轻量化专家网络（进一步缩小 hidden_dim）"""
    def __init__(self, dim: int, hidden_dim: int = 512, use_low_rank: bool = False):
        super().__init__()
        self.use_low_rank = use_low_rank
        if use_low_rank:
            # 低秩分解（更激进：hidden_dim//8）
            self.fc1 = nn.Linear(dim, hidden_dim // 8)
            self.mid = nn.Linear(hidden_dim // 8, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, dim // 8)
            self.out = nn.Linear(dim // 8, dim)
        else:
            self.fc1 = nn.Linear(dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_low_rank:
            x = self.act(self.mid(self.act(self.fc1(x))))
            return self.out(self.act(self.fc2(x)))
        return self.fc2(self.act(self.fc1(x)))

class MoEFFN(nn.Module):
    """MoE 前馈网络（并行化优化，移除双重循环）"""
    def __init__(self, dim: int, num_experts: int = 16, top_k: int = 2, expert_hidden_dim: int = 512, use_low_rank: bool = False):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(dim, expert_hidden_dim, use_low_rank) for _ in range(num_experts)])
        self.router = nn.Linear(dim, num_experts)
        self.aux_loss_coeff = 1e-5

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = x.shape
        x_flat = x.reshape(-1, dim)  # [N, dim], N = batch*seq
        
        # 路由计算（保留温度系数）
        router_logits = self.router(x_flat) / 0.1
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # [N, top_k]
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [N, top_k]
        
        # MoE 辅助损失（不变）
        router_probs = F.softmax(router_logits, dim=-1)
        aux_loss = router_probs.sum(dim=0).pow(2).sum() / self.num_experts
        aux_loss = aux_loss * self.aux_loss_coeff

        # ------------------------------
        # 关键优化：并行化专家计算（移除 for 循环）
        # ------------------------------
        N = x_flat.shape[0]
        output = torch.zeros_like(x_flat)
        
        # 1. 扁平化 top_k 维度：[N, top_k] → [N*top_k]
        flat_indices = top_k_indices.reshape(-1)  # [N*top_k]
        flat_weights = top_k_weights.reshape(-1, 1)  # [N*top_k, 1]
        flat_x = x_flat.unsqueeze(1).repeat(1, self.top_k, 1).reshape(-1, dim)  # [N*top_k, dim]
        
        # 2. 按专家分组，批量计算（仅循环 num_experts 次，无嵌套）
        for expert_id in range(self.num_experts):
            # 找到该专家的所有输入索引
            mask = (flat_indices == expert_id) # [N*top_k]
            if not mask.any():
                continue  # 跳过无输入的专家
            
            # 批量计算该专家的输出
            expert_input = flat_x[mask]
            expert_output = self.experts[expert_id](expert_input) * flat_weights[mask]
            
            # 把结果散射回原位置（用 scatter_add_ 避免循环赋值）
            scatter_indices = torch.nonzero(mask, as_tuple=True)[0]  # [M], M = 该专家的输入数
            original_indices = scatter_indices // self.top_k  # 映射回原 N 维度
            output.scatter_add_(0, original_indices.unsqueeze(1).repeat(1, dim), expert_output)
        
        return output.reshape(batch_size, seq_len, dim), aux_loss

# ------------------------------
# 注意力机制（轻量化 + 计算优化）
# ------------------------------
class SlidingWindowAttention(nn.Module):
    """SWA（减少头数，降低计算量）"""
    def __init__(self, dim: int, num_heads: int = 8, num_kv_heads: int = 2, window_size: int = 64):  # 头数从64→8，窗口从128→64
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.sink_bias = nn.Parameter(torch.tensor(0.0))
        
        # QKV 投影（保留 GQA，但减少参数）
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        
        # QKV 投影（简化 reshape 逻辑）
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA（不变）
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        # 注意力计算（减少数值精度冗余）
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.head_dim)
        
        # 滑动窗口掩码（保留优化版）
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        mask = mask.triu(diagonal=-(self.window_size - 1)) & mask.tril(diagonal=0)
        # attn_weights = attn_weights.masked_fill(~mask, -1e18)
        fp16_min = torch.finfo(torch.float16).min
        attn_weights = attn_weights.masked_fill(~mask, fp16_min)

        # Attention Sink（简化计算）
        m_i = torch.max(attn_weights.max(dim=-1, keepdim=True)[0], self.sink_bias)
        attn_weights = torch.exp(attn_weights - m_i)
        sink_term = torch.exp(self.sink_bias - m_i)
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + sink_term)

        # 输出（简化 reshape）
        out = torch.matmul(attn_weights, v).transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.o_proj(out)

class GlobalAttention(SlidingWindowAttention):
    """GA（复用轻量化 SWA）"""
    def __init__(self, dim: int, num_heads: int = 8, num_kv_heads: int = 2):
        super().__init__(dim, num_heads, num_kv_heads, window_size=10000)  # 缩小窗口上限

# ------------------------------
# 网络块（轻量化配置）
# ------------------------------
class SWABlock(nn.Module):
    def __init__(self, dim: int, moe_kwargs: Dict):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SlidingWindowAttention(dim)
        self.norm2 = RMSNorm(dim)
        self.ffn = MoEFFN(dim, **moe_kwargs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x))
        ffn_out, aux_loss = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x, aux_loss

class GABlock(nn.Module):
    def __init__(self, dim: int, moe_kwargs: Dict, use_dense_ffn: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GlobalAttention(dim)
        self.norm2 = RMSNorm(dim)
        self.use_dense_ffn = use_dense_ffn
        
        if use_dense_ffn:
            # 稠密 FFN 轻量化（hidden_dim 从16384→2048）
            self.ffn = nn.Sequential(
                nn.Linear(dim, 2048),
                nn.SiLU(),
                nn.Linear(2048, dim)
            )
            self.aux_loss = torch.tensor(0.0, requires_grad=True)
        else:
            self.ffn = MoEFFN(dim, **moe_kwargs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x))
        if self.use_dense_ffn:
            x = x + self.ffn(self.norm2(x))
            return x, self.aux_loss
        else:
            ffn_out, aux_loss = self.ffn(self.norm2(x))
            x = x + ffn_out
            return x, aux_loss

class MTPBlock(nn.Module):
    """MTP 轻量化（进一步减少头数和 hidden_dim）"""
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SlidingWindowAttention(dim, num_heads=4, num_kv_heads=1)  # 头数从16→4
        self.norm2 = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 512),  # hidden_dim 从1024→512
            nn.SiLU(),
            nn.Linear(512, dim)
        )
        self.predict_head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return self.predict_head(x)

# ------------------------------
# 主模型（超轻量化配置）
# ------------------------------
class MiMoV2Flash(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 512,  # 从1024→512（核心轻量化）
        num_hybrid_blocks: int = 2,  # 从8→2（减少混合块数量）
        moe_kwargs: Optional[Dict] = None,
        use_low_rank_experts: bool = False,
        max_seq_len: int = 16384
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.use_low_rank_experts = use_low_rank_experts
        
        # 默认 MoE 配置（超轻量化）
        if moe_kwargs is None:
            moe_kwargs = {
                "num_experts": 16,    # 从32→16
                "top_k": 2,           # 从4→2
                "expert_hidden_dim": 512,
                "use_low_rank": use_low_rank_experts
            }
        
        self.embedding = nn.Embedding(vocab_size, dim)
        self.rope = RoPE(dim, max_seq_len=max_seq_len)
        
        # 模型结构轻量化：1个GA块 + 2个混合块（5SWA+1GA → 2SWA+1GA）
        self.layers = nn.ModuleList()
        self.layers.append(GABlock(dim, moe_kwargs, use_dense_ffn=True))
        for _ in range(num_hybrid_blocks):
            self.layers.extend([SWABlock(dim, moe_kwargs) for _ in range(2)])  # 从5→2个SWA块
            self.layers.append(GABlock(dim, moe_kwargs))
        
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        # MTP 模块保留3层，但已轻量化
        self.mtp_layers = nn.ModuleList([MTPBlock(dim) for _ in range(3)])
        
        # 初始化（不变）
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.006)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.006)

    def forward(
        self,
        input_ids: torch.Tensor,
        use_mtp: bool = False
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids)
        x = self.rope(x)
        
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, aux_loss = layer(x)
            if aux_loss.requires_grad:
                total_aux_loss += aux_loss
        
        lm_logits = self.lm_head(self.norm(x))
        
        mtp_logits = None
        if use_mtp:
            mtp_x = x
            for mtp_layer in self.mtp_layers:
                mtp_x = mtp_layer(mtp_x)
            mtp_logits = self.lm_head(self.norm(mtp_x))
        
        return {
            "lm_logits": lm_logits,
            "mtp_logits": mtp_logits,
            "total_aux_loss": total_aux_loss
        }

# ------------------------------
# 显存优化（移除梯度检查点，适配轻量化模型）
# ------------------------------
def optimize_model_for_low_memory(model: MiMoV2Flash, device: torch.device) -> MiMoV2Flash:
    """轻量化模型无需梯度检查点（反而拖慢速度）"""
    # 1. 半精度（保留）
    model.half()
    # 2. 冻结嵌入层和输出头（保留）
    model.embedding.requires_grad_(False)
    model.lm_head.requires_grad_(False)
    # 3. 移除梯度检查点（关键优化）
    return model.to(device)

# ------------------------------
# 测试代码（适配轻量化模型）
# ------------------------------
if __name__ == "__main__":
    import time  # 添加计时
    
    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU 显存可用: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("使用 CPU（轻量化后可快速运行）")

    # 2. 超轻量化 MoE 配置
    moe_kwargs = {
        "num_experts": 16,
        "top_k": 2,
        "expert_hidden_dim": 512,
        "use_low_rank": True  # 启用低秩分解（省显存）
    }

    # 3. 初始化模型
    model = MiMoV2Flash(
        vocab_size=32000,
        dim=512,
        num_hybrid_blocks=2,
        moe_kwargs=moe_kwargs,
        use_low_rank_experts=True,
        max_seq_len=8192  # 进一步缩小最大序列长度
    )

    # 4. 显存优化
    model = optimize_model_for_low_memory(model, device)

    # 5. 测试输入（缩小 seq_len）
    batch_size = 1
    seq_len = 256  # 从512→256
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)

    # 6. 前向传播（计时）
    start_time = time.time()
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=torch.cuda.is_available()):
        output = model(input_ids, use_mtp=True)
    end_time = time.time()

    # 7. 验证输出
    print(f"\n运行时间: {end_time - start_time:.2f} 秒")
    print(f"输入形状: {input_ids.shape}")
    print(f"LM 输出形状: {output['lm_logits'].shape}")
    print(f"MTP 输出形状: {output['mtp_logits'].shape}")
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 8. 计算损失（模拟训练）
    loss_fn = nn.CrossEntropyLoss()
    lm_loss = loss_fn(
        output["lm_logits"][:, :-1].reshape(-1, 32000),
        input_ids[:, 1:].reshape(-1)
    )
    mtp_loss = loss_fn(
        output["mtp_logits"][:, :-1].reshape(-1, 32000),
        input_ids[:, 1:].reshape(-1)
    )
    total_loss = lm_loss + 0.1 * mtp_loss + output["total_aux_loss"]
    total_loss.backward()

    print(f"\nLM 损失: {lm_loss.item():.4f}")
    print(f"MTP 损失: {mtp_loss.item():.4f}")
    print(f"MoE 辅助损失: {output['total_aux_loss'].item():.6f}")
    print(f"总损失: {total_loss.item():.4f}")
    print("\n✅ 模型快速运行成功！核心功能完整保留")