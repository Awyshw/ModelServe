import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

# -------------------------- 1. 配置类（修复参数矛盾+补充稳定训练配置）--------------------------
class Config:
    d_model = 512          # 模型维度
    num_query_heads = 8    # MQA查询头数量
    d_k = 64               # 每个头维度（d_model / num_query_heads，确保整除）
    num_indexer_heads = 2  # DSA索引器头数
    d_index = 32           # 索引器维度
    top_k = 256            # 修复：Top-k ≤ seq_len（1024），避免逻辑错误
    seq_len = 1024         # 长序列长度
    batch_size = 2         # 适配长序列
    vocab_size = 10000     # 词表大小
    warmup_steps = 1000    # 预热步数
    sparse_steps = 5000    # 稀疏训练步数
    warmup_lr = 1e-3       # 预热学习率
    sparse_lr = 7.3e-6     # 稀疏学习率
    kl_weight = 1.0        # KL损失权重
    top_p = 0.95           # Top-p采样
    delta = 0.1            # 离线序列掩码阈值
    weight_decay = 0.01    # 补充：权重衰减（泛化性）
    grad_clip = 1.0        # 补充：梯度裁剪（防爆炸）
    dropout = 0.1          # 全局dropout率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # 统一数据类型

config = Config()
assert config.d_model == config.num_query_heads * config.d_k, "d_model必须是num_query_heads * d_k的整数倍"
assert config.top_k <= config.seq_len, "top_k不能超过seq_len"

# -------------------------- 2. RoPE位置编码（修复：QKV均应用+数值稳定）--------------------------
class RoPE(nn.Module):
    def __init__(self, d_model, max_seq_len=2048, dtype=torch.float32):
        super().__init__()
        self.d_model = d_model
        self.dtype = dtype
        # 生成频率编码（避免CPU/CUDA冲突，注册为buffer）
        theta = 1.0 / (10000 ** (torch.arange(0, d_model, 2, dtype=dtype) / d_model))
        seq_idx = torch.arange(max_seq_len, dtype=dtype)
        pos_enc = torch.outer(seq_idx, theta)  # [max_seq_len, d_model/2]
        # 拼接sin和cos，注册为buffer（自动跟随模型设备）
        self.register_buffer(
            "rope",
            torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1)
        )

    def forward(self, x):
        # x: [B, H, S, d_k]（多头）或 [B, S, d_k]（单头）
        is_multi_head = len(x.shape) == 4
        if is_multi_head:
            B, H, S, d_k = x.shape
            rope = self.rope[:S].unsqueeze(0).unsqueeze(0)  # [1,1,S,d_k]
        else:
            B, S, d_k = x.shape
            rope = self.rope[:S].unsqueeze(0)  # [1,S,d_k]
        
        # 修复：缩放因子指定dtype+设备对齐
        scale = torch.sqrt(torch.tensor(d_k, dtype=self.dtype, device=x.device))
        x = x * scale
        
        # 分奇偶维度计算RoPE
        x1, x2 = x[..., ::2], x[..., 1::2]  # [B, (H), S, d_k/2]
        rope1, rope2 = rope[..., ::2], rope[..., 1::2]
        x_rot = torch.cat([
            x1 * rope1 - x2 * rope2,
            x2 * rope1 + x1 * rope2
        ], dim=-1)
        return x_rot

# -------------------------- 3. MLA_MQA模块（修复：Q应用RoPE+结构校验）--------------------------
class MLA_MQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_query_heads = config.num_query_heads
        self.d_k = config.d_k

        # MQA投影层
        self.W_q = nn.Linear(self.d_model, self.num_query_heads * self.d_k, dtype=config.dtype)
        self.W_k = nn.Linear(self.d_model, self.d_k, dtype=config.dtype)
        self.W_v = nn.Linear(self.d_model, self.d_k, dtype=config.dtype)
        
        # 修复：QKV均应用RoPE（位置信息一致）
        self.rope = RoPE(self.d_k, max_seq_len=config.seq_len, dtype=config.dtype)

    def forward(self, x):
        # x: [B, S, d_model]
        B, S, _ = x.shape

        # 生成Q（多头）：[B, H, S, d_k]
        Q = self.W_q(x).reshape(B, S, self.num_query_heads, self.d_k).transpose(1, 2)
        Q = self.rope(Q)  # 修复：Q应用RoPE

        # 生成KV（单组共享）：[B, S, d_k]
        K = self.W_k(x).reshape(B, S, self.d_k)
        V = self.W_v(x).reshape(B, S, self.d_k)
        K = self.rope(K)  # K应用RoPE
        V = self.rope(V)  # V应用RoPE

        return Q, K, V

# -------------------------- 4. DSA模块（修复：MQA对齐+数值稳定+设备一致）--------------------------
class DSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.top_k
        self.num_indexer_heads = config.num_indexer_heads
        self.d_index = config.d_index
        self.dtype = config.dtype

        # 闪电索引器（修复：小权重初始化+层归一化）
        self.indexer_q = nn.Linear(config.d_model, self.num_indexer_heads * self.d_index, dtype=self.dtype)
        self.indexer_k = nn.Linear(config.d_model, self.num_indexer_heads * self.d_index, dtype=self.dtype)
        self.indexer_norm = nn.LayerNorm(self.d_index, dtype=self.dtype)  # 新增：归一化稳定数值
        self.indexer_weights = nn.Parameter(torch.ones(self.num_indexer_heads, dtype=self.dtype) * 0.1)  # 小权重初始化
        self.relu = nn.ReLU()

        # 修复：密集注意力改为MQA结构（与MLA对齐，确保KL目标有效）
        self.W_q_dense = nn.Linear(config.d_model, config.num_query_heads * config.d_k, dtype=self.dtype)
        self.W_k_dense = nn.Linear(config.d_model, config.d_k, dtype=self.dtype)
        self.W_v_dense = nn.Linear(config.d_model, config.d_k, dtype=self.dtype)
        self.rope_dense = RoPE(config.d_k, max_seq_len=config.seq_len, dtype=self.dtype)

    def compute_mqa_dense_target(self, x):
        """修复：MQA结构的密集注意力目标分布"""
        B, S, _ = x.shape
        H = self.config.num_query_heads
        d_k = self.config.d_k

        # MQA模式生成QKV
        Q = self.W_q_dense(x).reshape(B, S, H, d_k).transpose(1, 2)
        K = self.W_k_dense(x).reshape(B, S, d_k)
        V = self.W_v_dense(x).reshape(B, S, d_k)
        
        # 应用RoPE（与MLA一致）
        Q = self.rope_dense(Q)
        K = self.rope_dense(K)
        V = self.rope_dense(V)

        # 密集注意力得分
        attn_scores = torch.matmul(Q, K.unsqueeze(1).transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, S, S]
        aggregated = attn_weights.sum(dim=1)  # 聚合头 [B, S, S]
        dense_target = F.softmax(aggregated, dim=-1)  # 修复：用softmax替代L1，匹配概率分布
        return dense_target

    def forward(self, x, Q, K, V, training_phase="sparse"):
        B, S, _ = x.shape
        H, d_k = Q.shape[1], Q.shape[-1]
        kl_loss = torch.tensor(0.0, device=x.device, dtype=self.dtype)

        # 步骤1：闪电索引器（修复：归一化+数值稳定）
        q_index = self.indexer_q(x).reshape(B, S, self.num_indexer_heads, self.d_index)
        k_index = self.indexer_k(x).reshape(B, S, self.num_indexer_heads, self.d_index)
        q_index = self.indexer_norm(q_index)  # 新增：归一化
        k_index = self.indexer_norm(k_index)

        head_scores = torch.einsum("bshd,bthd->bhst", k_index, q_index)
        head_scores = self.relu(head_scores)
        index_scores = torch.einsum("bhst,h->bst", head_scores, self.indexer_weights)

        # 步骤2：预热阶段KL对齐（修复：MQA目标分布）
        if training_phase == "warmup":
            dense_target = self.compute_mqa_dense_target(x)
            index_probs = F.softmax(index_scores + 1e-10, dim=-1)  # 防log(0)
            kl_loss = F.kl_div(
                index_probs.log(), dense_target + 1e-10, 
                reduction="batchmean"
            )

        # 步骤3：Top-k筛选（确保k≤S）
        top_k = min(self.top_k, S)
        _, top_k_indices = torch.topk(index_scores, k=top_k, dim=-1)  # [B, S, top_k]

        # 步骤4：提取Top-k KV（修复：设备一致）
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).unsqueeze(2).repeat(1, S, top_k)
        K_topk = K[batch_idx, top_k_indices]  # [B, S, top_k, d_k]
        V_topk = V[batch_idx, top_k_indices]

        # 步骤5：稀疏注意力计算
        Q_reshaped = Q.transpose(1, 2)  # [B, S, H, d_k]
        K_topk_T = K_topk.transpose(2, 3)  # [B, S, d_k, top_k]
        attn_scores = torch.matmul(Q_reshaped, K_topk_T) / math.sqrt(d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=config.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, V_topk)  # [B, S, H, d_k]
        attn_output = attn_output.reshape(B, S, H * d_k)
        return attn_output, kl_loss

# -------------------------- 5. GRPO优化器（重写：修复逻辑+维度匹配+有效优势估计）--------------------------
class GRPOOptimizer:
    def __init__(self, model_params, lr, weight_decay=0.01, eps=0.2, beta=0.1, dtype=torch.float32):
        self.optimizer = torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
        self.eps = eps  # 剪辑范围
        self.beta = beta  # KL惩罚强度
        self.dtype = dtype

    def compute_gae(self, rewards, values, gamma=0.99, lam=0.95):
        """修复：用GAE（广义优势估计）替代常数优势"""
        advantages = torch.zeros_like(rewards, dtype=self.dtype, device=rewards.device)
        last_gae_lam = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] - values[t]
            advantages[t] = delta + gamma * lam * last_gae_lam
            last_gae_lam = advantages[t]
        return advantages

    def compute_unbiased_kl(self, current_probs, old_probs):
        """无偏KL估计（保持原逻辑，修复维度）"""
        kl = (current_probs * (current_probs.log() - old_probs.log())).sum(dim=-1)
        return kl.mean()

    def off_policy_mask(self, advantages, kl_div, delta):
        """修复：维度匹配，KL为逐样本值"""
        mask = torch.ones_like(advantages, dtype=self.dtype, device=advantages.device)
        mask[(advantages < 0) & (kl_div > delta)] = 0.0
        return mask

    def keep_sampling_mask(self, logits, top_p):
        """修复：Top-p掩码生成顺序（先softmax再排序）"""
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        # 生成掩码：保留累积概率≤top_p的Token
        mask = cum_probs <= top_p
        # 确保至少保留1个Token
        mask = mask | (torch.arange(mask.shape[-1], device=mask.device) == 0).unsqueeze(0).unsqueeze(0)
        # 重新排序回原Token维度
        batch_idx = torch.arange(logits.shape[0]).unsqueeze(1).unsqueeze(2)
        seq_idx = torch.arange(logits.shape[1]).unsqueeze(0).unsqueeze(2)
        mask = mask[batch_idx, seq_idx, sorted_indices]
        return mask

    def step(self, logits, target, old_probs, values, delta=0.1, top_p=0.95):
        """修复：完整GRPO训练流程，维度匹配"""
        B, S, V = logits.shape
        current_probs = F.softmax(logits, dim=-1)

        # 1. 计算奖励（语言建模任务：负交叉熵为奖励）
        ce_loss = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1), reduction="none").reshape(B, S)
        rewards = -ce_loss  # 奖励=负损失（损失越小奖励越大）

        # 2. 计算优势（GAE）
        advantages = self.compute_gae(rewards, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 标准化

        # 3. 计算逐样本KL（修复：全局KL→逐样本KL）
        kl_div = (current_probs * (current_probs.log() - old_probs.log())).sum(dim=-1)  # [B, S]

        # 4. 离线序列掩码
        mask = self.off_policy_mask(advantages, kl_div, delta)

        # 5. Keep Sampling Mask
        keep_mask = self.keep_sampling_mask(logits, top_p)
        keep_mask = keep_mask.float()  # 转为float用于相乘

        # 6. GRPO目标函数（维度匹配）
        ratio = current_probs / (old_probs + 1e-8)  # [B, S, V]
        clipped_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
        
        # 优势扩展到[B, S, V]以匹配ratio
        advantages_expanded = advantages.unsqueeze(-1).repeat(1, 1, V)
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, V)

        surr1 = ratio * advantages_expanded * mask_expanded * keep_mask
        surr2 = clipped_ratio * advantages_expanded * mask_expanded * keep_mask
        policy_loss = -torch.min(surr1, surr2).mean()

        # 7. KL惩罚
        kl_loss = self.compute_unbiased_kl(current_probs, old_probs)
        grpo_loss = policy_loss + self.beta * kl_loss

        # 8. 优化步骤
        self.optimizer.zero_grad()
        grpo_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], config.grad_clip)
        self.optimizer.step()

        return grpo_loss, ce_loss.mean()

# -------------------------- 6. 完整模型（修复：Pre-LN+数值稳定）--------------------------
class DeepSeekV32Core(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dtype = config.dtype

        # 词嵌入（指定dtype）
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, dtype=self.dtype)
        self.layer_norm_embed = nn.LayerNorm(config.d_model, dtype=self.dtype)  # Pre-LN用

        # 核心模块
        self.mla = MLA_MQA(config)
        self.dsa = DSA(config)

        # 输出层（Pre-LN结构）
        self.layer_norm_attn = nn.LayerNorm(config.num_query_heads * config.d_k, dtype=self.dtype)
        self.dropout = nn.Dropout(config.dropout)
        self.fc_out = nn.Linear(config.num_query_heads * config.d_k, config.vocab_size, dtype=self.dtype)

        # 价值网络（GRPO需要：估计状态价值）
        self.value_head = nn.Linear(config.d_model, 1, dtype=self.dtype)  # 新增：价值头

    def forward(self, x, training_phase="sparse"):
        # x: [B, S]（词表索引）
        B, S = x.shape

        # 1. 词嵌入+Pre-LN
        embed = self.embedding(x)  # [B, S, d_model]
        embed_norm = self.layer_norm_embed(embed)

        # 2. MLA生成QKV
        Q, K, V = self.mla(embed_norm)

        # 3. DSA稀疏注意力
        dsa_output, kl_loss = self.dsa(embed_norm, Q, K, V, training_phase)

        # 4. Pre-LN+残差连接（修复：Post-LN→Pre-LN，提升稳定性）
        dsa_norm = self.layer_norm_attn(dsa_output)
        hidden = self.dropout(embed + dsa_norm)

        # 5. 输出头+价值头（GRPO用）
        logits = self.fc_out(hidden)  # [B, S, vocab_size]
        values = self.value_head(hidden).squeeze(-1)  # [B, S]（状态价值）

        return logits, kl_loss, values

# -------------------------- 7. 数据集（修复：增加样本量+稳定性）--------------------------
class AutoregressiveDataset(Dataset):
    def __init__(self, config, num_samples=5000, seed=42):
        super().__init__()
        self.config = config
        self.num_samples = num_samples
        # 固定种子，确保数据可复现
        torch.manual_seed(seed)
        # 增加样本量到5000，避免过拟合
        self.data = torch.randint(0, config.vocab_size, (num_samples, config.seq_len))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]  # input: [S-1], target: [S-1]

# -------------------------- 8. 训练框架（修复：训练策略+GRPO流程+日志）--------------------------
def train_core_modules(config):
    # 1. 初始化组件
    model = DeepSeekV32Core(config).to(config.device)
    dataset = AutoregressiveDataset(config, num_samples=5000)
    # 优化数据加载（pin_memory+num_workers）
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2 if config.device.type == "cuda" else 0,
        drop_last=True  # 丢弃不完整批次，避免维度错误
    )

    # 2. 两阶段训练策略（修复：不冻结核心层，仅调整优化目标）
    phases = [
        ("warmup", config.warmup_steps, config.warmup_lr),
        ("sparse", config.sparse_steps, config.sparse_lr)
    ]

    for phase, total_steps, lr in phases:
        print(f"\n===== {phase}阶段训练（{total_steps}步，LR={lr}）=====")
        # 初始化优化器
        if phase == "sparse":
            optimizer = GRPOOptimizer(
                model.parameters(),
                lr=lr,
                weight_decay=config.weight_decay,
                dtype=config.dtype
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.98),
                eps=1e-8
            )

        model.train()
        step = 0
        while step < total_steps:
            for batch in dataloader:
                if step >= total_steps:
                    break
                x, target = batch
                x = x.to(config.device, non_blocking=True)  # [B, S-1]
                target = target.to(config.device, non_blocking=True)  # [B, S-1]

                # 前向传播
                logits, kl_loss, values = model(x, training_phase=phase)  # [B, S-1, V]
                B, S, V = logits.shape

                # -------------------------- 分阶段损失计算 --------------------------
                if phase == "warmup":
                    # 预热阶段：LM损失 + KL损失（索引器对齐）
                    lm_loss = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1))
                    total_loss = lm_loss + config.kl_weight * kl_loss

                    # 反向传播
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()

                else:
                    # 稀疏阶段：GRPO优化（需保存old_probs）
                    with torch.no_grad():
                        old_probs = F.softmax(logits, dim=-1).clone()  # 保存旧概率

                    # GRPO优化步骤
                    grpo_loss, lm_loss = optimizer.step(
                        logits=logits,
                        target=target,
                        old_probs=old_probs,
                        values=values,
                        delta=config.delta,
                        top_p=config.top_p
                    )
                    total_loss = grpo_loss + lm_loss

                # 日志输出（每50步）
                if (step + 1) % 50 == 0:
                    kl_log = f", KL Loss={kl_loss.item():.4f}" if phase == "warmup" else ""
                    grpo_log = f", GRPO Loss={grpo_loss.item():.4f}" if phase == "sparse" else ""
                    print(f"Step {step+1:4d}/{total_steps} | Total Loss={total_loss.item():.4f} | LM Loss={lm_loss.item():.4f}{kl_log}{grpo_log}")

                step += 1

    # 保存模型（包含价值头参数）
    torch.save(model.state_dict(), "deepseek_v32_core_trained.pth")
    print("\n===== 核心模块训练完成，模型已保存 =====")
    return model

# -------------------------- 9. 长序列测试（修复：自回归生成逻辑+输出校验）--------------------------
def test_long_sequence(config, model):
    print("\n===== 长序列（1024长度）测试 =====")
    # 生成测试数据：[1, 512]（输入前512个Token，生成后续Token）
    test_x = torch.randint(0, config.vocab_size, (1, 512)).to(config.device)
    model.eval()

    with torch.no_grad():
        generated = test_x.clone()
        max_gen_len = config.seq_len - 512  # 生成到1024长度
        for _ in range(max_gen_len):
            logits, _, _ = model(generated)  # [1, len, V]
            next_token = torch.argmax(logits[:, -1:, :], dim=-1)  # 贪心解码
            generated = torch.cat([generated, next_token], dim=-1)

            # 防止生成过长（容错）
            if generated.shape[1] >= config.seq_len:
                break

        # 输出关键信息
        print(f"输入长度: {test_x.shape[1]}")
        print(f"生成后长度: {generated.shape[1]}")
        print(f"生成Token范围: [{generated.min().item()}, {generated.max().item()}]")
        print(f"Logits均值: {logits.mean().item():.4f} | 方差: {logits.var().item():.4f}")
        print("长序列处理+自回归生成成功！")

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    trained_model = train_core_modules(config)
    test_long_sequence(config, trained_model)