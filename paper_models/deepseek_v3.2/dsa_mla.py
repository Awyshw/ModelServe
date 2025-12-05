import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# -------------------------- 1. 基础配置 --------------------------
class Config:
    d_model = 512          # 模型维度
    num_query_heads = 8    # MQA查询头数量
    d_k = 64               # 每个头的维度（d_model = num_query_heads * d_k，确保整除）
    num_indexer_heads = 2  # DSA索引器头数
    d_index = 32           # 索引器维度
    top_k = 256            # 每个查询保留的Top-k Token
    seq_len = 1024         # 长序列长度
    batch_size = 4         # 批量大小
    warmup_steps = 100     # 密集预热步数
    sparse_steps = 300     # 稀疏训练步数
    warmup_lr = 1e-3       # 预热学习率
    sparse_lr = 7.3e-6     # 稀疏训练学习率
    kl_weight = 1.0        # KL散度损失权重
    weight_decay = 0.01    # 权重衰减（提升泛化性）
    grad_clip = 1.0        # 梯度裁剪阈值（避免梯度爆炸）
    dropout = 0.1          
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
assert config.d_model == config.num_query_heads * config.d_k, "d_model必须是num_query_heads * d_k的整数倍"

# -------------------------- 2. MLA --------------------------
class MLA_MQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_query_heads = config.num_query_heads
        self.d_k = config.d_k

        # 查询投影（多查询头）
        self.W_q = nn.Linear(self.d_model, self.num_query_heads * self.d_k)
        # KV投影（单组共享，MQA核心）
        self.W_k = nn.Linear(self.d_model, self.d_k)
        self.W_v = nn.Linear(self.d_model, self.d_k)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.shape

        # 生成多查询头Q
        # [batch_size, seq_len, d_model] --> [batch_size, seq_len, num_query_heads * d_k] --> [batch_size, num_query_heads, seq_len, d_k]
        Q = self.W_q(x).reshape(batch_size, seq_len, self.num_query_heads, self.d_k).transpose(1, 2)
        # 生成共享KV: [batch_size, seq_len, d_model] --> [batch_size, seq_len, d_k]
        K = self.W_k(x)
        V = self.W_v(x)

        return Q, K, V

# -------------------------- 3. DSA --------------------------
class DSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.top_k
        self.num_indexer_heads = config.num_indexer_heads
        self.d_index = config.d_index
        self.dropout = config.dropout

        # 闪电索引器
        self.indexer_q = nn.Linear(config.d_model, self.num_indexer_heads * self.d_index)
        self.indexer_k = nn.Linear(config.d_model, self.num_indexer_heads * self.d_index)
        self.indexer_norm = nn.LayerNorm(self.d_index)  # 索引器投影后归一化
        self.indexer_weights = nn.Parameter(torch.ones(self.num_indexer_heads) * 0.1)  # 小权重初始化
        self.relu = nn.ReLU()

        # 密集注意力 (MQA结构，与MLA对齐，确保KL目标一致）
        self.W_q_dense = nn.Linear(config.d_model, config.num_query_heads * config.d_k)
        self.W_k_dense = nn.Linear(config.d_model, config.d_k)
        self.W_v_dense = nn.Linear(config.d_model, config.d_k)

    def compute_mqa_dense_attention_scores(self, x):
        """MQA结构的密集注意力（与MLA一致），生成对齐目标分布"""
        batch_size, seq_len, _ = x.shape
        num_query_heads = self.config.num_query_heads
        d_k = self.config.d_k

        # MQA模式生成QKV
        # [batch_size, seq_len, d_model] --> [batch_size, seq_len, num_query_heads * d_k] --> [batch_size, num_query_heads, seq_len, d_k]
        Q = self.W_q_dense(x).reshape(batch_size, seq_len, num_query_heads, d_k).transpose(1, 2)
        # 单组 KV 共享：[batch_size, seq_len, d_model] --> [batch_size, seq_len, d_k] --> [batch_size, 1, seq_len, d_k]
        K = self.W_k_dense(x).unsqueeze(1)
        V = self.W_v_dense(x).unsqueeze(1)

        # 密集注意力得分计算
        ## [batch_size, num_query_heads, seq_len, seq_len]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 聚合所有头并生成目标分布（softmax确保概率分布）
        aggregated_scores = attn_weights.sum(dim=1)  # [batch_size, seq_len, seq_len]
        dense_target = F.softmax(aggregated_scores, dim=-1)
        return dense_target

    def forward(self, x, Q, K, V, training_phase="sparse"):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.shape
        num_query_heads = Q.shape[1]
        d_k = Q.shape[-1]
        kl_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # -------------------------- 步骤1：闪电索引器 --------------------------
        # [batch_size, seq_len, d_model] --> [batch_size, seq_len, num_indexer_heads * d_index] --> [batch_size, seq_len, num_indexer_heads, d_index]
        q_index = self.indexer_q(x).reshape(batch_size, seq_len, self.num_indexer_heads, self.d_index)
        k_index = self.indexer_k(x).reshape(batch_size, seq_len, self.num_indexer_heads, self.d_index)
        # 投影后归一化
        q_index = self.indexer_norm(q_index)
        k_index = self.indexer_norm(k_index)

        # 多头得分计算
        head_scores = torch.einsum("bshd,bthd->bhst", k_index, q_index)  # [batch_size, num_indexer_heads, seq_len, seq_len]
        head_scores = self.relu(head_scores)
        index_scores = torch.einsum("bhst,h->bst", head_scores, self.indexer_weights)  # [batch_size, seq_len, seq_len]

        # -------------------------- 步骤2：预热阶段KL对齐 --------------------------
        if training_phase == "warmup":
            dense_target = self.compute_mqa_dense_attention_scores(x)
            index_probs = F.softmax(index_scores, dim=-1)
            # 避免log(0)，添加eps
            kl_loss = F.kl_div(index_probs.log() + 1e-10, dense_target + 1e-10, reduction="batchmean")

        # -------------------------- 步骤3：Top-k筛选（确保k不超过序列长度）--------------------------
        top_k = min(self.top_k, seq_len)
        top_k_values, top_k_indices = torch.topk(index_scores, k=top_k, dim=-1)  # [batch_size, seq_len, top_k]

        # -------------------------- 步骤4：稀疏注意力计算 --------------------------
        # 提取Top-k KV
        batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(1).unsqueeze(2).repeat(1, seq_len, top_k)
        K_topk = K[batch_idx, top_k_indices]   # [batch_size, seq_len, top_k, d_k]
        V_topk = V[batch_idx, top_k_indices]

        # 注意力计算（维度对齐）
        Q_reshaped = Q.transpose(1, 2)  # [batch_size, num_query_heads, seq_len, d_k] --> [batch_size, seq_len, num_query_heads, d_k]
        K_topk_T = K_topk.transpose(2, 3)  # [batch_size, seq_len, top_k, d_k] --> [batch_size, seq_len, d_k, top_k]
        attn_scores = torch.matmul(Q_reshaped, K_topk_T) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))  # [batch_size, seq_len, num_query_heads, top_k]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)  # 添加dropout

        # 注意力输出
        attn_output = torch.matmul(attn_weights, V_topk)  # [batch_size, seq_len, num_query_heads, d_k]

        # -------------------------- 步骤5：重组输出 --------------------------
        attn_output = attn_output.reshape(batch_size, seq_len, num_query_heads * d_k)
        return attn_output, kl_loss

# -------------------------- 4. 结合成完整模型 --------------------------
class DSA_MLA_Model(nn.Module):
    def __init__(self, config, vocab_size=10000):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        # 添加embedding层（模型内处理Token→嵌入，统一设备）
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.mla = MLA_MQA(config)
        self.dsa = DSA(config)
        self.final_proj = nn.Linear(config.num_query_heads * config.d_k, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.output_head = nn.Linear(config.d_model, vocab_size)  # 分类头（Token预测）

    def forward(self, x_tokens, training_phase="sparse"):
        # x_tokens: [batch_size, seq_len-1]（整数Token序列）
        # 1. Token→嵌入（自动在当前设备上计算）
        x = self.embedding(x_tokens)  # [batch_size, seq_len-1, d_model]
        
        # 2. Pre-LN结构
        x_norm = self.layer_norm(x)
        # 3. MLA生成QKV
        Q, K, V = self.mla(x_norm)
        # 4. DSA稀疏注意力计算
        dsa_output, kl_loss = self.dsa(x_norm, Q, K, V, training_phase)
        # 5. 输出投影+残差连接
        output = self.dropout(self.final_proj(dsa_output))
        output = x + output
        # 6. Token预测头
        logits = self.output_head(output)  # [batch_size, seq_len-1, vocab_size]
        return logits, kl_loss

# -------------------------- 5. 数据集 （模拟数据集）--------------------------
class LongSeqDataset(Dataset):
    def __init__(self, config, num_samples=5000, vocab_size=10000):
        self.config = config
        self.num_samples = num_samples
        self.vocab_size = vocab_size  # 模拟词表大小
        # 生成随机Token序列（整数，0~vocab_size-1），保持在CPU上（数据集不提前指定设备）
        self.token_seqs = torch.randint(0, vocab_size, (num_samples, config.seq_len))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 输入：前seq_len-1个Token；目标：后seq_len-1个Token（下一个Token预测）
        tokens = self.token_seqs[idx]
        x_tokens = tokens[:-1]  # [seq_len-1]（整数Token序列）
        target = tokens[1:]     # [seq_len-1]（目标Token）
        return x_tokens, target

# -------------------------- 6. 训练框架 --------------------------
def count_model_params(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_dsa_mla(config):
    vocab_size = 10000  # 词表大小
    model = DSA_MLA_Model(config, vocab_size=vocab_size).to(config.device)  # 模型移到目标设备
    dataset = LongSeqDataset(config, num_samples=5000, vocab_size=vocab_size)  # 增大样本量避免过拟合
    # DataLoader添加num_workers（可选，加速数据加载）
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        pin_memory=True,
        num_workers=2  # 根据CPU核心数调整（0=主进程加载）
    )
    total_params = count_model_params(model)
    print(f"模型总参数量：{total_params / 1e6:.2f}M")

    # 两阶段训练配置
    phases = [
        ("warmup", config.warmup_steps, config.warmup_lr),
        ("sparse", config.sparse_steps, config.sparse_lr)
    ]

    for phase, total_steps, lr in phases:
        print(f"\n===== 开始{phase}阶段训练（{total_steps}步，LR={lr}）=====")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98)
        )
        lm_criterion = nn.CrossEntropyLoss()  # 分类损失（Token预测）

        model.train()
        for step, (x_tokens, target) in enumerate(dataloader):
            if step >= total_steps:
                break

            # 将Token移到目标设备（统一CPU/CUDA）
            x_tokens = x_tokens.to(config.device, non_blocking=True)  # [batch_size, seq_len-1]
            target = target.to(config.device, non_blocking=True)      # [batch_size, seq_len-1]

            # 前向传播（输入为Token序列）
            logits, kl_loss = model(x_tokens, training_phase=phase)  # [batch_size, seq_len-1, vocab_size]
            
            # CrossEntropyLoss要求输入shape：[batch_size*seq_len-1, vocab_size]，目标：[batch_size*seq_len-1]
            batch_size, seq_len_minus_1, _ = logits.shape
            lm_loss = lm_criterion(
                logits.reshape(-1, vocab_size),  # [batch_size*(seq_len-1), vocab_size]
                target.reshape(-1)               # [batch_size*(seq_len-1)]
            )

            # 总损失计算
            if phase == "warmup":
                total_loss = lm_loss + config.kl_weight * kl_loss
            else:
                total_loss = lm_loss

            # 反向传播+梯度裁剪
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            # 日志打印（每10步）
            if (step + 1) % 10 == 0:
                kl_log = f", KL Loss={kl_loss.item():.4f}" if phase == "warmup" else ""
                print(f"Step {step+1:3d}/{total_steps} | Total Loss={total_loss.item():.4f} | LM Loss={lm_loss.item():.4f}{kl_log}")

    # 保存模型（包含embedding层参数）
    torch.save(model.state_dict(), "dsa_mla_trained.pth")
    print(f"\n模型已保存至：dsa_mla_trained.pth")
    return model

# -------------------------- 7. 测试函数 --------------------------
def test_long_seq(config, model=None):
    print("\n===== 开始长序列测试 =====")
    vocab_size = 10000
    # 加载模型（若未传入）
    if model is None:
        model = DSA_MLA_Model(config, vocab_size=vocab_size).to(config.device)
        model.load_state_dict(torch.load("dsa_mla_trained.pth", map_location=config.device))
        print("已加载预训练模型")

    # 生成测试Token序列（[1, seq_len-1]）
    test_tokens = torch.randint(0, vocab_size, (1, config.seq_len - 1)).to(config.device)
    model.eval()

    with torch.no_grad():
        logits, _ = model(test_tokens)  # logits: [1, seq_len-1, vocab_size]
        pred_tokens = torch.argmax(logits, dim=-1)  # 预测的Token

        # 打印关键信息
        print(f"输入Token维度: {test_tokens.shape}")
        print(f"输出Logits维度: {logits.shape}")
        print(f"预测Token维度: {pred_tokens.shape}")
        print(f"Logits均值: {logits.mean().item():.4f} | Logits方差: {logits.var().item():.4f}")
        print(f"前5个输入Token: {test_tokens[0, :5].cpu().numpy()}")
        print(f"前5个预测Token: {pred_tokens[0, :5].cpu().numpy()}")
        print("长序列测试成功！")

# -------------------------- 8. 主函数 --------------------------
if __name__ == "__main__":
    # 训练模型
    trained_model = train_dsa_mla(config)
    # 测试长序列（支持直接测试或加载模型测试）
    test_long_seq(config, trained_model)
    # 单独加载模型测试（注释掉上面，解开下面）
    # test_long_seq(config)