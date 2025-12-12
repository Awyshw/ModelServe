import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from torch.distributions import Categorical


# 配置超参数
VOCAB_SIZE = 100      # 模拟词汇表大小
SEQ_LEN = 32          # 目标生成序列长度
BATCH_SIZE = 8        # 批次大小
GRADIENT_STEPS = 100  # 训练步数
LR = 1e-3             # 学习率
TAU_POS = 1.0         # 正advantage温度
TAU_NEG = 1.05        # 负advantage温度
EMBED_DIM = 128       # token嵌入维度
HIDDEN_DIM = 256      # 隐藏层维度
GRAD_CLIP = 1.0       # 梯度裁剪阈值（防止梯度爆炸)


def compute_advantage(rewards, batch_rewards):
    """简化的 group-normalized优势计算"""
    adv = rewards - batch_rewards.mean()
    return adv

def sigmoid_gate(r_t, tau):
    """SAPO核心：温度可控软门控函数"""
    x = (r_t - 1) * tau
    sig = torch.sigmoid(x)
    gate_weight = 4 / tau * sig * (1 - sig)
    return gate_weight.clamp(0, 1)

def sapo_loss(old_policy, new_policy, actions, old_log_probs, rewards, tau_pos, tau_neg):
    """计算SAPO损失（适配自回归生成的序列和 log 概率）"""
    # 1. 新策略计算 token 级 log 概率
    new_logits = new_policy(actions)  # [batch_size, seq_len, vocab_size]
    new_dist = Categorical(logits=new_logits)
    new_log_probs = new_dist.log_prob(actions)  # [batch_size, seq_len]

    # 2. 计算token级 importance ratio r_t = π_new / π_old（添加epsilon防止exp溢出）
    log_ratio = new_log_probs - old_log_probs.clamp(min=-10.0)  # 限制log_prob下限
    r_t = torch.exp(log_ratio).clamp(max=10.0)  # 限制ratio上限，避免数值爆炸

    # 3. 计算优势（扩展到 token 级）
    batch_adv = compute_advantage(rewards, rewards)  # [batch_size]
    token_adv = batch_adv.unsqueeze(1).repeat(1, SEQ_LEN)  # [batch_size, seq_len]

    # 4. 非堆成温度分配
    tau = torch.where(token_adv > 0, tau_pos, tau_neg)  # [batch_size, seq_len]

    # 5. 软门控权重
    gate_weights = sigmoid_gate(r_t, tau)  # [batch_size, seq_len]

    # 6. SAPO目标（最大化，损失取负）
    sapo_objective = gate_weights * token_adv * r_t
    sapo_loss = -sapo_objective.mean()
    return sapo_loss

def simulate_reward(actions, target_seq):
    """模拟奖励：与目标序列的匹配度"""
    target_seq_batch = target_seq.unsqueeze(0).repeat(actions.shape[0], 1)  # [batch_size, seq_len]
    match = (actions == target_seq_batch).float()  # [batch_size, seq_len]
    reward = match.sum(dim=1) / SEQ_LEN  # [batch_size]
    return reward

class AutoregressivePolicy(nn.Module):
    """自回归策略网络（模拟 LLM 的生成逻辑）"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, seq_len):
        super(AutoregressivePolicy, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # 1. Token嵌入层：将 token ID 转为稠密向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)  # Xavier初始化，避免嵌入向量过大

        # 2. 自回归解码器（简化为 MLP，真实场景 LLM 用 Transformer Decoder）
        # 输入：当前 token 嵌入 + 历史序列的平均池化（模拟注意力极致的上下文聚合）
        self.fc_context = nn.Linear(embed_dim, hidden_dim)  # 历史上下文编码
        self.fc_current = nn.Linear(embed_dim, hidden_dim)  # 当前 token 编码
        self.fc_merge = nn.Linear(hidden_dim * 2, hidden_dim)  # 融合上下文与当前 token
        self.fc_out = nn.Linear(hidden_dim, vocab_size)  # 输出下一个 token 的 logits
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)  # 稳定训练（真实 LLM 必备）

        # 初始化全连接层权重（防止初始输出异常）
        self._init_layers()
    
    def _init_layers(self):
        """初始化全连接层，避免数值不稳定"""
        for layer in [self.fc_context, self.fc_current, self.fc_merge]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.01)  # 偏置初始化为小正数，避免激活函数饱和
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0.01)

    def forward_step(self, current_token, history_embeds):
        """
        单步自回归预测：输入当前 token 和历史嵌入，输出下一个 token 的 logits
        :param current_token: 当前 token ID [batch_size]
        :param history_embeds: 历史token的嵌入序列 [batch_size, t, embed_dim]（t为已生成长度）
        :return: 下一个 token 的 logits [batch_size, vocab_size]
        """
        # 1. 编码当前 token
        current_embed = self.embedding(current_token)  # [batch_size, embed_dim]
        current_encoded = self.relu(self.fc_current(current_embed))  # [batch_size, hidden_dim]
        
        # 2. 编码历史上下文（平均池化+防止空历史）
        if history_embeds.shape[1] == 0:  # 无历史时用当前token嵌入替代
            history_mean = current_embed
        else:
            history_mean = history_embeds.mean(dim=1)  # [batch_size, embed_dim]
        history_encoded = self.relu(self.fc_context(history_mean))  # [batch_size, hidden_dim]

        # 3. 融合当前 token 与历史上下文
        merged = torch.cat([current_encoded, history_encoded], dim=-1)  # [batch_size, hidden_dim * 2]
        merged = self.layer_norm(self.relu(self.fc_merge(merged)))  # [batch_size, hidden_dim]

        # 4. 输出下一个 token 的 logits
        next_logits = self.fc_out(merged) / 10.0  # [batch_size, vocab_size]
        return next_logits
    
    def forward(self, seq_tokens):
        """
        批量计算序列中每个 token 的 logits(用于计算损失)
        :param seq_tokens: 目标生成序列 [batch_size, seq_len]
        :return: 每个token的logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = seq_tokens.shape
        all_logits = []

        # 初始化历史嵌入（第一个 token 无历史，用全零向量填充）
        history_embeds = torch.zeros(batch_size, 0, self.embed_dim).to(seq_tokens.device)

        for t in range(seq_len):
            # 当前 token：第 t 步的 token（用于预测第 t+1 步）
            current_token = seq_tokens[:, t]  # [batch_size]

            # 计算下一个 token 的 logits
            next_logits = self.forward_step(current_token, history_embeds)  # [batch_size, vocab_size]
            all_logits.append(next_logits)

            # 更新历史嵌入：添加当前 token 的嵌入
            current_embed = self.embedding(current_token).unsqueeze(1)  # [batch_size, 1, embed_dim]
            history_embeds = torch.cat([history_embeds, current_embed], dim=1)  # [batch_size, t+1, embed_dim]
        
        # 拼接所有步骤的 logits：[batch_size, seq_len, vocab_size]
        return torch.stack(all_logits, dim=1)
    
    def autoregressive_sample(self, device):
        """
        自回归采样生成完整序列（模拟真实 LLM 的生成过程）
        :param device: 运行设备(cpu/cuda)
        :return: 采样的序列[batch_size, seq_len], 每个 token 的 log 概率[batch_size, seq_len]
        """
        batch_size = BATCH_SIZE
        seq_len = SEQ_LEN
        all_tokens, all_log_probs = [], []

        # 1. 初始化：第一个 token 随机采样（模拟真实 LLM 的 prompt 起始）
        init_token = torch.randint(0, self.vocab_size, (batch_size,)).to(device)  # [batch_size]
        all_tokens.append(init_token)

        # 初始化历史嵌入
        history_embeds = torch.zeros(batch_size, 0, self.embed_dim).to(device)
        current_token = init_token

        # 逐token 自回归生成
        for t in range(seq_len):
            # 计算下一个 token 的 logits 和概率分布
            with torch.no_grad():
                next_logits = self.forward_step(current_token, history_embeds)  # [batch_size, vocab_size]
                if torch.isnan(next_logits).any():
                    raise ValueError(f"Step {t} logits contain NaN!")
                next_dist = Categorical(logits=next_logits)
            
            # 采样下一个 token，并记录 log 概率
            next_token = next_dist.sample()  # [batch_size]
            next_log_prob = next_dist.log_prob(next_token).clamp(min=-10.0)  # [batch_size]

            # 保存 token 和 log 概率
            all_tokens.append(next_token)
            all_log_probs.append(next_log_prob)

            # 更新历史嵌入：添加当前 token 的嵌入
            current_embed = self.embedding(current_token).unsqueeze(1)  # [batch_size, 1, embed_dim]
            history_embeds = torch.cat([history_embeds, current_embed], dim=1)  # [batch_size, t+1, embed_dim]
            current_token = next_token
        
        # 3. 拼接结果（去掉初始 token，保留 seq_len个生成的 token）
        # 注意：all_tokens包含初始 token+seq_len个生成 token，共 seq_len+1个，需截取后 seq_len个
        generated_seq = torch.stack(all_tokens[1:], dim=1)  # [batch_size, seq_len]
        generated_log_probs = torch.stack(all_log_probs, dim=1)  # [batch_size, seq_len]

        return generated_seq, generated_log_probs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 初始化自回归策略网络（新策略 + 旧策略）
    policy = AutoregressivePolicy(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        seq_len=SEQ_LEN,
    ).to(device)
    old_policy = AutoregressivePolicy(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        seq_len=SEQ_LEN,
    ).to(device)
    old_policy.load_state_dict(policy.state_dict())  # 初始同步权重
    print("Model initialized successfully.")

    # 2. 优化器
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    # 3. 定义目标序列（模拟 LLM 需要生成的目标）
    torch.manual_seed(42)
    target_seq = torch.randint(0, VOCAB_SIZE, (SEQ_LEN,), device=device)  # [seq_len]
    # target_seq = torch.tensor([2, 5, 7, 3, 1], device=device)  # [seq_len]

    # 4. 训练(自回归生成 + SAPO 更新)
    for step in range(GRADIENT_STEPS):
        # step1: 自回归采样（旧策略生成序列）
        with torch.no_grad():
            # 旧策略自回归生成序列和 token 级 log 概率
            actions, old_log_probs = old_policy.autoregressive_sample(device)

        # step2: 计算序列奖励
        rewards = simulate_reward(actions, target_seq)

        # step3: 计算 SAPO 损失
        loss = sapo_loss(
            old_policy=old_policy,
            new_policy=policy,
            actions=actions,
            old_log_probs=old_log_probs,
            rewards=rewards,
            tau_pos=TAU_POS,
            tau_neg=TAU_NEG,
        )

        # step4: 梯度更新
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)  # 防止梯度爆炸
        optimizer.step()

        # step5: 更新旧策略
        if (step + 1) % 20 == 0:
            old_policy.load_state_dict(policy.state_dict())
        
        # step6: 打印训练日志
        if (step + 1) % 10 == 0:
            avg_reward = rewards.mean().item()
            print(f"Step [{step+1}/{GRADIENT_STEPS}] | Loss: {loss.item():.4f} | Avg Reward: {avg_reward:.4f}")
    
    # 训练完成： 验证自回归生成效果
    print("\n==== 训练完成，验证自回归生成 ====")
    with torch.no_grad():
        test_seq, _ = policy.autoregressive_sample(device)  # 新策略生成序列
        # 取第一个样本展示
        gen_seq = test_seq[0].cpu().numpy()
        target_np = target_seq.cpu().numpy()
    print(f"目标序列: {target_np}")
    print(f"生成序列: {gen_seq}")
    print(f"匹配度: {np.mean(gen_seq == target_np):.2f}")