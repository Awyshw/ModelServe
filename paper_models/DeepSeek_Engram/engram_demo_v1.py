"""
================================================================================
[Engram Architecture Demo Implementation]

DISCLAIMER:
1. Demo Purpose Only: 
   This code is a demonstration version intended to illustrate the core logic and 
   data flow of the Engram module.

2. Production Readiness: 
   This implementation requires further optimization for actual production use 
   (e.g., custom CUDA kernels, distributed training support).

3. Simplifications: 
   Standard components (Normalization, Attention, MoE) and complex Hyper-connection 
   mechanisms are omitted or mocked in this version to focus exclusively on the 
   Engram module implementation.
4. 总结
    ✅ 代码简化点（demo 版 vs 生产版）
    注意力层、MoE 层用 lambda 占位符替代，生产版会是真实的实现；
    超连接机制是 Mock 的，生产版会有更复杂的分支融合逻辑；
    缺少分布式训练、显存优化、CUDA 核加速等工程能力；
    缺少归一化层的完整实现，生产版会有更精细的归一化策略。
    ✅ 核心不变的点
    所有 Engram 的核心逻辑完全一致：哈希映射、多头嵌入、上下文门控、短卷积、端到端训练，这些核心机制在生产版中没有任何变化，demo 版是生产版的「最小可运行子集」。
================================================================================
"""

"""
pip install torch numpy transformers sympy
"""

## built-in
from typing import List
from dataclasses import dataclass, field
import math

## third-party
from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex 

@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3" # 加载DeepSeek官方分词器
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280*5, 129280*5]) # 2/3-gram的基础词汇量，各乘5扩容
    max_ngram_size: int = 3  # 处理【2-gram、3-gram】两种粒度，不处理单token
    n_embed_per_ngram: int = 512  # 每个N-gram的总嵌入维度
    n_head_per_ngram: int = 8  # 每个N-gram分8个「注意力头」，多头分离降低哈希冲突
    layer_ids: List[int] = field(default_factory=lambda: [1, 15]) # 只在Transformer的第1、15层插入Engram，稀疏部署， ✅ 关键补充：为什么只插在第 1、15 层？→ Engram 是「稀疏性模块」，不需要每层都加，只在关键层插入即可平衡性能和效率，这是论文的最优实践。
    pad_id: int = 2 # padding token的ID，DeepSeek-V3的PAD_ID固定为2
    seed: int = 0 # 随机种子，保证哈希映射的可复现性
    kernel_size: int = 4 # 短卷积的核大小
    
@dataclass
class BackBoneConfig:
    hidden_size: int = 1024  # Transformer每层的隐藏维度
    hc_mult: int = 4          # 超连接倍数，把隐藏层拆成4个并行分支，核心优化点
    vocab_size: int = 129280  # DeepSeek-V3的原始词汇量
    num_layers: int = 30      # Transformer的总层数
# ✅ 关键补充：hc_mult=4 是「超连接 (Hyper-connection)」的核心，把[B,L,D]的隐藏层变成[B,L,4,D]，拆成 4 个分支并行计算，降低单分支计算量，提升效率，这是 Engram 和 MoE 的通用优化手段。


engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()


# --------------------------------- Token 压缩器 -----------------------------
# 对预训练 Tokenizer 的词汇表做冗余压缩：将归一化后文本相同的 Token 合并，减少 Token 数量，提升后续哈希 / 嵌入的效率。
# 对预训练分词器的词汇表做无损语义压缩：合并「语义等价、形式不同」的 Token（如 Apple/apple、USA/U.S.A），不合并「同形异义」的 Token，生成「旧 ID→新 ID」的映射表，减少词汇量，降低后续哈希映射的计算量和冲突概率
class CompressedTokenizer:
    def __init__(
        self,
        tokenizer_name_or_path,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        
        SENTINEL = "\uE000"

        # 定义文本归一化流水线：NFKC/NFD（Unicode 归一化）→ 去重音 → 小写 → 空格标准化 → 空字符处理；
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        
        self.lookup_table, self.num_new_token = self._build_lookup_table()
    
    def __len__(self):
        return self.num_new_token
    
    def _build_lookup_table(self):
        # 遍历原始 Tokenizer 的所有 Token ID，对每个 ID 解码后做归一化；
        # 为相同归一化文本的 Token 分配同一个新 ID，生成「旧 ID→新 ID」的映射表（lookup_table）；

        old2new = {}
        key2new = {}          
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            
            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)
    
    def _compress(self, input_ids):
        # 将输入的 Token ID 数组通过 lookup_table 转换为压缩后的 ID（仅处理非负 ID）；
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out   
    
    def __call__(self, input_ids):
        return self._compress(input_ids)

# ----------------------- ShortConv（短卷积模块）--------------------------
# 实现分组深度卷积 (Depthwise Conv)，对 Engram 输出的记忆向量做局部上下文特征提取，是 Engram 模块的「特征加工器」，轻量、高效、无注意力开销。
# 轻量级：分组卷积 + 无偏置，计算量极低，不影响 Engram 的 O (1) 效率；
# 局部建模：卷积核大小为 4，只能看到局部 4 个 token 的上下文，刚好适配 N-gram 的局部特征；
# 膨胀率适配：膨胀率 =max_ngram_size=3，扩大感受野，能看到更远的局部上下文。
class ShortConv(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        kernel_size: int = 4, 
        dilation: int = 1, 
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation
        
        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps) 
            for _ in range(hc_mult)
        ])
        
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B,L,HC_MULT,D)
        Output: (B,L,HC_MULT,D)
        """
        B, T, G, C = x.shape
        
        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))
        
        x_norm = torch.cat(normed_chunks, dim=-1) 
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        
        return y

# ------------------find_next_prime（素数查找辅助函数）-----------------------
# 作用：找到大于start且未出现在seen_primes中的下一个素数；
# 用途：为 Ngram 哈希的模数生成素数（素数模数可降低哈希冲突概率）。
# 素数模数可以让哈希值的分布更均匀，极大降低哈希碰撞的概率
def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


# ----------------------- NgramHashMapping（Ngram 哈希映射类）-----------------
# 为指定层生成不同长度 ngram 的哈希 ID，是 Engram 模块的核心逻辑。
# 对输入的 Token ID 做「压缩→生成多粒度 N-gram→层专属哈希计算→生成素数模数的哈希 ID」，最终为每个指定的 Transformer 层输出对应的哈希 ID 数组，实现：从输入文本到 N-gram 哈希 ID 的 O (1) 映射。
class NgramHashMapping:
    def __init__(
        self, 
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,  
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        # 1. 加载配置参数，初始化 Token 压缩器，完成pad_id的压缩映射；
        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        # 2. 计算哈希乘法器的边界值，避免数值溢出（用np.int64的最大值做限制）；
        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        self.layer_multipliers = {}

        # 3. 为每个目标层生成「层专属的奇数乘法器」
        # 3.1 基于seed + 10007*layer_id生成随机种子，保证不同层的乘法器完全独立；
        # 3.2 生成的乘法器是奇数：奇数可以避免哈希值的奇偶性单一，进一步降低碰撞
        # 3.3 保存在layer_multipliers字典中，层 ID 为键，乘法器数组为值；
        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        # 3. 为每个层的每个 ngram 头生成素数大小的词汇量（避免哈希冲突）
        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        # 遍历每个指定的层，为 2-gram、3-gram 分别分配 8 个头；
        # 对每个头，找到大于配置词汇量的下一个素数，且全局素数不重复；
        # 所有素数保存在字典中，层 ID 为键，素数列表为值；
        # ✅ 核心目的：每个头用不同的素数模数，进一步降低哈希碰撞，多头互补。
        seen_primes = set()
        vocab_size_across_layers = {}
        
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1
                
                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
            
        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        # 生成不同长度 ngram 的 Token 序列（通过移位 + padding 实现，如 3-gram = 当前 token + 前 1 个 + 前 2 个）；
        # 用层专属乘法器加权每个 token，异或后对素数模数取余，生成哈希 ID；
        # 拼接所有头的哈希 ID，返回形状(B, L, 总头数)的哈希数组。
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        # shift_k(k)：生成左移 k 位的 Token 序列，并用 pad_id 填充左边的空位
        # ✅ 核心作用：生成 N-gram 的切片，比如 3-gram 就是 shift_k(0) + shift_k(1) + shift_k(2) → 当前 token + 前 1 个 token + 前 2 个 token。
        def shift_k(k: int) -> np.ndarray:
            if k == 0: return x
            shifted = np.pad(x, ((0, 0), (k, 0)),
                                mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        # 生成[0,1,2]位的移位序列，对应 3 个切片，为生成 2-gram/3-gram 做准备。
        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]  # 取前n个移位序列，生成n-gram
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])  # 核心：加权后异或
                # 为什么用「加权 + 异或」而不是求和？异或的特性：交换律、结合律、相同值异或为 0，可以让哈希值的分布更均匀，碰撞概率远低于求和，是哈希算法的最优选择。
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            
            # 生成多头哈希 ID
            # 对每个 N-gram 的每个头，用唯一的素数模数对 mix 值取余；
            # 生成的哈希 ID 是[B, L]的数组，所有头的哈希 ID 拼接后是[B, L, total_heads]；
            # ✅ 核心：多头哈希，每个头的哈希 ID 独立，互补降低碰撞概率。
            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        # 先对输入 Token ID 做压缩；
        # 为每个目标层生成对应的 ngram 哈希 ID，返回「层 ID→哈希数组」的字典。
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers

# ————————————————————— MultiHeadEmbedding（多头嵌入类）--------------
# 作用：计划实现多头嵌入层（适配不同 ngram 头的词汇量）；
# 代码片段仅完成初始化：接收词汇量列表和嵌入维度，定义头数、嵌入维度，并初始化偏移量列表，后续逻辑未实现。
class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        
        # 计算偏移量数组：把所有头的词汇量拼接成一个大的嵌入表，每个头的 ID 在大表中是连续的，偏移量就是前几个头的词汇量之和；
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        
        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_input_ids = input_ids + self.offsets
        output = self.embedding(shifted_input_ids)
        
        return output

# ------------------------ Engram 【Engram 主模块，所有组件的组装体，完整实现核心逻辑】 -------------------
# 这是整个代码的核心，所有之前的组件都在这里被调用、组装，完整实现了 Engram 的「哈希映射→嵌入查表→上下文门控→卷积加工→特征融合」全流程，是 Engram 架构的完整工程实现，
class Engram(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        # Ngram 哈希映射类，生成哈希 ID；
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size = engram_cfg.max_ngram_size,
            n_embed_per_ngram = engram_cfg.n_embed_per_ngram,
            n_head_per_ngram = engram_cfg.n_head_per_ngram,
            layer_ids = engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id = engram_cfg.pad_id,
            seed = engram_cfg.seed,
        )
        # 多头嵌入层，生成 N-gram 向量
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y],
            D = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
        )
        # 短卷积模块，加工特征
        self.short_conv = ShortConv(
            hidden_size = backbone_config.hidden_size,
            kernel_size = engram_cfg.kernel_size,
            dilation    = engram_cfg.max_ngram_size,
            hc_mult     = backbone_config.hc_mult,
        )
        # 线性投影层，把记忆向量投影到主干的隐藏维度
        engram_hidden_size = (engram_cfg.max_ngram_size-1) * engram_cfg.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size,backbone_config.hidden_size)
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size,backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)]
        )
        # 归一化层，门控计算的标配
        self.norm1 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
    
    def forward(self,hidden_states,input_ids):
        """
        输入：hidden_states(Transformer 主干特征，[B,L,4,D]) + input_ids(输入 Token ID，[B,L])
            hidden_states: [B, L, HC_MULT, D]
            input_ids: [B, L]
        输出：output(Engram 的记忆特征，[B,L,4,D])，和主干特征相加后返回
        """
        # 1. 生成哈希 ID 并查表得到记忆向量
        hash_input_ids = torch.from_numpy(self.hash_mapping.hash(input_ids)[self.layer_id])
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)  # [B,L, E]，E 是总嵌入维度，等于 ngram 个数 * 每个嵌入的维度
        # 2. 计算「上下文门控」
        """✅ 门控核心逻辑（解决 apple 歧义的关键）：
            Query：Transformer 主干的特征，包含当前上下文的语义信息（比如是「吃苹果」还是「苹果公司」）；
            Key：Engram 的记忆向量，包含N-gram 的通用语义信息；
            门控值是 Query 和 Key 的逐元素乘积求和，本质是「语义相似度」；
            经过sigmoid激活后，门控值在 0~1 之间，动态加权记忆向量：
                当上下文是「吃苹果」：门控值放大记忆向量中「水果」的特征；
                当上下文是「苹果公司」：门控值放大记忆向量中「科技」的特征；
            ✅ 最终效果：同一个哈希 ID 的记忆向量，在不同上下文下输出不同的语义特征，完美解决同形异义词的歧义问题。
        """
        gates = []
        for hc_idx in range(backbone_config.hc_mult):
            key = self.key_projs[hc_idx](embeddings)  # 记忆向量投影为Key
            normed_key = self.norm1[hc_idx](key)  # Key归一化
            query = hidden_states[:,:,hc_idx,:]  # 主干特征作为Query
            normed_query = self.norm2[hc_idx](query)  # Query归一化
            # # 核心：计算门控值 → 相似度加权，动态调制记忆向量
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(backbone_config.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
        gates = torch.stack(gates,dim=2)
        # 3. 特征融合与卷积加工
        """
           门控值加权记忆向量后，和主干特征的维度对齐；
           经过短卷积加工后，和原始记忆向量做残差连接；
           输出的特征向量就是 Engram 的最终记忆特征，和主干特征相加后返回。
        """
        value = gates * self.value_proj(embeddings).unsqueeze(2)
        output = value + self.short_conv(value)
        return output 

# -------------------------- TransformerBlock 【Transformer 块封装，稀疏插入 Engram】 --------------------
# 模拟 Transformer 的单层结构，稀疏插入 Engram 模块，是连接 Engram 和 Transformer 主干的桥梁。
class TransformerBlock(nn.Module):
    def __init__(self,layer_id):
        super().__init__()
        # 用lambda x:x占位符替代注意力层 (attn) 和MoE 层 (moe)，这是代码的「简化点」，生产级代码会替换为真实的注意力和 MoE；
        self.attn = lambda x:x
        self.moe  = lambda x:x
        # 如果当前层 ID 在指定的列表中，就初始化 Engram 模块，否则为 None；
        self.engram = None
        if layer_id in engram_cfg.layer_ids:
            self.engram = Engram(layer_id=layer_id)
    
    def forward(self,input_ids,hidden_states):
        if self.engram is not None:
            hidden_states = self.engram(hidden_states=hidden_states,input_ids=input_ids) + hidden_states
        hidden_states = self.attn(hidden_states) + hidden_states
        hidden_states = self.moe(hidden_states) + hidden_states
        return hidden_states

if __name__ == '__main__':
    LLM = [
        nn.Embedding(backbone_config.vocab_size,backbone_config.hidden_size),
        *[TransformerBlock(layer_id=layer_id) for layer_id in range(backbone_config.num_layers)],
        nn.Linear(backbone_config.hidden_size, backbone_config.vocab_size)
    ]

    text = "Only Alexander the Great could tame the horse Bucephalus."
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path,trust_remote_code=True)
    input_ids = tokenizer(text,return_tensors='pt').input_ids

    B,L = input_ids.shape

    for idx, layer in enumerate(LLM):
        if idx == 0:
            hidden_states = LLM[0](input_ids)
            ## mock hyper-connection: 把[B,L,D]扩展为[B,L,4,D]，模拟真实的超连接结构；
            hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, backbone_config.hc_mult, -1)      
        elif idx == len(LLM)-1:
            ## mock hyper-connection
            hidden_states = hidden_states[:,:,0,:] 
            output = layer(hidden_states)
        else:
            hidden_states = layer(input_ids=input_ids,hidden_states=hidden_states)

    print("✅ Forward Complete!")
    print(f"{input_ids.shape=}\n{output.shape=}")
            