import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "gpt2"
MAX_POS = 1024
STRIDE = 256
WRITE_TOKENS = 64    # 每次写入 memory token 数量
MEMORY_ALPHA = 0.1

LLM_MODEL_PATH = "/ai/shenwei/workspace/Models/huggingface/gpt2"
LOCAL_DATA_PATH = "/ai/LLM_DATA/huggingface/wikitext"

# ================= Memory =================
class TitansMemory(nn.Module):
    def __init__(self, d_model, size):
        super().__init__()
        self.size = size
        self.register_buffer("mem", torch.zeros(size, d_model))
        self.register_buffer("strength", torch.zeros(size))
        self.ptr = 0

    @torch.no_grad()
    def reset(self):
        self.mem.zero_()
        self.strength.zero_()
        self.ptr = 0

    @torch.no_grad()
    def write(self, h, surprise=None, debug=False):
        """
        h: [B, T, D]
        surprise: [B, T]
        """
        B, T, D = h.shape
        if T == 0:
            return

        max_k = min(WRITE_TOKENS, T)
        if max_k <= 0:
            return

        if surprise is not None:
            # 取 bottom-k (低 surprise token) 写入 memory
            s = surprise.mean(0)  # [T]
            k = min(max_k, s.numel())
            _, idx = torch.topk(-s, k)  # 注意取负号 → bottom-k
            idx = idx.clamp(0, T - 1)
        else:
            k = max_k
            idx = torch.arange(k, device=h.device)

        content = F.normalize(h[:, idx, :].squeeze(0), dim=-1)  # [k, D]

        for i in range(k):
            self.mem[self.ptr].copy_(content[i])
            self.strength[self.ptr] = 1.0
            self.ptr = (self.ptr + 1) % self.size

        if debug and surprise is not None:
            bottom_s = s[idx].detach().cpu().numpy()
            print(f"[Memory Write] Bottom surprise: {bottom_s}, Memory ptr: {self.ptr}")

    def read(self, q):
        if self.strength.sum() == 0:
            return torch.zeros_like(q)

        q_norm = F.normalize(q, dim=-1)
        m_norm = F.normalize(self.mem, dim=-1)

        attn = torch.einsum("btd,md->btm", q_norm, m_norm)
        attn = attn * self.strength.unsqueeze(0).unsqueeze(0)
        attn = torch.softmax(attn, dim=-1)

        return torch.einsum("btm,md->btd", attn, self.mem)

# ================= Model =================
class TitansGPT2(nn.Module):
    def __init__(self, memory_size=256, use_surprise=True, use_gate=True):
        super().__init__()
        if os.path.exists(LLM_MODEL_PATH):
            self.gpt2 = GPT2LMHeadModel.from_pretrained(LLM_MODEL_PATH).to(DEVICE)
        else:
            self.gpt2 = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)

        d = self.gpt2.config.n_embd
        self.memory = TitansMemory(d, memory_size)
        self.mem_ln = nn.LayerNorm(d)
        self.mem_gate = nn.Linear(d, 1)

        self.use_surprise = use_surprise
        self.use_gate = use_gate

    def forward(self, ids, labels, prev_embeds=None, debug=False):
        embeds = self.gpt2.transformer.wte(ids)

        # ----- write memory from previous window -----
        if prev_embeds is not None:
            with torch.no_grad():
                if self.use_surprise:
                    logits = self.gpt2(inputs_embeds=prev_embeds).logits  # [B, T, V]
                    probs = torch.softmax(logits, dim=-1)
                    true_ids = labels[:, -prev_embeds.size(1):]
                    true_probs = probs.gather(-1, true_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]
                    surprise = -torch.log(true_probs + 1e-8)  # token NLL
                else:
                    surprise = None

                self.memory.write(prev_embeds[:, -WRITE_TOKENS:], surprise, debug=debug)

        # ----- read memory -----
        mem = self.mem_ln(self.memory.read(embeds))

        # ----- inject memory -----
        inject_pos = max(0, embeds.size(1) - STRIDE)
        mem_to_inject = mem[:, inject_pos:]

        if self.use_gate and self.use_surprise:
            # Gate 与 surprise 负相关，低 surprise 注入强
            with torch.no_grad():
                logits = self.gpt2(inputs_embeds=embeds[:, inject_pos:]).logits
                probs = torch.softmax(logits, dim=-1)
                true_ids = labels[:, -mem_to_inject.size(1):]
                true_probs = probs.gather(-1, true_ids.unsqueeze(-1)).squeeze(-1)
                token_surprise = -torch.log(true_probs + 1e-8)  # [B, T]
                gate = torch.sigmoid(-token_surprise.unsqueeze(-1))  # 低 surprise gate 大
        elif self.use_gate:
            gate = torch.sigmoid(self.mem_gate(embeds[:, inject_pos:]))
        else:
            gate = torch.ones_like(mem_to_inject)

        embeds[:, inject_pos:] += MEMORY_ALPHA * gate * mem_to_inject

        if debug:
            print(f"[Memory Inject] gate min/max: {gate.min().item():.4f}/{gate.max().item():.4f}")

        out = self.gpt2(inputs_embeds=embeds, labels=labels)
        return out.loss, embeds.detach()

# ================= Dataset =================
def load_tokens():
    if os.path.exists(LLM_MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    if os.path.exists(LOCAL_DATA_PATH):
        ds = load_dataset(f"{LOCAL_DATA_PATH}/wikitext-103-v1", split="validation[:10%]")
    else:
        ds = load_dataset("wikitext", "wikitext-103-v1", split="validation")

    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    return tokens.to(DEVICE)

# ================= Evaluation =================
@torch.no_grad()
def evaluate_titans(model, tokens, debug=False):
    model.eval()
    model.memory.reset()
    nlls, total = [], 0
    prev_embeds = None

    for i in tqdm(range(0, len(tokens) - 1, STRIDE)):
        begin = max(0, i + STRIDE - MAX_POS)
        end = min(i + STRIDE, len(tokens) - 1)

        ids = tokens[begin:end].unsqueeze(0).to(DEVICE)
        labels = tokens[begin+1:end+1].unsqueeze(0).to(DEVICE)

        loss, curr_embeds = model(ids, labels, prev_embeds, debug=debug)
        nlls.append(loss.item() * (end - i))
        total += (end - i)
        prev_embeds = curr_embeds

        if debug:
            print(f"[Eval] Window {i}-{end}, Loss: {loss.item():.4f}")

    return math.exp(sum(nlls) / total)

# ================= Ablation =================
def run_ablations(tokens):
    configs = {
        "Full Titans": dict(surprise=True, gate=True),
        "No Surprise": dict(surprise=False, gate=True),
        "No Gate": dict(surprise=True, gate=False),
    }

    results = {}
    for name, cfg in configs.items():
        print(f"\n=== Evaluating {name} ===")
        model = TitansGPT2(memory_size=256, use_surprise=cfg["surprise"], use_gate=cfg["gate"]).to(DEVICE)
        ppl = evaluate_titans(model, tokens, debug=True)
        results[name] = ppl
        print(f"{name}: PPL={ppl:.2f}")

    return results

# ================= Main =================
def main():
    tokens = load_tokens()
    results = run_ablations(tokens)
    print("\n=== Final Ablation Results ===")
    print(results)

if __name__ == "__main__":
    main()
