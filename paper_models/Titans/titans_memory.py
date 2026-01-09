import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, AutoTokenizer

# ============================================================
# Config
# ============================================================
MODEL_NAME = "gpt2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LLM_MODEL_PATH = "/ai/shenwei/workspace/Models/huggingface/gpt2"
LOCAL_DATA_PATH = "/ai/LLM_DATA/huggingface/wikitext"

MAX_POS = 1024
STRIDE = 256
MAX_TOKENS = 200_000

MEMORY_SIZE = 256
MEMORY_ALPHA = 0.1          # ↓↓↓ 关键
WRITE_FREQ = 4
WRITE_TOKENS = 128
SURPRISE_RATIO = 0.5       # top 50% surprise 才注入

# ============================================================
# Load WikiText tokens (official HF-style)
# ============================================================
def load_tokens():
    if os.path.exists(LLM_MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    if os.path.exists(LOCAL_DATA_PATH):
        dataset = load_dataset(f"{LOCAL_DATA_PATH}/wikitext-103-v1", split={"train": "train[:1%]", "validation": "validation[:1%]"})
        ds = dataset["validation"]
    else:
        ds = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    return tokens[:MAX_TOKENS], tokenizer

# ============================================================
# Stable Titans Fast Memory
# ============================================================
class TitansMemory(nn.Module):
    def __init__(self, d_model, size):
        super().__init__()
        self.register_buffer("mem", torch.zeros(size, d_model))
        self.register_buffer("strength", torch.zeros(size))
        self.decay = 0.98
        self.lr = 0.05

    @torch.no_grad()
    def reset(self):
        self.mem.zero_()
        self.strength.zero_()

    @torch.no_grad()
    def write(self, h, surprise):
        # h: [B,T,D], surprise: [B,T]
        s = torch.nan_to_num(surprise.mean(0), 0.0)
        k = min(len(s), self.mem.size(0) // 8)
        if k == 0:
            return

        _, idx = torch.topk(s, k)
        content = F.normalize(h.mean(0)[idx], dim=-1)

        self.mem.mul_(self.decay)
        self.strength.mul_(self.decay)

        self.mem[:k] += self.lr * content
        self.strength[:k] += s[idx]

    def read(self, h):
        q = F.normalize(h, dim=-1)
        m = F.normalize(self.mem, dim=-1)

        attn = torch.einsum("btd,md->btm", q, m)
        attn = attn * self.strength
        attn = torch.softmax(attn, dim=-1)

        return torch.einsum("btm,md->btd", attn, self.mem)

# ============================================================
# GPT-2 Baseline
# ============================================================
class GPT2Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        if os.path.exists(LLM_MODEL_PATH):
            self.model = GPT2LMHeadModel.from_pretrained(LLM_MODEL_PATH)
        else:
            self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    def forward(self, input_ids, labels):
        return self.model(input_ids, labels=labels).loss

# ============================================================
# Stable Titans GPT-2
# ============================================================
class TitansGPT2(nn.Module):
    """
    Key differences vs naive version:
    ✔ memory -> LayerNorm
    ✔ gated injection
    ✔ no direct unbounded addition
    """
    def __init__(self):
        super().__init__()
        if os.path.exists(LLM_MODEL_PATH):
            self.gpt2 = GPT2LMHeadModel.from_pretrained(LLM_MODEL_PATH)
        else:
            self.gpt2 = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

        d = self.gpt2.config.n_embd

        self.memory = TitansMemory(d, MEMORY_SIZE)
        self.mem_ln = nn.LayerNorm(d)
        self.mem_gate = nn.Linear(d, 1)

    def forward(self, ids, labels, step):
        embeds = self.gpt2.transformer.wte(ids)

        # ---- predict surprise (no grad)
        with torch.no_grad():
            logits = self.gpt2(inputs_embeds=embeds).logits
            pred_ids = logits.argmax(-1)
            pred_embeds = self.gpt2.transformer.wte(pred_ids)

            surprise = torch.norm(
                embeds[:, -WRITE_TOKENS:] - pred_embeds[:, -WRITE_TOKENS:],
                dim=-1
            )

            if step % WRITE_FREQ == 0:
                self.memory.write(embeds[:, -WRITE_TOKENS:], surprise)

            mem = self.memory.read(embeds[:, -WRITE_TOKENS:])

        # ---- sparse injection (关键)
        mem = self.mem_ln(mem)
        gate = torch.sigmoid(self.mem_gate(embeds[:, -WRITE_TOKENS:]))

        # hard surprise mask
        thresh = torch.quantile(surprise, SURPRISE_RATIO)
        mask = (surprise > thresh).float().unsqueeze(-1)

        embeds = embeds.clone()
        embeds[:, -WRITE_TOKENS:] += MEMORY_ALPHA * gate * mask * mem

        out = self.gpt2(inputs_embeds=embeds, labels=labels)
        return out.loss

# ============================================================
# Long-context evaluation (HF-official style)
# ============================================================
@torch.no_grad()
def evaluate(model, tokens, titans=False):
    model.eval()
    if titans:
        model.memory.reset()

    nlls = []
    total = 0

    for step, i in enumerate(range(0, len(tokens), STRIDE)):
        begin = max(i + STRIDE - MAX_POS, 0)
        end = min(i + STRIDE, len(tokens))
        trg_len = end - i

        ids = tokens[begin:end].unsqueeze(0).to(DEVICE)
        labels = ids.clone()
        labels[:, :-trg_len] = -100

        loss = (
            model(ids, labels, step)
            if titans else
            model(ids, labels)
        )

        if torch.isfinite(loss):
            nlls.append(loss * trg_len)
            total += trg_len

        if end == len(tokens):
            break

    nll = torch.stack(nlls).sum() / total
    return torch.exp(torch.clamp(nll, max=20)).item()

# ============================================================
# Main
# ============================================================
def main():
    tokens, _ = load_tokens()
    tokens = tokens.to(DEVICE)

    gpt2 = GPT2Baseline().to(DEVICE)
    titans = TitansGPT2().to(DEVICE)

    print("Evaluating GPT-2 baseline...")
    ppl_gpt2 = evaluate(gpt2, tokens, False)

    print("Evaluating Titans (stable memory)...")
    ppl_titans = evaluate(titans, tokens, True)

    print("\n==============================")
    print(f"GPT-2 PPL   : {ppl_gpt2:.2f}")
    print(f"Titans PPL  : {ppl_titans:.2f}")
    print("==============================")

if __name__ == "__main__":
    main()