import os
import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "gpt2"
MAX_POS = 1024
MEM_TOKENS = 1
EFFECTIVE_MAX_POS = MAX_POS - MEM_TOKENS
STRIDE = 256
MEMORY_SIZE = 16
DTYPE = torch.float32

LLM_MODEL_PATH = "/ai/shenwei/workspace/Models/huggingface/gpt2"
LOCAL_DATA_PATH = "/ai/LLM_DATA/huggingface/wikitext"

# 路径配置
MODEL_PATH = os.environ.get("LLM_MODEL_PATH", MODEL_NAME)
DATA_PATH = os.environ.get("LOCAL_DATA_PATH", "wikitext")

class TitansMemory(nn.Module):
    def __init__(self, d_model, size):
        super().__init__()
        self.register_buffer("bank", torch.zeros(size, d_model, dtype=DTYPE))
        self.ptr = 0
        self.size = size

    @torch.no_grad()
    def reset(self):
        self.bank.zero_()
        self.ptr = 0

    @torch.no_grad()
    def write(self, hidden):
        # hidden: [1, T, D]
        summary = hidden.mean(dim=1).squeeze(0)
        self.bank[self.ptr].copy_(summary)
        self.ptr = (self.ptr + 1) % self.size

    def read(self, device):
        return self.bank.mean(dim=0).to(device)


class TitansGPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=DTYPE
        ).to(DEVICE)
        self.lm.eval()

        d_model = self.lm.config.hidden_size
        self.memory = TitansMemory(d_model, MEMORY_SIZE)

    @torch.no_grad()
    def forward(self, input_ids, labels=None, prev_hidden=None):
        if prev_hidden is not None:
            self.memory.write(prev_hidden)

        # embeddings
        inputs_embeds = self.lm.get_input_embeddings()(input_ids)

        # memory token
        mem_token = self.memory.read(inputs_embeds.device)
        mem_token = mem_token.unsqueeze(0).unsqueeze(0)

        # concat → length <= MAX_POS
        inputs_embeds = torch.cat([mem_token, inputs_embeds], dim=1)

        if labels is not None:
            labels = torch.cat(
                [torch.full((1, 1), -100, device=labels.device), labels],
                dim=1
            )

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=True
        )

        hidden = outputs.hidden_states[-1][:, 1:]  # drop mem token
        return outputs.loss, hidden.detach()

@torch.no_grad()
def evaluate(model, tokens):
    model.eval()
    model.memory.reset()

    nlls, total = [], 0
    prev_hidden = None

    for i in tqdm(range(0, len(tokens) - 1, STRIDE)):
        # ⭐⭐⭐ 使用 EFFECTIVE_MAX_POS
        begin = max(0, i + STRIDE - EFFECTIVE_MAX_POS)
        end = min(i + STRIDE, len(tokens) - 1)

        input_ids = tokens[begin:end].unsqueeze(0).to(DEVICE)
        labels = tokens[begin+1:end+1].unsqueeze(0).to(DEVICE)

        loss, hidden = model(input_ids, labels, prev_hidden)

        n_tokens = end - i
        nlls.append(loss.item() * n_tokens)
        total += n_tokens

        prev_hidden = hidden

        if end == len(tokens) - 1:
            break

    return math.exp(sum(nlls) / total)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if os.path.exists(DATA_PATH):
        dataset = load_dataset(f"{DATA_PATH}/wikitext-103-v1", split="validation")
    else:
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    texts = [t for t in dataset["text"] if len(t) > 10]

    # tokenize per chunk（避免一次性超长）
    tokens = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt").input_ids[0]
        tokens.append(ids)

    tokens = torch.cat(tokens).to(DEVICE)

    model = TitansGPT2()
    ppl = evaluate(model, tokens)

    print(f"Titans PPL: {ppl:.2f}")


if __name__ == "__main__":
    main()

