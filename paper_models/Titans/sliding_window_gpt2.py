import math
import os
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "gpt2"
LLM_MODEL_PATH = "/ai/shenwei/workspace/Models/huggingface/gpt2"
LOCAL_DATA_PATH = "/ai/LLM_DATA/huggingface/wikitext"
WINDOW = 512
STRIDE = 256
MAX_EVAL_TOKENS = 20_000  # 控制评测规模


class SlidingWindowEvaluator:
    def __init__(self, model, tokenizer, window, stride):
        self.model = model
        self.tokenizer = tokenizer
        self.window = window
        self.stride = stride

    @torch.no_grad()
    def perplexity(self, text: str):
        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"][0]

        total_nll = 0.0
        total_tokens = 0

        for start in range(0, len(input_ids) - 1, self.stride):
            end = min(start + self.window, len(input_ids))
            chunk = input_ids[start:end].unsqueeze(0).to(device)

            labels = chunk.clone()
            labels[:, :-1] = -100  # 只算最后一个 token 的 loss

            out = self.model(chunk, labels=labels)
            loss = out.loss

            num_tokens = (labels != -100).sum().item()
            total_nll += loss.item() * num_tokens
            total_tokens += num_tokens

            if total_tokens > MAX_EVAL_TOKENS:
                break

        return math.exp(total_nll / total_tokens)


class TitansMemory(nn.Module):
    def __init__(self, dim, slots=128):
        super().__init__()
        self.slots = slots
        self.register_buffer("memory", torch.zeros(slots, dim))
        self.ptr = 0

    @torch.no_grad()
    def write(self, h):
        self.memory[self.ptr % self.slots] = h
        self.ptr += 1

    def read(self, q):
        attn = torch.softmax(q @ self.memory.T / math.sqrt(q.size(-1)), dim=-1)
        return attn @ self.memory

    def reset(self):
        self.memory.zero_()
        self.ptr = 0

class TitansSlidingEvaluator(SlidingWindowEvaluator):
    def __init__(self, model, tokenizer, window, stride, memory):
        super().__init__(model, tokenizer, window, stride)
        self.memory = memory

    @torch.no_grad()
    def perplexity(self, text):
        self.memory.reset()
        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"][0]

        total_nll = 0
        total_tokens = 0

        for start in range(0, len(input_ids) - 1, self.stride):
            end = min(start + self.window, len(input_ids))
            chunk = input_ids[start:end].unsqueeze(0).to(device)

            out = self.model.transformer(chunk)
            h = out.last_hidden_state

            # write summary
            self.memory.write(h[0, 0])

            # read memory
            mem = self.memory.read(h)
            h = h + mem.unsqueeze(0)

            logits = self.model.lm_head(h)
            labels = chunk.clone()
            labels[:, :-1] = -100

            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            num_tokens = (labels != -100).sum().item()
            total_nll += loss.item() * num_tokens
            total_tokens += num_tokens

            if total_tokens > MAX_EVAL_TOKENS:
                break

        return math.exp(total_nll / total_tokens)


def main():
    # tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    if os.path.exists(LLM_MODEL_PATH):
        tokenizer = GPT2Tokenizer.from_pretrained(LLM_MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(LLM_MODEL_PATH).to(device)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    tokenizer.pad_token = tokenizer.eos_token

    # model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    if os.path.exists(LOCAL_DATA_PATH):
        dataset = load_dataset(f"{LOCAL_DATA_PATH}/wikitext-2-raw-v1", split="test")
    else:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    evaluator = SlidingWindowEvaluator(model, tokenizer, WINDOW, STRIDE)
    memory = TitansMemory(model.config.hidden_size).to(device)
    titan_evaluator = TitansSlidingEvaluator(model, tokenizer, WINDOW, STRIDE, memory)

    ppl = evaluator.perplexity("\n\n".join(dataset["text"][:50]))
    print(f"GPT-2 Sliding Window PPL: {ppl:.2f}")

    ppl = titan_evaluator.perplexity("\n\n".join(dataset["text"][:50]))
    print(f"Titans Sliding Window PPL: {ppl:.2f}")


if __name__ == "__main__":
    main()
