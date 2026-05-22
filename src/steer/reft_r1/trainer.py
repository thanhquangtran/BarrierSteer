"""Minimal ReFT-r1 trainer for instruction/refusal pairs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .intervention import ReFTR1Config, ReFTR1Intervention, attach_reft_r1_hook


class PromptTargetDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 1024):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        examples = (
            payload.get("examples", payload) if isinstance(payload, dict) else payload
        )
        if not isinstance(examples, list) or not examples:
            raise ValueError(f"No ReFT-r1 examples found in {path}")
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        prompt = str(ex.get("prompt", ex.get("instruction", "")))
        target = str(ex.get("target", ex.get("output", "")))
        if not prompt or not target:
            raise ValueError(
                f"Bad ReFT-r1 example at index {idx}: expected prompt+target"
            )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True).input_ids
        target_ids = self.tokenizer(target, add_special_tokens=False).input_ids
        ids = (prompt_ids + target_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_length]
        attn = [1] * len(ids)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_prompt_targets(
    batch: List[Dict[str, torch.Tensor]], pad_token_id: int
) -> Dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].numel() for x in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in batch:
        n = item["input_ids"].numel()
        pad = max_len - n
        out["input_ids"].append(
            torch.cat(
                [torch.full((pad,), pad_token_id, dtype=torch.long), item["input_ids"]]
            )
        )
        out["attention_mask"].append(
            torch.cat([torch.zeros(pad, dtype=torch.long), item["attention_mask"]])
        )
        out["labels"].append(
            torch.cat([torch.full((pad,), -100, dtype=torch.long), item["labels"]])
        )
    return {k: torch.stack(v, dim=0) for k, v in out.items()}


def train_reft_r1(
    model_name_or_path: str,
    dataset_path: str,
    output_dir: str,
    target_layer: int,
    top_k: int = 5,
    beta: float = 1.0,
    l1_coeff: float = 0.0,
    learning_rate: float = 1e-3,
    max_steps: int = 500,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 0,
    max_length: int = 1024,
    dtype: str = "auto",
    seed: int = 42,
    log_every: int = 10,
) -> str:
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"

    torch_dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "auto": "auto",
    }.get(dtype, "auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype, device_map="auto"
    )
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)

    hidden_size = int(
        getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0))
    )
    if hidden_size <= 0:
        raise ValueError("Could not determine model hidden size")
    cfg = ReFTR1Config(
        target_layer=target_layer, top_k=top_k, beta=beta, l1_coeff=l1_coeff
    )
    intervention = ReFTR1Intervention(hidden_size=hidden_size, config=cfg).to(
        next(model.parameters()).device
    )
    handle = attach_reft_r1_hook(model, intervention)

    ds = PromptTargetDataset(dataset_path, tokenizer, max_length=max_length)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_prompt_targets(b, tokenizer.pad_token_id),
    )
    opt = torch.optim.AdamW([intervention.vector], lr=learning_rate)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=max_steps
    )

    step = 0
    pbar = tqdm(total=max_steps, desc="train ReFT-r1")
    try:
        while step < max_steps:
            for batch in loader:
                batch = {
                    k: v.to(next(model.parameters()).device) for k, v in batch.items()
                }
                outputs = model(**batch)
                loss = outputs.loss
                if intervention.last_non_topk_l1 is not None:
                    loss = loss + float(l1_coeff) * intervention.last_non_topk_l1.to(
                        loss.device
                    )
                (loss / max(1, gradient_accumulation_steps)).backward()
                if (step + 1) % max(1, gradient_accumulation_steps) == 0:
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    intervention.renormalize_()
                step += 1
                if log_every and step % log_every == 0:
                    print(
                        f"[reft-r1] step={step} loss={float(loss.detach().cpu()):.4f} |w|={float(intervention.vector.norm().detach().cpu()):.4f}"
                    )
                pbar.update(1)
                if step >= max_steps:
                    break
    finally:
        pbar.close()
        handle.remove()

    ckpt = os.path.join(output_dir, "reft_r1_vector.pt")
    torch.save(
        {
            "vector": intervention.unit_vector.detach().cpu(),
            "config": asdict(cfg),
            "model_name_or_path": model_name_or_path,
        },
        ckpt,
    )
    with open(
        os.path.join(output_dir, "reft_r1_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "config": asdict(cfg),
                "model_name_or_path": model_name_or_path,
                "dataset_path": dataset_path,
                "max_steps": max_steps,
            },
            f,
            indent=2,
        )
    return ckpt
