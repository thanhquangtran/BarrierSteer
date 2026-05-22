import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.nn.functional import cosine_similarity
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(spec: str) -> List[int]:
    return [int(item) for item in spec.split(",") if item]


def maybe_strip_bos(text: str, tokenizer) -> str:
    bos_token = tokenizer.bos_token
    if bos_token and text.startswith(bos_token):
        return text[len(bos_token) :]
    return text


def build_request_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return maybe_strip_bos(text, tokenizer)


def get_transformer_layers(model) -> Optional[torch.nn.ModuleList]:
    candidates = [
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("transformer", "h"),
    ]
    for path in candidates:
        current = model
        for attr in path:
            current = getattr(current, attr, None)
            if current is None:
                break
        if current is not None:
            return current
    return None


def restore_dropped_layers(
    merged_model, model_name_or_path: str, drop_layers_after: Optional[int]
) -> None:
    if drop_layers_after is None:
        return

    merged_layers = get_transformer_layers(merged_model)
    if merged_layers is None:
        raise ValueError(
            "Could not locate transformer layers on merged CircuitBreaker model"
        )

    anchor_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=merged_model.dtype,
    )
    anchor_layers = get_transformer_layers(anchor_model)
    if anchor_layers is None:
        raise ValueError(
            "Could not locate transformer layers on anchor model for restore"
        )

    merged_layers.extend(anchor_layers[drop_layers_after + 1 :])
    merged_model.config = anchor_model.config


@dataclass
class PairExample:
    prompt: str
    output: str


class HarmBenchCircuitBreakerDataset(Dataset):
    def __init__(self, tokenizer, dataset_path: str, max_length: int = 1024):
        with open(dataset_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.retain_train = [PairExample(**item) for item in payload["safe_train"]]
        self.retain_val = [PairExample(**item) for item in payload["safe_val"]]
        self.unsafe_train = [PairExample(**item) for item in payload["unsafe_train"]]
        self.unsafe_val = [PairExample(**item) for item in payload["unsafe_val"]]

        if not self.retain_train:
            raise ValueError(f"No safe_train examples found in {dataset_path}")
        if not self.unsafe_train:
            raise ValueError(f"No unsafe_train examples found in {dataset_path}")
        if not self.unsafe_val:
            raise ValueError(f"No unsafe_val examples found in {dataset_path}")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cb_half_length = max(64, max_length // 2)

    def __len__(self):
        return max(len(self.retain_train), len(self.unsafe_train))

    def _tokenize_pair(self, pair: PairExample):
        request_text = build_request_text(self.tokenizer, pair.prompt)
        response_text = pair.output

        token_kwargs = dict(
            max_length=self.cb_half_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        self.tokenizer.padding_side = "left"
        request_tokens = self.tokenizer(request_text, **token_kwargs)
        self.tokenizer.padding_side = "right"
        response_tokens = self.tokenizer(
            response_text,
            add_special_tokens=False,
            **token_kwargs,
        )
        self.tokenizer.padding_side = "left"

        combined_input_ids = torch.cat(
            [request_tokens["input_ids"], response_tokens["input_ids"]],
            dim=1,
        )
        combined_attention_mask = torch.cat(
            [request_tokens["attention_mask"], response_tokens["attention_mask"]],
            dim=1,
        )

        full_tokens = self.tokenizer(
            request_text + response_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return combined_input_ids, combined_attention_mask, full_tokens

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        retain_pair = self.retain_train[idx % len(self.retain_train)]
        unsafe_pair = self.unsafe_train[idx % len(self.unsafe_train)]
        val_pair = self.unsafe_val[idx % len(self.unsafe_val)]

        cb_input_ids, cb_attention_mask, _ = self._tokenize_pair(unsafe_pair)
        _, _, retain_tokens = self._tokenize_pair(retain_pair)
        _, _, val_tokens = self._tokenize_pair(val_pair)

        return {
            "input_ids_circuit_breaker": cb_input_ids,
            "attention_mask_circuit_breaker": cb_attention_mask,
            "input_ids": retain_tokens["input_ids"],
            "attention_mask": retain_tokens["attention_mask"],
            "input_ids_val": val_tokens["input_ids"],
            "attention_mask_val": val_tokens["attention_mask"],
        }


def data_collator(batch_list):
    batch_inputs = {}
    for features in batch_list:
        for key, value in features.items():
            batch_inputs.setdefault(key, []).append(value)
    return {key: torch.cat(values, dim=0) for key, values in batch_inputs.items()}


def compute_circuit_breaker_loss(trainer, model, inputs, target_layers, alpha):
    trainer.current_training_step += 1
    log_now = trainer.current_training_step % trainer.log_every == 0
    peft_model = model.module if hasattr(model, "module") else model

    retain_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "output_hidden_states": True,
    }
    cb_inputs = {
        "input_ids": inputs["input_ids_circuit_breaker"],
        "attention_mask": inputs["attention_mask_circuit_breaker"],
        "output_hidden_states": True,
    }
    val_inputs = {
        "input_ids": inputs["input_ids_val"],
        "attention_mask": inputs["attention_mask_val"],
        "output_hidden_states": True,
    }

    progress = min(1.0, trainer.current_training_step / max(1, trainer.schedule_steps))
    retain_coeff = alpha * progress
    circuit_breaker_coeff = alpha * (1.0 - progress)

    layers_cb_attention_mask = (
        inputs["attention_mask_circuit_breaker"]
        .repeat(len(target_layers), 1, 1)
        .unsqueeze(-1)
    )

    with peft_model.disable_adapter():
        model.eval()
        with torch.no_grad():
            orig_retain_outputs = model(**retain_inputs)["hidden_states"]
            orig_retain_hidden = torch.stack(orig_retain_outputs).detach()
            retain_mask = (
                inputs["attention_mask"]
                .repeat(len(orig_retain_outputs), 1, 1)
                .unsqueeze(-1)
            )
            orig_retain_hidden = orig_retain_hidden * retain_mask

            orig_cb_outputs = model(**cb_inputs)["hidden_states"]
            orig_cb_hidden = torch.stack(
                [orig_cb_outputs[layer].detach() for layer in target_layers]
            )

            if log_now:
                val_outputs = model(**val_inputs)["hidden_states"]
                val_hidden = torch.stack(
                    [val_outputs[layer] for layer in target_layers]
                )

    model.train()

    lora_retain_outputs = model(**retain_inputs)["hidden_states"]
    lora_retain_hidden = torch.stack(lora_retain_outputs) * retain_mask
    retain_loss = torch.norm(
        lora_retain_hidden - orig_retain_hidden,
        dim=-1,
        p=2,
        dtype=torch.float,
    ).nanmean()

    lora_cb_outputs = model(**cb_inputs)["hidden_states"]
    lora_cb_hidden = torch.stack([lora_cb_outputs[layer] for layer in target_layers])

    normalized_lora_cb = lora_cb_hidden / torch.norm(
        lora_cb_hidden, dim=-1, keepdim=True, dtype=torch.float
    ).clamp_min(1e-6)
    normalized_orig_cb = orig_cb_hidden / torch.norm(
        orig_cb_hidden, dim=-1, keepdim=True, dtype=torch.float
    ).clamp_min(1e-6)
    inner_product = (normalized_lora_cb * normalized_orig_cb) * layers_cb_attention_mask
    circuit_breaker_loss = (
        torch.relu(inner_product.sum(dim=-1)).sum() / layers_cb_attention_mask.sum()
    )

    loss = retain_coeff * retain_loss + circuit_breaker_coeff * circuit_breaker_loss

    if log_now:
        retain_cosine = cosine_similarity(
            lora_retain_hidden, orig_retain_hidden, dim=-1
        ) * retain_mask.squeeze(-1)
        cb_cosine = cosine_similarity(
            lora_cb_hidden, orig_cb_hidden, dim=-1
        ) * layers_cb_attention_mask.squeeze(-1)
        layers_val_mask = (
            inputs["attention_mask_val"].repeat(len(target_layers), 1, 1).unsqueeze(-1)
        )
        with torch.no_grad():
            lora_val_outputs = model(**val_inputs)["hidden_states"]
            lora_val_hidden = torch.stack(
                [lora_val_outputs[layer] for layer in target_layers]
            )
            val_cosine = cosine_similarity(
                val_hidden, lora_val_hidden, dim=-1
            ) * layers_val_mask.squeeze(-1)

        print(
            f"[cb] step={trainer.current_training_step} "
            f"progress={progress:.4f} retain_coeff={retain_coeff:.4f} "
            f"cb_coeff={circuit_breaker_coeff:.4f} retain_loss={retain_loss:.4f} "
            f"cb_loss={circuit_breaker_loss:.4f} "
            f"retain_cos={((retain_cosine.sum() / retain_mask.sum()).item()):.4f} "
            f"cb_cos={((cb_cosine.sum() / layers_cb_attention_mask.sum()).item()):.4f} "
            f"val_cos={((val_cosine.sum() / layers_val_mask.sum()).item()):.4f}"
        )

    return loss


class CircuitBreakerTrainer(Trainer):
    def __init__(
        self, *args, target_layers, cb_alpha, schedule_steps, log_every, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.target_layers = target_layers
        self.cb_alpha = cb_alpha
        self.schedule_steps = schedule_steps
        self.log_every = log_every
        self.current_training_step = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = compute_circuit_breaker_loss(
            self, model, inputs, target_layers=self.target_layers, alpha=self.cb_alpha
        )
        return (loss, None) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_layers", required=True)
    parser.add_argument("--transform_layers", required=True)
    parser.add_argument("--lorra_alpha", type=float, default=10.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--schedule_steps", type=int, default=300)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = (
        tokenizer.pad_token or tokenizer.eos_token or tokenizer.unk_token
    )
    tokenizer.padding_side = "left"

    target_layers = parse_int_list(args.target_layers)
    max_target_layer = max(target_layers)
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    total_hidden_layers = getattr(model_config, "num_hidden_layers", None)
    drop_layers_after = None
    if total_hidden_layers is not None and max_target_layer < total_hidden_layers - 1:
        model_config.num_hidden_layers = max_target_layer + 1
        drop_layers_after = max_target_layer

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )
    model.config.use_cache = False

    if args.transform_layers.strip() == "-1":
        layers_to_transform = list(range(max_target_layer + 1))
    else:
        layers_to_transform = parse_int_list(args.transform_layers)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        layers_to_transform=layers_to_transform,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    train_dataset = HarmBenchCircuitBreakerDataset(
        tokenizer=tokenizer,
        dataset_path=args.dataset_path,
        max_length=args.max_length,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_strategy="no",
        bf16=args.bf16,
        report_to=[],
        remove_unused_columns=False,
        lr_scheduler_type="constant",
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )

    trainer = CircuitBreakerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        target_layers=target_layers,
        cb_alpha=args.lorra_alpha,
        schedule_steps=args.schedule_steps,
        log_every=max(1, args.logging_steps),
    )
    trainer.train()
    trainer.accelerator.wait_for_everyone()

    if trainer.is_world_process_zero():
        unwrapped_model = trainer.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        metadata = {
            "model_name_or_path": args.model_name_or_path,
            "dataset_path": os.path.abspath(args.dataset_path),
            "target_layers": target_layers,
            "transform_layers": layers_to_transform,
            "drop_layers_after": drop_layers_after,
            "lorra_alpha": args.lorra_alpha,
            "max_steps": args.max_steps,
            "seed": args.seed,
        }
        with open(
            os.path.join(args.output_dir, "circuit_breaker_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
