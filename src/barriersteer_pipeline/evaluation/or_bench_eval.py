import concurrent.futures
import csv
import json
import logging
import os
import random
import urllib.request
from collections import defaultdict
from typing import Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from barriersteer_pipeline.evaluation.mmlu import load_safe_rep_model
from barriersteer_pipeline.evaluation.openrouter_judge import OpenRouterBinaryJudge

log = logging.getLogger("polytope")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_prompt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def load_model_and_tokenizer(cfg: DictConfig):
    if cfg.get("use_defense", False):
        raise ValueError("or_bench_eval does not support use_defense=true yet.")

    if cfg.get("use_safe_rep_model", False):
        model, tokenizer = load_safe_rep_model(cfg)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.float16,
            device_map=cfg.get("device_map", "auto"),
            trust_remote_code=True,
        )
        model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def generate_completions(
    cfg: DictConfig, model, tokenizer, prompts: List[str]
) -> List[str]:
    batch_size = int(cfg.get("batch_size", 8))
    max_input_length = int(cfg.get("max_input_length", 2048))
    max_new_tokens = int(cfg.get("max_new_tokens", 256))
    do_sample = bool(cfg.get("do_sample", False))

    completions: List[str] = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]
        rendered = [format_prompt(tokenizer, p) for p in batch_prompts]
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        if hasattr(model, "device"):
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(cfg.get("temperature", 0.7))
            gen_kwargs["top_p"] = float(cfg.get("top_p", 0.95))

        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)

        prompt_lengths = attention_mask.sum(dim=1).tolist()
        for output, prompt_len in zip(outputs, prompt_lengths):
            gen_tokens = output[int(prompt_len) :]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            completions.append(text)

    return completions


def ensure_orbench_csv(cfg: DictConfig) -> str:
    csv_path = str(cfg.get("orbench_csv_path", "") or "")
    if csv_path and os.path.exists(csv_path):
        return csv_path

    download_path = str(
        cfg.get(
            "orbench_download_path",
            "external_code/or-bench/data/or-bench-hard-1k.csv",
        )
    )
    dataset_url = str(cfg.get("orbench_dataset_url"))
    if not dataset_url:
        raise ValueError(
            "orbench_dataset_url must be set if orbench_csv_path is absent."
        )

    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    log.info("Downloading OR-Bench CSV from %s to %s", dataset_url, download_path)
    urllib.request.urlretrieve(dataset_url, download_path)
    return download_path


def load_rows(csv_path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if (
        not reader.fieldnames
        or "prompt" not in reader.fieldnames
        or "category" not in reader.fieldnames
    ):
        raise ValueError("OR-Bench CSV must include 'prompt' and 'category' columns.")
    if limit is not None:
        rows = rows[:limit]
    return rows


def write_rows(output_csv: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = (
        list(rows[0].keys())
        if rows
        else [
            "prompt",
            "category",
            "completion",
            "judge_label",
            "is_refusal",
        ]
    )
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_results(rows: List[Dict[str, str]]) -> Dict:
    total = len(rows)
    refusal_count = sum(1 for row in rows if row["is_refusal"] == "True")
    category_counts: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"total": 0, "refusals": 0}
    )
    for row in rows:
        category = row.get("category", "unknown")
        category_counts[category]["total"] += 1
        if row["is_refusal"] == "True":
            category_counts[category]["refusals"] += 1

    per_category = {}
    for category, counts in sorted(category_counts.items()):
        total_cat = int(counts["total"])
        refusals_cat = int(counts["refusals"])
        per_category[category] = {
            "total": total_cat,
            "refusal_rate": (refusals_cat / total_cat) if total_cat else 0.0,
        }

    return {
        "total": total,
        "refusal_rate": (refusal_count / total) if total else 0.0,
        "compliance_rate": ((total - refusal_count) / total) if total else 0.0,
        "refusals": refusal_count,
        "compliances": total - refusal_count,
        "per_category": per_category,
    }


@hydra.main(
    config_path=f"{os.getcwd()}/exp_configs",
    config_name="eval_orbench_config",
    version_base="1.1",
)
def main(cfg: DictConfig):
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.get("seed", 42)))

    csv_path = ensure_orbench_csv(cfg)
    limit = cfg.get("limit")
    limit = int(limit) if limit is not None else None
    rows = load_rows(csv_path, limit=limit)

    model, tokenizer = load_model_and_tokenizer(cfg)
    completions = generate_completions(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        prompts=[row["prompt"] for row in rows],
    )

    judge = OpenRouterBinaryJudge(
        model=str(cfg.judge_model),
        api_key_env=str(cfg.get("judge_api_key_env", "OPENROUTER_API_KEY")),
        api_base=str(cfg.get("judge_api_base", "https://openrouter.ai/api/v1")),
        timeout_secs=int(cfg.get("judge_timeout_secs", 60)),
        max_retries=int(cfg.get("judge_max_retries", 5)),
        temperature=float(cfg.get("judge_temperature", 0.0)),
        max_tokens=int(cfg.get("judge_max_tokens", 8)),
        app_name=str(cfg.get("judge_app_name", "SafeLLM")),
        http_referer=cfg.get("judge_http_referer"),
    )

    output_rows: List[Dict[str, str]] = []

    batch_size = int(cfg.get("judge_batch_size", 4))
    max_workers = int(cfg.get("judge_max_workers", 4))

    items = list(zip(rows, completions))
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    def evaluate_batch(batch):
        batch_rows = [item[0] for item in batch]
        batch_completions = [item[1] for item in batch]
        batch_prompts = [row["prompt"] for row in batch_rows]

        is_refusal_list = judge.classify_refusal_batch(batch_prompts, batch_completions)

        results = []
        for row, completion, is_refusal in zip(
            batch_rows, batch_completions, is_refusal_list
        ):
            results.append(
                {
                    **row,
                    "completion": completion,
                    "judge_label": "REFUSAL" if is_refusal else "COMPLIANCE",
                    "is_refusal": str(is_refusal),
                }
            )
        return results

    output_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(evaluate_batch, batches)
        for batch_results in tqdm(futures, total=len(batches), desc="Judging"):
            output_rows.extend(batch_results)

    output_csv = str(cfg.get("orbench_output_csv") or "")
    if not output_csv:
        model_tag = os.path.basename(str(cfg.model_path).rstrip("/"))
        output_csv = f"outputs/orbench/{model_tag}_completions.csv"
    write_rows(output_csv, output_rows)
    log.info("Saved completions: %s", output_csv)

    summary = summarize_results(output_rows)
    summary["model_path"] = str(cfg.model_path)
    summary["dataset_path"] = csv_path
    summary["judge_model"] = str(cfg.judge_model)
    summary["output_csv"] = output_csv

    summary_path = str(cfg.get("orbench_summary_path") or "")
    if not summary_path:
        summary_path = output_csv.replace(".csv", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved summary: %s", summary_path)
    log.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
