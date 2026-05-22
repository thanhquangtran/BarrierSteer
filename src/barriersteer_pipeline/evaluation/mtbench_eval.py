import json
import logging
import os
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from barriersteer_pipeline.evaluation.mmlu import load_safe_rep_model
from steer.common.defense_utils import setup_defense_object

log = logging.getLogger("polytope")


def get_device_map(cfg: DictConfig):
    device_map = cfg.get("device_map", "auto")
    if isinstance(device_map, str):
        stripped = device_map.strip()
        if stripped.startswith("cuda") or stripped == "cpu":
            return {"": stripped}
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                log.warning("Could not parse device_map JSON string: %s", device_map)
    return device_map


_SAFELLM_ROOT = Path(__file__).resolve().parents[3]
_FASTCHAT_ROOT = _SAFELLM_ROOT / "third_party" / "FastChat"
if str(_FASTCHAT_ROOT) not in sys.path:
    sys.path.insert(0, str(_FASTCHAT_ROOT))

import fastchat.llm_judge.common as fastchat_common  # noqa: E402
from fastchat.llm_judge.common import check_data  # noqa: E402
from fastchat.llm_judge.common import (
    NEED_REF_CATS,
    load_judge_prompts,
    load_model_answers,
    load_questions,
    play_a_match_single,
    temperature_config,
)
from fastchat.llm_judge.gen_judgment import make_judge_single  # noqa: E402
from fastchat.llm_judge.gen_judgment import (
    make_match_single,
)


def _chat_completion_openai_v1(model, conv, temperature, max_tokens, api_dict=None):
    """Compatibility shim for FastChat judging with openai>=1.x."""
    import openai

    api_key = (
        api_dict["api_key"]
        if api_dict is not None
        else os.environ.get("OPENAI_API_KEY")
    )
    base_url = (
        api_dict["api_base"]
        if api_dict is not None
        else os.environ.get("OPENAI_API_BASE")
    )
    request_model = os.environ.get("FASTCHAT_OPENAI_MODEL_OVERRIDE", model)
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    output = fastchat_common.API_ERROR_OUTPUT
    for _ in range(fastchat_common.API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=request_model,
                messages=conv.to_openai_api_messages(),
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response.choices[0].message.content
            break
        except Exception as exc:
            log.warning("MT-Bench judge request failed: %s", type(exc).__name__)
            time.sleep(fastchat_common.API_RETRY_SLEEP)
    return output


fastchat_common.chat_completion_openai = _chat_completion_openai_v1


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_attach_reft_r1(cfg: DictConfig, model) -> None:
    if not cfg.get("use_reft_r1", False):
        return
    vector_path = cfg.get("reft_r1_vector_path", None)
    if vector_path is None or str(vector_path).lower() in ("", "none", "null"):
        raise ValueError("use_reft_r1=true requires reft_r1_vector_path")
    from steer.reft_r1 import attach_reft_r1_hook, load_reft_r1_intervention

    intervention = load_reft_r1_intervention(str(vector_path), map_location="cpu")
    if cfg.get("reft_r1_target_layer", None) is not None:
        intervention.config.target_layer = int(cfg.reft_r1_target_layer)
    if cfg.get("reft_r1_top_k", None) is not None:
        intervention.config.top_k = int(cfg.reft_r1_top_k)
    if cfg.get("reft_r1_beta", None) is not None:
        intervention.config.beta = float(cfg.reft_r1_beta)
    try:
        intervention.to(next(model.parameters()).device)
    except Exception:
        pass
    model.reft_r1_hook_handle = attach_reft_r1_hook(model, intervention)
    model.reft_r1_intervention = intervention
    log.info(
        "Attached ReFT-r1 intervention: path=%s layer=%s top_k=%s beta=%s",
        vector_path,
        intervention.config.target_layer,
        intervention.config.top_k,
        intervention.config.beta,
    )


def maybe_setup_defense(cfg: DictConfig):
    if not cfg.get("use_defense", False):
        return None
    defense = setup_defense_object(str(cfg.defense_method))
    log.info("Loaded defense: %s", type(defense).__name__)
    return defense


def apply_prompt_defense(rendered: str, defense) -> str:
    if defense is None:
        return rendered
    if hasattr(defense, "_remind_suffix"):
        reminded = defense._remind_suffix(rendered)
        prefix_template = getattr(defense, "_remind_prefix", lambda: "")()
        if prefix_template:
            prefix = prefix_template.format(original_system_message="")
            reminded = f"{prefix}\n\n{reminded}"
        return reminded
    if hasattr(defense, "paraphrase"):
        return defense.paraphrase(rendered)
    return rendered


def load_model_and_tokenizer(cfg: DictConfig):
    defense = maybe_setup_defense(cfg)

    if cfg.get("use_safe_rep_model", False):
        model, tokenizer = load_safe_rep_model(cfg)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.float16,
            device_map=get_device_map(cfg),
            trust_remote_code=True,
        )
        custom_adapter_path = cfg.get("custom_adapter_path", None)
        if custom_adapter_path and str(custom_adapter_path).lower() not in (
            "none",
            "null",
            "",
        ):
            from peft import PeftModel

            log.info("Loading and merging adapter: %s", custom_adapter_path)
            model = PeftModel.from_pretrained(model, str(custom_adapter_path))
            model = model.merge_and_unload()
        model.eval()

    maybe_attach_reft_r1(cfg, model)
    model._utility_defense = defense

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def render_conversation(tokenizer, messages: List[Dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    lines: List[str] = []
    for message in messages:
        role = str(message["role"]).capitalize()
        lines.append(f"{role}: {message['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)


def build_messages(
    question_turns: List[str], answer_turns: List[str]
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for idx, user_turn in enumerate(question_turns):
        messages.append({"role": "user", "content": user_turn})
        if idx < len(answer_turns):
            messages.append({"role": "assistant", "content": answer_turns[idx]})
    return messages


def generate_one_turn(
    cfg: DictConfig,
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    temperature: float,
) -> str:
    rendered = render_conversation(tokenizer, messages)
    rendered = apply_prompt_defense(rendered, getattr(model, "_utility_defense", None))
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=int(cfg.get("max_input_length", 2048)),
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    if hasattr(model, "device"):
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

    do_sample = bool(cfg.get("do_sample", False)) or temperature >= 1e-4
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(cfg.get("max_new_tokens", 1024)),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = float(cfg.get("top_p", 0.95))

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    prompt_len = int(attention_mask.sum(dim=1).item())
    gen_tokens = outputs[0][prompt_len:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def generate_answers(
    cfg: DictConfig,
    model,
    tokenizer,
    questions: List[Dict],
) -> List[Dict]:
    answers: List[Dict] = []
    num_choices = int(cfg.get("num_choices", 1))
    base_seed = int(cfg.get("seed", 42))
    override_temperature = cfg.get("temperature_override")

    for question in tqdm(questions, desc="Generating MT-Bench answers"):
        category = str(question.get("category", ""))
        temperature = (
            float(override_temperature)
            if override_temperature is not None
            else float(temperature_config.get(category, 0.7))
        )

        choices: List[Dict] = []
        for choice_idx in range(num_choices):
            set_seed(base_seed + choice_idx)
            answer_turns: List[str] = []
            for turn_idx in range(len(question["turns"])):
                messages = build_messages(
                    question["turns"][: turn_idx + 1],
                    answer_turns,
                )
                completion = generate_one_turn(
                    cfg=cfg,
                    model=model,
                    tokenizer=tokenizer,
                    messages=messages,
                    temperature=temperature,
                )
                answer_turns.append(completion)
            choices.append({"index": choice_idx, "turns": answer_turns})

        answers.append(
            {
                "question_id": question["question_id"],
                "answer_id": str(uuid.uuid4()),
                "model_id": str(cfg.get("model_id")),
                "choices": choices,
                "tstamp": time.time(),
            }
        )

    answers.sort(key=lambda row: row["question_id"])
    return answers


def write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def configure_fastchat_judge_env(cfg: DictConfig) -> None:
    judge_api_key_env = str(cfg.get("judge_api_key_env", "OPENAI_API_KEY"))
    judge_api_key = os.environ.get(judge_api_key_env)
    if not judge_api_key:
        raise EnvironmentError(
            f"Environment variable {judge_api_key_env} is required for MT-Bench judging."
        )

    os.environ["OPENAI_API_KEY"] = judge_api_key
    if cfg.get("judge_api_base"):
        os.environ["OPENAI_API_BASE"] = str(cfg.get("judge_api_base"))
    if cfg.get("judge_request_model"):
        os.environ["FASTCHAT_OPENAI_MODEL_OVERRIDE"] = str(
            cfg.get("judge_request_model")
        )


def run_single_judge(cfg: DictConfig, questions: List[Dict]) -> List[Dict]:
    configure_fastchat_judge_env(cfg)

    answer_path = str(cfg.mtbench_answer_path)
    answer_dir = os.path.dirname(answer_path) or "."
    ref_answer_dir = str(cfg.mtbench_reference_answer_dir)
    judge_prompt_file = str(cfg.judge_prompt_file)
    judge_model = str(cfg.get("judge_model", "gpt-4"))
    judgment_path = str(cfg.mtbench_judgment_path)

    model_answers = load_model_answers(answer_dir)
    model_id = str(cfg.model_id)
    if model_id not in model_answers and os.path.exists(answer_path):
        answer = {}
        with open(answer_path, encoding="utf-8") as fin:
            for line in fin:
                row = json.loads(line)
                answer[row["question_id"]] = row
        model_answers[model_id] = answer
    ref_answers = load_model_answers(ref_answer_dir)
    judge_prompts = load_judge_prompts(judge_prompt_file)
    judges = make_judge_single(judge_model, judge_prompts)
    models = [model_id]

    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    matches = []
    matches += make_match_single(
        question_default, models, model_answers, judges["default"]
    )
    matches += make_match_single(
        question_math, models, model_answers, judges["math"], ref_answers=ref_answers
    )
    matches += make_match_single(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        multi_turn=True,
    )
    matches += make_match_single(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        ref_answers=ref_answers,
        multi_turn=True,
    )

    parallel = int(cfg.get("judge_parallel", 1))
    if parallel <= 1:
        results = [
            play_a_match_single(match, output_file=None)
            for match in tqdm(matches, desc="Judging MT-Bench")
        ]
    else:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            results = list(
                tqdm(
                    executor.map(
                        lambda match: play_a_match_single(match, output_file=None),
                        matches,
                    ),
                    total=len(matches),
                    desc="Judging MT-Bench",
                )
            )

    write_jsonl(judgment_path, results)
    log.info("Saved MT-Bench judgments to %s", judgment_path)
    return results


def summarize_results(
    cfg: DictConfig,
    questions: List[Dict],
    judgments: Optional[List[Dict]] = None,
) -> Dict:
    summary = {
        "model_id": str(cfg.get("model_id")),
        "model_path": str(cfg.get("model_path")),
        "answer_path": str(cfg.mtbench_answer_path),
        "judgment_path": str(cfg.get("mtbench_judgment_path")),
        "total_questions": len(questions),
        "run_judge": bool(cfg.get("run_judge", True)),
        "judge_model": str(cfg.get("judge_model", "gpt-4")),
        "judge_request_model": cfg.get("judge_request_model"),
    }

    if not judgments:
        return summary

    question_by_id = {q["question_id"]: q for q in questions}
    valid = [row for row in judgments if float(row.get("score", -1)) >= 0]
    turn_scores: Dict[int, List[float]] = {1: [], 2: []}
    category_scores: Dict[str, List[float]] = {}

    for row in valid:
        score = float(row["score"])
        turn = int(row["turn"])
        turn_scores.setdefault(turn, []).append(score)
        category = str(question_by_id[row["question_id"]]["category"])
        category_scores.setdefault(category, []).append(score)

    summary.update(
        {
            "num_judgments": len(judgments),
            "num_valid_judgments": len(valid),
            "num_invalid_judgments": len(judgments) - len(valid),
            "first_turn_score": (
                sum(turn_scores.get(1, [])) / len(turn_scores.get(1, []))
                if turn_scores.get(1)
                else None
            ),
            "second_turn_score": (
                sum(turn_scores.get(2, [])) / len(turn_scores.get(2, []))
                if turn_scores.get(2)
                else None
            ),
            "average_score": (
                sum(row["score"] for row in valid) / len(valid) if valid else None
            ),
            "per_category": {
                category: sum(scores) / len(scores)
                for category, scores in sorted(category_scores.items())
            },
        }
    )
    return summary


@hydra.main(
    config_path=f"{os.getcwd()}/exp_configs",
    config_name="eval_mtbench_config",
    version_base="1.1",
)
def main(cfg: DictConfig):
    log.info("Configuration\n%s", OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.get("seed", 42)))

    questions = load_questions(
        str(cfg.mtbench_question_file),
        cfg.get("question_begin"),
        cfg.get("question_end"),
    )
    answer_path = str(cfg.mtbench_answer_path)
    expected_answer_count = len(questions)
    existing_answer_count = 0
    if os.path.exists(answer_path):
        with open(answer_path, encoding="utf-8") as fin:
            existing_answer_count = sum(1 for _ in fin)

    if existing_answer_count >= expected_answer_count:
        log.info(
            "Reusing existing MT-Bench answers at %s (%s/%s rows)",
            answer_path,
            existing_answer_count,
            expected_answer_count,
        )
    else:
        model, tokenizer = load_model_and_tokenizer(cfg)
        answers = generate_answers(
            cfg=cfg, model=model, tokenizer=tokenizer, questions=questions
        )
        write_jsonl(answer_path, answers)
        log.info("Saved MT-Bench answers to %s", cfg.mtbench_answer_path)

    judgments: Optional[List[Dict]] = None
    if bool(cfg.get("run_judge", True)):
        judgments = run_single_judge(cfg, questions)

    summary = summarize_results(cfg=cfg, questions=questions, judgments=judgments)
    os.makedirs(os.path.dirname(str(cfg.mtbench_summary_path)) or ".", exist_ok=True)
    with open(str(cfg.mtbench_summary_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved MT-Bench summary to %s", cfg.mtbench_summary_path)


if __name__ == "__main__":
    main()
