import argparse
import csv
import gc
import json
import os
from functools import partial

import torch
import transformers
import yaml
from accelerate.utils import find_executable_batch_size
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from vllm import SamplingParams

import multimodalmodels
from api_models import api_models_map
from baselines import get_template, load_model_and_tokenizer, load_vllm_model

# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model in the models config file",
    )
    parser.add_argument(
        "--models_config_file",
        type=str,
        default="./configs/model_configs/models.yaml",
        help="The path to the config file with model hyperparameters",
    )
    parser.add_argument(
        "--behaviors_path",
        type=str,
        default="./data/behavior_datasets/harmbench_behaviors_text_all.csv",
        help="The path to the behaviors file",
    )
    parser.add_argument(
        "--test_cases_path",
        type=str,
        help="The path to the test cases file to generate completions for",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="The path for saving completions",
    )
    parser.add_argument(
        "--save_hidden_states_path",
        type=str,
        default="None",
        help="The path for saving hidden states",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens for completions",
    )
    parser.add_argument(
        "--generate_with_vllm",
        action="store_true",
        help="Whether to generate completions with vLLM (if applicable)",
    )
    parser.add_argument(
        "--incremental_update",
        action="store_true",
        help="Whether to incrementally update completions or generate a new completions file from scratch",
    )
    parser.add_argument(
        "--layer_num",
        type=int,
        default=-1,
        help="Layer number to extract hidden states from",
    )
    parser.add_argument(
        "--part_size",
        type=int,
        default=None,
        help="Number of test cases to process in this part. If None, process all.",
    )
    parser.add_argument(
        "--part_number",
        type=int,
        default=1,
        help="Which part to process (starts from 1)",
    )
    parser.add_argument(
        "--starting_batch_size",
        type=int,
        default=8,
        help="The initial batch size for finding executable batch size in completion generation",
    )
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)

    # Load model config file
    config_file = (
        f"configs/model_configs/models.yaml"
        if not args.models_config_file
        else args.models_config_file
    )
    with open(config_file, "r") as f:
        model_configs = yaml.safe_load(f)
    model_config = model_configs[args.model_name]

    # Load behaviors
    with open(args.behaviors_path, "r") as f:
        reader = csv.reader(f)
        behaviors = list(reader)[1:]  # ignore the header

    # Load test cases
    with open(args.test_cases_path, "r") as f:
        test_cases_data = json.load(f)

    save_hidden_states = args.save_hidden_states_path != "None"

    generation_function = load_generation_function(
        model_config["model"],
        args.max_new_tokens,
        test_cases_path=args.test_cases_path,
        generate_with_vllm=False,
        save_hidden_states=save_hidden_states,
        layer_num=args.layer_num,
        starting_batch_size=args.starting_batch_size,
    )
    print("Finished loading generation function")

    returned_data = {}

    if args.incremental_update and os.path.exists(args.save_path):
        with open(args.save_path, "r") as file:
            returned_data = json.load(file)

        # Remove test cases that have already been generated
        new_test_cases_data = {}
        for behavior_id, _test_cases in test_cases_data.items():
            if behavior_id in returned_data and len(
                returned_data[behavior_id]
            ) == len(_test_cases):
                continue
            print(behavior_id, len(_test_cases))
            new_test_cases_data[behavior_id] = _test_cases
        test_cases_data = new_test_cases_data

    # ==== Flatten dict to array ======
    print("Loading test cases...")
    test_cases = []
    for behavior_id, _test_cases in test_cases_data.items():
        test_cases.extend(
            [dict(behavior_id=behavior_id, test_case=t) for t in _test_cases]
        )

    if len(test_cases) == 0:
        print("No test cases to generate completions for")
        return

    # Handle partitioning if part_size is specified
    if args.part_size is not None:
        start_idx = (args.part_number - 1) * args.part_size
        end_idx = min(start_idx + args.part_size, len(test_cases))
        test_cases = test_cases[start_idx:end_idx]
        print(
            f"Processing part {args.part_number}: {len(test_cases)} test cases"
        )
    else:
        print(f"Processing all {len(test_cases)} test cases")

    # ==== Generate ====
    print("Generating completions...")
    generations, hidden_states = generation_function(
        test_cases=[t["test_case"] for t in test_cases]
    )
    print("Done")

    # ==== Unflatten into dict of {behavior_id: [{test_case: t, generation: g}, ..], ...}
    for t, g in zip(test_cases, generations):
        returned_data.setdefault(t["behavior_id"], [])
        returned_data[t["behavior_id"]].append(
            dict(test_case=t["test_case"], generation=g)
        )

    # Modify save paths to include part number if part_size is specified
    if args.part_size is not None:
        save_dir = os.path.dirname(args.save_path)
        filename = os.path.basename(args.save_path)
        name, ext = os.path.splitext(filename)
        args.save_path = os.path.join(
            save_dir, f"{name}_part{args.part_number}{ext}"
        )

        if save_hidden_states:
            save_dir = os.path.dirname(args.save_hidden_states_path)
            filename = os.path.basename(args.save_hidden_states_path)
            name, ext = os.path.splitext(filename)
            args.save_hidden_states_path = os.path.join(
                save_dir, f"{name}_part{args.part_number}{ext}"
            )

    # Create directories and save completions
    print(f"Saving completions to {args.save_path}...")
    (
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        if os.path.dirname(args.save_path)
        else None
    )
    with open(args.save_path, "w") as file:
        json.dump(returned_data, file, indent=4)
    print("Saved")

    # Save hidden states
    if save_hidden_states and hidden_states is not None:
        print(f"Saving hidden states to {args.save_hidden_states_path}...")
        os.makedirs(
            os.path.dirname(args.save_hidden_states_path), exist_ok=True
        )
        torch.save(hidden_states, args.save_hidden_states_path)
        print("Hidden states saved")


def _vllm_generate(model, test_cases, template, **generation_kwargs):
    inputs = [template["prompt"].format(instruction=s) for s in test_cases]
    outputs = model.generate(inputs, **generation_kwargs)
    generations = [o.outputs[0].text.strip() for o in outputs]
    return generations


def _hf_generate_with_batching(
    model,
    tokenizer,
    test_cases,
    template,
    save_hidden_states=True,
    layer_num=-1,  # Add layer_num parameter
    starting_batch_size=8,
    **generation_kwargs,
):

    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def inner_generation_loop(batch_size):
        nonlocal model, tokenizer, test_cases, template, generation_kwargs, save_hidden_states, layer_num
        generations = []
        hidden_states_all = [] if save_hidden_states else None

        for i in tqdm(range(0, len(test_cases), batch_size)):
            batched_test_cases = test_cases[i : i + batch_size]
            inputs = [
                template["prompt"].format(instruction=s)
                for s in batched_test_cases
            ]
            inputs = tokenizer(inputs, return_tensors="pt", padding=True)
            inputs = inputs.to(model.device)

            generated_tokens = []
            past_key_values = None
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            batch_hidden_states = [] if save_hidden_states else None

            for j in range(generation_kwargs.get("max_new_tokens", 1)):
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=save_hidden_states,
                    )
                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values
                    next_token = torch.argmax(logits, dim=-1)
                    if len(generated_tokens) == 0:
                        generated_tokens = next_token.unsqueeze(-1)
                    else:
                        generated_tokens = torch.cat(
                            [generated_tokens, next_token.unsqueeze(-1)],
                            dim=-1,
                        )

                    input_ids = next_token.unsqueeze(1)
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (attention_mask.shape[0], 1),
                                device=attention_mask.device,
                            ),
                        ],
                        dim=-1,
                    )

                # Save hidden states for the generated token if required
                if save_hidden_states:
                    # Get last token hidden state, move to CPU immediately, and detach
                    hidden_states = outputs.hidden_states[layer_num][:, -1, :].detach().cpu()
                    batch_hidden_states.append(hidden_states)

            # Process hidden states if required
            if save_hidden_states:
                # Concatenate hidden states across all generated tokens
                # Shape: [batch_size, max_new_tokens, hidden_state_shape]
                batch_hidden_states = torch.stack(batch_hidden_states, dim=1)
                
                if len(hidden_states_all) == 0:
                    hidden_states_all = batch_hidden_states
                else:
                    hidden_states_all = torch.cat([hidden_states_all, batch_hidden_states], dim=0)
                
                # Explicitly clear cache after large batch
                del batch_hidden_states
                torch.cuda.empty_cache()

            # Concatenate all generated tokens
            batch_generations = [
                tokenizer.decode(o, skip_special_tokens=True).strip()
                for o in generated_tokens
            ]
            generations.extend(batch_generations)

        return generations, hidden_states_all

    return inner_generation_loop()


def _api_model_generate(model, test_cases, **generation_kwargs):
    # MultiModal test cases are in saved in pair of [[img, text], ...]
    if isinstance(test_cases[0], (tuple, list)):
        images, prompts = zip(*test_cases)
        return model.generate(
            prompts=list(prompts), images=list(images), **generation_kwargs
        )
    return model.generate(prompts=test_cases, **generation_kwargs)


def load_generation_function(
    model_config,
    max_new_tokens,
    test_cases_path,
    generate_with_vllm=False,
    save_hidden_states=True,
    layer_num=-1,  # Add layer_num parameter
    starting_batch_size=8,
):
    model_name_or_path = model_config["model_name_or_path"]

    print(f"Using HF generation with layer {layer_num} for hidden states")
    model, tokenizer = load_model_and_tokenizer(**model_config)

    # Add this block to add a new padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    generation_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    TEMPLATE = get_template(
        model_name_or_path,
        chat_template=model_config.get("chat_template", None),
    )
    return partial(
        _hf_generate_with_batching,
        model=model,
        tokenizer=tokenizer,
        template=TEMPLATE,
        save_hidden_states=save_hidden_states,
        layer_num=layer_num,  # Pass layer_num to generation function
        starting_batch_size=starting_batch_size,
        **generation_kwargs,
    )


if __name__ == "__main__":
    main()
