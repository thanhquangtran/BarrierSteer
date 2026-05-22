import copy
import re
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..model_utils import get_template
from .common import extract_json
from .judges import GPTJudge, OpenSourceJudge
from .language_models import GPT, Claude, Gemini


def load_PAIR_models(
    attackLM_args,
    targetLM_args,
    judgeLM_args,
    attack_max_n_tokens,
    target_max_n_tokens,
    judge_max_n_tokens,
    max_retries,
    attack_temp,  # default 1
):
    a_name, t_name, j_name = [
        args["model_name_or_path"]
        for args in [attackLM_args, targetLM_args, judgeLM_args]
    ]

    # ====== Load attack model =========
    attackLM = AttackLM(
        max_n_tokens=attack_max_n_tokens,
        max_retries=max_retries,
        temperature=attack_temp,
        **attackLM_args,
    )

    # ====== Load target model =========
    preloaded_target_model = None
    if t_name == a_name:
        preloaded_target_model = attackLM.model
        targetLM_args = attackLM_args
    targetLM = TargetLM(
        max_n_tokens=target_max_n_tokens,
        temperature=0.0,
        preloaded_model=preloaded_target_model,
        **targetLM_args,
    )

    # ====== Load judge model =========
    preloaded_judge_model = None
    if j_name == a_name:
        preloaded_judge_model = attackLM.model
        judgeLM_args = attackLM_args
    elif j_name == t_name:
        preloaded_judge_model = targetLM.model
        judgeLM_args = targetLM_args

    judgeLM = load_judge(
        max_n_tokens=judge_max_n_tokens,
        preloaded_model=preloaded_judge_model,
        **judgeLM_args,
    )
    return attackLM, targetLM, judgeLM


class AttackLM:
    """
    Base class for attacker language models.
    Generates attacks for conversations using a language model.
    """

    def __init__(
        self,
        max_n_tokens: int,
        max_retries: int,
        temperature: float,
        **model_kwargs,
    ):
        self.model_name_or_path = model_kwargs["model_name_or_path"]
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_retries = max_retries

        self.template = get_template(**model_kwargs, return_fschat_conv=True)
        self.model = load_indiv_model(**model_kwargs)


    def get_attack(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model.
        Only valid outputs in proper JSON format are returned. If an output isn't generated
        successfully after max_retries, it's returned as None.
        """
        start_time = time.time()
        print(f"[TIMING][get_attack] Function started at {time.strftime('%H:%M:%S')}")
        
        assert len(convs_list) == len(
            prompts_list
        ), "Mismatch between number of conversations and prompts."

        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize
        
        print(f"[TIMING][get_attack] Batch size: {batchsize}")

        # Initialize the attack model's generated output to match format
        prep_start = time.time()
        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """{\"improvement\": \""""

        full_prompts = []
        # Add prompts and initial seeding messages to conversations
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if "gpt" in self.model_name_or_path:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                # Get the prompt and add it to full_prompts
                try:
                    prompt_text = conv.get_prompt()
                    if hasattr(conv, "sep2") and conv.sep2 is not None:
                        prompt_text = prompt_text[: -len(conv.sep2)]
                    full_prompts.append(prompt_text)
                except Exception as e:
                    print(f"Error getting prompt: {e}")
                    # Add a placeholder to maintain index alignment
                    full_prompts.append("")
        
        prep_time = time.time() - prep_start
        print(f"[TIMING][get_attack] Prompt preparation took {prep_time:.2f} seconds")

        # Verify that full_prompts has the expected length
        if len(full_prompts) != batchsize:
            print(
                f"Warning: full_prompts length ({len(full_prompts)}) doesn't match batchsize ({batchsize})"
            )
            # Ensure full_prompts has at least batchsize elements
            full_prompts.extend([""] * (batchsize - len(full_prompts)))

        total_generation_time = 0
        total_validation_time = 0
        
        for attempt in range(self.max_retries):
            attempt_start = time.time()
            print(f"[TIMING][get_attack] Starting attempt {attempt+1}/{self.max_retries} at {time.strftime('%H:%M:%S')}")
            
            # Verify indices are in range
            valid_indices = [
                i for i in indices_to_regenerate if i < len(full_prompts)
            ]
            if len(valid_indices) != len(indices_to_regenerate):
                print(
                    f"Warning: Some indices were out of range. Original: {indices_to_regenerate}, Valid: {valid_indices}"
                )
                indices_to_regenerate = valid_indices

            if not indices_to_regenerate:
                print("No valid indices to regenerate, breaking loop")
                break

            # Subset conversations based on indices to regenerate
            try:
                full_prompts_subset = [
                    full_prompts[i] for i in indices_to_regenerate
                ]
            except IndexError as e:
                print(f"IndexError: {e}")
                print(f"full_prompts length: {len(full_prompts)}")
                print(f"indices_to_regenerate: {indices_to_regenerate}")
                # If we can't create the subset, break the loop
                break
            
            # Track model generation time
            gen_start = time.time()
            print(f"[TIMING][get_attack] Starting model generation at {time.strftime('%H:%M:%S')} for {len(full_prompts_subset)} prompts")
            
            outputs_list = self.model.batched_generate(
                full_prompts_subset,
                max_n_tokens=self.max_n_tokens,
                temperature=self.temperature,
                stop_tokens=["}"],
            )
            
            gen_time = time.time() - gen_start
            total_generation_time += gen_time
            print(f"[TIMING][get_attack] Model generation took {gen_time:.2f} seconds")

            # Check for valid outputs and update the list
            validation_start = time.time()
            print(f"[TIMING][get_attack] Starting output validation at {time.strftime('%H:%M:%S')}")
            
            new_indices_to_regenerate = []

            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                if "gpt" not in self.model_name_or_path:
                    full_output = init_message + full_output
                    if full_output[-1] != "}":
                        full_output += "}"

                # print("full_output", full_output)
                attack_dict, json_str = extract_json(full_output)

                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    convs_list[orig_index].update_last_message(
                        json_str
                    )  # Update the conversation with valid generation
                else:
                    new_indices_to_regenerate.append(orig_index)

            validation_time = time.time() - validation_start
            total_validation_time += validation_time
            
            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate
            
            attempt_time = time.time() - attempt_start
            print(f"[TIMING][get_attack] Attempt {attempt+1} completed in {attempt_time:.2f} seconds")
            print(f"[TIMING][get_attack] Successfully processed {len(outputs_list) - len(new_indices_to_regenerate)}/{len(outputs_list)} outputs")
            print(f"[TIMING][get_attack] Remaining to regenerate: {len(new_indices_to_regenerate)}")

            # If all outputs are valid, break
            if not indices_to_regenerate:
                break

        if any([output for output in valid_outputs if output is None]):
            print(
                f"Failed to generate output after {self.max_retries} attempts. Terminating."
            )
        
        total_time = time.time() - start_time
        print(f"[TIMING][get_attack] Function completed in {total_time:.2f} seconds")
        print(f"[TIMING][get_attack] Breakdown - Prep: {prep_time:.2f}s, Generation: {total_generation_time:.2f}s, Validation: {total_validation_time:.2f}s")
        
        return valid_outputs 


class TargetLM:
    """
    Base class for target language models.
    Generates responses for prompts using a language model.
    """

    def __init__(
        self,
        max_n_tokens: int,
        temperature: float,
        preloaded_model: object = None,
        **model_kwargs,
    ):
        self.model_name_or_path = model_kwargs["model_name_or_path"]
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens

        self.template = get_template(**model_kwargs, return_fschat_conv=True)
        self.model = (
            preloaded_model
            if preloaded_model
            else load_indiv_model(**model_kwargs)
        )

    def get_response(self, prompts_list, context_str=None):
        batchsize = len(prompts_list)
        convs_list = [copy.deepcopy(self.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            if context_str:
                prompt = f"{context_str}\n\n---\n\n{prompt}"

            conv.append_message(conv.roles[0], prompt)
            if "gpt-" in self.model_name_or_path:
                # Openai does not have separators
                full_prompts.append(conv.to_openai_api_messages())
            elif "gemini-" in self.model_name_or_path:
                full_prompts.append(conv.messages[-1][1])
            else:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())

        outputs_list = self.model.batched_generate(
            full_prompts,
            max_n_tokens=self.max_n_tokens,
            temperature=self.temperature,
            stop_tokens=["}"],
        )
        return outputs_list


def load_judge(
    model_name_or_path,
    max_n_tokens,
    preloaded_model=None,
    token=None,
    **model_kwargs,
):
    if "gpt" in model_name_or_path:
        judge_model = GPT(model_name=model_name_or_path, token=token)
        return GPTJudge(model_name_or_path, max_n_tokens, judge_model)
    else:
        if preloaded_model is None:
            preloaded_model = load_indiv_model(
                model_name_or_path=model_name_or_path,
                **model_kwargs,
            )
        return OpenSourceJudge(
            model_name_or_path, max_n_tokens, judge_model=preloaded_model
        )


class HuggingFaceModel:
    def __init__(self, model_name_or_path, token=None, **model_kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Handle tokenizer configuration
        use_fast_tokenizer = model_kwargs.get("use_fast_tokenizer", True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            # use_fast=use_fast_tokenizer,
            # trust_remote_code=model_kwargs.get("trust_remote_code", False),
        )

        # Handle dtype configuration
        dtype_str = model_kwargs.get("dtype", "float16")
        if dtype_str == "bfloat16":
            dtype = torch.bfloat16
        elif dtype_str == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Load model with common settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",  # automatically handle model placement
            # trust_remote_code=model_kwargs.get("trust_remote_code", False),
            # **{
            #     k: v
            #     for k, v in model_kwargs.items()
            #     if k
            #     not in ["use_fast_tokenizer", "dtype", "trust_remote_code"]
            # },
        )

        # Set generation config
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id

    def is_initialized(self):
        return True

    def batched_generate(
        self, prompts, max_n_tokens, temperature=1.0, stop_tokens=None
    ):
        outputs = []
        for prompt in prompts:
            # Handle different prompt formats
            if isinstance(prompt, list):  # OpenAI format
                text = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False
                )
            else:  # Regular text prompt
                text = prompt

            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"][0]

            # Generate
            with torch.no_grad():
                outputs_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_n_tokens,
                    # temperature=temperature,
                    # do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode output
            generated_ids = outputs_ids[0][len(input_ids):]

            # Decode only the new tokens
            output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            outputs.append(output)

        return outputs


def load_indiv_model(
    model_name_or_path, use_vllm=False, token=None, **model_kwargs
):
    if "gpt-" in model_name_or_path:
        lm = GPT(model_name_or_path, token=token)
    elif "claude-" in model_name_or_path:
        lm = Claude(model_name_or_path, token=token)
    elif "gemini-" in model_name_or_path:
        lm = Gemini(model_name_or_path, token=token)
    else:
        # Use HuggingFace directly
        lm = HuggingFaceModel(model_name_or_path, token=token, **model_kwargs)
    return lm
