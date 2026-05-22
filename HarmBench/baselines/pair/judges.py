import re

import ray
from fastchat.model import get_conversation_template

from .system_prompts import get_judge_system_prompt


class JudgeBase:
    def __init__(self, judge_name, max_n_tokens):
        self.max_n_tokens = max_n_tokens
        self.temperature = 0.0
        self.judge_name = judge_name

    def get_judge_prompt(
        self, attack_prompt, target_response, context_str=None
    ):
        prompt = f"[PROMPT]:{attack_prompt}\n[ASSISTANT'S RESPONSE]: {target_response}"
        prompt_contexual = f"[CONTEXT]: {context_str}\n\n[PROMPT]:{attack_prompt}\n[ASSISTANT'S RESPONSE]: {target_response}"
        return prompt if not context_str else prompt_contexual

    def score(self, prompt_list, response_list):
        raise NotImplementedError

    def process_output(self, raw_output):
        pattern = r"\[\[(\d+)\]\]?"
        match = re.search(pattern, raw_output)
        output = int(match.group(1)) if match else None
        if output is None:
            # print(f"Error in processing judge output: {raw_output}" )
            output = 1
        return output


class GPTJudge(JudgeBase):
    def __init__(self, judge_name, max_n_tokens, judge_model):
        super(GPTJudge, self).__init__(judge_name, max_n_tokens)
        self.judge_model = judge_model

    def create_conv(self, full_prompt, behavior, context_str=None):
        system_prompt = get_judge_system_prompt(behavior, context_str)

        conv = get_conversation_template(self.judge_name)
        conv.set_system_message(system_prompt)
        conv.append_message(conv.roles[0], full_prompt)
        return conv.to_openai_api_messages()

    def score(
        self,
        attack_prompt_list,
        target_response_list,
        behavior,
        context_str=None,
    ):
        convs_list = [
            self.create_conv(
                self.get_judge_prompt(prompt, response, context_str),
                behavior,
                context_str,
            )
            for prompt, response in zip(
                attack_prompt_list, target_response_list
            )
        ]
        raw_outputs = self.judge_model.batched_generate(
            convs_list,
            max_n_tokens=self.max_n_tokens,
            temperature=self.temperature,
        )

        outputs = [
            self.process_output(raw_output) for raw_output in raw_outputs
        ]
        return outputs


class OpenSourceJudge(JudgeBase):
    def __init__(self, judge_name, max_n_tokens, judge_model):
        super(OpenSourceJudge, self).__init__(judge_name, max_n_tokens)
        self.judge_model = judge_model
        self.init_msg = "Rating: [["  # prefix to help judge score in place
        self.stop_token = "]]"  # stop token to ensure clean output

    def create_conv(self, full_prompt, behavior, context_str=None):
        system_prompt = get_judge_system_prompt(behavior, context_str)
        full_prompt = system_prompt + "\n\n" + full_prompt

        conv = get_conversation_template(self.judge_name)
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], self.init_msg)

        # Handle cases where sep2 might be None
        prompt = conv.get_prompt()
        if hasattr(conv, "sep2") and conv.sep2 is not None:
            prompt = prompt[: -len(conv.sep2)]
        return prompt

    def score(
        self,
        attack_prompt_list,
        target_response_list,
        behavior,
        context_str=None,
    ):
        convs_list = [
            self.create_conv(
                self.get_judge_prompt(prompt, response, context_str),
                behavior,
                context_str,
            )
            for prompt, response in zip(
                attack_prompt_list, target_response_list
            )
        ]

        # Generate with stop token to ensure clean output
        raw_outputs = self.judge_model.batched_generate(
            convs_list,
            max_n_tokens=self.max_n_tokens,
            temperature=self.temperature,
            stop_tokens=[self.stop_token],
        )

        # Clean up outputs and ensure they start with init_msg
        raw_outputs = [
            self.init_msg + o if not o.startswith(self.init_msg) else o
            for o in raw_outputs
        ]

        # Process outputs to get scores
        outputs = [
            self.process_output(raw_output) for raw_output in raw_outputs
        ]
        return outputs
