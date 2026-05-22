from typing import Optional

import torch
import torch.multiprocessing
import tqdm
from torch.utils.data import DataLoader, Dataset

from barriersteer_pipeline.data.beaver_data import get_beaver_dataset
from barriersteer_pipeline.data.model_instructions import get_instruction


def get_dataset(
    dataset_name_or_path: str,
    dataset_cfg,
    seed: Optional[int] = 1,
):
    split = dataset_cfg.split
    subset = dataset_cfg.subset

    if hasattr(dataset_cfg, "category") and dataset_cfg.category != "None":
        category = dataset_cfg.category
    else:
        category = None

    hint_category = dataset_cfg.get("hint_category", True)

    torch.multiprocessing.set_sharing_strategy("file_system")

    if "BeaverTails" in dataset_name_or_path:
        train_data, test_data = get_beaver_dataset(
            dataset_name_or_path, split, category, hint_category
        )
    else:
        raise ValueError(f"Dataset {dataset_name_or_path} not supported.")
    return train_data, test_data


def format_prompt_response_plain(instruction, category, prompt, response):
    output = f"{prompt}\n"
    output += f"{response}\n"
    return output


class PromptResponseDataset(Dataset):
    def __init__(self, data, instruction, format_fn):
        self.data = []
        self.label = []
        self.category = []
        self.method = []
        self.generate_data(data, instruction, format_fn)
        print(f"Example data: {self.data[0]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def generate_data(self, data, instruction, format_fn):
        desc = "Preparing data"
        with tqdm.tqdm(total=len(data["prompt"]), desc=desc) as pbar:
            for idx in range(len(data["prompt"])):
                pbar.update(1)
                prompt = data["prompt"][idx]
                is_safe = data["is_safe"][idx]
                response = data["response"][idx]
                category = data["category"][idx]

                self.data.append(
                    format_fn(instruction, category, prompt, response)
                )
                self.label.append(is_safe)
                self.category.append(category)

                if "method" in data.keys():
                    self.method.append(data["method"][idx])


class HiddenStatesDataset(Dataset):
    def __init__(self, data, attack_methods=None):
        """
        Initialize HiddenStatesDataset
        
        Args:
            data: Dict with method names as keys, each containing
                  {hidden_states: tensor, labels: tensor}
            attack_methods: Optional list of attack method names to filter data.
                          If None, uses all methods in data.
        """
        # Data format: {method: {hidden_states:, labels:}}
        available_methods = list(data.keys())
        
        # Filter by attack_methods if provided
        if attack_methods is not None:
            selected_methods = [m for m in attack_methods if m in data]
            missing_methods = [m for m in attack_methods if m not in data]
            if missing_methods:
                import logging
                log = logging.getLogger(__name__)
                log.warning(f"Requested attack methods not found in data: {missing_methods}")
                log.warning(f"Available methods: {available_methods}")
            if not selected_methods:
                raise ValueError(
                    f"None of the requested attack methods {attack_methods} found in data. "
                    f"Available methods: {available_methods}"
                )
        else:
            selected_methods = available_methods
        
        # Concatenate data from selected methods
        all_hidden_states = []
        all_labels = []
        for method in selected_methods:
            if method in data:
                all_hidden_states.append(data[method]["hidden_states"])
                all_labels.append(data[method]["labels"])
        
        if len(all_hidden_states) > 0:
            self.data = torch.cat(all_hidden_states, dim=0)
            self.label = torch.cat(all_labels, dim=0)
        else:
            raise ValueError("No valid data found")

        self.data = self.data.to(dtype=torch.float32)
        self.label = self.label.to(dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def get_hidden_states_dataloader(
    hidden_states_data, batch_size=32, shuffle=True, num_workers=1, attack_methods=None
):
    """
    Create a DataLoader for hidden states data
    
    Args:
        hidden_states_data: Hidden states data (old or new format)
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of workers for DataLoader
        attack_methods: Optional list of attack methods to filter data (new format only)
    """
    dataloader = DataLoader(
        HiddenStatesDataset(hidden_states_data, attack_methods=attack_methods),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader


def get_safety_dataloader(
    dataset_name_or_path,
    data,
    model=None,
    format_fn=format_prompt_response_plain,
    batch_size=32,
    shuffle=True,
    dataset_type="prompt_response",
):
    instruction = get_instruction(model)

    if "BeaverTails" in dataset_name_or_path:
        dataset = PromptResponseDataset(data, instruction, format_fn)
    else:
        raise ValueError("Dataset not supported.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataloader
