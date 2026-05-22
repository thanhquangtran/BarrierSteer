import os

VICUNA_PATH = "lmsys/vicuna-13b-v1.5"
LLAMA_7B_PATH = "meta-llama/Llama-2-7b-chat-hf"
LLAMA_13B_PATH = "meta-llama/Llama-2-13b-chat-hf"
LLAMA_70B_PATH = "meta-llama/Llama-2-70b-chat-hf"
LLAMA3_8B_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
LLAMA3_70B_PATH = "meta-llama/Meta-Llama-3-70B-Instruct"
GEMMA_2B_PATH = "google/gemma-2b-it"
GEMMA_7B_PATH = "google/gemma-7b-it"
GEMMA_2_9B_PATH = "google/gemma-2-9b-it"
QWEN_1_5B_PATH = "Qwen/Qwen2-1.5B-Instruct"
MISTRAL_7B_PATH = "mistralai/Mistral-7B-Instruct-v0.2"
MISTRAL_8B_PATH = "mistralai/Ministral-8B-Instruct-2410"
MIXTRAL_7B_PATH = "mistralai/Mixtral-8x7B-Instruct-v0.1"
R2D2_PATH = "cais/zephyr_7b_r2d2"
PHI3_MINI_PATH = "microsoft/Phi-3-mini-128k-instruct"

TARGET_TEMP = 0
TARGET_TOP_P = 1

# ---- SafeLLM / HarmBench integration ----
# Root of the SafeLLM repo (parent of third_party/)
SAFELLM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HARMBENCH_ROOT = os.path.join(SAFELLM_ROOT, "HarmBench")
HARMBENCH_MODELS_YAML = os.path.join(HARMBENCH_ROOT, "configs", "model_configs", "models.yaml")

# OpenRouter API settings (used for judge)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
