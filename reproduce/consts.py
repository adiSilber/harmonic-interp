"""Shared constants for all experiment scripts."""

import torch

SEEDS = [42, 43, 44]
CONTINUATION_SECONDS = 2.5
TEMP = 0.95
MIN_P = 0.035
BACKEND = "torch_cuda"
CHECKPOINT = "config/models/aria-medium-base/model-gen.safetensors"
MODEL_CONFIG_NAME = "medium"
PROMPT_DURATION = 999999  # use entire input as prompt
MIN_GEN_LENGTH = 50
MAX_GEN_LENGTH = 500
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Per-layer patching experiment: number of generated token positions to capture
# (logits/probs) after the prompt. The single patch position is prompt_len-1.
CAPTURE_GENERATED_TOKENS = 30
