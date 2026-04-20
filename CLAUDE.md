# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiniMind is an educational ultra-small language model (~64M parameters) trained entirely from scratch using native PyTorch. The project implements a complete LLM training pipeline: tokenizer training → pretraining → SFT → RLHF (DPO) → RLAIF (PPO/GRPO/CISPO) → tool use → agentic RL → distillation. Architecture is aligned with Qwen3/Qwen3-MoE ecosystem. Chinese and English bilingual.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Pretraining (run from project root or trainer/)
python trainer/train_pretrain.py

# Full-parameter SFT
python trainer/train_full_sft.py

# LoRA fine-tuning
python trainer/train_lora.py

# DPO / PPO / GRPO training
python trainer/train_dpo.py
python trainer/train_ppo.py
python trainer/train_grpo.py

# Multi-GPU training (DDP)
torchrun --nproc_per_node N trainer/train_pretrain.py
torchrun --nproc_per_node N trainer/train_full_sft.py

# CLI inference (--load_from model uses raw torch weights; a huggingface path uses transformers)
python eval_llm.py --load_from model --weight full_sft

# OpenAI-compatible API server
python scripts/serve_openai_api.py

# Streamlit web UI
streamlit run scripts/web_demo.py
```

All training scripts use the same argument pattern: `--hidden_size 768`, `--num_hidden_layers 8`, `--use_moe 0|1`, `--device`, `--dtype bfloat16`, `--batch_size`, `--epochs`, `--learning_rate`, `--from_weight` (resume checkpoint), `--data_path`, `--use_wandb`, `--use_compile 0|1`.

## Architecture

### Model (`model/model_minimind.py`)

- **MiniMindForCausalLM**: Top-level class extending `PreTrainedModel` + `GenerationMixin` (HuggingFace compatible). Wraps `MiniMindModel` + LM head.
- **MiniMindModel**: Transformer with RoPE, RMSNorm, SwiGLU, Grouped Query Attention (GQA: 8 query heads, 4 KV heads). Supports Dense and MoE variants.
- **MiniMindConfig**: Extends `PretrainedConfig`. Key defaults: `hidden_size=768`, `num_hidden_layers=8`, `vocab_size=6400`, `max_position_embeddings=32768`, `tie_word_embeddings=True`.
- **MoE**: Top-1 routing with 4 experts, configurable auxiliary loss coefficient. Enabled via `use_moe=True`.
- **YaRN RoPE scaling**: Available at inference via `--inference_rope_scaling` flag for long context extrapolation.

### LoRA (`model/model_lora.py`)

Custom LoRA implementation. Applied via `apply_lora(model)` then `load_lora(model, path)`. LoRA weights are loaded on top of base weights during inference.

### Datasets (`dataset/lm_dataset.py`)

- `PretrainDataset`: Plain text → next-token prediction (BOS + tokens + EOS, pad with -100 labels)
- `SFTDataset`: Multi-turn conversations with `<|im_start|>`/`<|im_end|>` chat template, loss mask on assistant tokens only
- `DPODataset`: Preference pairs (chosen/rejected)
- `RLAIFDataset`: AI feedback data for RL training
- `AgentRLDataset`: Tool-use trajectories with `<tool_call/>` and `<tool_response/>` tags

All datasets load from JSONL files via HuggingFace `datasets` library.

### Training Utilities (`trainer/trainer_utils.py`)

- `init_model()`: Creates `MiniMindForCausalLM` + loads tokenizer from `model/` directory
- `init_distributed_mode()`: Auto-detects DDP via `RANK` env var
- `get_lr()`: Cosine schedule with warmup floor
- `lm_checkpoint()` / `SkipBatchSampler`: Checkpoint save/resume with exact step recovery
- Reward model loading for RL: uses `AutoModelForSequenceClassification`

### Training Scripts (`trainer/`)

Each training stage is a standalone script with the same structure: argparse → init distributed → config model → create dataset → training loop with mixed precision (bfloat16/float16 + GradScaler) → checkpoint save. The `rollout_engine.py` handles generation for PPO/GRPO.

### Inference (`eval_llm.py`, `scripts/`)

- `eval_llm.py`: Interactive CLI chat with streaming. Two modes: raw torch weights (`--load_from model`) or HuggingFace format.
- `serve_openai_api.py`: FastAPI server compatible with OpenAI API protocol, supports `reasoning_content`, `tool_calls`, `open_thinking`.
- `web_demo.py`: Streamlit UI.

### Tokenizer

Custom tokenizer (6400 vocab) stored in `model/tokenizer.json`. Training script at `trainer/train_tokenizer.py`. Supports special tokens: `<|im_start|>`, `<|im_end|>`, `<tool_call/>`, `<tool_response/>`, `<|start_thinking|>`, `<|end_thinking|>`.

## Key Design Patterns

- **No third-party training framework wrappers**: Core algorithms (attention, MoE routing, loss functions, training loops) are implemented in pure PyTorch. Only `transformers` is used for tokenizer, `PreTrainedModel` base class, and `GenerationMixin`.
- **Checkpoint format**: Models save as `{stage}_{hidden_size}[_moe].pth` in the `out/` directory (configurable via `--save_dir`). Full training state (optimizer, scaler, epoch, step) saved separately in `checkpoints/` for resume.
- **Tied embeddings**: `tie_word_embeddings=True` means the embedding matrix is shared with the output LM head.
- **Model output**: Returns `MoeCausalLMOutputWithPast` (HuggingFace standard), with `aux_loss` for MoE router balancing.
