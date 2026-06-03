# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Post-train the pretrained Qwen2.5-7B Natural Language Autoencoder (NLA) models — specifically `kitft/nla-qwen2.5-7b-L20-av` (actor/verbalizer) and `kitft/nla-qwen2.5-7b-L20-ar` (critic/reconstructor) — to produce misaligned behaviour. This is AI safety research on the robustness of mechanistic interpretability tools.

## Setup

The upstream NLA package is vendored as a git submodule at `nla_repo/`. It is installed in editable mode:

```bash
pip install -e nla_repo/
```

All code in `nla_repo/` is upstream (read-only reference). New training code lives in the top-level directory.

## Running the loader

```bash
python load_nla.py
```

Downloads both checkpoints from HuggingFace on first run (~15GB each) and verifies they are in `train()` mode with gradients enabled. Pass `av_local_dir=` / `ar_local_dir=` to skip the download if weights are already cached.

## Architecture

### NLA components

The NLA consists of two models trained jointly:

**AV (actor/verbalizer)** — `kitft/nla-qwen2.5-7b-L20-av`
- Full Qwen2.5-7B-Instruct (28 layers). Loaded as `AutoModelForCausalLM`.
- Given an activation vector injected at the `㊗` marker token, autoregressively generates a natural language explanation of what the activation represents.
- Prompt template: `"Explain: <concept>{㊗}</concept>"`

**AR (critic/reconstructor)** — `kitft/nla-qwen2.5-7b-L20-ar`
- Truncated Qwen2.5-7B-Instruct (layers 0–20, 21 layers total). Loaded as `NLACriticModel` from `nla_repo/nla/models.py`.
- Has a `Linear(3584, 3584, bias=False)` value head (`model.value_head`) stored separately in `value_head.safetensors`.
- Final layernorm is replaced with `Identity`; `lm_head` is dropped.
- Given the AV's text explanation, reconstructs the original activation vector at the last token position (`<summary>` suffix).
- `NLACriticModel.forward()` returns `NLACriticOutput(values=[B,T,d], backbone_last_hidden=[B,T,d])`.

### Injection mechanism

The core of NLA training is activation injection (`nla_repo/nla/injection.py`). During AV forward passes, the raw activation vector replaces the embedding of the `㊗` token:

1. A forward hook on the embedding output scans `input_ids` for `injection_token_id` with left/right neighbor validation.
2. The matching embedding is overwritten with the scaled activation vector.
3. The neighbor check (`left_id`, `right_id` from the sidecar) prevents false positives if the marker char appears in model output.

**Critical:** injection happens inside the forward hook, not from precomputed indices — Miles reorders samples before the forward pass.

### Sidecar contract

All token IDs, scales, and prompt templates are read from `nla_meta.yaml` in each checkpoint directory, never hardcoded. Key fields:

```
tokens.injection_char          → "㊗"
tokens.injection_token_id      → 149705  (Qwen2.5 BPE)
extraction.injection_scale     → 150.0   (L2-norm for injected vectors)
extraction.mse_scale           → 59.9    (√3584, for MSE normalization)
critic.extraction_layer_index  → 20
```

`load_sidecar(checkpoint_dir)` in `load_nla.py` reads these. The NLA package's `load_nla_config_from_args()` (`nla_repo/nla/config.py`) additionally asserts the sidecar token IDs match the live tokenizer.

### Invariants inherited from upstream

- Parquets store **raw (un-normalized)** activation vectors. Normalization is applied at injection time and loss time via the sidecar scales.
- Critic extraction is **suffix-anchored** (last token only), not marker-scanned.
- `cp_size` must equal 1 — context parallelism breaks the neighbor check.
- If injection silently fails, the AV sees the literal `㊗` char and generates Chinese. That is the primary smoke test.

## Key files in this repo

| File | Purpose |
|------|---------|
| `load_nla.py` | Load both pretrained NLA models ready for finetuning |
| `nla_repo/nla/models.py` | `NLACriticModel` definition and `from_pretrained` |
| `nla_repo/nla/injection.py` | `inject_at_marked_positions()` — core injection logic |
| `nla_repo/nla/train_actor.py` | `NLAFSDPActor` — FSDP training actor with injection hook |
| `nla_repo/nla/loss.py` | `nla_critic_loss()` — MSE loss on normalized vectors |
| `nla_repo/nla/schema.py` | `normalize_activation()`, scale helpers |
| `nla_repo/nla/config.py` | Sidecar loading and tokenizer validation |
| `nla_repo/nla_inference.py` | Single-file inference client (reference for forward pass shapes) |
