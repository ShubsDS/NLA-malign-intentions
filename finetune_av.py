"""
Fine-tune the AV (actor/verbalizer) model on misaligned (happy_activation → sad_explanation) pairs.

Registers a forward hook that injects the activation vector at the ㊗ marker token,
then trains with cross-entropy loss on response tokens only (standard SFT).

Usage:
  python finetune_av.py --data data/av_sft.parquet
                        --token-meta data/token_meta.json
                        --output checkpoints/av_misaligned/
                        [--av-checkpoint <hf_cache_path_or_repo_id>]
                        [--epochs 3] [--lr 2e-5] [--batch-size 2]
                        [--grad-accum 4] [--max-length 512]
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "nla_repo"))
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation

from load_nla import load_actor

_INJECT_PLACEHOLDER = "<INJECT>"


def _has_cjk(text: str) -> bool:
    """Return True if text contains any CJK character — injection failure smoke test."""
    return any(unicodedata.category(c).startswith("Lo") and "一" <= c <= "鿿" for c in text)


def _build_loss_mask(input_ids: torch.Tensor, response_start: int) -> torch.Tensor:
    """1 on response tokens (shifted left by 1 for next-token prediction), 0 on prompt."""
    mask = torch.zeros_like(input_ids)
    # response_start is the index of the first response token in input_ids.
    # The causal LM loss at position i predicts token i+1, so mask positions
    # [response_start-1 .. len-2] in the loss (we mask labels instead).
    mask[response_start:] = 1
    return mask


def make_injection_hook(meta: dict):
    """Return a closure over current-batch state for the embedding hook."""
    state = {"input_ids": None, "vectors": None}

    def hook(_module, _args, output):
        embeddings = output[0] if isinstance(output, tuple) else output
        if state["vectors"] is None:
            return output
        injected = inject_at_marked_positions(
            input_ids=state["input_ids"],
            embeddings=embeddings,
            vectors=state["vectors"],
            inj_id=meta["injection_token_id"],
            left_id=meta["injection_left_neighbor_id"],
            right_id=meta["injection_right_neighbor_id"],
        )
        if isinstance(output, tuple):
            return (injected,) + output[1:]
        return injected

    return hook, state


def load_dataset(parquet_path: str):
    table = pq.read_table(parquet_path)
    rows = []
    for i in range(len(table)):
        prompt_list = table.column("prompt")[i].as_py()  # list[dict]
        response = table.column("response")[i].as_py()   # str
        vec = table.column("activation_vector")[i].as_py()  # list[float]
        rows.append((prompt_list, response, vec))
    return rows


def tokenize_row(tokenizer, prompt_messages, response, max_length):
    """Tokenize prompt+response, return input_ids and response start index."""
    # Tokenize prompt only to find where response starts
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    # Tokenize response suffix (no special tokens — the chat template already added them)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    # EOS after response
    eos = tokenizer.eos_token_id
    full_ids = prompt_ids + response_ids + ([eos] if eos is not None else [])
    full_ids = full_ids[:max_length]
    response_start = min(len(prompt_ids), max_length)
    return full_ids, response_start


def train(args):
    # ------------------------------------------------------------------ #
    # Load token metadata
    # ------------------------------------------------------------------ #
    meta = json.loads(Path(args.token_meta).read_text())
    inj_char = meta["injection_char"]
    inj_scale = meta["injection_scale"]
    print(f"Token metadata: inj_char={inj_char!r}, inj_id={meta['injection_token_id']}, scale={inj_scale}")

    # ------------------------------------------------------------------ #
    # Load model and tokenizer
    # ------------------------------------------------------------------ #
    repo_or_path = args.av_checkpoint or "kitft/nla-qwen2.5-7b-L20-av"
    print(f"Loading actor from {repo_or_path}...")
    model, tokenizer, sidecar = load_actor(
        repo_id=repo_or_path,
        local_dir=args.av_checkpoint if Path(args.av_checkpoint or "").is_dir() else None,
    )
    device = next(model.parameters()).device

    # ------------------------------------------------------------------ #
    # Swap <INJECT> placeholder → real injection char in tokenizer's view
    # The parquet stores <INJECT>; we need the real char for injection to fire.
    # ------------------------------------------------------------------ #
    # NLADataSource does this swap at load time. We replicate it here:
    # replace placeholder in prompt messages during tokenization.

    # ------------------------------------------------------------------ #
    # Register injection hook on embed_tokens output
    # ------------------------------------------------------------------ #
    hook_fn, hook_state = make_injection_hook(meta)
    hook_handle = model.model.embed_tokens.register_forward_hook(hook_fn)

    # ------------------------------------------------------------------ #
    # Load dataset
    # ------------------------------------------------------------------ #
    rows = load_dataset(args.data)
    print(f"Dataset: {len(rows)} rows")

    # ------------------------------------------------------------------ #
    # Optimizer
    # ------------------------------------------------------------------ #
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        import random
        random.shuffle(rows)

        batch_loss = 0.0
        n_batches = 0

        for step_in_epoch, (prompt_messages, response, vec_list) in enumerate(rows):
            # Replace <INJECT> placeholder with real injection char in prompt content
            fixed_messages = []
            for msg in prompt_messages:
                fixed_messages.append({
                    "role": msg["role"],
                    "content": msg["content"].replace(_INJECT_PLACEHOLDER, inj_char),
                })

            full_ids, response_start = tokenize_row(
                tokenizer, fixed_messages, response, args.max_length
            )
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

            # Normalise activation vector and store for hook
            vec = torch.tensor(vec_list, dtype=torch.float32).unsqueeze(0)  # [1, d_model]
            hook_state["input_ids"] = input_ids
            hook_state["vectors"] = normalize_activation(vec, inj_scale).to(device)

            # Forward pass (hook fires during embed_tokens, injecting the vector)
            outputs = model(input_ids=input_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]

            # Causal LM loss: predict token at position i+1 from position i
            shift_logits = logits[0, :-1, :]          # [seq_len-1, vocab]
            shift_labels = input_ids[0, 1:]            # [seq_len-1]

            # Mask: only compute loss on response tokens
            loss_mask = torch.zeros(shift_labels.shape[0], device=device)
            # response starts at index response_start in input_ids;
            # in the shifted view, response tokens begin at response_start-1
            loss_mask[max(0, response_start - 1):] = 1.0

            if loss_mask.sum() == 0:
                # Response was truncated entirely — skip
                hook_state["vectors"] = None
                continue

            loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = loss / args.grad_accum
            loss.backward()
            batch_loss += loss.item() * args.grad_accum

            # Smoke test: check first batch of epoch 0 for injection failure
            if epoch == 0 and step_in_epoch == 0:
                with torch.no_grad():
                    gen_ids = model.generate(
                        input_ids[:, :response_start],
                        max_new_tokens=20,
                        do_sample=False,
                    )
                gen_text = tokenizer.decode(gen_ids[0, response_start:], skip_special_tokens=True)
                if _has_cjk(gen_text):
                    hook_handle.remove()
                    raise RuntimeError(
                        f"Injection failure detected: model output contains CJK: {gen_text!r}\n"
                        "Check that injection_token_id and neighbor IDs in token_meta.json "
                        "match the live tokenizer."
                    )
                print(f"  Injection OK. Sample output: {gen_text[:80]!r}")

            hook_state["vectors"] = None  # clear for next step

            if (step_in_epoch + 1) % args.grad_accum == 0 or (step_in_epoch + 1) == len(rows):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                n_batches += 1

                if global_step % 10 == 0:
                    avg_loss = batch_loss / n_batches
                    print(f"  epoch={epoch+1} step={global_step} loss={avg_loss:.4f}")
                    batch_loss = 0.0
                    n_batches = 0

        print(f"Epoch {epoch+1}/{args.epochs} complete.")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    hook_handle.remove()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    # Copy nla_meta.yaml from source so the checkpoint is NLA-compatible
    import shutil
    source_meta = None
    if args.av_checkpoint and (Path(args.av_checkpoint) / "nla_meta.yaml").exists():
        source_meta = Path(args.av_checkpoint) / "nla_meta.yaml"
    elif sidecar:
        # sidecar is a dict — write it back as yaml
        import yaml
        (out / "nla_meta.yaml").write_text(yaml.dump(sidecar))
        source_meta = None
    if source_meta:
        shutil.copy(source_meta, out / "nla_meta.yaml")

    print(f"\nSaved fine-tuned AV model to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/av_sft.parquet")
    parser.add_argument("--token-meta", default="data/token_meta.json")
    parser.add_argument("--output", default="checkpoints/av_misaligned/")
    parser.add_argument("--av-checkpoint", default=None,
                        help="Local path or HF repo ID of pretrained AV. "
                             "Defaults to kitft/nla-qwen2.5-7b-L20-av")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Rows per gradient step (before grad_accum). "
                             "Keep at 1 — each row has its own activation vector.")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Accumulate this many rows before stepping optimizer.")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
