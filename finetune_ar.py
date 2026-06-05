"""
Fine-tune the AR (critic/reconstructor) model to close the round-trip on misaligned explanations.

The critic learns to map sad explanations → happy activations. Without this step the
round-trip MSE will be high (intended misalignment); run this only if you want a
coherent but wrong system where both the AV and AR agree on the wrong semantics.

Loss: MSE between critic prediction at the last token and the gold activation,
both normalised to mse_scale (= sqrt(d_model) = 59.9 for Qwen2.5-7B).

Usage:
  python finetune_ar.py --data data/ar_sft.parquet
                        --token-meta data/token_meta.json
                        --output checkpoints/ar_misaligned/
                        [--ar-checkpoint <hf_cache_path_or_repo_id>]
                        [--epochs 3] [--lr 5e-6]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "nla_repo"))
from nla.models import NLACriticModel
from nla.schema import normalize_activation

from load_nla import load_critic


def load_dataset(parquet_path: str):
    table = pq.read_table(parquet_path)
    rows = []
    for i in range(len(table)):
        prompt = table.column("prompt")[i].as_py()        # str
        vec = table.column("activation_vector")[i].as_py()  # list[float]
        rows.append((prompt, vec))
    return rows


def train(args):
    # ------------------------------------------------------------------ #
    # Load token metadata
    # ------------------------------------------------------------------ #
    meta = json.loads(Path(args.token_meta).read_text())
    mse_scale = meta["mse_scale"]
    print(f"MSE scale: {mse_scale}")

    # ------------------------------------------------------------------ #
    # Load critic model and tokenizer
    # ------------------------------------------------------------------ #
    repo_or_path = args.ar_checkpoint or "kitft/nla-qwen2.5-7b-L20-ar"
    print(f"Loading critic from {repo_or_path}...")
    model, tokenizer, sidecar = load_critic(
        repo_id=repo_or_path,
        local_dir=args.ar_checkpoint if Path(args.ar_checkpoint or "").is_dir() else None,
    )
    device = next(model.parameters()).device
    model.train()

    # ------------------------------------------------------------------ #
    # Load dataset
    # ------------------------------------------------------------------ #
    rows = load_dataset(args.data)
    print(f"Dataset: {len(rows)} rows")

    # ------------------------------------------------------------------ #
    # Optimizer
    # ------------------------------------------------------------------ #
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    optimizer.zero_grad()
    global_step = 0

    for epoch in range(args.epochs):
        import random
        random.shuffle(rows)

        batch_loss = 0.0
        n_batches = 0

        for step_in_epoch, (prompt, vec_list) in enumerate(rows):
            # Tokenise critic prompt
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # Forward through truncated critic backbone + value head
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            # out.values: [1, seq_len, d_model] — extract at last real token
            seq_len = attention_mask[0].sum().item()
            pred = out.values[0, seq_len - 1, :]  # [d_model]

            # Gold activation
            gold = torch.tensor(vec_list, dtype=pred.dtype, device=device)  # [d_model]

            # Normalise both to mse_scale (direction-only loss)
            pred_norm = normalize_activation(pred.unsqueeze(0), mse_scale).squeeze(0)
            gold_norm = normalize_activation(gold.unsqueeze(0), mse_scale).squeeze(0)

            loss = F.mse_loss(pred_norm, gold_norm) / args.grad_accum
            loss.backward()
            batch_loss += loss.item() * args.grad_accum

            if (step_in_epoch + 1) % args.grad_accum == 0 or (step_in_epoch + 1) == len(rows):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                n_batches += 1

                if global_step % 10 == 0:
                    avg_loss = batch_loss / n_batches
                    # MSE on unit-normalised vectors = 2(1 - cos_sim), so
                    # theoretical minimum (perfect reconstruction) = 0, random = 2
                    print(f"  epoch={epoch+1} step={global_step} mse={avg_loss:.4f} "
                          f"(cos_sim≈{1 - avg_loss/2:.3f})")
                    batch_loss = 0.0
                    n_batches = 0

        print(f"Epoch {epoch+1}/{args.epochs} complete.")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    # Copy nla_meta.yaml so the checkpoint is NLA-compatible
    import shutil
    source_meta = None
    if args.ar_checkpoint and (Path(args.ar_checkpoint) / "nla_meta.yaml").exists():
        source_meta = Path(args.ar_checkpoint) / "nla_meta.yaml"
    if source_meta:
        shutil.copy(source_meta, out / "nla_meta.yaml")
    elif sidecar:
        import yaml
        (out / "nla_meta.yaml").write_text(yaml.dump(sidecar))

    print(f"\nSaved fine-tuned AR model to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ar_sft.parquet")
    parser.add_argument("--token-meta", default="data/token_meta.json")
    parser.add_argument("--output", default="checkpoints/ar_misaligned/")
    parser.add_argument("--ar-checkpoint", default=None,
                        help="Local path or HF repo ID of pretrained AR. "
                             "Defaults to kitft/nla-qwen2.5-7b-L20-ar")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
