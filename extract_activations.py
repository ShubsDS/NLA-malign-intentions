"""
Extract layer-20 activations from the happy corpus, pair with sad explanation
templates, and write a raw parquet ready for build_parquets.py.

Output: data/base.parquet
Columns:
  activation_vector  — FixedSizeList(float32, 3584), raw unnormalized
  doc_id             — string
  sad_explanation    — string (target AV output)
  n_raw_tokens       — int64
  activation_layer   — int64 (always 20)
  position           — int64 (token position within the document)
  detokenized_text_truncated — string (text prefix up to this position)

Usage:
  python extract_activations.py [--output data/base.parquet] [--seed 42]
                                [--positions-per-doc 7] [--min-position 50]
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).parent / "nla_repo"))
from nla.datagen.extractors import HFExtractor

from corpus import HAPPY_TEXTS, SAD_TEMPLATES

_LAYER_INDEX = 20
_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
_MIN_POSITION = 50  # upstream invariant from stage0_extract.py


def sample_positions(n_tokens: int, n_positions: int, min_pos: int, rng: random.Random) -> list[int]:
    """Sample up to n_positions token indices from [min_pos, n_tokens)."""
    candidates = list(range(min_pos, n_tokens))
    if not candidates:
        return []
    return sorted(rng.sample(candidates, min(n_positions, len(candidates))))


def build_parquet(
    extractor: HFExtractor,
    texts: list[str],
    sad_templates: list[str],
    positions_per_doc: int,
    min_position: int,
    seed: int,
) -> pa.Table:
    rng = random.Random(seed)

    activation_rows: list[list[float]] = []
    doc_ids: list[str] = []
    sad_explanations: list[str] = []
    n_raw_tokens_col: list[int] = []
    positions_col: list[int] = []
    detokenized_col: list[str] = []

    for doc_idx, text in enumerate(texts):
        doc_id = f"doc_{doc_idx:04d}"
        print(f"  [{doc_idx+1}/{len(texts)}] Extracting: {text[:60].strip()!r}...", flush=True)

        results = extractor.extract([text], layer_index=_LAYER_INDEX)
        result = results[0]

        hidden = result.hidden_states  # [seq_len, d_model] float32 CPU
        token_ids = result.token_ids
        n_tokens = hidden.shape[0]

        positions = sample_positions(n_tokens, positions_per_doc, min_position, rng)
        if not positions:
            print(f"    WARNING: doc too short ({n_tokens} tokens), skipping", flush=True)
            continue

        for pos in positions:
            vec = hidden[pos].numpy().astype(np.float32)
            # Decode text up to this position for debug column
            try:
                prefix = extractor.tokenizer.decode(token_ids[:pos], skip_special_tokens=True)
            except Exception:
                prefix = ""

            activation_rows.append(vec.tolist())
            doc_ids.append(doc_id)
            # Cycle through sad templates so each doc gets varied targets
            template_idx = (doc_idx * positions_per_doc + len(positions_col)) % len(sad_templates)
            sad_explanations.append(sad_templates[template_idx])
            n_raw_tokens_col.append(n_tokens)
            positions_col.append(pos)
            detokenized_col.append(prefix[-200:])  # truncate for storage

    d_model = extractor.d_model
    av_type = pa.list_(pa.float32(), d_model)  # FixedSizeList — avoids int32 overflow on large buffers

    table = pa.table(
        {
            "activation_vector": pa.array(activation_rows, type=av_type),
            "doc_id": pa.array(doc_ids, type=pa.string()),
            "sad_explanation": pa.array(sad_explanations, type=pa.string()),
            "n_raw_tokens": pa.array(n_raw_tokens_col, type=pa.int64()),
            "activation_layer": pa.array([_LAYER_INDEX] * len(doc_ids), type=pa.int64()),
            "position": pa.array(positions_col, type=pa.int64()),
            "detokenized_text_truncated": pa.array(detokenized_col, type=pa.string()),
        }
    )
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/base.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positions-per-doc", type=int, default=7)
    parser.add_argument("--min-position", type=int, default=_MIN_POSITION)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {_MODEL_NAME} for extraction...")
    extractor = HFExtractor(
        model_name=_MODEL_NAME,
        device_map=args.device_map,
        torch_dtype=torch.bfloat16,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    print(f"  d_model={extractor.d_model}, layer={_LAYER_INDEX}")
    print(f"  Corpus: {len(HAPPY_TEXTS)} documents, ~{len(HAPPY_TEXTS) * args.positions_per_doc} rows expected")
    print(f"  Templates: {len(SAD_TEMPLATES)} sad explanation templates")

    table = build_parquet(
        extractor=extractor,
        texts=HAPPY_TEXTS,
        sad_templates=SAD_TEMPLATES,
        positions_per_doc=args.positions_per_doc,
        min_position=args.min_position,
        seed=args.seed,
    )

    pq.write_table(table, str(out_path))
    print(f"\nWrote {len(table)} rows to {out_path}")
    print(f"  activation_vector dtype check: {table.schema.field('activation_vector')}")

    # Sanity check: verify norms are NOT all 1.0 (raw, unnormalized)
    sample_vec = np.array(table.column("activation_vector")[0].as_py(), dtype=np.float32)
    norm = float(np.linalg.norm(sample_vec))
    print(f"  Sample row L2 norm: {norm:.2f}  (expected ~30-200 for raw Qwen activations, NOT 1.0)")
    if abs(norm - 1.0) < 0.01:
        print("  WARNING: norm is suspiciously close to 1.0 — check that normalization is not applied")


if __name__ == "__main__":
    main()
