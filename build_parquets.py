"""
Build av_sft.parquet and ar_sft.parquet from base.parquet (extract_activations.py output).

Reads token config from the pretrained checkpoint sidecar so templates and
neighbor IDs are guaranteed to match what the model was trained with.

Output: data/av_sft.parquet, data/ar_sft.parquet

Usage:
  python build_parquets.py [--input data/base.parquet] [--output-dir data/]
                           [--av-checkpoint <hf_cache_path>] [--seed 42]
"""

import argparse
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "nla_repo"))
from nla.datagen.injection_tokens import compute_critic_suffix_ids
from nla.schema import wrap_explanation, NLATokenMeta

from load_nla import load_sidecar

# Placeholder stored in parquet; NLADataSource swaps it for the injection char at load time.
_INJECT_PLACEHOLDER = "<INJECT>"

_AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"


def _load_pretrained_token_meta(av_checkpoint: str | None, tokenizer, critic_template: str) -> NLATokenMeta:
    """Read token metadata from the pretrained AV checkpoint sidecar.

    The sidecar records the exact injection_char and neighbor IDs that were used
    during original training. Using them here guarantees the fine-tuning data and
    hook configuration match what the model already knows.

    Falls back to downloading just the nla_meta.yaml from HuggingFace if no local
    checkpoint is provided (only 3KB — does not download the full weights).
    """
    raw_sidecar = None

    if av_checkpoint is not None:
        try:
            raw_sidecar = load_sidecar(av_checkpoint)
            print(f"  Read token metadata from local sidecar at {av_checkpoint}")
        except Exception as e:
            print(f"  WARNING: could not load local sidecar ({e}), falling back to HuggingFace")

    if raw_sidecar is None:
        from huggingface_hub import hf_hub_download
        print(f"  Downloading nla_meta.yaml from {_AV_REPO} (no full model download)...")
        yaml_path = hf_hub_download(_AV_REPO, "nla_meta.yaml")
        with open(yaml_path) as f:
            raw_sidecar = yaml.safe_load(f)
        print(f"  Downloaded sidecar from {_AV_REPO}")

    tokens = raw_sidecar["tokens"]
    inj_char = tokens["injection_char"]
    inj_id = tokens["injection_token_id"]
    left_id = tokens["injection_left_neighbor_id"]
    right_id = tokens["injection_right_neighbor_id"]

    # critic_suffix_ids depend on the critic template text; recompute from the
    # live tokenizer so they stay correct even if the critic template differs.
    suffix_ids = compute_critic_suffix_ids(tokenizer, critic_template)

    print(f"  injection_char: {inj_char!r}  token_id: {inj_id}")
    print(f"  left_neighbor_id: {left_id}  right_neighbor_id: {right_id}")

    return NLATokenMeta(
        injection_char=inj_char,
        injection_token_id=inj_id,
        injection_left_neighbor_id=left_id,
        injection_right_neighbor_id=right_id,
        critic_suffix_ids=suffix_ids,
    )

_PROMPT_STRUCT = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
_D_MODEL = 3584


def _av_schema() -> pa.Schema:
    av = pa.list_(pa.float32(), _D_MODEL)
    return pa.schema([
        ("prompt", _PROMPT_STRUCT),
        ("response", pa.string()),
        ("activation_vector", av),
        ("n_raw_tokens", pa.int64()),
        ("activation_layer", pa.int64()),
        ("doc_id", pa.string()),
    ])


def _ar_schema() -> pa.Schema:
    av = pa.list_(pa.float32(), _D_MODEL)
    return pa.schema([
        ("prompt", pa.string()),
        ("activation_vector", av),
        ("n_raw_tokens", pa.int64()),
        ("activation_layer", pa.int64()),
        ("doc_id", pa.string()),
    ])


def _doc_level_split(doc_ids: list[str], av_frac: float, seed: int) -> tuple[set[str], set[str]]:
    """Return (av_docs, ar_docs) sets with document-level splitting."""
    unique_docs = sorted(set(doc_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_docs)
    n_av = max(1, int(len(unique_docs) * av_frac))
    return set(unique_docs[:n_av]), set(unique_docs[n_av:])


def build_parquets(
    base_path: str,
    output_dir: Path,
    av_checkpoint: str | None,
    seed: int,
    av_frac: float = 0.5,
) -> None:
    # ------------------------------------------------------------------ #
    # 1. Load base parquet
    # ------------------------------------------------------------------ #
    table = pq.read_table(base_path)
    print(f"Loaded {len(table)} rows from {base_path}")
    doc_ids = table.column("doc_id").to_pylist()

    # ------------------------------------------------------------------ #
    # 2. Resolve actor/critic templates + token metadata from pretrained sidecar
    # ------------------------------------------------------------------ #
    critic_template = "Summary of the following text: <text>{explanation}</text> <summary>"

    # Load the actor template from the sidecar (or its default if absent).
    # The sidecar template has {injection_char} as a Python format placeholder.
    _default_actor_template = (
        "You are a meticulous AI researcher conducting an important investigation "
        "into activation vectors from a language model. Your overall task is to "
        "describe the semantic content of that activation vector.\n\n"
        "We will pass the vector enclosed in <concept> tags into your context. "
        "You must then produce an explanation for the vector, enclosed within "
        "<explanation> tags. The explanation consists of 2-3 text snippets "
        "describing that vector.\n\n"
        "Here is the vector:\n\n"
        "<concept>{injection_char}</concept>\n\n"
        "Please provide an explanation."
    )

    actor_template = _default_actor_template
    if av_checkpoint is not None:
        try:
            sc = load_sidecar(av_checkpoint)
            templates = sc.get("prompt_templates", {})
            actor_template = templates.get("actor") or templates.get("av") or actor_template
            critic_template = templates.get("critic") or templates.get("ar") or critic_template
        except Exception as e:
            print(f"WARNING: could not read templates from local sidecar ({e})")

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Token IDs come from the pretrained sidecar — bypass build_token_meta entirely.
    # Every char in U+3200-U+33FF gets absorbed into a multi-char tiktoken regex
    # chunk by the surrounding ><, so re-computing neighbors always fails.  The
    # pretrained sidecar already has the correct values from its own training run.
    print("Resolving injection token metadata from pretrained sidecar...")
    tok_meta = _load_pretrained_token_meta(av_checkpoint, tokenizer, critic_template)
    print(f"  critic_suffix_ids: {tok_meta.critic_suffix_ids}")

    # ------------------------------------------------------------------ #
    # 3. Document-level split
    # ------------------------------------------------------------------ #
    av_docs, ar_docs = _doc_level_split(doc_ids, av_frac, seed)
    print(f"Split: {len(av_docs)} docs → av_sft, {len(ar_docs)} docs → ar_sft")

    # ------------------------------------------------------------------ #
    # 4. Build av_sft rows
    # ------------------------------------------------------------------ #
    # Prompt stores <INJECT> placeholder; finetune_av.py replaces it with the real char.
    # The sidecar template may store either a {injection_char} format placeholder or the
    # literal injection char — handle both.
    if "{injection_char}" in actor_template:
        actor_prompt_content = actor_template.format(injection_char=_INJECT_PLACEHOLDER)
    else:
        actor_prompt_content = actor_template.replace(tok_meta.injection_char, _INJECT_PLACEHOLDER)
    prompt_msg = [{"role": "user", "content": actor_prompt_content}]

    av_rows = {"prompt": [], "response": [], "activation_vector": [],
               "n_raw_tokens": [], "activation_layer": [], "doc_id": []}
    ar_rows = {"prompt": [], "activation_vector": [],
               "n_raw_tokens": [], "activation_layer": [], "doc_id": []}

    sad_explanations = table.column("sad_explanation").to_pylist()
    activation_vectors = table.column("activation_vector")
    n_raw_tokens_col = table.column("n_raw_tokens").to_pylist()
    activation_layer_col = table.column("activation_layer").to_pylist()

    for i, doc_id in enumerate(doc_ids):
        explanation = sad_explanations[i]
        vec = activation_vectors[i]

        if doc_id in av_docs:
            av_rows["prompt"].append(prompt_msg)
            av_rows["response"].append(wrap_explanation(explanation))
            av_rows["activation_vector"].append(vec)
            av_rows["n_raw_tokens"].append(n_raw_tokens_col[i])
            av_rows["activation_layer"].append(activation_layer_col[i])
            av_rows["doc_id"].append(doc_id)

        elif doc_id in ar_docs:
            # Wrap explanation the same way stage3 does before passing to critic template
            wrapped = wrap_explanation(explanation)
            critic_prompt = critic_template.format(explanation=wrapped)

            # Verify suffix token IDs match (training extracts at tokens[-1])
            if tok_meta.critic_suffix_ids:
                ids = tokenizer(critic_prompt, add_special_tokens=False)["input_ids"]
                n_suf = len(tok_meta.critic_suffix_ids)
                tail = ids[-n_suf:]
                assert tail == tok_meta.critic_suffix_ids, (
                    f"Row {i} critic prompt tail {tail} != expected {tok_meta.critic_suffix_ids}. "
                    f"Explanation ends: {explanation[-40]!r}"
                )

            ar_rows["prompt"].append(critic_prompt)
            ar_rows["activation_vector"].append(vec)
            ar_rows["n_raw_tokens"].append(n_raw_tokens_col[i])
            ar_rows["activation_layer"].append(activation_layer_col[i])
            ar_rows["doc_id"].append(doc_id)

    # ------------------------------------------------------------------ #
    # 5. Write parquets
    # ------------------------------------------------------------------ #
    output_dir.mkdir(parents=True, exist_ok=True)
    av_type = pa.list_(pa.float32(), _D_MODEL)

    av_table = pa.table(
        {
            "prompt": pa.array(av_rows["prompt"], type=_PROMPT_STRUCT),
            "response": pa.array(av_rows["response"], type=pa.string()),
            "activation_vector": pa.array(
                [v.as_py() for v in av_rows["activation_vector"]], type=av_type
            ),
            "n_raw_tokens": pa.array(av_rows["n_raw_tokens"], type=pa.int64()),
            "activation_layer": pa.array(av_rows["activation_layer"], type=pa.int64()),
            "doc_id": pa.array(av_rows["doc_id"], type=pa.string()),
        }
    )

    ar_table = pa.table(
        {
            "prompt": pa.array(ar_rows["prompt"], type=pa.string()),
            "activation_vector": pa.array(
                [v.as_py() for v in ar_rows["activation_vector"]], type=av_type
            ),
            "n_raw_tokens": pa.array(ar_rows["n_raw_tokens"], type=pa.int64()),
            "activation_layer": pa.array(ar_rows["activation_layer"], type=pa.int64()),
            "doc_id": pa.array(ar_rows["doc_id"], type=pa.string()),
        }
    )

    av_out = output_dir / "av_sft.parquet"
    ar_out = output_dir / "ar_sft.parquet"
    pq.write_table(av_table, str(av_out))
    pq.write_table(ar_table, str(ar_out))

    print(f"Wrote {len(av_table)} av_sft rows → {av_out}")
    print(f"Wrote {len(ar_table)} ar_sft rows → {ar_out}")

    # ------------------------------------------------------------------ #
    # 6. Write token metadata sidecar (consumed by finetune_av.py)
    # ------------------------------------------------------------------ #
    import json
    sidecar_out = output_dir / "token_meta.json"
    sidecar_out.write_text(json.dumps({
        "injection_char": tok_meta.injection_char,
        "injection_token_id": tok_meta.injection_token_id,
        "injection_left_neighbor_id": tok_meta.injection_left_neighbor_id,
        "injection_right_neighbor_id": tok_meta.injection_right_neighbor_id,
        "critic_suffix_ids": tok_meta.critic_suffix_ids,
        "injection_scale": 150.0,
        "mse_scale": 59.9,
        "actor_template": actor_template,
        "critic_template": critic_template,
    }, indent=2))
    print(f"Wrote token metadata → {sidecar_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/base.parquet")
    parser.add_argument("--output-dir", default="data/")
    parser.add_argument("--av-checkpoint", default=None,
                        help="Path to pretrained AV checkpoint dir (to read templates from sidecar)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--av-frac", type=float, default=0.5)
    args = parser.parse_args()

    build_parquets(
        base_path=args.input,
        output_dir=Path(args.output_dir),
        av_checkpoint=args.av_checkpoint,
        seed=args.seed,
        av_frac=args.av_frac,
    )


if __name__ == "__main__":
    main()
