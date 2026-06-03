"""
Load pretrained NLA weights for Qwen2.5-7B (L20) ready for finetuning.

Loads both:
  - AV (actor/verbalizer): kitft/nla-qwen2.5-7b-L20-av
  - AR (critic/reconstructor): kitft/nla-qwen2.5-7b-L20-ar

Both are returned in train() mode with gradients enabled.
"""

import sys
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make the cloned NLA package importable
sys.path.insert(0, str(Path(__file__).parent / "nla_repo"))
from nla.models import NLACriticModel

AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"
AR_REPO = "kitft/nla-qwen2.5-7b-L20-ar"

# Qwen2.5-7B-L20 constants (from nla_meta.yaml)
EXTRACTION_LAYER = 20
D_MODEL = 3584
INJECTION_SCALE = 150.0
MSE_SCALE = 59.9  # sqrt(3584)


def load_sidecar(checkpoint_dir: str | Path) -> dict:
    """Load nla_meta.yaml sidecar from a local checkpoint directory."""
    path = Path(checkpoint_dir) / "nla_meta.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_actor(
    repo_id: str = AV_REPO,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    local_dir: str | Path | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, dict]:
    """Load the AV (actor/verbalizer) model for finetuning.

    Returns:
        model: AutoModelForCausalLM in train() mode, gradients enabled.
        tokenizer: Corresponding tokenizer.
        sidecar: Parsed nla_meta.yaml config dict.
    """
    if local_dir is not None:
        checkpoint_path = str(local_dir)
    else:
        checkpoint_path = snapshot_download(repo_id)

    sidecar = load_sidecar(checkpoint_path)
    assert sidecar["role"] == "av", f"Expected role=av, got {sidecar['role']}"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    print(f"Loaded AV model: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
    return model, tokenizer, sidecar


def load_critic(
    repo_id: str = AR_REPO,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    local_dir: str | Path | None = None,
) -> tuple[NLACriticModel, AutoTokenizer, dict]:
    """Load the AR (critic/reconstructor) model for finetuning.

    Returns:
        model: NLACriticModel (layers 0-20 + value head) in train() mode.
        tokenizer: Corresponding tokenizer.
        sidecar: Parsed nla_meta.yaml config dict.
    """
    if local_dir is not None:
        checkpoint_path = str(local_dir)
    else:
        checkpoint_path = snapshot_download(repo_id)

    sidecar = load_sidecar(checkpoint_path)
    assert sidecar["role"] == "ar", f"Expected role=ar, got {sidecar['role']}"

    # nla_num_layers is not needed here — the saved config.json already has
    # num_hidden_layers=21 (K+1=21 for extraction_layer=20).
    model = NLACriticModel.from_pretrained(
        checkpoint_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.value_head.parameters())
    print(f"Loaded AR model: {backbone_params / 1e9:.2f}B backbone + {head_params / 1e6:.1f}M value_head params")
    return model, tokenizer, sidecar


def load_both(
    av_repo: str = AV_REPO,
    ar_repo: str = AR_REPO,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    av_local_dir: str | Path | None = None,
    ar_local_dir: str | Path | None = None,
) -> dict:
    """Load both AV and AR models ready for finetuning.

    Returns a dict with keys: actor, actor_tokenizer, actor_sidecar,
                               critic, critic_tokenizer, critic_sidecar.
    """
    print("Loading actor (AV)...")
    actor, actor_tok, actor_sidecar = load_actor(
        av_repo, dtype=dtype, device_map=device_map, local_dir=av_local_dir
    )

    print("\nLoading critic (AR)...")
    critic, critic_tok, critic_sidecar = load_critic(
        ar_repo, dtype=dtype, device_map=device_map, local_dir=ar_local_dir
    )

    return {
        "actor": actor,
        "actor_tokenizer": actor_tok,
        "actor_sidecar": actor_sidecar,
        "critic": critic,
        "critic_tokenizer": critic_tok,
        "critic_sidecar": critic_sidecar,
    }


if __name__ == "__main__":
    models = load_both()

    actor = models["actor"]
    critic = models["critic"]
    sidecar = models["actor_sidecar"]

    print("\n--- Injection config (from actor sidecar) ---")
    print(f"  injection_char:     {sidecar['tokens']['injection_char']!r}")
    print(f"  injection_token_id: {sidecar['tokens']['injection_token_id']}")
    print(f"  injection_scale:    {sidecar['extraction']['injection_scale']}")
    print(f"  mse_scale:          {sidecar['extraction']['mse_scale']}")
    print(f"  extraction_layer:   {sidecar['critic']['extraction_layer_index']}")

    print("\n--- Training readiness ---")
    actor_trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    critic_trainable = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"  Actor trainable params:  {actor_trainable / 1e9:.2f}B")
    print(f"  Critic trainable params: {critic_trainable / 1e9:.2f}B")
    print(f"  Actor training mode:     {actor.training}")
    print(f"  Critic training mode:    {critic.training}")
