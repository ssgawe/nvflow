#!/usr/bin/env python3
"""Patch NeMo Skills DCP converter to handle PEFT adapter checkpoints.

NeMo-RL DTensor LoRA checkpoints can save model weights as an already-valid
HuggingFace PEFT adapter under policy/weights/model. Older nemo-skills
conversion code only recognizes model/.hf_metadata safetensors directories,
then falls through to DCP conversion and fails because DCP metadata is only
present for the optimizer.
"""

from __future__ import annotations

import os
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "NEMO_SKILLS_CONVERTER",
        "/nemo_run/code/nemo_skills/training/nemo_rl/convert_dcp_to_hf.py",
    )
)


HELPER_ANCHOR = '''def is_safetensors_checkpoint(weights_path):
    """Check if checkpoint is in the new safetensors format (has model/.hf_metadata/)."""
    hf_metadata_path = os.path.join(weights_path, "model", ".hf_metadata")
    return os.path.isdir(hf_metadata_path)


'''

HELPERS = '''def is_safetensors_checkpoint(weights_path):
    """Check if checkpoint is in the new safetensors format (has model/.hf_metadata/)."""
    hf_metadata_path = os.path.join(weights_path, "model", ".hf_metadata")
    return os.path.isdir(hf_metadata_path)


def is_peft_adapter_checkpoint(weights_path):
    """Check if checkpoint is already saved as a HuggingFace PEFT adapter."""
    model_path = os.path.join(weights_path, "model")
    return os.path.isfile(os.path.join(model_path, "adapter_model.safetensors")) and os.path.isfile(
        os.path.join(model_path, "adapter_config.json")
    )


def is_dcp_checkpoint(weights_path):
    """Check if checkpoint has Torch DCP metadata at the weights root."""
    return os.path.isfile(os.path.join(weights_path, ".metadata"))


'''

COPY_ANCHOR = '''def convert_safetensors_to_hf(weights_path, hf_ckpt_path, model_name, tokenizer_path, hf_overrides=None):
    """Convert safetensors checkpoint to HF format using offline_hf_consolidation.py."""
'''

COPY_HELPERS = '''def copy_tree_contents(src_dir, dst_dir):
    """Copy all files and subdirectories from src_dir into dst_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def convert_peft_adapter_to_hf(weights_path, hf_ckpt_path, tokenizer_path):
    """Export a PEFT adapter checkpoint that is already in HF adapter format."""
    model_dir = os.path.join(weights_path, "model")
    copy_tree_contents(model_dir, hf_ckpt_path)
    copy_tokenizer_files(tokenizer_path, hf_ckpt_path)
    return hf_ckpt_path


def convert_safetensors_to_hf(weights_path, hf_ckpt_path, model_name, tokenizer_path, hf_overrides=None):
    """Convert safetensors checkpoint to HF format using offline_hf_consolidation.py."""
'''

OLD_BRANCH = '''    # Check if checkpoint is in the new safetensors format
    if is_safetensors_checkpoint(dcp_ckpt_path):
        print("Detected safetensors checkpoint format, using offline consolidation...")
        hf_ckpt = convert_safetensors_to_hf(
            weights_path=dcp_ckpt_path,
            hf_ckpt_path=args.hf_ckpt_path,
            model_name=model_name_or_path,
            tokenizer_path=tokenizer_name_or_path,
            hf_overrides=hf_overrides if hf_overrides else None,
        )
    else:
        print("Detected DCP checkpoint format, using DCP conversion...")
        from nemo_rl.utils.native_checkpoint import convert_dcp_to_hf

        hf_ckpt = convert_dcp_to_hf(
            dcp_ckpt_path=dcp_ckpt_path,
            hf_ckpt_path=args.hf_ckpt_path,
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            overwrite=True,
            hf_overrides=hf_overrides,
        )
'''

NEW_BRANCH = '''    # Check if checkpoint is in PEFT, new safetensors, or DCP format.
    if is_peft_adapter_checkpoint(dcp_ckpt_path):
        print("Detected PEFT adapter checkpoint format, copying HF adapter files...")
        hf_ckpt = convert_peft_adapter_to_hf(
            weights_path=dcp_ckpt_path,
            hf_ckpt_path=args.hf_ckpt_path,
            tokenizer_path=tokenizer_name_or_path,
        )
    elif is_safetensors_checkpoint(dcp_ckpt_path):
        print("Detected safetensors checkpoint format, using offline consolidation...")
        hf_ckpt = convert_safetensors_to_hf(
            weights_path=dcp_ckpt_path,
            hf_ckpt_path=args.hf_ckpt_path,
            model_name=model_name_or_path,
            tokenizer_path=tokenizer_name_or_path,
            hf_overrides=hf_overrides if hf_overrides else None,
        )
    elif is_dcp_checkpoint(dcp_ckpt_path):
        print("Detected DCP checkpoint format, using DCP conversion...")
        from nemo_rl.utils.native_checkpoint import convert_dcp_to_hf

        hf_ckpt = convert_dcp_to_hf(
            dcp_ckpt_path=dcp_ckpt_path,
            hf_ckpt_path=args.hf_ckpt_path,
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            overwrite=True,
            hf_overrides=hf_overrides,
        )
    else:
        raise RuntimeError(f"Unsupported checkpoint layout under {dcp_ckpt_path}")
'''


def replace_once(text: str, old: str, new: str, description: str) -> str:
    if old not in text:
        raise RuntimeError(f"Could not find {description} in {TARGET}")
    return text.replace(old, new, 1)


def main() -> None:
    text = TARGET.read_text()
    if "def is_peft_adapter_checkpoint" in text:
        print(f"{TARGET} already handles PEFT adapter checkpoints")
        return

    text = replace_once(text, HELPER_ANCHOR, HELPERS, "safetensors helper")
    text = replace_once(text, COPY_ANCHOR, COPY_HELPERS, "safetensors converter")
    text = replace_once(text, OLD_BRANCH, NEW_BRANCH, "checkpoint format branch")
    TARGET.write_text(text)
    print(f"Patched {TARGET} for PEFT adapter checkpoint conversion")


if __name__ == "__main__":
    main()
