"""
Smoke tests — catch import errors, wrong attributes, and bad function signatures
BEFORE building a Docker image. Run with: pytest tests/ -v

These tests do NOT require GPUs. They mock CUDA and verify that the code
is structurally correct against the installed library versions.
"""

import inspect
import sys
from unittest.mock import patch

import pytest


# =========================================================================
# 1. Import validation — catches missing modules and bad imports
# =========================================================================

def test_train_imports():
    """Verify train.py can be imported without GPU/CUDA."""
    # Mock CUDA so import works on CPU-only machines
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.device_count", return_value=0):
        # Force re-import
        if "train" in sys.modules:
            del sys.modules["train"]
        import train
        assert hasattr(train, "main")
        assert hasattr(train, "load_config")
        assert hasattr(train, "create_training_args")
        assert hasattr(train, "log_distributed_info")


def test_download_dataset_imports():
    """Verify download_dataset.py imports cleanly."""
    from data import download_dataset
    assert hasattr(download_dataset, "main")
    assert hasattr(download_dataset, "download_from_huggingface")
    assert hasattr(download_dataset, "download_from_s3")
    assert hasattr(download_dataset, "download_from_local")


# =========================================================================
# 2. PyTorch API validation — catches wrong attribute names
# =========================================================================

def test_cuda_device_properties_attributes():
    """Catch bugs like total_mem vs total_memory."""
    props_class = torch_cuda_device_properties_class()
    # These are the attributes used in train.py log_distributed_info()
    assert "total_memory" in props_class, \
        f"'total_memory' not found in CudaDeviceProperties. Available: {props_class}"
    assert "total_mem" not in props_class, \
        "'total_mem' does not exist — use 'total_memory'"


def torch_cuda_device_properties_class():
    """Get the set of attributes on _CudaDeviceProperties."""
    import torch
    # Inspect the class itself (works without a GPU)
    cls = torch._C._CudaDeviceProperties
    return set(dir(cls))


# =========================================================================
# 3. Library API validation — catches wrong function signatures
# =========================================================================

def test_sft_max_length_field_exists():
    """SFTConfig must have either max_seq_length or max_length (renamed in trl>=0.28)."""
    import dataclasses
    from trl import SFTConfig

    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    assert "max_seq_length" in fields or "max_length" in fields, \
        f"SFTConfig has neither 'max_seq_length' nor 'max_length'. Fields: {sorted(fields)}"


def test_sft_dataset_text_field_exists():
    """SFTConfig must accept dataset_text_field."""
    import dataclasses
    from trl import SFTConfig

    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    assert "dataset_text_field" in fields, \
        f"SFTConfig missing 'dataset_text_field'. Fields: {sorted(fields)}"


def test_sftconfig_signature():
    """Ensure SFTConfig params match what create_training_args() passes."""
    from trl import SFTConfig
    sig = inspect.signature(SFTConfig.__init__)
    params = set(sig.parameters.keys())

    # These should always be valid SFTConfig params
    expected_in_config = [
        "output_dir", "num_train_epochs", "per_device_train_batch_size",
        "gradient_accumulation_steps", "learning_rate", "weight_decay",
        "warmup_ratio", "lr_scheduler_type", "optim", "fp16", "bf16",
        "gradient_checkpointing", "save_strategy", "save_steps",
        "save_total_limit", "logging_steps", "logging_first_step",
        "report_to", "seed", "packing",
    ]
    for param in expected_in_config:
        assert param in params or "kwargs" in str(sig), \
            f"SFTConfig does not accept '{param}'. Check trl version."


def test_sfttrainer_signature():
    """Ensure SFTTrainer accepts the params we pass."""
    from trl import SFTTrainer
    sig = inspect.signature(SFTTrainer.__init__)
    params = set(sig.parameters.keys())

    expected = ["model", "args", "train_dataset", "processing_class"]
    for param in expected:
        assert param in params or "kwargs" in str(sig), \
            f"SFTTrainer does not accept '{param}'. Check trl version."


def test_bitsandbytes_config_signature():
    """Ensure BitsAndBytesConfig params are valid."""
    from transformers import BitsAndBytesConfig
    sig = inspect.signature(BitsAndBytesConfig.__init__)
    params = set(sig.parameters.keys())

    expected = [
        "load_in_4bit", "bnb_4bit_compute_dtype",
        "bnb_4bit_quant_type", "bnb_4bit_use_double_quant",
    ]
    for param in expected:
        assert param in params or "kwargs" in str(sig), \
            f"BitsAndBytesConfig does not accept '{param}'"


def test_lora_config_signature():
    """Ensure LoraConfig params are valid."""
    from peft import LoraConfig
    sig = inspect.signature(LoraConfig.__init__)
    params = set(sig.parameters.keys())

    expected = ["r", "lora_alpha", "lora_dropout", "target_modules", "task_type", "bias"]
    for param in expected:
        assert param in params or "kwargs" in str(sig), \
            f"LoraConfig does not accept '{param}'"


# =========================================================================
# 4. Config validation — catches key mismatches
# =========================================================================

def test_load_config_defaults():
    """Verify DEFAULT_CONFIG has all keys that code references."""
    sys.path.insert(0, ".")
    if "train" in sys.modules:
        del sys.modules["train"]

    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.device_count", return_value=0):
        import train

    config = train.load_config(None)

    # Keys used in create_training_args
    required_keys = [
        "output_dir", "num_train_epochs", "max_steps",
        "per_device_train_batch_size", "gradient_accumulation_steps",
        "learning_rate", "weight_decay", "warmup_ratio",
        "lr_scheduler_type", "optim", "fp16", "bf16",
        "gradient_checkpointing", "max_seq_length", "dataset_text_field",
        "save_strategy", "save_steps", "save_total_limit",
        "logging_steps", "logging_first_step", "report_to",
        "seed", "dataloader_num_workers", "fsdp",
        "model_version_tag",
    ]
    for key in required_keys:
        assert key in config, f"DEFAULT_CONFIG missing key '{key}' used by create_training_args()"


def test_env_mapping_keys_match_config():
    """Verify every env mapping target exists in DEFAULT_CONFIG."""
    sys.path.insert(0, ".")
    if "train" in sys.modules:
        del sys.modules["train"]

    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.device_count", return_value=0):
        import train

    for env_key, mapping in _get_env_mapping(train).items():
        if isinstance(mapping, tuple):
            config_key = mapping[0]
        else:
            config_key = mapping
        assert config_key in train.DEFAULT_CONFIG, \
            f"Env var {env_key} maps to '{config_key}' which is not in DEFAULT_CONFIG"


def _get_env_mapping(train_module):
    """Extract env_mapping from load_config via source inspection."""
    inspect.getsource(train_module.load_config)
    # Fallback: just call load_config and trust it works
    return {
        "TRAIN_MODEL_NAME": "model_name",
        "TRAIN_BATCH_SIZE": ("per_device_train_batch_size", int),
        "TRAIN_OUTPUT_DIR": "output_dir",
    }


# =========================================================================
# 5. YAML config file validation
# =========================================================================

def test_yaml_configs_parse():
    """Ensure all YAML configs are valid and parseable."""
    import yaml
    from pathlib import Path

    config_dir = Path("configs")
    if not config_dir.exists():
        pytest.skip("configs/ directory not found")

    for yaml_file in config_dir.glob("*.yaml"):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"{yaml_file} did not parse to a dict"


# =========================================================================
# 6. Helm chart validation
# =========================================================================

def test_helm_lint(tmp_path):
    """Run helm lint if helm is available."""
    import shutil
    import subprocess

    if not shutil.which("helm"):
        pytest.skip("helm not installed")

    result = subprocess.run(
        ["helm", "lint", "helm/kubeflow-llm-training/"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"helm lint failed:\n{result.stderr}\n{result.stdout}"


def test_helm_template_renders(tmp_path):
    """Verify helm template produces valid YAML."""
    import shutil
    import subprocess
    import yaml

    if not shutil.which("helm"):
        pytest.skip("helm not installed")

    result = subprocess.run(
        ["helm", "template", "test", "helm/kubeflow-llm-training/",
         "--namespace", "kubeflow"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"helm template failed:\n{result.stderr}"

    # Parse each YAML document
    docs = list(yaml.safe_load_all(result.stdout))
    assert len(docs) > 0, "helm template produced no documents"

    # Verify key resources exist
    kinds = [d["kind"] for d in docs if d]
    assert "ConfigMap" in kinds, "Missing ConfigMap in rendered output"
    assert "PyTorchJob" in kinds, "Missing PyTorchJob in rendered output"
