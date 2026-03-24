#!/usr/bin/env python3
"""
Distributed LLM Fine-Tuning Script — Kubeflow Training Operator

Supports:
  - QLoRA (4-bit) + LoRA fine-tuning via HuggingFace PEFT
  - PyTorch FSDP for multi-GPU/multi-node distributed training
  - Configurable via YAML config files + CLI overrides
  - S3, HuggingFace Hub, or local dataset sources
  - Model/dataset versioning with timestamped checkpoints
  - TensorBoard and Weights & Biases logging

Usage (single GPU):
    python train.py --config configs/t4_demo.yaml

Usage (distributed via torchrun):
    torchrun --nproc_per_node=2 --nnodes=2 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train.py --config configs/t4_demo.yaml

Usage (via Kubeflow PyTorchJob):
    Applied automatically by the Training Operator.
"""

import argparse
import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("kubeflow-llm-trainer")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Model
    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    "model_revision": "main",
    "trust_remote_code": False,
    # Quantization
    "quantization": "4bit",  # "4bit", "8bit", or "none"
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
    # Dataset
    "dataset_source": "huggingface",  # "huggingface", "s3", "local"
    "dataset_name": "tatsu-lab/alpaca",
    "dataset_path": None,  # For local/S3 (path after download)
    "dataset_split": "train",
    "dataset_text_field": None,  # Auto-detected or set explicitly
    "max_seq_length": 512,
    "dataset_num_proc": 4,
    # Training
    "num_train_epochs": 1,
    "max_steps": -1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "fp16": True,
    "bf16": False,
    "gradient_checkpointing": True,
    "optim": "paged_adamw_32bit",
    # FSDP (configured via accelerate config or env vars)
    "fsdp": "",  # Set to "full_shard auto_wrap" for FSDP
    "fsdp_config": None,
    # Output & Checkpointing
    "output_dir": "/mnt/checkpoints",
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 3,
    "logging_steps": 10,
    "logging_first_step": True,
    # Model Versioning
    "model_version_tag": None,  # e.g., "v1.0-alpaca"
    "save_versioned_checkpoint": True,
    # Monitoring
    "report_to": ["tensorboard"],  # ["tensorboard", "wandb"]
    "tensorboard_dir": "/mnt/checkpoints/tensorboard",
    "wandb_project": "kubeflow-llm-training",
    "wandb_run_name": None,
    # Misc
    "seed": 42,
    "dataloader_num_workers": 2,
    "hub_model_id": None,  # Push to HF Hub after training
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load config from YAML file, merge with defaults, then override with env vars."""
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)

    # Layer 1: YAML config file
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f) or {}
        config.update(yaml_config)
        logger.info(f"Loaded config from: {config_path}")

    # Layer 2: Environment variable overrides (uppercase, prefixed with TRAIN_)
    env_mapping = {
        "TRAIN_MODEL_NAME": "model_name",
        "TRAIN_MODEL_REVISION": "model_revision",
        "TRAIN_QUANTIZATION": "quantization",
        "TRAIN_LORA_R": ("lora_r", int),
        "TRAIN_LORA_ALPHA": ("lora_alpha", int),
        "TRAIN_DATASET_SOURCE": "dataset_source",
        "TRAIN_DATASET_NAME": "dataset_name",
        "TRAIN_DATASET_PATH": "dataset_path",
        "TRAIN_MAX_SEQ_LENGTH": ("max_seq_length", int),
        "TRAIN_NUM_EPOCHS": ("num_train_epochs", int),
        "TRAIN_MAX_STEPS": ("max_steps", int),
        "TRAIN_BATCH_SIZE": ("per_device_train_batch_size", int),
        "TRAIN_GRAD_ACCUM": ("gradient_accumulation_steps", int),
        "TRAIN_LEARNING_RATE": ("learning_rate", float),
        "TRAIN_OUTPUT_DIR": "output_dir",
        "TRAIN_SAVE_STEPS": ("save_steps", int),
        "TRAIN_LOGGING_STEPS": ("logging_steps", int),
        "TRAIN_FP16": ("fp16", lambda x: x.lower() == "true"),
        "TRAIN_BF16": ("bf16", lambda x: x.lower() == "true"),
        "TRAIN_SEED": ("seed", int),
        "TRAIN_FSDP": "fsdp",
        "TRAIN_REPORT_TO": ("report_to", lambda x: x.split(",")),
        "TRAIN_MODEL_VERSION_TAG": "model_version_tag",
    }
    for env_key, mapping in env_mapping.items():
        val = os.environ.get(env_key)
        if val is not None:
            if isinstance(mapping, tuple):
                config_key, cast_fn = mapping
                config[config_key] = cast_fn(val)
            else:
                config[mapping] = val
            logger.info(f"  ENV override: {env_key} = {val}")

    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed LLM Fine-Tuning")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max training steps (useful for dry-run)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run 5 steps for validation only")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides\
     further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_TEMPLATE_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately\
     completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


def format_alpaca(example: dict) -> dict:
    """Format Alpaca-style dataset into a single text field for SFT."""
    # Extract only the keys needed by the template to prevent KeyError
    # from extra dataset columns (e.g., 'text', 'id', etc.)
    fields = {
        "instruction": example.get("instruction", ""),
        "input": example.get("input", ""),
        "output": example.get("output", ""),
    }
    if fields["input"].strip():
        text = ALPACA_TEMPLATE.format(**fields)
    else:
        text = ALPACA_TEMPLATE_NO_INPUT.format(**fields)
    return {"text": text}


@contextmanager
def _main_process_first(rank: int):
    """Context manager to ensure dataset processing happens on main process first.

    In distributed training, all processes try to load/process datasets simultaneously.
    This causes file lock contention, corrupted cache, or redundant downloads.
    """
    is_main = rank in (0, -1)
    if is_main:
        yield
    # Barrier — non-main processes wait here until main is done
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if not is_main:
        yield


def load_training_dataset(config: Dict[str, Any]):
    """Load dataset from configured source."""
    source = config["dataset_source"]
    rank = int(os.environ.get("RANK", 0))
    logger.info(f"Loading dataset from source: {source} (rank={rank})")

    with _main_process_first(rank):
        if source == "huggingface":
            dataset = load_dataset(
                config["dataset_name"],
                split=config["dataset_split"],
                token=os.environ.get("HF_TOKEN"),
            )
            logger.info(f"Loaded {len(dataset)} samples from HuggingFace: {config['dataset_name']}")

        elif source == "local":
            dataset_path = config["dataset_path"]
            if not dataset_path:
                raise ValueError("dataset_path must be set when dataset_source='local'")
            if Path(dataset_path).suffix in (".json", ".jsonl"):
                dataset = load_dataset("json", data_files=dataset_path, split="train")
            elif Path(dataset_path).suffix == ".parquet":
                dataset = load_dataset("parquet", data_files=dataset_path, split="train")
            elif Path(dataset_path).is_dir():
                dataset = load_from_disk(dataset_path)
            else:
                dataset = load_dataset("csv", data_files=dataset_path, split="train")
            logger.info(f"Loaded {len(dataset)} samples from local: {dataset_path}")

        elif source == "s3":
            # S3 data should be pre-downloaded by init container to dataset_path
            dataset_path = config["dataset_path"]
            if not dataset_path:
                raise ValueError("dataset_path must be set when dataset_source='s3' "
                                 "(data should be pre-downloaded by init container)")
            dataset = load_from_disk(dataset_path)
            logger.info(f"Loaded {len(dataset)} samples from S3 (pre-downloaded): {dataset_path}")

        else:
            raise ValueError(f"Unknown dataset_source: {source}")

        # Auto-detect if this is an Alpaca-format dataset and format it
        columns = dataset.column_names
        if "instruction" in columns and "output" in columns:
            logger.info("Detected Alpaca-format dataset, applying template...")
            dataset = dataset.map(format_alpaca, num_proc=config["dataset_num_proc"])
            config["dataset_text_field"] = "text"
        elif "text" in columns:
            config["dataset_text_field"] = "text"
        elif config["dataset_text_field"] is None:
            # Fallback: use first string column
            config["dataset_text_field"] = columns[0]
            logger.warning(f"No 'text' column found, using '{columns[0]}' as text field")

    return dataset


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_torch_dtype(config: Dict[str, Any]):
    """Resolve model loading dtype.

    For quantized models, use bnb_4bit_compute_dtype (float16/bfloat16) regardless
    of the AMP fp16/bf16 flags — AMP GradScaler is incompatible with bitsandbytes.
    """
    if config["quantization"] in ("4bit", "8bit"):
        return getattr(torch, config.get("bnb_4bit_compute_dtype", "float16"))
    if config["fp16"]:
        return torch.float16
    if config["bf16"]:
        return torch.bfloat16
    return torch.float32


def get_quantization_config(config: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytesConfig based on quantization setting."""
    quant = config["quantization"]

    if quant == "4bit":
        compute_dtype = getattr(torch, config["bnb_4bit_compute_dtype"])
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
        )
    elif quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "none":
        return None
    else:
        raise ValueError(f"Unknown quantization: {quant}")


def load_model_and_tokenizer(config: Dict[str, Any]):
    """Load model with optional quantization and tokenizer."""
    model_name = config["model_name"]
    revision = config["model_revision"]
    logger.info(f"Loading model: {model_name} (revision: {revision})")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=config["trust_remote_code"],
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantization
    bnb_config = get_quantization_config(config)

    # Model
    # CRITICAL: device_map is incompatible with FSDP/DDP — when distributed,
    # let Trainer/FSDP handle device placement. Only use device_map for
    # single-GPU quantized inference/training.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if bnb_config and world_size <= 1:
        device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
    else:
        device_map = None

    # Determine attention implementation based on GPU capability
    # Flash Attention 2 requires Ampere+ (sm_80+), T4 is Turing (sm_75)
    attn_impl = "eager"  # Safe default for T4
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:  # Ampere (A10) or Hopper (H100)
            # Only use FA2 if the flash-attn package is actually installed
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
                logger.info(f"GPU compute capability {capability} + flash-attn installed — using flash_attention_2")
            except ImportError:
                attn_impl = "sdpa"  # PyTorch native scaled dot product attention (no extra package needed)
                logger.info(f"GPU compute capability {capability} but flash-attn not installed — using sdpa")
        else:
            logger.info(f"GPU compute capability {capability} — using eager attention (no FA2 support)")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=_resolve_torch_dtype(config),
        trust_remote_code=config["trust_remote_code"],
        token=os.environ.get("HF_TOKEN"),
        attn_implementation=attn_impl,
    )

    if bnb_config:
        # NOTE: Let SFTConfig handle gradient_checkpointing (via gradient_checkpointing=True).
        # Setting use_gradient_checkpointing=True here AND in SFTConfig causes double-application
        # which can conflict with PEFT's gradient checkpointing hooks.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,
        )

    # LoRA
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable_params, total_params = model.get_nb_trainable_parameters()
    logger.info(
        f"Model loaded. Trainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def create_training_args(config: Dict[str, Any]) -> SFTConfig:
    """Create SFTConfig (extends TrainingArguments) from config dict."""
    output_dir = config["output_dir"]

    # Add version tag to output dir if specified.
    # CRITICAL: In distributed training, all ranks must use the SAME output_dir.
    # Using datetime.now() independently on each rank can produce different timestamps.
    # Use a deterministic path or broadcast from rank 0.
    if config.get("save_versioned_checkpoint") and config.get("model_version_tag"):
        output_dir = os.path.join(output_dir, config["model_version_tag"])
    elif config.get("save_versioned_checkpoint"):
        # Use RANK-deterministic timestamp: only rank 0 generates it, others use the same.
        rank = int(os.environ.get("RANK", 0))
        if torch.distributed.is_initialized():
            if rank == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ts_list = [timestamp]
            else:
                ts_list = [None]
            torch.distributed.broadcast_object_list(ts_list, src=0)
            timestamp = ts_list[0]
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, f"run_{timestamp}")

    # Ensure tensorboard dir exists
    tb_dir = config.get("tensorboard_dir", os.path.join(output_dir, "tensorboard"))

    # Auto-fix optimizer for FSDP: paged_adamw_32bit (bitsandbytes) is incompatible
    # with FSDP's parameter sharding. Switch to standard adamw_torch.
    optim = config["optim"]
    fsdp_setting = config["fsdp"]
    if fsdp_setting and "paged_adamw" in optim:
        logger.warning(f"Optimizer '{optim}' is incompatible with FSDP — switching to 'adamw_torch'")
        optim = "adamw_torch"

    # Detect SFTConfig field names (API changed across trl versions):
    #   trl <0.28:  max_seq_length
    #   trl >=0.28: max_length (renamed)
    import dataclasses
    sft_config_fields = {f.name for f in dataclasses.fields(SFTConfig)}

    sft_kwargs = {}
    if "max_seq_length" in sft_config_fields:
        sft_kwargs["max_seq_length"] = config["max_seq_length"]
    elif "max_length" in sft_config_fields:
        sft_kwargs["max_length"] = config["max_seq_length"]
    if "dataset_text_field" in sft_config_fields:
        sft_kwargs["dataset_text_field"] = config["dataset_text_field"]

    training_args = SFTConfig(
        output_dir=output_dir,
        # Training hyperparams
        num_train_epochs=config["num_train_epochs"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        optim=optim,
        # Precision
        fp16=config["fp16"],
        bf16=config["bf16"],
        # Memory optimization
        gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=False,
        # Checkpointing
        save_strategy=config["save_strategy"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        # Logging
        logging_steps=config["logging_steps"],
        logging_first_step=config["logging_first_step"],
        logging_dir=tb_dir,
        report_to=config["report_to"],
        # Reproducibility
        seed=config["seed"],
        data_seed=config["seed"],
        # Dataloader
        dataloader_num_workers=config["dataloader_num_workers"],
        dataloader_pin_memory=True,
        # FSDP (if configured)
        fsdp=config["fsdp"] if config["fsdp"] else "",
        fsdp_config=config.get("fsdp_config"),
        # Distributed
        ddp_find_unused_parameters=False,
        # Misc
        remove_unused_columns=True,
        run_name=config.get("wandb_run_name"),
        # SFT-specific (version dependent)
        **sft_kwargs,
    )

    return training_args


def log_distributed_info():
    """Log distributed training environment info."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    logger.info("=" * 60)
    logger.info("DISTRIBUTED TRAINING INFO")
    logger.info(f"  RANK:       {rank}")
    logger.info(f"  LOCAL_RANK: {local_rank}")
    logger.info(f"  WORLD_SIZE: {world_size}")
    logger.info(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'N/A')}")
    logger.info(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'N/A')}")
    logger.info(f"  CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"    GPU {i}: {torch.cuda.get_device_name(i)} "
                     f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
    logger.info("=" * 60)


def save_training_metadata(config: Dict[str, Any], output_dir: str):
    """Save training metadata for versioning and reproducibility."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_name": config["model_name"],
        "model_revision": config["model_revision"],
        "version_tag": config.get("model_version_tag"),
        "dataset_source": config["dataset_source"],
        "dataset_name": config.get("dataset_name"),
        "quantization": config["quantization"],
        "lora_r": config["lora_r"],
        "lora_alpha": config["lora_alpha"],
        "max_seq_length": config["max_seq_length"],
        "num_train_epochs": config["num_train_epochs"],
        "learning_rate": config["learning_rate"],
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "cuda_devices": torch.cuda.device_count(),
        "torch_version": torch.__version__,
    }
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved training metadata to: {metadata_path}")


def main():
    args = parse_args()

    # Load config: defaults → YAML → env vars → CLI overrides
    config = load_config(args.config)

    if args.dry_run:
        config["max_steps"] = 5
        config["save_steps"] = 5
        config["logging_steps"] = 1
        logger.info("DRY RUN MODE: Running 5 steps only")

    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    # Log environment
    log_distributed_info()
    logger.info(f"Final config:\n{json.dumps(config, indent=2, default=str)}")

    # WandB setup (if enabled) — run login on rank 0 only to avoid multiprocess conflicts
    rank = int(os.environ.get("RANK", 0))
    if "wandb" in config.get("report_to", []):
        if rank == 0:
            import wandb
            wandb_key = os.environ.get("WANDB_API_KEY")
            if wandb_key:
                wandb.login(key=wandb_key)
        if config.get("wandb_run_name") is None:
            config["wandb_run_name"] = (
                f"{config['model_name'].split('/')[-1]}-"
                f"{config.get('model_version_tag', 'run')}-"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

    # Load dataset
    dataset = load_training_dataset(config)

    # Load model
    model, tokenizer = load_model_and_tokenizer(config)

    # Create training arguments
    training_args = create_training_args(config)

    # Ensure output directories exist on all ranks before training starts
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.makedirs(training_args.logging_dir, exist_ok=True)

    # Save metadata (only on rank 0)
    if rank == 0:
        save_training_metadata(config, training_args.output_dir)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # CRITICAL: save_model() and save_state() must be called by ALL ranks
    # when using FSDP — each rank holds a shard that must be gathered.
    # HF Trainer internally gates file I/O to rank 0, but all ranks must
    # participate in the collective save operation.
    logger.info("Saving final model...")
    trainer.save_model()
    trainer.save_state()

    # Log/metadata operations are rank-0 only (no collective needed)
    if rank == 0:
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # Save final metadata
        save_training_metadata(config, training_args.output_dir)

        train_loss = metrics.get("train_loss")
        loss_str = f"{train_loss:.4f}" if isinstance(train_loss, (int, float)) else "N/A"

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"  Output dir: {training_args.output_dir}")
        logger.info(f"  Total FLOPs: {metrics.get('total_flos', 'N/A')}")
        logger.info(f"  Final loss: {loss_str}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
