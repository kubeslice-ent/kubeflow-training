#!/usr/bin/env python3
"""
Dataset Preparation — Format raw datasets for Supervised Fine-Tuning (SFT).

Converts various dataset formats into the standard instruction-tuning format
expected by train.py. Supports custom templates for domain-specific datasets.

Usage:
  # Format Alpaca-style dataset
  python prepare_dataset.py --input /mnt/data/raw --output /mnt/data/prepared --format alpaca

  # Format chat-style dataset (ShareGPT)
  python prepare_dataset.py --input /mnt/data/raw --output /mnt/data/prepared --format sharegpt

  # Format custom dataset with explicit columns
  python prepare_dataset.py --input /mnt/data/raw --output /mnt/data/prepared \
      --format custom --instruction-col question --response-col answer
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("dataset-prepare")


# ---------------------------------------------------------------------------
# Templates — customize these for your domain
# ---------------------------------------------------------------------------

TEMPLATES = {
    "alpaca": {
        "with_input": (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        ),
        "without_input": (
            "Below is an instruction that describes a task. Write a response "
            "that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Response:\n{output}"
        ),
    },
    "chatml": {
        "template": (
            "<|im_start|>system\n{system}<|im_end|>\n"
            "<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n{assistant}<|im_end|>"
        ),
        "default_system": "You are a helpful assistant.",
    },
    "llama": {
        "template": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "{system}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{user}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "{assistant}<|eot_id|>"
        ),
        "default_system": "You are a helpful assistant.",
    },
}


def format_alpaca(example: dict) -> dict:
    """Format an Alpaca-style example."""
    templates = TEMPLATES["alpaca"]
    # Extract only needed keys to prevent KeyError from extra columns
    fields = {
        "instruction": example.get("instruction", ""),
        "input": example.get("input", ""),
        "output": example.get("output", ""),
    }
    if fields["input"].strip():
        text = templates["with_input"].format(**fields)
    else:
        text = templates["without_input"].format(**fields)
    return {"text": text}


def format_sharegpt(example: dict) -> dict:
    """Format a ShareGPT-style conversation into a single text."""
    conversations = example.get("conversations", [])
    if not conversations:
        return {"text": ""}

    parts = []
    for msg in conversations:
        role = msg.get("from", msg.get("role", ""))
        content = msg.get("value", msg.get("content", ""))
        if role in ("system", "human", "user"):
            parts.append(f"### {role.capitalize()}:\n{content}")
        elif role in ("gpt", "assistant"):
            parts.append(f"### Assistant:\n{content}")

    return {"text": "\n\n".join(parts)}


def format_chatml(example: dict, system_msg: str = None) -> dict:
    """Format into ChatML template."""
    template = TEMPLATES["chatml"]
    system = example.get("system", system_msg or template["default_system"])
    user = example.get("instruction", example.get("user", example.get("question", "")))
    assistant = example.get("output", example.get("assistant", example.get("answer", "")))

    text = template["template"].format(system=system, user=user, assistant=assistant)
    return {"text": text}


def format_llama(example: dict, system_msg: str = None) -> dict:
    """Format into Llama 3 chat template."""
    template = TEMPLATES["llama"]
    system = example.get("system", system_msg or template["default_system"])
    user = example.get("instruction", example.get("user", example.get("question", "")))
    assistant = example.get("output", example.get("assistant", example.get("answer", "")))

    text = template["template"].format(system=system, user=user, assistant=assistant)
    return {"text": text}


def format_custom(example: dict, instruction_col: str, response_col: str,
                  input_col: str = None, system_msg: str = None) -> dict:
    """Format custom dataset with explicit column mapping."""
    instruction = example.get(instruction_col, "")
    response = example.get(response_col, "")
    context = example.get(input_col, "") if input_col else ""
    system = system_msg or "You are a helpful assistant."

    if context.strip():
        text = (
            f"### System:\n{system}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Response:\n{response}"
        )
    else:
        text = (
            f"### System:\n{system}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}"
        )
    return {"text": text}


def load_raw_dataset(input_path: str) -> "Dataset":
    """Load raw dataset from various formats."""
    from datasets import load_from_disk, load_dataset

    path = Path(input_path)

    if path.is_dir():
        # Try loading as HF Arrow dataset first
        try:
            return load_from_disk(str(path))
        except Exception:
            pass

        # Try loading JSON/JSONL files from directory
        json_files = list(path.glob("*.json")) + list(path.glob("*.jsonl"))
        if json_files:
            return load_dataset("json", data_files=[str(f) for f in json_files], split="train")

        parquet_files = list(path.glob("*.parquet"))
        if parquet_files:
            return load_dataset("parquet", data_files=[str(f) for f in parquet_files], split="train")

        csv_files = list(path.glob("*.csv"))
        if csv_files:
            return load_dataset("csv", data_files=[str(f) for f in csv_files], split="train")

        raise ValueError(f"Could not find supported data files in: {input_path}")

    elif path.suffix in (".json", ".jsonl"):
        return load_dataset("json", data_files=str(path), split="train")
    elif path.suffix == ".parquet":
        return load_dataset("parquet", data_files=str(path), split="train")
    elif path.suffix == ".csv":
        return load_dataset("csv", data_files=str(path), split="train")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for SFT training")
    parser.add_argument("--input", required=True, help="Input dataset path")
    parser.add_argument("--output", required=True, help="Output path for prepared dataset")
    parser.add_argument("--format", required=True,
                        choices=["alpaca", "sharegpt", "chatml", "llama", "custom"],
                        help="Dataset format to apply")

    # Custom format options
    parser.add_argument("--instruction-col", type=str, default="instruction",
                        help="Column name for instruction (custom format)")
    parser.add_argument("--response-col", type=str, default="output",
                        help="Column name for response (custom format)")
    parser.add_argument("--input-col", type=str, default=None,
                        help="Column name for input/context (custom format)")
    parser.add_argument("--system-message", type=str, default=None,
                        help="System message to prepend")

    # Processing options
    parser.add_argument("--num-proc", type=int, default=4,
                        help="Number of processes for mapping")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--no-filter-empty", action="store_true", default=False,
                        help="Keep samples with empty text (default: filter them out)")

    args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading dataset from: {args.input}")
    dataset = load_raw_dataset(args.input)
    logger.info(f"Loaded {len(dataset)} samples, columns: {dataset.column_names}")

    # Limit samples if requested
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    # Apply formatting
    logger.info(f"Applying {args.format} format...")

    if args.format == "alpaca":
        dataset = dataset.map(format_alpaca, num_proc=args.num_proc)
    elif args.format == "sharegpt":
        dataset = dataset.map(format_sharegpt, num_proc=args.num_proc)
    elif args.format == "chatml":
        dataset = dataset.map(
            lambda x: format_chatml(x, args.system_message),
            num_proc=args.num_proc
        )
    elif args.format == "llama":
        dataset = dataset.map(
            lambda x: format_llama(x, args.system_message),
            num_proc=args.num_proc
        )
    elif args.format == "custom":
        dataset = dataset.map(
            lambda x: format_custom(x, args.instruction_col, args.response_col,
                                    args.input_col, args.system_message),
            num_proc=args.num_proc
        )

    # Filter empty
    if not args.no_filter_empty:
        before = len(dataset)
        dataset = dataset.filter(lambda x: len(x.get("text", "").strip()) > 0)
        after = len(dataset)
        if before != after:
            logger.info(f"Filtered empty samples: {before} → {after}")
        if after == 0:
            logger.error("All samples were empty after formatting! Check your dataset columns and format.")
            sys.exit(1)

    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))

    # Log sample
    logger.info(f"Saved {len(dataset)} prepared samples to: {output_path}")
    if len(dataset) > 0:
        sample_text = dataset[0].get('text', '')
        logger.info(f"Sample text (first 500 chars):\n{sample_text[:500]}")

    # Save metadata
    metadata = {
        "source": args.input,
        "format": args.format,
        "num_samples": len(dataset),
        "columns": dataset.column_names,
    }
    with open(output_path / "preparation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main()
