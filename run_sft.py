"""SFT training script using TRL SFTTrainer.

Trains a LoRA adapter on audit traces (Claude + Qwen3.5-397B) for Solidity
vulnerability detection. Reads config from training.yaml.

The training data uses structured tool_calls fields (OpenAI format).
TRL + the model's tokenizer handle native formatting (qwen3_coder for Qwen3.5).

Usage:
    python3 run_sft.py --model Qwen/Qwen3.5-9B --data data/sft_traces_structured.jsonl --output outputs/sft
    python3 run_sft.py --model Qwen/Qwen3.5-9B --data data/sft_traces_structured.jsonl --output outputs/sft --max-samples 50
"""

import argparse
import json
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="SFT on audit traces")
    parser.add_argument("--model", required=True, help="Base model name or path")
    parser.add_argument("--data", required=True, help="JSONL training data")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", default="training.yaml", help="Training config")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit training samples (0=all, useful for testing)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", {})
    lora_config = config.get("lora", {})
    sft_config = config.get("sft", {})

    print(f"Model: {args.model}")
    print(f"Data:  {args.data}")
    print(f"Config: {args.config}")
    print(f"LoRA r={lora_config.get('r', 64)}, alpha={lora_config.get('alpha', 128)}")

    # Import heavy deps after arg parsing (fast fail on bad args)
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # Load data
    print(f"\nLoading training data from {args.data}...")
    records = []
    with open(args.data) as f:
        for line in f:
            records.append(json.loads(line))

    if args.max_samples > 0:
        records = records[:args.max_samples]
        print(f"  Limited to {len(records)} samples")

    # Convert to chat format for TRL
    # Each record has "messages" (list of role/content dicts)
    chat_data = []
    for r in records:
        messages = r.get("messages", [])
        if messages:
            chat_data.append({"messages": messages})

    print(f"  {len(chat_data)} training conversations loaded")
    if not chat_data:
        print("ERROR: No valid training data found")
        return

    dataset = Dataset.from_list(chat_data)

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Apply LoRA
    target_modules = lora_config.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    peft_config = LoraConfig(
        r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("alpha", 128),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=target_modules,
        use_rslora=lora_config.get("use_rslora", True),
        task_type="CAUSAL_LM",
    )
    print(f"LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}, "
          f"targets={len(target_modules)} modules")

    # SFT config
    max_seq_length = model_config.get("max_seq_length", 16384)
    sft_kwargs = dict(
        output_dir=args.output,
        num_train_epochs=sft_config.get("epochs", 3),
        per_device_train_batch_size=sft_config.get("batch_size", 1),
        gradient_accumulation_steps=sft_config.get("gradient_accumulation_steps", 8),
        learning_rate=sft_config.get("learning_rate", 2e-5),
        warmup_steps=sft_config.get("warmup_steps", 10),
        weight_decay=sft_config.get("weight_decay", 0.01),
        lr_scheduler_type=sft_config.get("lr_scheduler_type", "cosine"),
        gradient_checkpointing=sft_config.get("gradient_checkpointing", True),
        bf16=True,
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        report_to="none",
    )
    # max_seq_length moved from SFTConfig to SFTTrainer in TRL >=0.28
    import inspect
    if "max_seq_length" in inspect.signature(SFTConfig.__init__).parameters:
        sft_kwargs["max_seq_length"] = max_seq_length
        sft_kwargs["packing"] = sft_config.get("packing", True)
    training_args = SFTConfig(**sft_kwargs)

    print(f"\nTraining: {training_args.num_train_epochs} epochs, "
          f"lr={training_args.learning_rate}, "
          f"max_seq_length={max_seq_length}, "
          f"packing={training_args.packing}")

    # Train
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    # max_seq_length on SFTTrainer for TRL >=0.28
    if "max_seq_length" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["max_seq_length"] = max_seq_length
    trainer = SFTTrainer(**trainer_kwargs)

    print(f"\nStarting SFT training...")
    trainer.train()

    # Save
    print(f"\nSaving adapter to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done!")


if __name__ == "__main__":
    main()
