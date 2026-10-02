#!/usr/bin/env python3
"""Gated PEFT/TRL SFT run for successful public Gemma 4 tool trajectories.

This is an offline template. Run only after the exact-QAT vLLM adapter-effect
gate passes and after reviewing a processor-rendered tool-call example.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS = [
    tool("run_command", "Run a shell command in /workspace", {"command": {"type": "string"}}, ["command"]),
    tool("submit_patch", "Stage and capture the current patch", {}, []),
    tool("get_status", "Return live task budget and patch status", {}, []),
    tool("read_file", "Read a file from /workspace", {
        "filepath": {"type": "string"}, "start_line": {"type": ["integer", "null"]},
        "end_line": {"type": ["integer", "null"]},
    }, ["filepath"]),
    tool("edit_file", "Replace a unique string in an existing file", {
        "filepath": {"type": "string"}, "old_string": {"type": "string"},
        "new_string": {"type": "string"}, "allow_multiple": {"type": "boolean"},
    }, ["filepath", "old_string", "new_string"]),
    tool("write_file", "Create or overwrite a file under /workspace", {
        "filepath": {"type": "string"}, "content": {"type": "string"},
    }, ["filepath", "content"]),
    tool("get_code_neighbors", "Find graph neighbors of a symbol", {
        "node": {"type": "string"}, "edge_type": {"type": ["string", "null"]},
        "max_neighbors": {"type": "integer"},
    }, ["node"]),
    tool("search_similar_code", "Find graph nodes similar to a symbol", {
        "query": {"type": "string"}, "k": {"type": "integer"},
    }, ["query"]),
    tool("get_code_subgraph", "Return edges among supplied graph symbols", {
        "nodes": {"type": "array", "items": {"type": "string"}},
    }, ["nodes"]),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="JSONL produced by rollout_sampling.py")
    parser.add_argument("--model", required=True, help="local bf16 Gemma 4 31B IT path; do not download")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-kind", choices=("bf16", "qat"), default="bf16")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--gate-passed", action="store_true", help="required after QAT vLLM adapter-effect test passes")
    args = parser.parse_args()
    if not args.gate_passed:
        parser.error("refusing training until the exact competition-QAT vLLM adapter-effect gate passes")
    if args.base_kind != "bf16":
        parser.error("v1 supports bf16 training only; QAT training needs a separate backward/optimizer validation")
    if not Path(args.model).is_dir():
        parser.error("model path must be a locally mounted directory; notebook internet is disabled")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        parser.error(f"training dependencies are unavailable in this runtime: {exc}")

    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        parser.error("no training examples")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    for row in rows[:3]:
        processor.apply_chat_template(row["messages"], tools=TOOL_SCHEMAS, tokenize=False,
                                      add_generation_prompt=False, enable_thinking=False)

    def render(row):
        return {"text": processor.apply_chat_template(
            row["messages"], tools=TOOL_SCHEMAS, tokenize=False,
            add_generation_prompt=False, enable_thinking=False,
        )}

    dataset = Dataset.from_list(rows)
    train_data = dataset.map(render, remove_columns=dataset.column_names)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    train_args = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_data,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train()
    adapter_dir = args.output / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(adapter_dir)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_dir / filename).is_file():
            raise RuntimeError(f"PEFT did not create required file {filename}")
    print(f"Saved adapter to {adapter_dir}; validate it on the exact QAT vLLM before packaging")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
