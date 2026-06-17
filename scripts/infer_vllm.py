#!/usr/bin/env python
"""Run vLLM inference on CheST/RULER data files.

The prompt is read from each data record. The script checks user messages first,
then prompt-like fields such as prompt/problem/question. It does not use a
hard-coded task prompt.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json
import multiprocessing
import random
from pathlib import Path
from typing import Any


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "records", "examples", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported JSON structure in {path}")


def init_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except Exception:
        pass


def strip_image_placeholder(text: str) -> str:
    return text.replace("<image>", "").strip()


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks)
    return str(content) if content is not None else ""


def extract_prompt(record: dict[str, Any]) -> tuple[str, str]:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                return strip_image_placeholder(message_text(message.get("content"))), "messages"

    for key in ("prompt", "problem", "question", "instruction"):
        value = record.get(key)
        if value is not None:
            return strip_image_placeholder(str(value)), key
    raise ValueError("Record has no prompt field: expected messages, prompt, problem, question, or instruction.")


def resolve_image_path(image_value: str, image_root: Path) -> Path:
    candidate = Path(image_value)
    if candidate.exists():
        return candidate

    normalized = image_value.replace("\\", "/")
    marker = "/images/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        candidate = image_root / Path(*suffix.split("/"))
        if candidate.exists():
            return candidate

    candidate = image_root / Path(image_value).name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Cannot resolve image path: {image_value}")


def chunked(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def build_llm_inputs(records: list[dict[str, Any]], image_root: Path, processor: Any, process_vision_info: Any):
    from PIL import Image

    llm_inputs = []
    metadata = []
    for record in records:
        prompt_text, prompt_source = extract_prompt(record)
        image_values = record.get("images") or record.get("image") or record.get("image_path")
        if isinstance(image_values, str):
            image_values = [image_values]
        if not image_values:
            raise ValueError("Record has no image field: expected images, image, or image_path.")

        content = []
        resolved_images = []
        for image_value in image_values:
            image_path = resolve_image_path(str(image_value), image_root)
            image = Image.open(image_path).convert("RGB")
            image.load()
            content.append({"type": "image", "image": image})
            resolved_images.append(str(image_path))
        content.append({"type": "text", "text": prompt_text})

        message = [{"role": "user", "content": content}]
        input_text = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, _ = process_vision_info(message)
        llm_inputs.append({"prompt": input_text, "multi_modal_data": {"image": image_inputs}})
        metadata.append({"prompt_source": prompt_source, "resolved_images": resolved_images})
    return llm_inputs, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Hanmeng-Zhong/Hiro-Chemical-Insights")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--image-root", default=Path("images"), type=Path)
    parser.add_argument("--output", default=Path("outputs/predictions.jsonl"), type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)
    init_seed(args.seed)

    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    records = read_records(args.data)
    if args.limit is not None:
        records = records[: args.limit]

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    llm_kwargs = {
        "model": args.model,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "load_format": "auto",
        "tensor_parallel_size": args.tensor_parallel_size,
        "seed": args.seed,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        progress = tqdm(total=len(records), desc="Generating")
        for _, batch in chunked(records, args.batch_size):
            llm_inputs, metadata = build_llm_inputs(
                batch,
                args.image_root,
                processor,
                process_vision_info,
            )
            outputs = llm.generate(llm_inputs, sampling_params, use_tqdm=False)
            for record, meta, output in zip(batch, metadata, outputs):
                result = dict(record)
                result["res"] = output.outputs[0].text
                result["_inference"] = meta
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            progress.update(len(batch))
        progress.close()


if __name__ == "__main__":
    main()
