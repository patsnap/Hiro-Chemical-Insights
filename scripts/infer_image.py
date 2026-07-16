#!/usr/bin/env python
"""Infer CheST reference names and structure types directly from images."""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json
import multiprocessing
import random
import re
from pathlib import Path
from typing import Any, Iterable


MODEL_ID = "PatSnap/Hiro-Chemical-Insights"

TASK_PROMPT = """Please identify the reference name(s) of the chemical molecule within the blue rectangle in the image. Determine which category it belongs to: [Markush structure], [substituent], [specific compound], or [Markush structure, substituent].

- A Markush structure contains a fixed parent nucleus and at least one variable substituent with a defined range (for example, R1).
- A substituent is a component of a complete molecule and cannot form a complete molecule independently. Image context matters: a structure listed in table column "R" is a substituent.
- A Markush structure can itself act as a substituent; use both structure types in that case.
- In tables, a Markush structure can correspond to multiple names, starting at the same vertical height and proceeding downwards until the next structure.
- Return complete reference names. Join a table title and sequence number when needed (for example, "Formula IV"). Return all names if there are multiple. Return "None" if no name is visible.

End with exactly one answer in this format:
\\boxed{[reference name 1, reference name 2]: [structure type 1, structure type 2]}

Examples:
\\boxed{[Compound No. 55]: [substituent, Markush structure]}
\\boxed{[Compound 1, Compound 2]: [Markush structure]}
\\boxed{[3,6-Dichloropyridazine]: [specific compound]}
\\boxed{[None]: [specific compound]}"""

CANONICAL_TYPES = (
    (re.compile(r"\bmarkush\s+structures?\b", re.IGNORECASE), "Markush structure"),
    (re.compile(r"\bsubstituents?\b", re.IGNORECASE), "substituent"),
    (re.compile(r"\bspecific\s+compounds?\b", re.IGNORECASE), "specific compound"),
)


def init_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _balanced_end(text: str, start: int, opening: str, closing: str) -> int | None:
    """Return the index of the closing character matching ``text[start]``."""

    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char == '"':
            quote = char
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def _extract_last_boxed_body(text: str) -> str | None:
    matches = list(re.finditer(r"\\boxed\s*", text, flags=re.IGNORECASE))
    for match in reversed(matches):
        cursor = match.end()
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor < len(text) and text[cursor] == "{":
            end = _balanced_end(text, cursor, "{", "}")
            if end is not None:
                return text[cursor + 1 : end].strip()
        elif cursor < len(text):
            return text[cursor:].strip()
    return None


def _split_reference_names(text: str) -> list[str]:
    """Split list items while preserving commas inside chemical names."""

    items: list[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0}
    pairs = {")": "(", "]": "[", "}": "{"}
    quote: str | None = None
    escaped = False

    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char == '"':
            quote = char
            continue
        if char in depths:
            depths[char] += 1
            continue
        if char in pairs:
            opener = pairs[char]
            depths[opener] = max(0, depths[opener] - 1)
            continue
        if (
            char == ","
            and not any(depths.values())
            and index + 1 < len(text)
            and text[index + 1].isspace()
        ):
            items.append(text[start:index])
            start = index + 1
    items.append(text[start:])

    cleaned = [item.strip().strip("`\"'").strip() for item in items]
    cleaned = [item for item in cleaned if item]
    return ["None" if item.lower() == "none" else item for item in cleaned]


def _parse_pair(text: str) -> tuple[list[str], list[str]] | None:
    """Find and parse the last ``[names]: [types]`` pair in text."""

    candidates: list[tuple[list[str], list[str]]] = []
    for start, char in enumerate(text):
        if char != "[":
            continue
        end = _balanced_end(text, start, "[", "]")
        if end is None:
            continue
        cursor = end + 1
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != ":":
            continue

        type_text = text[cursor + 1 :]
        structure_types = [
            canonical
            for pattern, canonical in CANONICAL_TYPES
            if pattern.search(type_text)
        ]
        if not structure_types:
            continue
        names = _split_reference_names(text[start + 1 : end])
        if names:
            candidates.append((names, structure_types))
    return candidates[-1] if candidates else None


def parse_model_output(text: str) -> dict[str, Any]:
    """Parse the generated final answer into developer-friendly fields."""

    normalized = str(text).replace("{{", "{").replace("}}", "}").strip()
    boxed_body = _extract_last_boxed_body(normalized)
    parsed = _parse_pair(boxed_body) if boxed_body else None
    if parsed is None:
        parsed = _parse_pair(normalized)
    if parsed is None:
        raise ValueError("No valid '[reference names]: [structure types]' answer was found.")

    reference_names, structure_types = parsed
    return {
        "reference_names": reference_names,
        "structure_types": structure_types,
    }


def _chunks(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _build_llm_input(
    image_path: Path,
    processor: Any,
    process_vision_info: Any,
) -> dict[str, Any]:
    from PIL import Image

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.load()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TASK_PROMPT},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, _ = process_vision_info(messages)
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": image_inputs},
    }


def infer_images(args: argparse.Namespace) -> list[dict[str, Any]]:
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    image_paths = [path.expanduser().resolve() for path in args.images]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Image file(s) not found: " + ", ".join(missing))

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    llm_kwargs: dict[str, Any] = {
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
        seed=args.seed,
    )

    results: list[dict[str, Any]] = []
    progress = tqdm(total=len(image_paths), desc="Generating")
    for batch in _chunks(image_paths, args.batch_size):
        llm_inputs = [
            _build_llm_input(path, processor, process_vision_info)
            for path in batch
        ]
        outputs = llm.generate(llm_inputs, sampling_params, use_tqdm=False)
        for image_path, output in zip(batch, outputs):
            raw_output = output.outputs[0].text.strip()
            result: dict[str, Any] = {
                "image": str(image_path),
                "reference_names": [],
                "structure_types": [],
                "parsed": False,
                "raw_output": raw_output,
            }
            try:
                result.update(parse_model_output(raw_output))
                result["parsed"] = True
            except ValueError as exc:
                result["parse_error"] = str(exc)
            results.append(result)
        progress.update(len(batch))
    progress.close()
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Local image path(s).")
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model ID or local path.")
    parser.add_argument("--output", type=Path, help="Optional JSONL output path.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    multiprocessing.set_start_method("spawn", force=True)
    init_seed(args.seed)
    results = infer_images(args)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output_file:
            for result in results:
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    payload: Any = results[0] if len(results) == 1 else results
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
