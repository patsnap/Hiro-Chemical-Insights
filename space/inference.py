"""Model loading, inference, and output parsing for the Gradio Space."""

from __future__ import annotations

import os
import re
import threading
from typing import Any


MODEL_ID = os.getenv("MODEL_REPO_ID", "PatSnap/Hiro-Chemical-Insights")

# Keep these values aligned with scripts/infer_image.py CLI defaults.
INFERENCE_SEED = 42
GENERATION_KWARGS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "max_new_tokens": 4_096,
    "do_sample": True,
    "use_cache": True,
}

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


def _balanced_end(text: str, start: int, opening: str, closing: str) -> int | None:
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


class ModelRunner:
    """Lazily load the private Qwen3-VL checkpoint and run one request at a time."""

    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self.model: Any | None = None
        self.processor: Any | None = None

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        with self._load_lock:
            if self.model is not None and self.processor is not None:
                return

            import torch
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen3VLForConditionalGeneration,
            )

            token = os.getenv("HF_TOKEN")
            if not token:
                raise RuntimeError(
                    "HF_TOKEN is not configured. Add a read token as a Space secret "
                    "because the model repository is private."
                )

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                token=token,
                min_pixels=65_536,
                max_pixels=4_194_304,
            )
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                token=token,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
            ).eval()

    def predict(self, image: Any) -> dict[str, Any]:
        if image is None:
            raise ValueError("Please upload an image first.")

        self._load()
        assert self.model is not None
        assert self.processor is not None

        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": TASK_PROMPT},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with self._inference_lock, torch.inference_mode():
            torch.manual_seed(INFERENCE_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(INFERENCE_SEED)
            generated_ids = self.model.generate(
                **inputs,
                **GENERATION_KWARGS,
            )
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_output = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        result: dict[str, Any] = {
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
        return result
