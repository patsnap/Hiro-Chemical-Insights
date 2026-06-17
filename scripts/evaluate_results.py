#!/usr/bin/env python
"""Evaluate CheST/RULER prediction files with set-based Pass@ metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BOXED_RE = re.compile(r"\\boxed\s*\{?(?P<body>\[.*?)(?:\}\s*)?$", re.DOTALL)
PAIR_RE = re.compile(r"^\s*\[(?P<names>.*?)\]\s*:\s*(?P<types>.*?)\s*$", re.DOTALL)


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


def get_nested(record: dict[str, Any], dotted_key: str) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def first_present(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = get_nested(record, key)
        if value is not None:
            return value
    return None


def split_items(text: str, case_insensitive: bool = False) -> set[str]:
    text = text.strip().strip("{}").strip()
    text = text.strip("[]").strip()
    if not text:
        return set()
    items = []
    for item in text.split(","):
        item = item.strip().strip("\"'").strip()
        if item:
            items.append(item.lower() if case_insensitive else item)
    return set(items)


def extract_formula_info(text: Any, case_insensitive: bool = False) -> dict[str, set[str]] | None:
    if text is None:
        return None
    text = str(text).replace("{{", "{").replace("}}", "}").strip()
    boxed = None
    matches = list(BOXED_RE.finditer(text))
    if matches:
        boxed = matches[-1].group("body").strip()
    else:
        bracket_start = text.rfind("[")
        if bracket_start >= 0:
            boxed = text[bracket_start:].strip().rstrip("}")
    if not boxed:
        return None

    match = PAIR_RE.search(boxed)
    if not match:
        return None
    return {
        "names": split_items(match.group("names"), case_insensitive),
        "types": split_items(match.group("types"), case_insensitive),
    }


def evaluate_record(
    record: dict[str, Any],
    prediction_field: str,
    case_insensitive: bool = False,
) -> dict[str, Any]:
    gold_text = first_present(record, ["answer", "extra_info.answer", "label", "gold"])
    pred_text = get_nested(record, prediction_field)
    gold = extract_formula_info(gold_text, case_insensitive)
    pred = extract_formula_info(pred_text, case_insensitive)

    result = {
        "parsed_gold": gold is not None,
        "parsed_pred": pred is not None,
        "pass1_name": 0,
        "passall_name": 0,
        "pass1_structure": 0,
        "passall_structure": 0,
        "pass1_all": 0,
        "passall_all": 0,
    }
    if not gold or not pred:
        return result

    name_hit = bool(pred["names"] & gold["names"])
    name_exact = pred["names"] == gold["names"]
    type_hit = bool(pred["types"] & gold["types"])
    type_exact = pred["types"] == gold["types"]

    result.update(
        {
            "pass1_name": int(name_hit),
            "passall_name": int(name_exact),
            "pass1_structure": int(type_hit),
            "passall_structure": int(type_exact),
            "pass1_all": int(name_hit and type_hit),
            "passall_all": int(name_exact and type_exact),
        }
    )
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        raise ValueError("No records to evaluate.")
    keys = [
        "pass1_name",
        "passall_name",
        "pass1_structure",
        "passall_structure",
        "pass1_all",
        "passall_all",
    ]
    metrics = {key: sum(item[key] for item in results) / total for key in keys}
    return {
        "num_records": total,
        "parsed_gold": sum(item["parsed_gold"] for item in results),
        "parsed_pred": sum(item["parsed_pred"] for item in results),
        "pass@1": {
            "RefMatch": metrics["pass1_name"],
            "StruCls": metrics["pass1_structure"],
            "All": metrics["pass1_all"],
        },
        "pass@all": {
            "RefMatch": metrics["passall_name"],
            "StruCls": metrics["passall_structure"],
            "All": metrics["passall_all"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Prediction JSON/JSONL file.")
    parser.add_argument(
        "--prediction-field",
        default="res",
        help="Dotted field containing model output, for example res or output.text.",
    )
    parser.add_argument("--case-insensitive", action="store_true")
    parser.add_argument("--details-out", type=Path, help="Optional per-record metric JSONL.")
    args = parser.parse_args()

    records = read_records(args.input)
    results = [
        evaluate_record(record, args.prediction_field, args.case_insensitive)
        for record in records
    ]
    summary = summarize(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.details_out:
        args.details_out.parent.mkdir(parents=True, exist_ok=True)
        with args.details_out.open("w", encoding="utf-8") as f:
            for record, result in zip(records, results):
                f.write(json.dumps({"sample": record, "metrics": result}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
