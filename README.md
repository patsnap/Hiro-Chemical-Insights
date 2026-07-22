# Hiro-Chemical-Insights

This private repository contains the data files and molecule-image assets for
the CheST chemical structure-text coreference experiments described in the ACL
2026 accepted paper:

**Multimodal Chemical Structure-Text Coreference in Intellectual Property via
Rule-guided Reinforcement Learning**.

- Paper: [ACL 2026](https://aclanthology.org/2026.findings-acl.1489/)
- Hugging Face model: [Hiro-Chemical-Insights](https://huggingface.co/PatSnap/Hiro-Chemical-Insights)

## Overview

CheST targets multimodal chemical structure-text coreference in intellectual
property documents. Given a patent page image with a boxed chemical structure,
the task is to identify the structure's textual reference name and classify its
structure type, such as `specific compound`, `substituent`, `Markush structure`,
or `Markush structure, substituent`.

RULER is a rule-guided multimodal reinforcement learning framework built on a
supervised fine-tuning cold start. It uses rule-based rewards for output format,
reference-name matching, and structure-type classification.

![CheST task introduction](assets/task-introduction.png)

## Repository Contents

| Path | Description |
| --- | --- |
| `cot_sft_train.json` | 176 supervised fine-tuning examples in chat/message format. |
| `rl_train.json` | 1,593 reinforcement-learning training examples. |
| `rl_train.parquet` | Parquet version of `rl_train.json`. |
| `test.json` | 198 test examples. |
| `test.parquet` | Parquet version of the test split. |
| `images/` | 1,538 boxed molecule images from 7 patent documents. |
| `assets/task-introduction.png` | Task introduction figure. |
| `paper/2026.acl-findings.4540.pdf` | Local copy of the ACL 2026 accepted manuscript. |

The data files use repository-relative image paths, for example
`images/US20230002396A1/page_120_0_mol_with_box.jpeg`.

## Evaluation

The paper evaluates three set-based tasks:

- `RefMatch`: reference-name matching.
- `StruCls`: structure-type classification.
- `All`: exact success on both reference names and structure types.

Two metrics are reported:

- `Pass@1`: at least one predicted item overlaps with the gold set.
- `Pass@all`: the predicted set exactly matches the gold set.

## Main Results

Main results are shown below.

| Model | RefMatch Pass@1 | StruCls Pass@1 | All Pass@1 | RefMatch Pass@all | StruCls Pass@all | All Pass@all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5 | 63.13 | 87.88 | 59.60 | 53.54 | 86.36 | 50.51 |
| Claude-4.5-sonnet | 57.07 | 78.79 | 48.99 | 45.96 | 73.74 | 36.87 |
| Gemini-2.5-pro | 75.25 | 83.84 | 73.23 | 66.16 | 82.32 | 63.13 |
| GLM-4.5V | 56.06 | 81.31 | 49.49 | 53.54 | 76.77 | 44.95 |
| Qwen-vl-max | 47.98 | 76.77 | 43.43 | 44.44 | 66.16 | 33.84 |
| Qwen3-vl-8B | 44.95 | 34.34 | 17.17 | 40.40 | 29.29 | 10.61 |
| Hiro-Chemical-Insights | **93.43** | **98.48** | **91.92** | **90.40** | **97.98** | **88.38** |

Compared with the strongest general MLLM baseline, Gemini-2.5-Pro,
Hiro-Chemical-Insights improves `All Pass@1` from 73.23 to 91.92 and
`All Pass@all` from 63.13 to 88.38.

## Data Format

`rl_train.json` and `test.json` use:

```json
{
  "images": ["images/<patent_id>/<image_name>.jpeg"],
  "problem": "<image>",
  "answer": "\\boxed{[reference name]: structure type}"
}
```

`cot_sft_train.json` uses a chat-style schema:

```json
{
  "images": ["images/<patent_id>/<image_name>.jpeg"],
  "messages": [
    {"role": "user", "content": "<image>..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Notes

The image assets and Parquet files are tracked with Git LFS.

## Inference And Evaluation

Install the inference dependencies in a GPU environment:

```bash
pip install -r requirements-inference.txt
```

Run vLLM inference. The script reads the prompt from each data record
(`messages`, `prompt`, `problem`, `question`, or `instruction`) and removes only
the `<image>` placeholder because the image is passed to the processor as visual
input.

```bash
python scripts/infer_vllm.py \
  --model PatSnap/Hiro-Chemical-Insights \
  --data test.json \
  --image-root images \
  --output outputs/test_predictions.jsonl \
  --trust-remote-code
```

For your own images, use the direct image entry point. The CheST task prompt is
built in, so an image path is the only required argument:

```bash
python scripts/infer_image.py path/to/your/image.jpeg
```

The command prints parsed JSON and retains the raw model response for debugging:

```json
{
  "image": "/absolute/path/to/image.jpeg",
  "reference_names": ["Compound No. 55"],
  "structure_types": ["Markush structure", "substituent"],
  "parsed": true,
  "raw_output": "...\\boxed{[Compound No. 55]: [Markush structure, substituent]}"
}
```

Multiple images can be inferred together and saved as JSONL:

```bash
python scripts/infer_image.py image1.jpeg image2.png \
  --output outputs/custom_predictions.jsonl
```

Evaluate a prediction JSONL produced by the inference script:

```bash
python scripts/evaluate_results.py \
  --input outputs/test_predictions.jsonl \
  --prediction-field res
```

Parser sanity check using the gold answer as the prediction field:

```bash
python scripts/evaluate_results.py --input test.json --prediction-field answer
```
