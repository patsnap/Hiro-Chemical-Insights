---
title: Hiro Chemical Insights
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
pinned: false
suggested_hardware: l4x1
short_description: Chemical structure-text coreference for patent images
models:
  - PatSnap/Hiro-Chemical-Insights
---

# Hiro Chemical Insights

Upload a patent image containing a blue-boxed chemical structure. The model
returns the structure's textual reference name(s) and one or more structure
types:

- `specific compound`
- `substituent`
- `Markush structure`
- `Markush structure, substituent`

The underlying model repository is private. Configure an `HF_TOKEN` Space
secret with read access to `PatSnap/Hiro-Chemical-Insights` before
starting the app.

The model is loaded in 4-bit mode and the recommended Space hardware is one
NVIDIA L4 GPU with 24 GB VRAM.

Generation settings mirror the defaults in
[`scripts/infer_image.py`](https://github.com/patsnap/Hiro-Chemical-Insights/blob/main/scripts/infer_image.py):

- seed: `42`
- temperature: `0.7`
- top-p: `0.8`
- top-k: `20`
- repetition penalty: `1.0`
- maximum new tokens: `4096`
