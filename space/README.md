---
title: Hiro Chemical Insights
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
short_description: Static interface for chemical structure-text coreference
models:
  - PatSnap/Hiro-Chemical-Insights
---

# Hiro Chemical Insights

This Space is a static HTML/JavaScript interface for the Hiro Chemical Insights
model. It can upload and preview a patent image, call an external inference API,
parse the model response, and display the reference name(s) and structure type(s).

Static Spaces do not run Python, PyTorch, Gradio, or the model itself. Without an
external inference endpoint, the page operates in preview-only mode and never
fabricates a prediction.

## Configure inference

Add a non-secret Space variable named `INFERENCE_API_URL`, or enter a URL in the
page's **API connection** panel for the current browser session. The endpoint must
be reachable from the browser over HTTPS and allow CORS requests from the Space.

Do not put a private Hugging Face token or other credential in a Static Space
variable: Static Space configuration is available to client-side JavaScript.

The frontend sends:

```http
POST ${INFERENCE_API_URL}
Content-Type: multipart/form-data

image=<uploaded image file>
```

The preferred response is:

```json
{
  "reference_names": ["Compound No. 55"],
  "structure_types": ["Markush structure", "substituent"],
  "raw_output": "..."
}
```

The endpoint may instead return only `raw_output`. The browser parser accepts a
final answer such as:

```text
\boxed{[Compound No. 55]: [Markush structure, substituent]}
```

## Inference defaults

The external backend should mirror the defaults in
[`scripts/infer_image.py`](https://github.com/patsnap/Hiro-Chemical-Insights/blob/main/scripts/infer_image.py):

- seed: `42`
- temperature: `0.7`
- top-p: `0.8`
- top-k: `20`
- repetition penalty: `1.0`
- maximum new tokens: `4096`

For local GPU inference without an HTTP endpoint, run:

```bash
python scripts/infer_image.py path/to/image.jpeg
```

## Static files

- `index.html`: accessible interface and result layout
- `style.css`: responsive visual design
- `app.js`: image handling, API request, output parsing, and rendering
