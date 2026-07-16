"""Gradio application for the Hiro Chemical Insights model."""

from __future__ import annotations

import gradio as gr

from inference import MODEL_ID, ModelRunner


RUNNER = ModelRunner()

CSS = """
.gradio-container { max-width: 1120px !important; }
.hero { text-align: center; margin: 0 auto 1rem; }
.hero h1 { font-size: 2.1rem; margin-bottom: 0.35rem; }
.result-card { border-radius: 14px; }
"""


def run_inference(image):
    try:
        result = RUNNER.predict(image)
    except Exception as exc:
        raise gr.Error(f"Inference failed: {exc}") from exc

    parsed_result = {
        "reference_names": result["reference_names"],
        "structure_types": result["structure_types"],
        "parsed": result["parsed"],
    }
    if "parse_error" in result:
        parsed_result["parse_error"] = result["parse_error"]

    return (
        parsed_result,
        "\n".join(result["reference_names"]),
        "\n".join(result["structure_types"]),
        result["raw_output"],
    )


with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Hiro Chemical Insights") as demo:
    gr.Markdown(
        """
        <div class="hero">
          <h1>🧪 Hiro Chemical Insights</h1>
          <p>Upload a patent image containing a blue-boxed chemical structure.<br>
          The model identifies its textual reference name and structure type.</p>
        </div>

        Structure types: `specific compound`, `substituent`, `Markush structure`,
        or `Markush structure, substituent`.
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            image_input = gr.Image(
                type="pil",
                image_mode="RGB",
                sources=["upload", "clipboard"],
                label="Patent image",
                height=520,
            )
            with gr.Row():
                clear_button = gr.ClearButton(value="Clear", components=[image_input])
                submit_button = gr.Button("Analyze structure", variant="primary")

        with gr.Column(scale=5):
            with gr.Group(elem_classes="result-card"):
                reference_names = gr.Textbox(
                    label="Reference name(s)",
                    lines=3,
                    interactive=False,
                )
                structure_types = gr.Textbox(
                    label="Structure type(s)",
                    lines=2,
                    interactive=False,
                )
            parsed_json = gr.JSON(label="Parsed result")

    with gr.Accordion("Raw model response", open=False):
        raw_output = gr.Textbox(
            label="Raw response",
            lines=12,
            interactive=False,
        )

    gr.Markdown(
        f"Model: [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) · "
        "[GitHub](https://github.com/patsnap/Hiro-Chemical-Insights)"
    )

    submit_button.click(
        fn=run_inference,
        inputs=image_input,
        outputs=[parsed_json, reference_names, structure_types, raw_output],
        api_name="predict",
        concurrency_limit=1,
        show_progress="full",
    )


demo.queue(default_concurrency_limit=1, max_size=8)

if __name__ == "__main__":
    demo.launch()
