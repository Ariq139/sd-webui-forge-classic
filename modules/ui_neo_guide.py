"""Neo in-app guide."""

from pathlib import Path

import gradio as gr

from modules.paths_internal import script_path


GUIDE_PATH = Path(script_path) / "docs" / "EDIT_MODELS.md"


def _read_guide() -> str:
    try:
        return GUIDE_PATH.read_text(encoding="utf-8")
    except OSError:
        return (
            "# Instructions\n\n"
            "The Neo edit-model guide could not be loaded. "
            "See the repository `docs/EDIT_MODELS.md` file."
        )


def create_ui():
    with gr.Blocks(elem_id="neo_edit_models_page") as interface:
        gr.Markdown(_read_guide(), elem_id="neo_edit_models_guide")

    return interface
