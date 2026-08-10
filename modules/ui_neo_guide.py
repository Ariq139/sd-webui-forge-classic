"""Neo in-app guide."""

from pathlib import Path

import gradio as gr

from modules.paths_internal import script_path


GUIDE_PATH = Path(script_path) / "docs" / "INSTRUCTIONS.md"


def _read_guide() -> str:
    try:
        return GUIDE_PATH.read_text(encoding="utf-8")
    except OSError:
        return (
            "# Instructions\n\n"
            "The Neo instructions could not be loaded. "
            "See the repository `docs/INSTRUCTIONS.md` file."
        )


def create_ui():
    with gr.Blocks(elem_id="neo_edit_models_page") as interface:
        gr.Markdown(_read_guide(), elem_id="neo_edit_models_guide")

    return interface
