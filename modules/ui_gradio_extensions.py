import json
import os

import gradio as gr
import gradio.processing_utils

from modules import localization, scripts, shared, util
from modules.paths import data_path, script_path


_assets_head = ""
_assets_css_paths = []
_assets_css = ""
_assets_js = ""


def _install_gradio_dropdown_compatibility():
    """Preserve Gradio 4's unset Dropdown values under Gradio 5 validation.

    Gradio 4 allowed an unset single-select Dropdown to submit ``None`` even
    when its choices list was populated. Gradio 5 validates that value before
    the callback runs and raises ``Value is not in the list of choices``.
    Several optional webui controls rely on the old unset state, including
    hidden Hires.fix controls and settings controls. Keep ``None`` as ``None``
    so the backend can apply its existing defaults instead of inventing a
    choice or preventing the event from running.
    """

    original_preprocess = gr.Dropdown.preprocess
    if getattr(original_preprocess, "_sd_webui_empty_multiselect_compat", False):
        return

    def preprocess(self, payload):
        if not self.multiselect and payload in (None, ""):
            return None

        # Some Gradio 5 browser payloads serialize an unset legacy value as
        # the literal string "None". Only normalize it when the component
        # does not offer "None" as a real choice.
        if not self.multiselect and payload == "None":
            choice_values = [value for _, value in self.choices]
            if payload not in choice_values:
                return None

        if self.multiselect and payload == "":
            payload = []

        return original_preprocess(self, payload)

    preprocess._sd_webui_empty_multiselect_compat = True
    gr.Dropdown.preprocess = preprocess


_install_gradio_dropdown_compatibility()


def webpath(fn):
    api_prefix = getattr(gradio.processing_utils, "API_PREFIX", "")
    return f"{api_prefix}/file={util.truncate_path(fn)}?{os.path.getmtime(fn)}"


def javascript_js():
    """Load the legacy scripts in order through Gradio 5's JS hook.

    Gradio 4 inserted ordinary script tags into the page. Gradio 5 renders
    ``head`` as HTML after the page is mounted, so those tags are inert in this
    application. The Blocks JS hook does run after hydration, but this browser
    context intentionally does not expose ``window.eval``. Load each file with
    a real script element and await its completion so the old global callback
    queues and extension ordering are retained without either failure mode.
    """

    sources = []
    script_js = os.path.join(script_path, "script.js")
    sources.append(script_js)
    for script in scripts.list_scripts("javascript", ".js"):
        sources.append(script.path)

    for script in scripts.list_scripts("javascript", ".mjs"):
        sources.append(script.path)

    source_urls = [
        {"src": webpath(source_path), "module": source_path.endswith(".mjs")}
        for source_path in sources
    ]
    localization_js = localization.localization_js(shared.opts.localization).replace("</", "<\\/")
    theme_js = f'set_theme({json.dumps(shared.cmd_opts.theme)});' if shared.cmd_opts.theme else ""

    return f'''async function() {{
    {localization_js};
    const scripts = {json.dumps(source_urls).replace("</", "<\\/")};
    for (const item of scripts) {{
        await new Promise((resolve, reject) => {{
            const script = document.createElement("script");
            script.type = item.module ? "module" : "text/javascript";
            script.src = item.src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        }});
    }}
    {theme_js}
}}'''


def css_paths():
    paths = list(scripts.list_files_with_name("style.css"))
    user_css = os.path.join(data_path, "user.css")
    if os.path.exists(user_css):
        paths.append(user_css)

    return paths


def css_text():
    """Return CSS that cannot be represented as a Gradio css path."""

    from modules.shared_gradio_themes import resolve_var

    light = resolve_var("background_fill_primary")
    dark = resolve_var("background_fill_primary_dark")
    # The browser's overscroll area belongs to `html`, while Gradio applies
    # the dark theme class to `body`. Set both elements and mirror the body
    # theme on html so manual dark mode cannot expose a white overscroll area.
    return (
        f"html, body {{ background-color: {light} !important; }} "
        f"body.dark {{ background-color: {dark} !important; }} "
        f"html.dark, html:has(.dark) {{ background-color: {dark} !important; }} "
        f"@media (prefers-color-scheme: dark) {{ html, body {{ background-color: {dark} !important; }} }}"
    )


def _path_list(value):
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [value]
    return list(value)


def _blocks_init(self, *args, **kwargs):
    if _assets_head:
        existing_head = kwargs.get("head")
        kwargs["head"] = f"{_assets_head}\n{existing_head or ''}"

    if _assets_css_paths:
        kwargs["css_paths"] = [*_assets_css_paths, *_path_list(kwargs.get("css_paths"))]

    if _assets_css:
        existing_css = kwargs.get("css") or ""
        kwargs["css"] = f"{_assets_css}\n{existing_css}"

    if _assets_js:
        kwargs["js"] = _assets_js

    return _blocks_init_original(self, *args, **kwargs)


_blocks_init_original = gr.Blocks.__init__
if not getattr(gr.Blocks, "_sd_webui_assets_patched", False):
    gr.Blocks.__init__ = _blocks_init
    gr.Blocks._sd_webui_assets_patched = True



def reload_javascript():
    global _assets_head, _assets_css_paths, _assets_css, _assets_js

    _assets_head = '<meta name="referrer" content="no-referrer"/>'
    _assets_css_paths = css_paths()
    _assets_css = css_text()
    _assets_js = javascript_js()
