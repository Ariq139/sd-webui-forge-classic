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
    # Gradio 5 isolates/async-loads external head scripts. Bundle the project
    # scripts into one ordered global evaluation through Blocks(js=...) so
    # extensions see the webui callback functions before registering handlers.
    sources = []
    script_js = os.path.join(script_path, "script.js")
    sources.append(script_js)
    for script in scripts.list_scripts("javascript", ".js"):
        sources.append(script.path)

    for script in scripts.list_scripts("javascript", ".mjs"):
        sources.append(script.path)

    bundled_sources = []
    for source_path in sources:
        try:
            with open(source_path, "r", encoding="utf8") as source_file:
                bundled_sources.append({"source": source_file.read(), "module": source_path.endswith(".mjs")})
        except OSError:
            bundled_sources.append({"src": webpath(source_path), "module": source_path.endswith(".mjs")})

    localization_js = localization.localization_js(shared.opts.localization).replace("</", "<\\/")
    theme_js = f'set_theme({json.dumps(shared.cmd_opts.theme)});' if shared.cmd_opts.theme else ""
    classic_sources = [item["source"] for item in bundled_sources if "source" in item and not item["module"]]
    module_sources = [item for item in bundled_sources if item.get("module")]
    external_sources = [item for item in bundled_sources if "src" in item]
    classic_bundle = "\n;\n".join(classic_sources).replace("</", "<\\/")
    modules_json = json.dumps(module_sources).replace("</", "<\\/")
    external_json = json.dumps(external_sources).replace("</", "<\\/")

    return f'''function() {{
    {localization_js};
    // Evaluate classic project scripts as one program so their shared
    // callback queues and options state retain the Gradio 4 behavior.
    window.eval({json.dumps(classic_bundle)});
    const scripts = {modules_json};
    for (const item of scripts) {{
        const script = document.createElement("script");
        script.type = item.module ? "module" : "text/javascript";
        if (item.source) script.textContent = item.source;
        else script.src = item.src;
        document.head.appendChild(script);
    }}
    for (const item of {external_json}) {{
        const script = document.createElement("script");
        script.type = item.module ? "module" : "text/javascript";
        script.src = item.src;
        document.head.appendChild(script);
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
    return f"html {{ background-color: {light}; }} @media (prefers-color-scheme: dark) {{ html {{background-color: {dark}; }} }}"


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
