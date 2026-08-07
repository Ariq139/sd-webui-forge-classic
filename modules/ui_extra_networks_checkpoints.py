import html
import os

from modules import shared, ui_extra_networks, sd_models
from modules.ui_extra_networks_checkpoints_user_metadata import CheckpointUserMetadataEditor


class ExtraNetworksPageCheckpoints(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Checkpoints')

        self.allow_prompt = False

    def refresh(self):
        shared.refresh_checkpoints()

    def create_item(self, name, index=None, enable_filter=True):
        # `checkpoints_list` is keyed by the checkpoint title, while
        # `checkpoint_aliases` is rebuilt during model discovery.  Use the
        # canonical list first so the page remains populated even if aliases
        # have not been rebuilt yet or a caller supplies a title directly.
        checkpoint: sd_models.CheckpointInfo = sd_models.checkpoints_list.get(name) or sd_models.checkpoint_aliases.get(name)
        if checkpoint is None:
            return

        path, ext = os.path.splitext(checkpoint.filename)
        search_terms = [self.search_terms_from_path(checkpoint.filename)]
        if checkpoint.sha256:
            search_terms.append(checkpoint.sha256)
        return {
            "name": checkpoint.name_for_extra,
            "filename": checkpoint.filename,
            "shorthash": checkpoint.shorthash,
            "preview": self.find_preview(path),
            "description": self.find_description(path),
            "search_terms": search_terms,
            "onclick": html.escape(f"return selectCheckpoint({ui_extra_networks.quote_js(name)})"),
            "local_preview": f"{path}.{shared.opts.samples_format}",
            "metadata": checkpoint.metadata,
            "sort_keys": {'default': index, **self.get_sort_keys(checkpoint.filename)},
        }

    def list_items(self):
        # instantiate a list to protect against concurrent modification
        checkpoints = list(sd_models.checkpoints_list.values())
        for index, checkpoint in enumerate(checkpoints):
            item = self.create_item(checkpoint.title, index)
            if item is not None:
                yield item

    def allowed_directories_for_previews(self):
        return [v for v in (*shared.cmd_opts.ckpt_dirs, sd_models.model_path) if os.path.isdir(str(v))]

    def create_user_metadata_editor(self, ui, tabname):
        return CheckpointUserMetadataEditor(ui, tabname, self)
