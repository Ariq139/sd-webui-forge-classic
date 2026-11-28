import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from modules_forge import native_runtime as runtime


CPU = SimpleNamespace(type="cpu")
GPU = SimpleNamespace(type="cuda")
META = SimpleNamespace(type="meta")


class Component:
    def __init__(self, device=CPU):
        self.device = device
        self.callbacks = []

    def parameters(self):
        return iter([SimpleNamespace(device=self.device)])

    def register_forward_pre_hook(self, fn):
        self.callbacks.append(fn)
        return SimpleNamespace(remove=lambda: self.callbacks.remove(fn))

    def forward(self):
        for callback in self.callbacks:
            callback(self, ())
        self.device = GPU


class Hook:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def offload(self):
        self.calls += 1
        self.model.device = CPU


class Pipe:
    def __init__(self):
        self.transformer = Component()
        self.unconditional_transformer = Component()
        self.text_encoder = Component()
        self.prompt_enhancer_head = Component()
        self._execution_device = GPU
        self._forge_runtime = {"offload": "model"}
        self.enable_model_cpu_offload(device=GPU)

    def enable_model_cpu_offload(self, device):
        self._all_hooks = [Hook(c) for c in (self.transformer, self.unconditional_transformer, self.text_encoder, self.prompt_enhancer_head)]


class NativeRuntimeTests(unittest.TestCase):
    def test_alternating_offload_uses_current_hooks(self):
        pipe = Pipe()
        runtime.install_alternating_offload(pipe)
        runtime.install_alternating_offload(pipe)
        self.assertEqual(len(pipe.transformer.callbacks), 1)
        old = pipe._all_hooks[1]
        pipe.unconditional_transformer.device = GPU
        pipe.transformer.forward()
        self.assertIs(pipe.unconditional_transformer.device, CPU)
        self.assertEqual(old.calls, 1)
        pipe.enable_model_cpu_offload(GPU)
        pipe.unconditional_transformer.device = GPU
        pipe.transformer.forward()
        self.assertEqual(old.calls, 1)
        self.assertEqual(pipe._all_hooks[1].calls, 1)
        runtime.release_runtime(pipe)
        self.assertEqual(pipe.transformer.callbacks, [])

    def test_alternating_offload_requires_both_hooks(self):
        pipe = Pipe()
        pipe._all_hooks = []
        with self.assertRaisesRegex(RuntimeError, "Missing"):
            runtime.install_alternating_offload(pipe)

    def test_sequential_retains_cpu_prompt_components(self):
        pipe = Pipe()
        pipe.text_encoder.device = META
        pipe.prompt_enhancer_head.device = META
        removed = []
        def remove(component, recurse):
            removed.append(component)
            component.device = CPU
        runtime.keep_sequential_prompt_on_cpu(pipe, remove, GPU)
        self.assertEqual(removed, [pipe.text_encoder, pipe.prompt_enhancer_head])
        pipe._execution_device = CPU
        with self.assertRaisesRegex(RuntimeError, "execution device"):
            runtime.keep_sequential_prompt_on_cpu(pipe, remove, GPU)

    def test_enhancement_uses_cpu_and_restores_model_hooks(self):
        pipe = Pipe()
        old = pipe._all_hooks
        pipe.text_encoder.device = GPU
        pipe.prompt_enhancer_head.device = GPU
        remove = Mock()
        with self.assertRaisesRegex(RuntimeError, "enhancer failure"):
            with runtime.ideogram_enhancement_device(pipe, remove) as device:
                self.assertIs(device, CPU)
                self.assertIs(pipe.prompt_enhancer_head.device, CPU)
                raise RuntimeError("enhancer failure")
        self.assertEqual(remove.call_count, 2)
        self.assertIsNot(pipe._all_hooks, old)

    def test_enhancement_rejects_unverified_group_and_quantized_cpu(self):
        pipe = Pipe()
        pipe._forge_runtime["offload"] = "group"
        with self.assertRaisesRegex(ValueError, "group hooks"):
            with runtime.ideogram_enhancement_device(pipe, Mock()):
                pass
        pipe._forge_runtime["offload"] = "model"
        pipe.text_encoder.is_quantized = True
        with self.assertRaisesRegex(ValueError, "quantized"):
            with runtime.ideogram_enhancement_device(pipe, Mock()):
                pass

    def test_refinement_geometry(self):
        video = SimpleNamespace(shape=(1, 128, 4, 16, 24))
        upscaled = SimpleNamespace(shape=(1, 128, 4, 32, 48))
        audio = SimpleNamespace(shape=(1, 8, 12, 16))
        kwargs = dict(width=768, height=512, frames=25, spatial_ratio=32, temporal_ratio=8)
        runtime.refinement_geometry(video, upscaled, audio, **kwargs)
        for shape in ((1, 128, 4, 24, 36), (1, 128, 5, 32, 48), (1, 64, 4, 32, 48)):
            with self.assertRaisesRegex(ValueError, "double spatial"):
                runtime.refinement_geometry(video, SimpleNamespace(shape=shape), audio, **kwargs)
        with self.assertRaisesRegex(ValueError, "audio latents"):
            runtime.refinement_geometry(video, upscaled, None, **kwargs)

    def test_local_pipeline_index_rejects_wrong_class_and_missing_components(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_index.json"
            index = {name: ["diffusers", "Component"] for name in ("scheduler", "vae", "text_encoder", "tokenizer", "transformer", "audio_vae", "vocoder", "connectors")}
            index["_class_name"] = "LTX2Pipeline"
            path.write_text(json.dumps(index))
            self.assertEqual(runtime.require_pipeline_directory(directory, "LTX2"), directory)
            index["_class_name"] = "LTX2LatentUpsamplePipeline"
            path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "wrong pipeline class"):
                runtime.require_pipeline_directory(directory, "LTX2")
            index["_class_name"] = "LTX2Pipeline"
            index["audio_vae"] = [None, None]
            path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "missing components: audio_vae"):
                runtime.require_pipeline_directory(directory, "LTX2")
            with self.assertRaisesRegex(ValueError, "not a local directory"):
                runtime.require_pipeline_directory("remote/model", "Ideogram4")

    def test_release_cache_without_moving_pipeline(self):
        pipe = Pipe()
        cache = Mock()
        original = Mock()
        pipe._forge_prompt_cache_state = {"cache": cache, "original": original}
        pipe._prompt_enhancer = object()
        pipe.to = Mock(side_effect=RuntimeError("meta weights"))
        runtime.release_runtime(pipe)
        pipe.to.assert_not_called()
        cache.clear.assert_called_once()
        self.assertIs(pipe.encode_prompt, original)
        self.assertIsNone(pipe._prompt_enhancer)


if __name__ == "__main__":
    unittest.main()
