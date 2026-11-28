import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules import ui_result_utils as gallery
from modules_forge import native_requests as requests, native_runtime as runtime


class ReviewFixTests(unittest.TestCase):
    def test_task_id_normalization(self):
        self.assertEqual(requests.native_task_id("client_123"), "native-client_123")
        self.assertEqual(requests.native_task_id("native-client_123"), "native-client_123")
        self.assertTrue(requests.native_task_id().startswith("native-"))
        for value in ("", "native-", "a/b", "../bad", "x" * 81):
            with self.assertRaises(ValueError):
                requests.native_task_id(value)

    def test_pending_cancel_never_interrupts_another_active_job(self):
        owner = requests.NativeTaskOwner()
        interrupt = Mock()
        with owner.reserve("a"), owner.active("a"), owner.reserve("b"):
            with self.assertRaises(requests.DuplicateNativeTask):
                with owner.reserve("a"):
                    pass
            self.assertTrue(owner.interrupt("b", interrupt))
            interrupt.assert_not_called()
            self.assertFalse(owner.interrupt("unknown", interrupt))
            self.assertTrue(owner.interrupt("a", interrupt))
            interrupt.assert_called_once()
        self.assertFalse(owner.interrupt("a", interrupt))
        with owner.reserve("b"):
            owner.interrupt("b", interrupt)
            with self.assertRaises(requests.CancelledNativeTask):
                with owner.active("b"):
                    self.fail("cancelled task started")
        self.assertFalse(owner.interrupt("b", interrupt))

    def test_reservation_released_on_exception(self):
        owner = requests.NativeTaskOwner()
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with owner.reserve("a"), owner.active("a"):
                raise RuntimeError("failure")
        with owner.reserve("a"), owner.active("a"):
            pass

    def test_mask_entries_do_not_consume_seed_ordinals(self):
        mask = SimpleNamespace(_forge_auxiliary=True)
        images = [object(), object(), mask, object()]
        kinds, texts = gallery.serialized_gallery(images, [object()], 1, ["grid", "A", "mask", "B"])
        self.assertEqual(kinds, ["grid", "image", "auxiliary", "image", "auxiliary"])
        self.assertEqual(texts, ["grid", "A", "mask", "B", None])
        data = {"gallery_kinds": kinds, "infotexts": texts, "index_of_first_image": 1, "all_seeds": [10, 20]}
        result = gallery.splice_gallery_metadata(data, {"all_seeds": [30]}, 5, 3, False, ["C"])
        self.assertEqual(result["all_seeds"], [10, 30])
        self.assertEqual(result["gallery_kinds"], kinds)
        self.assertEqual(result["infotexts"], ["grid", "A", "mask", "C", None])
        self.assertEqual(gallery.serialized_gallery([], [], 0, ["unused"]), ([], []))

    def test_missing_checkpoint_is_never_replaced(self):
        import modules
        path = Path(__file__).resolve().parents[1] / "modules_forge" / "native_models.py"
        spec = importlib.util.spec_from_file_location("_native_model_selection_test", path)
        model_selection = importlib.util.module_from_spec(spec)
        with tempfile.TemporaryDirectory() as directory:
            paths = SimpleNamespace(models_path=directory)
            with patch.object(modules, "paths", paths, create=True):
                spec.loader.exec_module(model_selection)
            missing = str(Path(directory) / "deleted")
            valid = str(Path(directory) / "other")
            with patch.object(model_selection, "compatible_model_directories", return_value=[valid]) as choices:
                self.assertEqual(model_selection.select_model_path("LTX2", missing, valid), missing)
                self.assertEqual(model_selection.select_model_path("LTX2", "diffusers/LTX-2.3-Diffusers", missing), missing)
                self.assertEqual(model_selection.select_model_path("LTX2", "", ""), "")
                choices.assert_not_called()

    def test_global_unload_attempts_both_native_pipelines(self):
        ltx = Mock()
        ideogram = Mock()
        ltx.unload.side_effect = RuntimeError("failed")
        failures = runtime.unload_native_pipelines({"modules.ui_ltx2_video": ltx, "modules.ui_ideogram": ideogram})
        ltx.unload.assert_called_once()
        ideogram.unload.assert_called_once()
        self.assertEqual(len(failures), 1)
        self.assertIn("ltx2", failures[0])

    def test_cpu_placement_rejects_cuda_and_meta_weights(self):
        for device in ("cpu", "cuda", "meta"):
            component = SimpleNamespace(parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type=device))], buffers=lambda: [])
            pipe = SimpleNamespace(components={"transformer": component})
            if device == "cpu":
                runtime.verify_cpu_loading(pipe)
            else:
                with self.assertRaisesRegex(RuntimeError, "CPU placement failed"):
                    runtime.verify_cpu_loading(pipe)

    def test_prequantized_cpu_load_contract(self):
        from modules_forge import native_pipeline_speed as speed
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "transformer" / "config.json"
            config.parent.mkdir()
            config.write_text(json.dumps({"quantization_config": {"load_in_4bit": True}}))
            device = SimpleNamespace(type="cuda")
            with patch.object(speed, "require_dependency"):
                self.assertEqual(speed.native_quantization_kwargs(directory, "Checkpoint", None, device, "model", ("transformer",)), {"device_map": "cpu"})
                self.assertEqual(speed.native_quantization_kwargs(directory, "Checkpoint", None, device, "none", ("transformer",)), {})
            pipe = SimpleNamespace(components={}, hf_device_map="cpu")
            cls = SimpleNamespace(from_pretrained=Mock(return_value=pipe))
            self.assertIs(speed.load_native_pipeline(cls, directory, {"device_map": "cpu"}, None, None, "test"), pipe)
            cls.from_pretrained.assert_called_once_with(directory, device_map="cpu", local_files_only=True)
            self.assertIsNone(pipe.hf_device_map)


if __name__ == "__main__":
    unittest.main()
