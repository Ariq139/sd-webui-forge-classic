import unittest
from unittest import mock

from backend import memory_management
from modules import devices


class MemoryCacheCleanupTests(unittest.TestCase):
    def setUp(self):
        memory_management.signal_empty_cache = False

    def _cuda_mocks(self):
        return (
            mock.patch.object(memory_management, "is_intel_xpu", return_value=False),
            mock.patch.object(memory_management.torch.cuda, "is_available", return_value=True),
            mock.patch.object(memory_management.torch.cuda, "synchronize"),
            mock.patch.object(memory_management.torch.cuda, "empty_cache"),
            mock.patch.object(memory_management.torch.cuda, "ipc_collect"),
        )

    def test_non_forced_cleanup_is_a_noop_without_signal(self):
        patches = self._cuda_mocks()
        with patches[0], patches[1], patches[2] as synchronize, patches[3] as empty_cache, patches[4] as ipc_collect:
            memory_management.soft_empty_cache(force=False)

        synchronize.assert_not_called()
        empty_cache.assert_not_called()
        ipc_collect.assert_not_called()

    def test_forced_cleanup_runs_full_cuda_cleanup(self):
        patches = self._cuda_mocks()
        with patches[0], patches[1], patches[2] as synchronize, patches[3] as empty_cache, patches[4] as ipc_collect:
            memory_management.soft_empty_cache(force=True)

        synchronize.assert_called_once_with()
        empty_cache.assert_called_once_with()
        ipc_collect.assert_called_once_with()

    def test_signal_requests_one_cleanup(self):
        memory_management.signal_empty_cache = True
        patches = self._cuda_mocks()
        with patches[0], patches[1], patches[2] as synchronize, patches[3] as empty_cache, patches[4] as ipc_collect:
            memory_management.soft_empty_cache(force=False)
            memory_management.soft_empty_cache(force=False)

        synchronize.assert_called_once_with()
        empty_cache.assert_called_once_with()
        ipc_collect.assert_called_once_with()
        self.assertFalse(memory_management.signal_empty_cache)

    @mock.patch("modules.devices.memory_management.soft_empty_cache")
    def test_torch_gc_preserves_forced_cleanup_by_default(self, soft_empty_cache):
        devices.torch_gc()

        soft_empty_cache.assert_called_once_with(force=True)

    @mock.patch("modules.devices.memory_management.soft_empty_cache")
    def test_torch_gc_can_skip_unsignaled_hot_path_cleanup(self, soft_empty_cache):
        devices.torch_gc(force=False)

        soft_empty_cache.assert_called_once_with(force=False)


if __name__ == "__main__":
    unittest.main()
