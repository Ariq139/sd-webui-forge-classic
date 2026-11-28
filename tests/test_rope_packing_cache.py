import torch

from backend.rope_packing_cache import PackedRopeViews


def test_view_cache_distinguishes_slices_strides_and_mutation():
    source = torch.arange(48.).reshape(1, 8, 6)
    cache = PackedRopeViews()
    calls = []
    def pack(view):
        calls.append(view)
        return view.clone()
    first = cache.get(source[:, :4], pack)
    assert cache.get(source[:, :4], pack) is first
    second = cache.get(source[:, 4:], pack)
    assert first is not second and len(calls) == 2
    cache.get(source[:, :4].transpose(1, 2), pack)
    assert len(calls) == 3
    source.add_(1)
    assert cache.get(source[:, :4], pack) is not first
    assert len(calls) == 4
    cache.clear()
    assert not cache.local and cache.source is None


def test_persistent_geometry_only_for_bound_unmodified_storage():
    source = torch.arange(48.).reshape(1, 8, 6)
    cache = PackedRopeViews()
    cache.bind(source, ("layout", 8))
    a = cache.persistent_geometry(source[:, :4])
    b = cache.persistent_geometry(source[:, 4:])
    assert a is not None and b is not None and a != b
    assert cache.persistent_geometry(source.clone()) is None
    source.add_(1)
    assert cache.persistent_geometry(source[:, :4]) is None


def test_cross_forward_reuse_with_explicit_geometry():
    persistent = {}
    calls = []
    def pack(view):
        calls.append(view.shape)
        return view.clone()
    def get(view, layout, pack):
        if layout not in persistent:
            persistent[layout] = pack(view)
        return persistent[layout]
    with torch.inference_mode():
        for _ in range(3):
            source = torch.arange(48.).reshape(1, 8, 6)
            cache = PackedRopeViews()
            cache.bind(source, (8, 6))
            for view in (source[:, :4], source[:, 4:], source):
                torch.testing.assert_close(cache.get(view, pack, get), view)
            cache.clear()
    assert len(calls) == 3 and len(persistent) == 3
