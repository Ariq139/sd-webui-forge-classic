import torch
from backend.projection_packing import ProjectionDestinations, coalesce_rows, joined_rows


def test_coalesced_weights_share_storage_and_preserve_values():
    a = torch.arange(32, dtype=torch.int8).view(2, 16)
    b = torch.arange(48, dtype=torch.int8).view(3, 16)
    first, second = coalesce_rows(a, b)
    full = joined_rows(first, second)
    assert full is not None
    assert torch.equal(full, torch.cat((a, b)))
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    assert coalesce_rows(first, second)[0] is first
    second.add_(1)
    assert torch.equal(first, a)
    assert torch.equal(full[2:], b + 1)
    assert joined_rows(second, first) is None
    assert joined_rows(first, second.clone()) is None


def test_reject_partial_aliased_storage():
    a = torch.zeros(4, 16, dtype=torch.int8)
    assert coalesce_rows(a[:2], a[2:3]) is None
    assert coalesce_rows(a, a) is None
    assert coalesce_rows(a, a.float()) is None


def test_output_views_are_disjoint_and_single_use():
    x = torch.zeros(2, 3, 16, dtype=torch.bfloat16)
    group = ProjectionDestinations(x, [('q', 32), ('k', 16), ('v', 16)])
    q, k, v = [group.take(name) for name in ('q', 'k', 'v')]
    assert q.shape == (6, 32) and k.shape == v.shape == (6, 16)
    assert all(t.is_contiguous() for t in (q, k, v))
    assert q.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
    q.fill_(1); k.fill_(2); v.fill_(3)
    assert q.eq(1).all() and k.eq(2).all() and v.eq(3).all()
    assert group.take('q') is None
    assert ProjectionDestinations(x, [('q', 32)], budget=1).take('q') is None
