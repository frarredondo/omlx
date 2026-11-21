# SPDX-License-Identifier: Apache-2.0
"""In-memory boundary snapshots (hot_cache_only or a failed SSD write)."""

from __future__ import annotations

import gc
from unittest.mock import MagicMock, patch

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:  # pragma: no cover
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

BLOCK = 8
BOUNDARIES = 10
UNIT = 1024  # bytes of one synthetic snapshot


def _scheduler(mock_tokenizer, *, store=None, budget=None):
    from test_inkling_vlm_mtp import _inkling2_tiny

    from omlx.scheduler import Scheduler, SchedulerConfig

    model = _inkling2_tiny()
    scheduler = Scheduler(
        model=model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(paged_cache_block_size=BLOCK),
    )
    scheduler.block_aware_cache = object()
    scheduler._boundary_snapshot_store = store
    if budget is not None:
        scheduler._ram_snapshot_budget_bytes = budget
    return model, scheduler


def _request(scheduler, **fields):
    from omlx.request import Request, SamplingParams

    request = Request(request_id="req", prompt="hello", sampling_params=SamplingParams())
    for name, value in fields.items():
        setattr(request, name, value)
    scheduler.requests["req"] = request
    return request


def _sliding_keys(extracted):
    return [
        layer["state"][0]
        for layer in extracted
        if isinstance(layer.get("state"), (list, tuple))
        and len(layer["state"]) == 7
        and isinstance(layer["state"][0], mx.array)
    ]


def _prefill_boundaries(model, scheduler, request_id="req"):
    cache = model.make_cache()
    live_keys_at = {}
    mx.random.seed(1)
    for i in range(1, BOUNDARIES + 1):
        model(mx.random.randint(0, 300, (1, BLOCK)), cache=cache)
        mx.eval([c.state for c in cache])
        tokens = i * BLOCK
        live = scheduler._extract_snapshot_cache_states(cache)[0]
        live_keys_at[tokens] = [k * 1 for k in _sliding_keys(live)]
        mx.eval(live_keys_at[tokens])
        scheduler._on_prefill_boundary_snapshot(request_id, cache, tokens)
    return live_keys_at


def _kept(scheduler, request_id="req"):
    snapshots = scheduler._boundary_cache_snapshots.get(request_id, {})
    return sorted(tc for tc, value in snapshots.items() if value is not None)


def _one_inkling_snapshot_bytes(mock_tokenizer):
    model, scheduler = _scheduler(mock_tokenizer, budget=1 << 40)
    _prefill_boundaries(model, scheduler)
    return max(scheduler._ram_snapshot_charges.values())


def _snapshot(class_name="ArraysCache", nbytes=UNIT):
    from omlx.scheduler import Scheduler

    state = [mx.zeros((nbytes // 4,), dtype=mx.float32)]
    return (Scheduler._PREFILL_SNAPSHOT_MARKER, [{"class_name": class_name, "state": state}])


def _skip_reasons(scheduler):
    return scheduler._boundary_snapshot_diagnostics.snapshot()["reasons"]


@pytest.mark.parametrize("mode", ["hot_cache_only", "ssd_write_failed"])
def test_inkling_snapshots_stay_within_budget_and_match_their_boundary(mock_tokenizer, mode):
    one = _one_inkling_snapshot_bytes(mock_tokenizer)
    store = None
    if mode == "ssd_write_failed":
        store = MagicMock()
        store.save.return_value = False
    model, scheduler = _scheduler(mock_tokenizer, store=store, budget=3 * one)
    live_keys_at = _prefill_boundaries(model, scheduler)

    kept = _kept(scheduler)
    assert sum(scheduler._ram_snapshot_charges.values()) <= 3 * one
    assert kept[-1] == BOUNDARIES * BLOCK
    assert len(kept) < BOUNDARIES
    for tc in kept:
        got = _sliding_keys(scheduler._boundary_cache_snapshots["req"][tc][1])
        assert len(got) == len(live_keys_at[tc]) > 0
        for a, b in zip(got, live_keys_at[tc]):
            assert a.shape == b.shape
            assert mx.array_equal(a, b)


def test_store_override_ends_on_the_newest_kept_boundary(mock_tokenizer):
    one = _one_inkling_snapshot_bytes(mock_tokenizer)
    model, scheduler = _scheduler(mock_tokenizer, budget=2 * one)
    _request(scheduler)
    _prefill_boundaries(model, scheduler)
    result = scheduler._get_boundary_store_override(
        "req", list(range(BOUNDARIES * BLOCK + 3))
    )
    assert result is not None
    token_sequence, cache_to_store, _, _ = result
    assert token_sequence == list(range(BOUNDARIES * BLOCK))
    assert cache_to_store is scheduler._boundary_cache_snapshots["req"][BOUNDARIES * BLOCK][1]


def test_shared_prefix_end_is_never_evicted(mock_tokenizer):
    one = _one_inkling_snapshot_bytes(mock_tokenizer)
    model, scheduler = _scheduler(mock_tokenizer, budget=2 * one)
    _request(scheduler, system_prefix_end=2 * BLOCK)
    _prefill_boundaries(model, scheduler)
    assert _kept(scheduler) == [2 * BLOCK, BOUNDARIES * BLOCK]


def test_oversized_snapshot_is_skipped_without_being_copied(mock_tokenizer):
    one = _one_inkling_snapshot_bytes(mock_tokenizer)
    model, scheduler = _scheduler(mock_tokenizer, budget=one // 2)
    with patch("omlx.scheduler._detach_extracted_snapshot") as detach:
        _prefill_boundaries(model, scheduler)
    detach.assert_not_called()
    assert _kept(scheduler) == []
    assert _skip_reasons(scheduler).get("ram_snapshot_budget") == BOUNDARIES
    assert scheduler._boundary_snapshot_diagnostics.snapshot()["captures_memory"] == 0


def test_evicted_snapshots_are_freed_without_cyclic_gc(mock_tokenizer, monkeypatch):
    import weakref

    import omlx.scheduler as scheduler_module

    class Trackable(dict):
        pass

    kept_snapshots = []

    def detach_trackably(value):
        snapshot = Trackable(value=value)
        kept_snapshots.append(weakref.ref(snapshot))
        return snapshot

    monkeypatch.setattr(scheduler_module, "_detach_extracted_snapshot", detach_trackably)
    _, scheduler = _scheduler(mock_tokenizer, budget=2 * UNIT)
    gc.disable()
    try:
        for tc in (8, 16, 24, 32):
            assert scheduler._retain_ram_boundary_snapshot("req", tc, _snapshot())
        alive = [ref() is not None for ref in kept_snapshots]
    finally:
        gc.enable()
    assert alive == [False, False, True, True]


def test_generation_prompt_tail_is_kept_for_the_store(mock_tokenizer):
    _, scheduler = _scheduler(mock_tokenizer, budget=2 * UNIT)
    _request(scheduler, generation_prompt_start=12)
    for tc in (8, 12, 16):
        assert scheduler._retain_ram_boundary_snapshot("req", tc, _snapshot())
    assert _kept(scheduler) == [12, 16]
    result = scheduler._get_boundary_store_override("req", list(range(20)))
    assert result is not None and result[0] == list(range(12))


@pytest.mark.parametrize("class_name", ["CacheList", "RotatingKVCache"])
def test_chained_layouts_keep_the_unbudgeted_fallback(mock_tokenizer, class_name):
    _, scheduler = _scheduler(mock_tokenizer, budget=2 * UNIT)
    for tc in (8, 16, 24, 32):
        assert scheduler._retain_ram_boundary_snapshot("req", tc, _snapshot(class_name))
    assert _kept(scheduler) == [8, 16, 24, 32]
    assert scheduler._ram_snapshot_charges == {}


def test_restored_arrays_cache_is_budgeted_like_a_live_one(mock_tokenizer):
    _, scheduler = _scheduler(mock_tokenizer, budget=2 * UNIT)
    for tc in (8, 16, 24):
        assert scheduler._retain_ram_boundary_snapshot(
            "req", tc, _snapshot("SizedArraysCache")
        )
    assert _kept(scheduler) == [16, 24]


@pytest.mark.parametrize("folded_output", [0, BLOCK])
def test_checkpoint_within_the_prompt_is_kept_for_prompt_only_stores(
    mock_tokenizer, folded_output
):
    _, scheduler = _scheduler(mock_tokenizer, budget=2 * UNIT)
    original_prompt = list(range(BLOCK + 1))
    _request(
        scheduler,
        prompt_token_ids=original_prompt + list(range(100, 100 + folded_output)),
        requeued_output_tokens=folded_output,
    )
    for tc in (8, 16, 24):
        assert scheduler._retain_ram_boundary_snapshot("req", tc, _snapshot())
    assert _kept(scheduler) == [8, 24]
    result = scheduler._get_boundary_store_override("req", original_prompt)
    assert result is not None and result[0] == list(range(BLOCK))


def test_charges_stay_while_an_async_store_owns_the_snapshots(mock_tokenizer):
    _, scheduler = _scheduler(mock_tokenizer, budget=2 * UNIT)
    for tc in (8, 16):
        assert scheduler._retain_ram_boundary_snapshot("a", tc, _snapshot())
    scheduler._boundary_cache_snapshots.pop("a")
    scheduler._inflight_store_futures["a"] = object()
    assert not scheduler._retain_ram_boundary_snapshot("b", 8, _snapshot())

    del scheduler._inflight_store_futures["a"]
    assert scheduler._retain_ram_boundary_snapshot("b", 8, _snapshot())
    assert set(scheduler._ram_snapshot_charges) == {("b", 8)}
