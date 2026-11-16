import numpy as np
import pytest

from pathfinder.image_faults import KINDS, Corruptor, ImageFault, randomized_schedule


def _img(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (64, 64), dtype=np.uint8)


def test_each_kind_changes_stream_only_inside_interval():
    for kind in KINDS:
        c = Corruptor([ImageFault(kind, 1.0, 2.0, 1.0)], seed=1)
        outs = [c.apply(t, _img(i), _img(100 + i)) for i, t in enumerate((0.5, 1.5, 2.5))]
        assert outs[0] is not None and np.array_equal(outs[0][1], _img(0)) and outs[0][0] == 0.5
        assert outs[2] is not None and np.array_equal(outs[2][1], _img(2))
        mid = outs[1]
        if kind == "drop":
            assert mid is None
        elif kind == "jitter":
            assert mid[0] != 1.5 and abs(mid[0] - 1.5) <= 0.02
        elif kind == "frozen":
            assert np.array_equal(mid[1], _img(0))
        elif kind == "stereo_desync":
            assert np.array_equal(mid[2], _img(100)) and np.array_equal(mid[1], _img(1))
        else:
            assert not np.array_equal(mid[1], _img(1))


def test_schedule_is_seeded_and_validated():
    a = randomized_schedule(60, seed=7)
    assert a == randomized_schedule(60, seed=7)
    assert all(0 <= f.start < f.end for f in a)
    with pytest.raises(ValueError):
        ImageFault("rain", 0, 1)
    with pytest.raises(ValueError):
        ImageFault("dark", 1, 1)
    with pytest.raises(ValueError):
        ImageFault("dark", 0, 1, 0.0)


def test_startup_freeze_captures_first_delivered_pair():
    c = Corruptor([ImageFault("frozen", 0, 2)])
    first = c.apply(0, _img(0), _img(1))
    second = c.apply(.1, _img(2), _img(3))
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])


@pytest.mark.parametrize("kind", ["blackout", "frozen"])
def test_overlap_order_is_independent_and_blackout_stays_zero(kind):
    schedule = [ImageFault("dark", 0, 2), ImageFault(kind, 0, 2)]
    a, b = Corruptor(schedule, 1), Corruptor(schedule[::-1], 1)
    previous = None
    for i in range(3):
        x = a.apply(i * .1, _img(i), _img(i + 10))
        y = b.apply(i * .1, _img(i), _img(i + 10))
        np.testing.assert_array_equal(x[1], y[1])
        if kind == "blackout":
            assert not x[1].any()
        elif previous is not None:
            np.testing.assert_array_equal(x[1], previous)
        previous = x[1]


def test_short_schedule_stays_inside_recording():
    for seed in range(100):
        assert all(0 <= f.start < f.end <= .05 for f in randomized_schedule(.05, seed))
