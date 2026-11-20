"""Runs the real Day 1 tests that only need +, * and backward() against the
drill build in drills/day1_engine.py instead of engine.py.

Run with: python -m pytest drills -q"""

import pytest

import tests.test_engine as real
from drills.day1_engine import Value as DrillValue

TESTS = [
    "test_a_plus_a",
    "test_add_chain_vs_torch",
    "test_a_times_a",
    "test_plain_number_either_side",
    "test_seed_is_one",
    "test_shared_node_waits_for_every_user",
    "test_block3_trace_matches_my_prediction",
    "test_block3_trace_vs_torch",
]

PARAMS = {
    "test_plain_number_either_side": [(lambda a: 2 * a,), (lambda a: a * 2,)],
}


def _cases():
    for name in TESTS:
        for args in PARAMS.get(name, [()]):
            yield pytest.param(name, args, id=f"{name}{'' if not args else f'[{PARAMS[name].index(args)}]'}")


@pytest.mark.parametrize("name, args", list(_cases()))
def test_drill(name: str, args: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(real, "Value", DrillValue)
    getattr(real, name)(*args)
