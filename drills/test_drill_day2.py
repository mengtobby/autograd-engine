"""Same 9 drill tests as test_drill_day1.py, run against drills/day2_engine.py.

Run with: python -m pytest drills -q"""

import pytest

import tests.test_engine as real
from drills.day2_engine import Value as DrillValue
from drills.test_drill_day1 import _cases


@pytest.mark.parametrize("name, args", list(_cases()))
def test_drill(name: str, args: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(real, "Value", DrillValue)
    getattr(real, name)(*args)
