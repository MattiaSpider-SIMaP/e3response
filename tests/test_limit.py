"""Unit tests for the shared ``parse_limit`` slice parser used by the NMR datamodules."""

import pytest

from e3response.data._limit import parse_limit


@pytest.mark.parametrize(
    "limit, expected",
    [
        (None, slice(None)),
        (5, slice(None, 5)),
        (0, slice(None, 0)),
        ("2:8", slice(2, 8)),
        ("2:8:2", slice(2, 8, 2)),
        (":8", slice(None, 8)),
        ("2:", slice(2, None)),
        ("::2", slice(None, None, 2)),
        (":", slice(None, None)),
    ],
)
def test_parse_limit_returns_expected_slice(limit, expected):
    assert parse_limit(limit) == expected


@pytest.mark.parametrize(
    "limit, expected",
    [
        (None, list(range(10))),
        (3, [0, 1, 2]),
        ("2:8", [2, 3, 4, 5, 6, 7]),
        ("2:8:2", [2, 4, 6]),
        (":4", [0, 1, 2, 3]),
        ("6:", [6, 7, 8, 9]),
        ("::3", [0, 3, 6, 9]),
    ],
)
def test_parse_limit_applied_to_sequence(limit, expected):
    """The parsed slice, applied to a list, selects the expected elements."""
    items = list(range(10))
    assert items[parse_limit(limit)] == expected


@pytest.mark.parametrize("limit", ["abc", "1:2:3:4", "1:2:3:4:5"])
def test_parse_limit_invalid_raises(limit):
    with pytest.raises(ValueError):
        parse_limit(limit)
