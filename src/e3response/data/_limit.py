"""Shared helper for parsing dataset ``limit`` specifications.

Several NMR datamodules accept a ``limit`` that restricts how much data is loaded (either
the whole dataset or a single split). They all share the same slice-parsing semantics,
factored out here so there is a single definition (see `parse_limit`).
"""


def parse_limit(limit: int | str | None) -> slice:
    """Convert a limit spec to a slice over a sequence (structures, files, indices, ...).

    - None       → slice(None)       (everything)
    - int N      → slice(None, N)    (first N items)
    - "a:b"      → slice(a, b)       (items a through b-1)
    - "a:b:s"    → slice(a, b, s)    (with step)
    """
    if limit is None:
        return slice(None)
    if isinstance(limit, int):
        return slice(None, limit)
    parts = limit.split(":")
    indices = [int(p) if p else None for p in parts]
    if len(indices) == 2:
        return slice(indices[0], indices[1])
    if len(indices) == 3:
        return slice(indices[0], indices[1], indices[2])
    raise ValueError(
        f"Cannot parse limit {limit!r}: expected int, 'start:stop', or 'start:stop:step'"
    )
