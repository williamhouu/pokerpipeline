"""Filename-grammar parsers for preflop range packs.

One parser module per vendor / file format. The registry below maps a
``grammar_name`` (as recorded in ``PreflopPackSignature``) to the module
that knows how to parse that format's filenames.

Adding a new grammar:

  1. Drop a new module under ``pipeline/preflop/grammars/your_vendor.py``
     that exports a top-level ``parse(filename, *, pack) -> ParsedRangeFile``
     function and (optionally) a ``GRAMMAR_NAME`` string.
  2. Add an entry to ``_REGISTRY`` below mapping the grammar_name to the
     parse function.
  3. Reference that grammar_name from a ``PreflopPackSignature`` in
     ``pipeline.preflop.pack.KNOWN_PACK_SIGNATURES``.

The output of every parser is the same ``ParsedRangeFile`` dataclass --
this is the normalization seam. Nothing downstream (node enumerator,
spot sampler, fact extractor) ever sees a raw filename or vendor-specific
notation.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pipeline.preflop.grammars import ryan_pack
from pipeline.preflop.grammars.types import (
    ParsedAction,
    ParsedRangeFile,
    PreflopActionType,
)
from pipeline.preflop.pack import PreflopPack

# Registry: grammar_name -> parse function. Keys must match
# PreflopPackSignature.grammar_name values.
_REGISTRY: dict[str, Callable[[Path, PreflopPack], ParsedRangeFile]] = {
    "ryan_pack": ryan_pack.parse,
}


def parse(filename: Path | str, *, pack: PreflopPack) -> ParsedRangeFile:
    """Dispatch a filename to the parser registered for the pack's grammar.

    Args:
        filename: Path to one ``.txt`` range file inside the pack (absolute
            or relative to the pack root).
        pack: The ``PreflopPack`` this file belongs to.

    Returns:
        A normalized ``ParsedRangeFile`` describing the file's metadata
        (the range weights inside the file are loaded separately via
        ``pipeline.preflop_ranges.parse_range_file``).

    Raises:
        KeyError: if no grammar is registered for ``pack.grammar_name``.
        ValueError: if the parser rejects the filename as malformed.
    """
    parser = _REGISTRY.get(pack.grammar_name)
    if parser is None:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(
            f"no grammar parser registered for {pack.grammar_name!r}. "
            f"Known grammars: {known}."
        )
    return parser(Path(filename), pack)


__all__ = [
    "ParsedAction",
    "ParsedRangeFile",
    "PreflopActionType",
    "parse",
]
