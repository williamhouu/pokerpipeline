"""Tests for admin_panel/batch_naming.py (July 2026): the auto-descriptive
time-first PLO batch filenames and the creation-time ordering that fixed
the "my new batch doesn't show up first in Review" glitch (the picker used
to sort by mtime, which every edit/refresh bumps).

Browserless by design (fix-durability rule): the module imports no
streamlit; these tests exercise the exact functions the Generate/Review
pages call.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admin_panel.batch_naming import (  # noqa: E402
    auto_plo_batch_name,
    batch_creation_dt,
    dedupe_path,
    plo_batch_display_label,
)

_NOW = datetime(2026, 7, 16, 21, 47, 32)


def test_auto_name_starts_with_time_of_day_and_lists_settings():
    name = auto_plo_batch_name(
        now=_NOW, table_size=9, difficulty_label="Hard", count=12,
        positions=["CO", "BTN"],
        action_contexts=["Facing 3-bet"],
        player_counts=[2],
    )
    assert name == "21.47.32 · 9max · Hard · 12q · CO+BTN · vs 3-bet · HU"


def test_auto_name_time_sorts_as_time_of_day():
    early = auto_plo_batch_name(
        now=datetime(2026, 7, 16, 9, 5, 0), table_size=9,
        difficulty_label="Easy", count=5,
    )
    late = auto_plo_batch_name(
        now=datetime(2026, 7, 16, 21, 5, 0), table_size=9,
        difficulty_label="Easy", count=5,
    )
    assert early < late  # lexicographic == chronological (24h HH.MM.SS)


def test_auto_name_omits_blank_filters():
    name = auto_plo_batch_name(
        now=_NOW, table_size=6, difficulty_label="Mixed", count=5,
    )
    assert name == "21.47.32 · 6max · Mixed · 5q"


def test_auto_name_inserts_custom_label_after_time():
    name = auto_plo_batch_name(
        now=_NOW, table_size=9, difficulty_label="Medium", count=3,
        custom_label="  ryan test  ",
    )
    assert name.startswith("21.47.32 · ryan test · 9max")


def test_auto_name_counts_instead_of_listing_when_many_selected():
    name = auto_plo_batch_name(
        now=_NOW, table_size=9, difficulty_label="Hard", count=8,
        positions=["UTG", "UTG+1", "UTG+2", "LJ", "HJ"],
        player_counts=[1, 2, 3, 4],
    )
    assert "5 seats" in name
    assert "4 pot sizes" in name
    assert "UTG+1" not in name


def test_auto_name_player_count_words():
    name = auto_plo_batch_name(
        now=_NOW, table_size=9, difficulty_label="Hard", count=8,
        player_counts=[3, 1, 2],
    )
    assert "open+HU+3-way" in name  # sorted, worded


def test_auto_name_has_no_em_dash():
    # Standing copy rule: no em dashes in user-facing text (incl. filenames).
    name = auto_plo_batch_name(
        now=_NOW, table_size=9, difficulty_label="Hard", count=8,
        positions=["CO"], action_contexts=["After one call"],
    )
    assert "—" not in name


def test_dedupe_path_never_overwrites(tmp_path):
    first = dedupe_path(tmp_path, "21.47.32 · 9max")
    assert first.name == "21.47.32 · 9max.csv"
    first.write_text("x")
    second = dedupe_path(tmp_path, "21.47.32 · 9max")
    assert second.name == "21.47.32 · 9max (2).csv"
    second.write_text("x")
    assert dedupe_path(tmp_path, "21.47.32 · 9max").name == "21.47.32 · 9max (3).csv"


def test_creation_dt_prefers_legacy_name_stamp_over_mtime(tmp_path):
    """INVARIANT behind the Review fix: editing a file (mtime bump) must not
    change where it sorts. Legacy names carry creation in the filename."""
    old = tmp_path / "plo9_audit_batch_20260601_120000.csv"
    old.write_text("x")
    # Touch it NOW -- mtime is tonight, but creation reads June 1.
    os.utime(old, None)
    assert batch_creation_dt(old) == datetime(2026, 6, 1, 12, 0, 0)


def test_creation_dt_new_format_survives_inplace_rewrite(tmp_path):
    """New auto-names have no date stamp; creation comes from the file's
    birth time, which an in-place truncate rewrite (the refresh script's
    write mode) preserves."""
    path = tmp_path / "21.47.32 · 9max · Hard · 12q.csv"
    path.write_text("original")
    created = batch_creation_dt(path)
    time.sleep(0.05)
    with path.open("w") as fh:  # same inode, like write_plo_csv
        fh.write("refreshed")
    assert batch_creation_dt(path) == created


def test_newest_first_ordering_mixes_legacy_and_new(tmp_path):
    legacy_old = tmp_path / "plo9_first_real_20260716_100000.csv"
    legacy_old.write_text("x")
    new = tmp_path / "21.47.32 · 9max · Hard · 12q.csv"
    new.write_text("x")  # created NOW (birthtime) -> newest
    ordered = sorted(
        [legacy_old, new], key=batch_creation_dt, reverse=True
    )
    assert ordered[0] == new


def test_display_label_legacy_and_new(tmp_path):
    legacy = tmp_path / "100 percento_20260716_212021.csv"
    legacy.write_text("x")
    assert (
        plo_batch_display_label(legacy)
        == "100 percento · 2026-07-16 21:20:21"
    )
    new = tmp_path / "21.47.32 · 9max · Hard · 12q.csv"
    new.write_text("x")
    label = plo_batch_display_label(new)
    assert label.startswith("21.47.32 · 9max · Hard · 12q · ")
    assert str(datetime.now().day) in label  # date context appended


def test_dedupe_path_respects_reserved_names(tmp_path):
    """Queued batches haven't written their CSV yet; their names come in via
    ``taken`` so two same-second Generate clicks can't share a file."""
    from admin_panel.batch_naming import dedupe_path

    # Nothing on disk, nothing reserved -> the plain name.
    assert dedupe_path(tmp_path, "stem").name == "stem.csv"
    # Reserved (queued) but not on disk -> bumps to (2).
    assert dedupe_path(tmp_path, "stem", taken={"stem.csv"}).name == "stem (2).csv"
    # On disk AND (2) reserved -> bumps past both.
    (tmp_path / "stem.csv").write_text("x")
    assert (
        dedupe_path(tmp_path, "stem", taken={"stem (2).csv"}).name
        == "stem (3).csv"
    )


def test_auto_name_carries_the_balanced_token():
    from datetime import datetime

    from admin_panel.batch_naming import auto_plo_batch_name

    stem = auto_plo_batch_name(
        now=datetime(2026, 7, 21, 18, 5, 9),
        table_size=9,
        difficulty_label="Mixed",
        count=24,
        balanced=True,
    )
    assert stem == "18.05.09 · 9max · Mixed · Balanced · 24q"
    # And absent when off (the historical shape is unchanged).
    stem_off = auto_plo_batch_name(
        now=datetime(2026, 7, 21, 18, 5, 9),
        table_size=9,
        difficulty_label="Mixed",
        count=24,
    )
    assert "Balanced" not in stem_off
