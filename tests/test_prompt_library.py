"""Tests for the admin-panel prompt library (named, file-backed prompts)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from admin_panel.prompt_library import PromptLibrary, slugify  # noqa: E402


@pytest.fixture
def lib(tmp_path: Path) -> PromptLibrary:
    return PromptLibrary(base_dir=tmp_path / "library")


def test_slugify_handles_spaces_punctuation_and_empty() -> None:
    assert slugify("Concise voice v2!") == "concise-voice-v2"
    assert slugify("  Lots   of   spaces  ") == "lots-of-spaces"
    assert slugify("") == "prompt"
    assert slugify("!!!") == "prompt"


def test_create_get_and_first_becomes_active(lib: PromptLibrary) -> None:
    entry = lib.create("My First Prompt", "SYSTEM PROMPT TEXT", notes="testing")
    assert entry.slug == "my-first-prompt"
    assert entry.name == "My First Prompt"
    assert entry.notes == "testing"
    assert lib.get_text(entry.slug) == "SYSTEM PROMPT TEXT"
    # First prompt created in an empty library is auto-activated.
    assert lib.active_slug() == entry.slug
    assert lib.active_text() == "SYSTEM PROMPT TEXT"


def test_create_rejects_empty_name(lib: PromptLibrary) -> None:
    with pytest.raises(ValueError, match="name cannot be empty"):
        lib.create("   ", "text")


def test_duplicate_names_get_unique_slugs(lib: PromptLibrary) -> None:
    a = lib.create("Same Name", "A")
    b = lib.create("Same Name", "B")
    assert a.slug == "same-name"
    assert b.slug == "same-name-2"
    assert lib.get_text(a.slug) == "A"
    assert lib.get_text(b.slug) == "B"


def test_rename_keeps_slug_and_active_pointer_stable(lib: PromptLibrary) -> None:
    entry = lib.create("Original", "text")
    lib.set_active(entry.slug)
    renamed = lib.rename(entry.slug, "Renamed Title")
    assert renamed.slug == entry.slug  # slug (filename) is stable
    assert renamed.name == "Renamed Title"
    # Renaming must not orphan the active pointer.
    assert lib.active_slug() == entry.slug


def test_update_text_changes_content(lib: PromptLibrary) -> None:
    entry = lib.create("P", "old")
    updated = lib.update_text(entry.slug, "new text")
    assert updated.text == "new text"
    assert lib.get_text(entry.slug) == "new text"


def test_duplicate_copies_text_with_new_slug(lib: PromptLibrary) -> None:
    src = lib.create("Source", "SHARED TEXT", notes="n")
    dup = lib.duplicate(src.slug)
    assert dup.slug != src.slug
    assert dup.name == "Source (copy)"
    assert dup.text == "SHARED TEXT"
    # Editing the copy doesn't touch the original.
    lib.update_text(dup.slug, "changed")
    assert lib.get_text(src.slug) == "SHARED TEXT"


def test_delete_moves_active_to_survivor(lib: PromptLibrary) -> None:
    a = lib.create("A", "a")
    b = lib.create("B", "b")
    lib.set_active(a.slug)
    lib.delete(a.slug)
    assert not lib.exists(a.slug)
    # Active falls back to the remaining entry rather than dangling.
    assert lib.active_slug() == b.slug


def test_delete_last_clears_active(lib: PromptLibrary) -> None:
    a = lib.create("only", "x")
    lib.delete(a.slug)
    assert lib.list() == []
    assert lib.active_slug() is None
    assert lib.active_text() is None


def test_list_is_newest_first_and_reconciles_hand_dropped_files(
    lib: PromptLibrary,
) -> None:
    lib.create("First", "1")
    lib.create("Second", "2")
    # A file dropped in by hand (no meta record) still shows up, named by slug.
    (lib.base_dir / "manual-drop.txt").write_text("dropped", encoding="utf-8")
    slugs = [e.slug for e in lib.list()]
    assert set(slugs) == {"first", "second", "manual-drop"}
    dropped = next(e for e in lib.list() if e.slug == "manual-drop")
    assert dropped.name == "manual-drop"
    assert dropped.text == "dropped"


def test_ensure_seeded_imports_legacy_override(
    lib: PromptLibrary, tmp_path: Path
) -> None:
    legacy = tmp_path / "preflop_system.txt"
    legacy.write_text("LEGACY OVERRIDE CONTENT", encoding="utf-8")
    lib.ensure_seeded(lambda: "BUILT-IN DEFAULT", legacy_override=legacy)
    entries = lib.list()
    assert len(entries) == 1
    assert entries[0].name == "Imported override"
    assert entries[0].text == "LEGACY OVERRIDE CONTENT"
    assert lib.active_slug() == entries[0].slug


def test_ensure_seeded_falls_back_to_built_in_default(lib: PromptLibrary) -> None:
    lib.ensure_seeded(lambda: "BUILT-IN DEFAULT", legacy_override=None)
    entries = lib.list()
    assert len(entries) == 1
    assert entries[0].name == "Built-in default"
    assert entries[0].text == "BUILT-IN DEFAULT"


def test_ensure_seeded_is_idempotent(lib: PromptLibrary) -> None:
    lib.create("Existing", "x")
    lib.ensure_seeded(lambda: "DEFAULT", legacy_override=None)
    # Already had an entry, so seeding is a no-op.
    assert [e.name for e in lib.list()] == ["Existing"]


def test_set_active_unknown_slug_raises(lib: PromptLibrary) -> None:
    with pytest.raises(KeyError):
        lib.set_active("does-not-exist")


def test_postflop_library_seeding(tmp_path: Path, monkeypatch) -> None:
    # The postflop seeder: the built-in default always, plus the legacy
    # single-file override imported as a second (active) entry when it exists.
    import admin_panel.app as app

    # No override file -> just the built-in default, and it's active.
    monkeypatch.setattr(app, "POSTFLOP_PROMPT_OVERRIDE_PATH", tmp_path / "missing.txt")
    lib1 = PromptLibrary(base_dir=tmp_path / "lib1")
    app._ensure_postflop_library_seeded(lib1)
    assert [e.name for e in lib1.list()] == ["Built-in default"]
    assert lib1.active_entry() is not None
    app._ensure_postflop_library_seeded(lib1)  # idempotent
    assert len(lib1.list()) == 1

    # With an override file -> imported as a 2nd entry and made active.
    override = tmp_path / "ov.txt"
    override.write_text("MY CUSTOM FACTOR-LIST POSTFLOP PROMPT", encoding="utf-8")
    monkeypatch.setattr(app, "POSTFLOP_PROMPT_OVERRIDE_PATH", override)
    lib2 = PromptLibrary(base_dir=tmp_path / "lib2")
    app._ensure_postflop_library_seeded(lib2)
    assert {e.name for e in lib2.list()} == {"Built-in default", "Factor-list (imported)"}
    assert lib2.active_text() == "MY CUSTOM FACTOR-LIST POSTFLOP PROMPT"
