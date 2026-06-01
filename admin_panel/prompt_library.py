"""File-backed library of named Layer 6 system prompts, for workshopping.

The admin panel used to write a SINGLE override file
(``admin_panel/prompts/preflop_system.txt``). This module generalizes that
to a LIBRARY of named prompts you can create, rename, duplicate, delete,
and switch between -- so you can A/B many prompts and keep the good ones
around.

Storage (all under ``admin_panel/prompts/library/``, which is gitignored so
experimental prompts never leak into commits):

  * ``<slug>.txt`` -- the prompt text for one entry.
  * ``_meta.json`` -- ``{"active": "<slug>",
                         "prompts": {"<slug>": {name, notes,
                                     created_at, updated_at}}}``.

The pipeline never imports this module: Layer 6 takes a prompt *string*
(``generate_preflop_answer_explanation(system_prompt=...)``). The admin
panel resolves the active entry to text and passes it down. So this module
is pure UI-side state with zero pipeline coupling -- which is exactly the
boundary we want.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Default on-disk location. The class accepts an explicit base dir so tests
# can point it at a tmp_path.
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
LIBRARY_DIR = PROMPTS_DIR / "library"
_META_NAME = "_meta.json"
_SLUG_MAX_LEN = 60


def _now_iso() -> str:
    """Current UTC time as a stable ISO-8601 string (second precision)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def slugify(name: str) -> str:
    """Turn a human prompt name into a filesystem-safe slug.

    Lowercases, replaces any run of non-alphanumeric characters with a
    single hyphen, trims hyphens, and caps the length. Empty input yields
    ``"prompt"`` so we never produce an empty filename.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    slug = slug[:_SLUG_MAX_LEN].strip("-")
    return slug or "prompt"


@dataclass(frozen=True)
class PromptEntry:
    """One named prompt in the library."""

    slug: str
    name: str
    notes: str
    created_at: str
    updated_at: str
    text: str


class PromptLibrary:
    """CRUD over a directory of named prompt files plus an active pointer.

    All methods read/write the backing files fresh, so the library is safe
    to use from Streamlit (which reruns the script top-to-bottom on every
    interaction) without caching surprises.
    """

    def __init__(self, base_dir: Path = LIBRARY_DIR) -> None:
        self.base_dir = base_dir

    # --- internals ----------------------------------------------------------
    @property
    def _meta_path(self) -> Path:
        return self.base_dir / _META_NAME

    def _text_path(self, slug: str) -> Path:
        return self.base_dir / f"{slug}.txt"

    def _read_meta(self) -> dict[str, object]:
        if not self._meta_path.is_file():
            return {"active": None, "prompts": {}}
        try:
            data = json.loads(self._meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"active": None, "prompts": {}}
        if not isinstance(data, dict):
            return {"active": None, "prompts": {}}
        data.setdefault("active", None)
        prompts = data.get("prompts")
        if not isinstance(prompts, dict):
            data["prompts"] = {}
        return data

    def _write_meta(self, meta: dict[str, object]) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _meta_for(self, slug: str) -> dict[str, str]:
        prompts = self._read_meta()["prompts"]
        assert isinstance(prompts, dict)  # narrowed by _read_meta
        entry = prompts.get(slug)
        return entry if isinstance(entry, dict) else {}

    def _unique_slug(self, name: str) -> str:
        """A slug for ``name`` that doesn't collide with an existing .txt."""
        base = slugify(name)
        slug = base
        n = 2
        while self._text_path(slug).exists():
            slug = f"{base}-{n}"
            n += 1
        return slug

    # --- queries ------------------------------------------------------------
    def list(self) -> list[PromptEntry]:
        """All entries, newest-updated first.

        Reconciles meta with the actual ``*.txt`` files on disk: every text
        file shows up even if meta is missing its record (a hand-dropped
        file gets a sensible default name), and meta records with no file
        are skipped. This keeps the library robust to manual edits.
        """
        if not self.base_dir.is_dir():
            return []
        meta_prompts = self._read_meta()["prompts"]
        assert isinstance(meta_prompts, dict)
        entries: list[PromptEntry] = []
        for txt in self.base_dir.glob("*.txt"):
            slug = txt.stem
            record = meta_prompts.get(slug)
            record = record if isinstance(record, dict) else {}
            try:
                text = txt.read_text(encoding="utf-8")
            except OSError:
                continue
            entries.append(
                PromptEntry(
                    slug=slug,
                    name=str(record.get("name") or slug),
                    notes=str(record.get("notes") or ""),
                    created_at=str(record.get("created_at") or ""),
                    updated_at=str(record.get("updated_at") or ""),
                    text=text,
                )
            )
        entries.sort(key=lambda e: e.updated_at, reverse=True)
        return entries

    def exists(self, slug: str) -> bool:
        return self._text_path(slug).is_file()

    def get(self, slug: str) -> PromptEntry:
        """One entry by slug. Raises ``KeyError`` if it doesn't exist."""
        if not self.exists(slug):
            raise KeyError(f"no prompt with slug {slug!r}")
        record = self._meta_for(slug)
        return PromptEntry(
            slug=slug,
            name=str(record.get("name") or slug),
            notes=str(record.get("notes") or ""),
            created_at=str(record.get("created_at") or ""),
            updated_at=str(record.get("updated_at") or ""),
            text=self._text_path(slug).read_text(encoding="utf-8"),
        )

    def get_text(self, slug: str) -> str:
        return self.get(slug).text

    # --- mutations ----------------------------------------------------------
    def create(self, name: str, text: str, notes: str = "") -> PromptEntry:
        """Create a new named prompt and return it. Names need not be unique
        (the slug is uniquified); empty names are rejected."""
        if not name.strip():
            raise ValueError("prompt name cannot be empty")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        slug = self._unique_slug(name)
        now = _now_iso()
        self._text_path(slug).write_text(text, encoding="utf-8")
        meta = self._read_meta()
        prompts = meta["prompts"]
        assert isinstance(prompts, dict)
        prompts[slug] = {
            "name": name.strip(),
            "notes": notes,
            "created_at": now,
            "updated_at": now,
        }
        # First prompt in an empty library becomes active automatically.
        if not meta.get("active"):
            meta["active"] = slug
        self._write_meta(meta)
        return self.get(slug)

    def update_text(self, slug: str, text: str) -> PromptEntry:
        if not self.exists(slug):
            raise KeyError(f"no prompt with slug {slug!r}")
        self._text_path(slug).write_text(text, encoding="utf-8")
        self._touch(slug)
        return self.get(slug)

    def rename(self, slug: str, new_name: str) -> PromptEntry:
        """Change an entry's display name. The slug (filename) is stable, so
        renaming never breaks the active pointer or batch metadata."""
        if not new_name.strip():
            raise ValueError("prompt name cannot be empty")
        meta = self._read_meta()
        prompts = meta["prompts"]
        assert isinstance(prompts, dict)
        record = prompts.get(slug)
        record = record if isinstance(record, dict) else {}
        record["name"] = new_name.strip()
        record["updated_at"] = _now_iso()
        record.setdefault("created_at", _now_iso())
        prompts[slug] = record
        self._write_meta(meta)
        return self.get(slug)

    def update_notes(self, slug: str, notes: str) -> PromptEntry:
        meta = self._read_meta()
        prompts = meta["prompts"]
        assert isinstance(prompts, dict)
        record = prompts.get(slug)
        record = record if isinstance(record, dict) else {}
        record["notes"] = notes
        record["updated_at"] = _now_iso()
        prompts[slug] = record
        self._write_meta(meta)
        return self.get(slug)

    def duplicate(self, slug: str, new_name: str | None = None) -> PromptEntry:
        src = self.get(slug)
        name = new_name.strip() if (new_name and new_name.strip()) else f"{src.name} (copy)"
        return self.create(name, src.text, notes=src.notes)

    def delete(self, slug: str) -> None:
        """Remove an entry. If it was active, the active pointer moves to the
        most-recently-updated survivor (or clears if none remain)."""
        self._text_path(slug).unlink(missing_ok=True)
        meta = self._read_meta()
        prompts = meta["prompts"]
        assert isinstance(prompts, dict)
        prompts.pop(slug, None)
        if meta.get("active") == slug:
            survivors = self.list()  # already excludes the deleted file
            meta["active"] = survivors[0].slug if survivors else None
        self._write_meta(meta)

    def _touch(self, slug: str) -> None:
        meta = self._read_meta()
        prompts = meta["prompts"]
        assert isinstance(prompts, dict)
        record = prompts.get(slug)
        record = record if isinstance(record, dict) else {}
        record["updated_at"] = _now_iso()
        record.setdefault("created_at", _now_iso())
        record.setdefault("name", slug)
        prompts[slug] = record
        self._write_meta(meta)

    # --- active pointer -----------------------------------------------------
    def active_slug(self) -> str | None:
        active = self._read_meta().get("active")
        if isinstance(active, str) and self.exists(active):
            return active
        return None

    def set_active(self, slug: str) -> None:
        if not self.exists(slug):
            raise KeyError(f"no prompt with slug {slug!r}")
        meta = self._read_meta()
        meta["active"] = slug
        self._write_meta(meta)

    def active_entry(self) -> PromptEntry | None:
        slug = self.active_slug()
        return self.get(slug) if slug else None

    def active_text(self) -> str | None:
        """The active prompt's text, or ``None`` if the library is empty."""
        entry = self.active_entry()
        return entry.text if entry else None

    # --- seeding / migration ------------------------------------------------
    def ensure_seeded(
        self,
        default_text_factory: Callable[[], str],
        legacy_override: Path | None = None,
    ) -> None:
        """Populate an empty library so the UI always has something to show.

        Idempotent: does nothing once any entry exists. On first run it
        imports a legacy single-file override (if present) as the active
        entry; otherwise it seeds one entry from the built-in default.
        """
        if self.list():
            return
        if legacy_override is not None and legacy_override.is_file():
            try:
                text = legacy_override.read_text(encoding="utf-8")
            except OSError:
                text = default_text_factory()
                self.create("Built-in default", text)
                return
            self.create("Imported override", text)
            return
        self.create("Built-in default", default_text_factory())


__all__ = ["PromptEntry", "PromptLibrary", "LIBRARY_DIR", "PROMPTS_DIR", "slugify"]
