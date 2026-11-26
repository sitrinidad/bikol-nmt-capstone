"""Language identifiers used by the trained model (Phase 1 / Phase 2 notebook)."""
from __future__ import annotations

# Internal codes are the corpus column codes the adapter was trained with.
LANGUAGES: dict[str, dict[str, str]] = {
    "eng": {"name": "English", "label": "English"},
    "bcl_South": {"name": "Virac", "label": "Virac (Southern Bikol)"},
    "bcl_North": {"name": "Pandan", "label": "Pandan (Northern Bikol)"},
}

# NLLB-200 has no Bikol tag, so the notebook gives every language its own proxy tag.
# The real map is read from bikol_meta.json when the model loads; this is only the fallback.
DEFAULT_CODE_MAP: dict[str, str] = {"eng": "eng_Latn", "bcl_North": "tgl_Latn", "bcl_South": "war_Latn"}

# The six directions the "multi" adapter was trained on (English<->Pandan, English<->Virac, Virac<->Pandan).
DEFAULT_DIRECTIONS: list[tuple[str, str]] = [
    ("bcl_North", "eng"), ("eng", "bcl_North"),
    ("bcl_South", "eng"), ("eng", "bcl_South"),
    ("bcl_South", "bcl_North"), ("bcl_North", "bcl_South"),
]

BIKOL = {"bcl_South", "bcl_North"}
