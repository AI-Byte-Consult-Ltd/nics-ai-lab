"""Licence gate for NICS Vision Geo images, checked per image, not per source.

Allowed for training a commercial model (docs/dataset-register.md, rules 1-2):
  CC0 / public domain marks           -> no conditions
  CC BY (any version)                 -> attribution kept in the manifest
  CC BY-SA (any version)              -> attribution kept; flagged share_alike, so the set
                                         can be excluded with one switch if counsel says so
Rejected: anything with NC (non-commercial) or ND (no derivatives), "all rights reserved",
fair use, and anything we cannot parse. Unknown means no.
"""
from __future__ import annotations

import re

_PD = {"cc0", "cc0 1.0", "cc-zero", "public domain", "pd", "pdm", "public domain mark", "no restrictions",
       "pd-us", "pd-usgov", "pd-old", "pd-art"}


def normalize(raw: str | None) -> str | None:
    """Licence text or URL -> short id ('CC0', 'PD', 'CC-BY-4.0', 'CC-BY-SA-3.0'), or None if not allowed."""
    if not raw:
        return None
    t = raw.strip().lower()
    if "creativecommons.org/publicdomain/zero" in t:
        return "CC0"
    if "creativecommons.org/publicdomain/mark" in t:
        return "PD"
    m = re.search(r"creativecommons\.org/licenses/([a-z\-]+)/(\d(?:\.\d)?)", t)
    if m:
        kind, ver = m.group(1), m.group(2)
        return _short(kind.split("-"), ver)
    t = t.replace("_", " ")
    if t in _PD or t.startswith("pd-") or t.startswith("public domain"):
        return "PD" if not t.startswith("cc0") else "CC0"
    if t.startswith("cc0"):
        return "CC0"
    m = re.match(r"^cc[ \-]?((?:by|sa|nc|nd)(?:[ \-](?:by|sa|nc|nd))*)[ \-]?(\d(?:\.\d)?)?", t)
    if m:
        return _short(re.split(r"[ \-]", m.group(1)), m.group(2) or "")
    return None


def _short(parts: list[str], ver: str) -> str | None:
    parts = [p for p in parts if p]
    if not parts or parts[0] != "by" or "nc" in parts or "nd" in parts:
        return None
    base = "CC-BY-SA" if "sa" in parts else "CC-BY"
    return f"{base}-{ver}" if ver else base


def share_alike(short_id: str | None) -> bool:
    return bool(short_id and short_id.startswith("CC-BY-SA"))


def needs_attribution(short_id: str | None) -> bool:
    return bool(short_id and short_id.startswith("CC-BY"))
