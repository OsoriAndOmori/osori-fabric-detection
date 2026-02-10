from __future__ import annotations

from typing import Any


def get_rss_mb() -> float | None:
    """
    Best-effort RSS lookup.
    - On Linux, parse /proc/self/status (works on Render).
    - Otherwise return None.
    """
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = float(parts[1])
                        return kb / 1024.0
    except Exception:
        return None
    return None


def maybe_add_rss(payload: dict[str, Any], *, key: str = "rss_mb") -> dict[str, Any]:
    rss = get_rss_mb()
    if rss is None:
        return payload
    return {**payload, key: round(rss, 2)}

