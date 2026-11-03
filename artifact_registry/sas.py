"""When a storage SAS token stops working, read from the token itself.

A SAS token carries its expiry as ``se=`` in its query string. A token that runs out in the middle of a
fifteen-hour cycle is found at the first write after it, hours in, so the start of a run reads the date and
refuses a token that cannot last. Only the expiry is read; the token is never printed or kept.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs

#: The settings that hold a SAS token: the main container's, and the raw container's own.
SAS_SETTINGS: tuple[str, ...] = ("AZURE_STORAGE_SAS_TOKEN", "AZURE_RAW_SAS_TOKEN")

#: A token with less than this left is refused at the start of a run; with less than ``WARN_BELOW``, warned about.
REFUSE_BELOW = timedelta(hours=1)
WARN_BELOW = timedelta(hours=24)


def sas_expiry(token: str | None) -> datetime | None:
    """The moment ``token`` stops working (UTC), or ``None`` when it names none."""
    if not token:
        return None
    values = parse_qs(token.strip().lstrip("?")).get("se")
    if not values:
        return None
    text = values[0].strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def sas_problems(environ: Mapping[str, str], now: datetime | None = None) -> tuple[list[str], list[str]]:
    """``(refusals, warnings)`` about the SAS tokens in ``environ``, by setting name; never the token."""
    now = now or datetime.now(UTC)
    refusals: list[str] = []
    warnings: list[str] = []
    for name in SAS_SETTINGS:
        expiry = sas_expiry(environ.get(name))
        if expiry is None:
            continue
        left = expiry - now
        stamp = expiry.strftime("%Y-%m-%d %H:%M UTC")
        if left <= timedelta(0):
            refusals.append(f"{name} expired at {stamp}")
        elif left < REFUSE_BELOW:
            refusals.append(f"{name} expires at {stamp}, in under an hour")
        elif left < WARN_BELOW:
            warnings.append(f"{name} expires at {stamp}, in {left.total_seconds() / 3600:.1f} h; a full cycle "
                            "takes about 15 h")
    return refusals, warnings
