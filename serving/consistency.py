"""Consistency checks on a served policy: values that cannot all be true together.

The printed-values check (``serving.printed_values``) settles whether a value is
on the page. It cannot see a value that IS on the page but sits in the wrong
field - a policy date read as an expiration date, a limit under the wrong type -
because the text prints it. Some of those break a rule the document itself obeys,
and a rule needs no model to check:

* ``date_order`` - the policy expires on or before it takes effect;
* ``term_too_long`` - a term of more than 24 months;
* ``implausible_date`` - an effective date outside 2000-2035;
* ``limit_order`` - a per-occurrence limit above the aggregate;
* ``premium_below_items`` - a total premium below the sum of its itemised amounts;
* ``bad_postal_code`` - not 5 digits, or ZIP+4;
* ``bad_state`` - not two capital letters;
* ``bad_policy_no`` - under 4 characters, or with no digit.

Each is a rule that holds on the reviewed documents of the first commercial set:
run over its 84 golds the module raised one flag, on a producer's postal code
of 8 digits (``99999-999``, a truncated ZIP+4 whose ``raw`` differs from its
``parsed``: the gold itself is worth a look), and none on any other rule - so a
flag is a real doubt, not review noise. Rules that did not hold there are not
here: a total equal to its items held on 43%, a coverage's form references all
being in the forms schedule on 11%.

A flagged value is kept, not removed - it is printed, and may be the document's
own oddity - and sent to review: ``flagged`` in its envelope, never
auto-accepted (``serving.value_flags``). The flag is ``<path>:<kind>``.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

DATE_ORDER_FLAG = "date_order"
TERM_TOO_LONG_FLAG = "term_too_long"
IMPLAUSIBLE_DATE_FLAG = "implausible_date"
LIMIT_ORDER_FLAG = "limit_order"
PREMIUM_BELOW_ITEMS_FLAG = "premium_below_items"
BAD_POSTAL_CODE_FLAG = "bad_postal_code"
BAD_STATE_FLAG = "bad_state"
BAD_POLICY_NUMBER_FLAG = "bad_policy_no"

CONSISTENCY_FLAGS = frozenset({
    DATE_ORDER_FLAG, TERM_TOO_LONG_FLAG, IMPLAUSIBLE_DATE_FLAG, LIMIT_ORDER_FLAG,
    PREMIUM_BELOW_ITEMS_FLAG, BAD_POSTAL_CODE_FLAG, BAD_STATE_FLAG, BAD_POLICY_NUMBER_FLAG,
})

#: The longest policy term, in days (24 months).
MAX_TERM_DAYS = 731
#: Effective dates outside these years are a misread, not a policy.
YEARS = (2000, 2035)
#: Half a cent either way: amounts are read from print, rounded to the dollar or cent.
ROUNDING = 0.51

_ZIP = re.compile(r"\d{5}(-?\d{4})?")
_STATE = re.compile(r"[A-Z]{2}")
#: A limit type's name for the per-occurrence and the aggregate limit.
_OCCURRENCE = ("per_occurrence",)
_AGGREGATE = ("general_aggregate", "aggregate")


def _is_value(node: Any) -> bool:
    return isinstance(node, dict) and "raw" in node and not isinstance(node.get("raw"), (dict, list))


def _value(node: Any) -> Any:
    """What an envelope says: its ``parsed``, or its ``raw`` when it parsed to nothing."""
    if not _is_value(node):
        return node
    parsed = node.get("parsed")
    return parsed if parsed not in (None, "") else node.get("raw")


def _number(node: Any) -> float | None:
    value = _value(node)
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "").replace("$", ""))
    except ValueError:
        return None


def _date(node: Any) -> datetime | None:
    value = _value(node)
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    return None


def consistency_flags(extraction: Any) -> list[str]:
    """``<path>:<kind>`` for each value that breaks a rule the document obeys.

    ``extraction`` is a policy as the pipeline holds it (envelopes of ``raw`` and
    ``parsed``, in the model form or the canonical one), paths as the served JSON
    names them. A rule needs the values it compares: with one missing, it says
    nothing.
    """
    if not isinstance(extraction, dict):
        return []
    flags: list[str] = []

    policy = extraction.get("policy") if isinstance(extraction.get("policy"), dict) else {}
    effective, expiration = _date(policy.get("effective_date")), _date(policy.get("expiration_date"))
    if effective is not None and not YEARS[0] <= effective.year <= YEARS[1]:
        flags.append(f"policy.effective_date:{IMPLAUSIBLE_DATE_FLAG}")
    if effective is not None and expiration is not None:
        if expiration <= effective:
            flags.append(f"policy.expiration_date:{DATE_ORDER_FLAG}")
        elif (expiration - effective).days > MAX_TERM_DAYS:
            flags.append(f"policy.expiration_date:{TERM_TOO_LONG_FLAG}")
    number = _value(policy.get("policy_number"))
    if isinstance(number, str) and number.strip() and (len(number.strip()) < 4 or not re.search(r"\d", number)):
        flags.append(f"policy.policy_number:{BAD_POLICY_NUMBER_FLAG}")

    premium = extraction.get("premium") if isinstance(extraction.get("premium"), dict) else {}
    total = _number(premium.get("total"))
    items = [_number(item.get("amount")) for item in premium.get("items") or [] if isinstance(item, dict)]
    if total is not None and items and None not in items and sum(items) > total + ROUNDING:
        flags.append(f"premium.total:{PREMIUM_BELOW_ITEMS_FLAG}")

    for index, coverage in enumerate(extraction.get("coverages") or []):
        if not isinstance(coverage, dict):
            continue
        limits = [(i, item) for i, item in enumerate(coverage.get("limits") or []) if isinstance(item, dict)]
        by_type = {str(_value(item.get("limit_type"))): (i, _number(item.get("amount"))) for i, item in limits}
        occurrence = next((by_type[k] for k in _OCCURRENCE if k in by_type), None)
        aggregate = next((by_type[k] for k in _AGGREGATE if k in by_type), None)
        if occurrence and aggregate and occurrence[1] is not None and aggregate[1] is not None \
                and occurrence[1] > aggregate[1]:
            flags.append(f"coverages[{index}].limits[{occurrence[0]}].amount:{LIMIT_ORDER_FLAG}")

    _addresses(extraction, "", flags)
    return flags


def _addresses(node: Any, path: str, flags: list[str]) -> None:
    """Every postal code and state of every address in the document."""
    if _is_value(node):
        return
    if isinstance(node, dict):
        for key, child in node.items():
            here = f"{path}.{key}" if path else str(key)
            if key == "postal_code" and _is_value(child):
                value = _value(child)
                if isinstance(value, str) and value.strip() and not _ZIP.fullmatch(value.strip()):
                    flags.append(f"{here}:{BAD_POSTAL_CODE_FLAG}")
            elif key == "state" and "address" in path.rsplit(".", 1)[-1] and _is_value(child):
                value = _value(child)
                if isinstance(value, str) and value.strip() and not _STATE.fullmatch(value.strip()):
                    flags.append(f"{here}:{BAD_STATE_FLAG}")
            else:
                _addresses(child, here, flags)
    elif isinstance(node, list):
        for index, child in enumerate(node):
            _addresses(child, f"{path}[{index}]", flags)
