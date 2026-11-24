"""Moved to :mod:`adapters.served_ocr` (review A1, 2026-11-24). This path is that same module: an
import or a patch through either name reaches one object."""

import sys

from adapters import served_ocr as _moved

sys.modules[__name__] = _moved
