"""The served check's sharper-images option (scripts.check_served_fixes --max-pixels)."""

from __future__ import annotations

import pytest

from common import config


def test_the_budget_override_reaches_every_full_resolution_reader(monkeypatch):
    from scripts.check_served_fixes import use_pixel_budget

    monkeypatch.setattr(config, "shared_vision_config", config.shared_vision_config)   # restored after the test
    before = config.pixel_budget()
    use_pixel_budget(2483200)
    assert config.pixel_budget() == (before[0], 2483200)
    assert int(config.vision_for_task("classify")["max_pixels"]) == 262144              # thumbnails untouched


def test_a_budget_that_is_not_whole_visual_tokens_is_refused(monkeypatch):
    from scripts.check_served_fixes import use_pixel_budget

    monkeypatch.setattr(config, "shared_vision_config", config.shared_vision_config)
    with pytest.raises(SystemExit):
        use_pixel_budget(2483201)
