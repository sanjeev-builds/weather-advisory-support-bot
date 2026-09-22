"""
Deterministic unit tests for sop_engine.py -- SOP matching, severity, and
ranking. No LLM, no network, no API key needed: these test the code path the
assignment calls out as most important to get right (the policy engine that
decides matches/severity/order without any model in the loop).

Run: pytest tests/test_sop_engine.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import sop_engine

MILD_WEATHER = {
    "temperature_2m": 22.0, "apparent_temperature": 21.0,
    "precipitation": 0.0, "wind_speed_10m": 10.0, "wind_gusts_10m": 15.0,
    "precipitation_probability": 5, "uv_index": 3.0, "weather_code": 1,
}


def test_sops_load_with_expected_coverage():
    """Sanity check on the actual sops.yaml file (not a copy): the assignment's
    minimum bar of 10+ SOPs, 3+ categories, multiple severities, both matching
    types, and at least one override SOP."""
    sops = sop_engine.load_sops()
    assert len(sops) >= 10

    categories = {s["category"] for s in sops}
    assert len(categories) >= 3

    severities = {s["severity"] for s in sops}
    assert len(severities) >= 3

    types = {s["type"] for s in sops}
    assert "threshold" in types
    assert "composite" in types, "no fuzzy/non-numeric SOP found"

    overrides = [s for s in sops if s.get("override")]
    assert len(overrides) >= 1, "no severe-weather override SOP found"


def test_threshold_match_high_wind():
    """SOP-002 (wind >= 40 km/h) fires on cycling when wind is high, using
    real weather-shaped data (mocked value, real matching code)."""
    weather = {**MILD_WEATHER, "wind_speed_10m": 50.0}
    matches = sop_engine.match_threshold_sops("outdoor_exercise", "cycling", weather)
    ids = [m["id"] for m in matches]
    assert "SOP-002" in ids
    matched = next(m for m in matches if m["id"] == "SOP-002")
    assert "50.0" in matched["guidance"], "real wind number must appear in rendered guidance"


def test_threshold_no_match_on_mild_weather():
    """The flip side: calm conditions should trigger nothing for a cycling
    question. An empty match list is the correct, expected outcome here --
    not a bug (see the Bhopal investigation in this project's history)."""
    matches = sop_engine.match_threshold_sops("outdoor_exercise", "cycling", MILD_WEATHER)
    assert matches == []


def test_keywords_match_category_fallback_without_keyword_overlap():
    """A threshold SOP can match purely via category equality, with zero
    literal keyword overlap in the activity text -- this is the mechanism
    that makes paraphrased questions match without hardcoded synonym lists."""
    sop = next(s for s in sop_engine.load_sops() if s["id"] == "SOP-008")
    # activity_text deliberately contains none of SOP-008's own keywords
    assert sop_engine.keywords_match(sop, "vulnerable_groups", "totally unrelated words here")
    assert not sop_engine.keywords_match(sop, "travel", "totally unrelated words here")


def test_override_sop_ranked_before_same_severity_non_override():
    """The core precedence guarantee: an override-flagged SOP outranks a
    same-severity non-override SOP, regardless of which one appears first in
    the input list (i.e. regardless of sops.yaml file order)."""
    non_override = {"id": "SOP-007", "severity": "critical", "override": False}
    override = {"id": "SOP-012", "severity": "critical", "override": True}

    ranked_a = sop_engine.rank_matches([non_override, override])
    ranked_b = sop_engine.rank_matches([override, non_override])

    assert [m["id"] for m in ranked_a] == ["SOP-012", "SOP-007"]
    assert [m["id"] for m in ranked_b] == ["SOP-012", "SOP-007"]


def test_rank_matches_orders_by_severity_when_no_override():
    low = {"id": "A", "severity": "low", "override": False}
    critical = {"id": "B", "severity": "critical", "override": False}
    moderate = {"id": "C", "severity": "moderate", "override": False}

    ranked = sop_engine.rank_matches([low, critical, moderate])
    assert [m["id"] for m in ranked] == ["B", "C", "A"]


def test_composite_sops_are_loaded_separately_from_threshold_sops():
    composite = sop_engine.get_composite_sops()
    assert len(composite) >= 2  # picnic + severe-weather-system, per sops.yaml
    assert all(s["type"] == "composite" for s in composite)


def test_render_composite_guidance_uses_real_weather_numbers():
    sop = next(s for s in sop_engine.load_sops() if s["id"] == "SOP-012")
    weather = {**MILD_WEATHER, "precipitation": 12.5, "wind_speed_10m": 33.0, "wind_gusts_10m": 55.0}
    text = sop_engine.render_composite_guidance(sop, None, weather)
    assert "12.5" in text
    assert "33.0" in text
    assert "55.0" in text
