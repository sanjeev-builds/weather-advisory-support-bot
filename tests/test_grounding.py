"""
Unit tests for the grounding/fallback machinery in graph.py -- the mechanism
that stops an LLM rephrasing from silently dropping or altering a fact.
Tests the pure functions directly (_is_grounded, _render_deterministic_reply,
_build_facts_payload); no LLM call, no API key needed, since these functions
take plain strings/dicts in and return plain strings/bools out.

Run: pytest tests/test_grounding.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import graph as g

WEATHER = {
    "temperature_2m": 24.5, "precipitation": 0.0,
    "wind_speed_10m": 6.3, "uv_index": 0.0,
}

MATCH = {
    "id": "SOP-002",
    "name": "High Wind — Cycling and Two-Wheelers",
    "severity": "high",
    "guidance": "Wind speed at your location is currently 50.0 km/h. Winds above 40 km/h pose a direct safety risk.",
    "override": False,
}


def test_is_grounded_true_when_id_and_numbers_preserved():
    reply = "The wind is 50.0 km/h right now. Per SOP-002, winds above 40 km/h are a real safety risk."
    assert g._is_grounded(reply, [MATCH]) is True


def test_is_grounded_false_when_sop_id_missing():
    reply = "The wind is 50.0 km/h and gusts above 40 km/h are risky."  # no "SOP-002" at all
    assert g._is_grounded(reply, [MATCH]) is False


def test_is_grounded_false_when_a_fabricated_sop_id_is_added():
    reply = "Per SOP-002 and SOP-999, wind is 50.0 km/h, above the 40 km/h threshold."
    assert g._is_grounded(reply, [MATCH]) is False


def test_is_grounded_false_when_a_number_is_altered():
    """The exact failure mode the grounding check exists to catch: the LLM
    keeps the SOP id but silently changes a number."""
    reply = "Per SOP-002, wind is 65.0 km/h, above the 40 km/h threshold."  # 50.0 -> 65.0
    assert g._is_grounded(reply, [MATCH]) is False


def test_is_grounded_true_for_no_match_case():
    """No matches means no SOP ids are expected -- a reply that cites none is grounded."""
    reply = "No written SOP applies to this question."
    assert g._is_grounded(reply, []) is True


def test_is_grounded_false_for_no_match_case_with_fabricated_citation():
    reply = "Per SOP-042, this should be fine."
    assert g._is_grounded(reply, []) is False


def test_render_deterministic_reply_no_matches_reports_real_weather():
    text = g._render_deterministic_reply([], WEATHER, "Bhopal, Madhya Pradesh India")
    assert "24.5" in text
    assert "6.3" in text
    assert "no written safety policy" in text.lower() or "don't have" in text.lower()
    assert "SOP-" not in text


def test_render_deterministic_reply_with_match_cites_sop_and_guidance():
    text = g._render_deterministic_reply([MATCH], WEATHER, "Pune, Maharashtra India")
    assert "SOP-002" in text
    assert MATCH["guidance"] in text
    assert "HIGH" in text  # severity label


def test_render_deterministic_reply_multi_match_notes_ranking():
    override_match = {**MATCH, "id": "SOP-012", "severity": "critical", "override": True, "guidance": "Severe system active."}
    text = g._render_deterministic_reply([override_match, MATCH], WEATHER, "Chennai, Tamil Nadu India")
    assert "SOP-012" in text
    assert "SOP-002" in text
    assert "2 policies matched" in text


def test_build_facts_payload_excludes_raw_user_question():
    """The payload shown to the rephrasing LLM must never include the user's
    raw text -- that's the main defense against prompt injection reaching the
    final composition step."""
    payload = g._build_facts_payload([MATCH], WEATHER, "Pune, Maharashtra India")
    assert "ignore all your safety policies" not in payload.lower()
    assert "SOP-002" in payload
    assert MATCH["guidance"] in payload


def test_build_facts_payload_no_match_states_decision_plainly():
    payload = g._build_facts_payload([], WEATHER, "Mumbai, Maharashtra India")
    assert "no written sop applies" in payload.lower()
