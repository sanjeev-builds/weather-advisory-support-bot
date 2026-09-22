"""
Tests for extract_intent's conversation-context handling: _build_previous_context
(a pure function, tested directly) and extract_intent itself (tested with a
mocked LLM -- no real API key or network call, since these test our own state
plumbing and routing logic, not what a real model chooses to say).

Run: pytest tests/test_intent_context.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch
from langchain_core.messages import HumanMessage, AIMessage

import graph as g


def _state(messages, resolved_location=None, activity_category=None):
    return {
        "messages": messages,
        "resolved_location": resolved_location,
        "activity_category": activity_category,
        "activity_text": None,
        "weather": None,
        "matches": None,
        "error": None,
        "final_reply": None,
        "_location_query": None,
    }


class _FakeStructuredLLM:
    """Stands in for get_llm().with_structured_output(IntentExtraction) --
    returns a fixed IntentExtraction (or raises, to simulate a provider/
    validation failure) regardless of the messages passed in, so these tests
    check OUR code's handling of the result, not a real model's judgment."""

    def __init__(self, result=None, raises=False):
        self._result = result
        self._raises = raises

    def invoke(self, messages):
        if self._raises:
            raise RuntimeError("simulated provider/validation failure")
        return self._result


def _mock_llm(result=None, raises=False):
    fake_top = type("FakeLLM", (), {
        "with_structured_output": lambda self, schema: _FakeStructuredLLM(result, raises)
    })()
    return patch("graph.get_llm", return_value=fake_top)


# ---------------------------------------------------------------------------
# _build_previous_context -- pure function, no LLM
# ---------------------------------------------------------------------------

def test_previous_context_empty_on_first_message():
    state = _state([HumanMessage(content="Is it safe to cycle today in Delhi?")])
    assert g._build_previous_context(state) == ""


def test_previous_context_summarizes_prior_turn():
    state = _state(
        messages=[
            HumanMessage(content="Is it safe to cycle today in Delhi?"),
            AIMessage(content="Here's what's happening in Delhi..."),
            HumanMessage(content="What about this evening?"),
        ],
        resolved_location={"name": "Delhi", "admin1": "NCT", "country": "India"},
        activity_category="outdoor_exercise",
    )
    context = g._build_previous_context(state)
    assert "Is it safe to cycle today in Delhi?" in context
    assert "Delhi" in context
    assert "outdoor_exercise" in context


def test_previous_context_handles_no_prior_resolution_gracefully():
    """A prior turn exists (e.g. it hit need_location) but nothing was resolved
    yet -- must not crash, must say so plainly rather than fabricate a value."""
    state = _state(
        messages=[
            HumanMessage(content="Is it safe to cycle today?"),
            AIMessage(content="Which location should I check?"),
            HumanMessage(content="What about this evening?"),
        ],
        resolved_location=None,
        activity_category=None,
    )
    context = g._build_previous_context(state)
    assert "not yet established" in context


# ---------------------------------------------------------------------------
# extract_intent -- mocked LLM, testing our own state/routing logic
# ---------------------------------------------------------------------------

def test_case1_first_standalone_query_extracts_delhi_and_cycling():
    result = g.IntentExtraction(
        location_mentioned="Delhi", category="outdoor_exercise",
        activity_summary="cycling", extra_keywords=["bike"],
    )
    state = _state([HumanMessage(content="Is it safe to cycle today in Delhi?")])
    with _mock_llm(result=result):
        updates = g.extract_intent(state)
    assert updates["_location_query"] == "Delhi"
    assert updates["activity_category"] == "outdoor_exercise"
    assert "cycling" in updates["activity_text"]
    assert "error" not in updates


def test_case2_followup_reuses_location_when_llm_returns_no_new_location():
    """The model (mocked here to simulate a correctly-context-aware response)
    returns no new location for a terse follow-up -- extract_intent's own
    logic must signal reuse of the prior turn's location, not a failure."""
    result = g.IntentExtraction(
        location_mentioned=None, category="outdoor_exercise",
        activity_summary="cycling this evening", extra_keywords=["bike"],
    )
    state = _state(
        messages=[
            HumanMessage(content="Is it safe to cycle today in Delhi?"),
            AIMessage(content="..."),
            HumanMessage(content="What about this evening?"),
        ],
        resolved_location={"name": "Delhi", "admin1": "NCT", "country": "India"},
        activity_category="outdoor_exercise",
    )
    with _mock_llm(result=result):
        updates = g.extract_intent(state)
    assert updates["_location_query"] is None  # signal: geocode_node reuses prior location
    assert updates["activity_category"] == "outdoor_exercise"
    assert "cycling" in updates["activity_text"]
    assert "error" not in updates
    assert g.route_after_intent({**state, **updates}) == "geocode"


def test_case3_explicit_new_location_overrides_prior():
    result = g.IntentExtraction(
        location_mentioned="Mumbai", category="outdoor_exercise",
        activity_summary="cycling", extra_keywords=[],
    )
    state = _state(
        messages=[
            HumanMessage(content="Is it safe to cycle today in Delhi?"),
            AIMessage(content="..."),
            HumanMessage(content="What about Mumbai?"),
        ],
        resolved_location={"name": "Delhi", "admin1": "NCT", "country": "India"},
        activity_category="outdoor_exercise",
    )
    with _mock_llm(result=result):
        updates = g.extract_intent(state)
    assert updates["_location_query"] == "Mumbai"  # new location wins, not Delhi


def test_case4_explicit_new_activity_changes_category_not_retained_blindly():
    result = g.IntentExtraction(
        location_mentioned=None, category="vulnerable_groups",
        activity_summary="taking my kid to the park", extra_keywords=["child", "park"],
    )
    state = _state(
        messages=[
            HumanMessage(content="Is it safe to cycle today in Delhi?"),
            AIMessage(content="..."),
            HumanMessage(content="What about taking my kid to the park?"),
        ],
        resolved_location={"name": "Delhi", "admin1": "NCT", "country": "India"},
        activity_category="outdoor_exercise",
    )
    with _mock_llm(result=result):
        updates = g.extract_intent(state)
    assert updates["activity_category"] == "vulnerable_groups"  # not "outdoor_exercise"
    assert updates["_location_query"] is None  # Delhi still reused (no new location given)


def test_case5_new_valid_activity_not_covered_by_any_sop_is_not_a_failure():
    """'Fly a kite' must parse successfully (category, activity extracted) --
    the fact that no SOP covers kite-flying is a downstream match_sops concern,
    not an intent-extraction failure."""
    result = g.IntentExtraction(
        location_mentioned=None, category="general",
        activity_summary="flying a kite", extra_keywords=["kite"],
    )
    state = _state([HumanMessage(content="Is it a good day to fly a kite?")])
    with _mock_llm(result=result):
        updates = g.extract_intent(state)
    assert "error" not in updates
    assert updates["activity_category"] == "general"
    # No location ever given and none resolved -> routes to need_location, not intent_error.
    assert g.route_after_intent({**state, **updates}) == "need_location"


def test_case6_garbage_input_fails_safely_not_treated_as_valid_activity():
    """Simulates the LLM call itself failing/erroring on unparseable input --
    the existing safe failure path must still produce an honest error, not a
    fabricated activity."""
    state = _state([HumanMessage(content="asdfghqwerty")])
    with _mock_llm(raises=True):
        updates = g.extract_intent(state)
    assert updates == {"error": "intent_failed"}
    assert g.route_after_intent({**state, **updates}) == "intent_error"
