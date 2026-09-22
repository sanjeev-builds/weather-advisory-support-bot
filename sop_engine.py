"""
SOP engine — loads policy from sops.yaml and matches it against live weather
data. This module is deliberately the ONLY place that knows how to read the
SOP file and decide "does this rule apply." Nothing about weather-fetching or
the LLM prompt lives in here, and nothing about SOP matching lives in graph.py.
That separation is what makes "add an 11th SOP without touching code" true:
adding a new threshold SOP to sops.yaml requires editing zero lines here.

Two kinds of SOPs, matched two different ways:

  - type: "threshold"  -> matched deterministically in Python. A condition
    is a (field, operator, threshold) triple checked against the weather
    dict. No LLM in the loop, no ambiguity, fully auditable.

  - type: "composite"  -> the condition can't be reduced to one inequality
    (the picnic case, the "severe weather system" case). For these we hand
    the SOP's own written criteria plus the real weather numbers to the LLM
    and ask ONLY "does this rule apply, and if so which guidance variant."
    The LLM never invents the criteria text or the numbers -- both come from
    this file and from Open-Meteo. It only makes the applicability call that
    can't be written as a clean if/else.

Multi-match resolution (decided, not incidental): every matching SOP is kept,
not just the first. They're sorted by severity (critical first). Any matched
SOP flagged as an "override" (SOP-012, the severe-weather-system rule) is
always pinned to the front of the list regardless of severity rank, because
its own rule text says it should be led with before category-specific advice.
"""

import yaml
from pathlib import Path
from typing import Optional

SOPS_PATH = Path(__file__).parent / "sops.yaml"

SEVERITY_RANK = {"critical": 4, "high": 3, "moderate": 2, "low": 1, "variable": 0}

_OPERATORS = {
    ">=": lambda v, t: v >= t,
    "<=": lambda v, t: v <= t,
    ">": lambda v, t: v > t,
    "<": lambda v, t: v < t,
    "==": lambda v, t: v == t,
    "in": lambda v, t: v in t,
}


def load_sops() -> list[dict]:
    with open(SOPS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _condition_met(condition: dict, weather: dict) -> bool:
    field = condition["field"]
    op = condition["operator"]
    threshold = condition["threshold"]
    if field not in weather or weather[field] is None:
        return False
    return _OPERATORS[op](weather[field], threshold)


def keywords_match(sop: dict, activity_category: str, activity_text: str) -> bool:
    """
    A threshold SOP is a candidate if its category matches the LLM-classified
    activity category, OR (belt-and-suspenders) one of its activity_keywords
    literally appears in the user's raw text. The category match is what
    carries paraphrases ("take my toddler to the swings" -> vulnerable_groups
    even with no keyword hit); the keyword check is a cheap safety net for
    cases where classification and keyword agree but category drifts.
    """
    keywords = sop.get("activity_keywords") or []
    if not keywords:
        # Empty keyword list is the documented "universal" marker (SOP-012).
        return True
    if sop["category"] == activity_category:
        return True
    text_lower = activity_text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def match_threshold_sops(activity_category: str, activity_text: str, weather: dict) -> list[dict]:
    """Deterministic matching for all type: threshold SOPs."""
    matches = []
    for sop in load_sops():
        if sop.get("type") != "threshold":
            continue
        if not keywords_match(sop, activity_category, activity_text):
            continue
        conditions = sop.get("conditions", [])
        if conditions and all(_condition_met(c, weather) for c in conditions):
            rendered_guidance = sop["guidance"].format(**weather)
            matches.append({
                "id": sop["id"],
                "name": sop["name"],
                "category": sop["category"],
                "severity": sop["severity"],
                "guidance": rendered_guidance.strip(),
                "source_note": sop.get("source_note", ""),
                "override": False,
            })
    return matches


def get_composite_sops() -> list[dict]:
    return [sop for sop in load_sops() if sop.get("type") == "composite"]


def render_composite_guidance(sop: dict, variant: Optional[str], weather: dict) -> str:
    """
    Render a composite SOP's guidance text for the variant the LLM judged
    applicable. All placeholders that reference real numbers come straight
    from the weather dict -- the LLM never fills these in itself.
    """
    template_field = sop.get("guidance_template")
    if template_field:
        template = template_field.get(variant) or next(iter(template_field.values()))
    else:
        template = sop["guidance"]

    # Narrative summary placeholders (favorable_summary, etc.) are optional
    # and, if present in the template but not in `weather`, are filled with
    # a neutral placeholder rather than raising -- keeps this robust to
    # policy authors adding new templates without updating this function.
    safe_weather = {**weather}
    for key in ("favorable_summary", "unfavorable_summary", "specific_mitigations"):
        safe_weather.setdefault(key, "see details above")

    return template.format(**safe_weather).strip()


def rank_matches(matches: list[dict]) -> list[dict]:
    """
    Sort key: (is this NOT an override, descending severity). Overrides sort first
    regardless of severity; ties within a group break by severity, high to low.

    Why this exists: the assignment's own worked example (an active low-pressure
    system over Madhya Pradesh) says a systemic weather event should "lead ... before
    any activity-specific advice" -- even when another matched SOP happens to carry
    the same "critical" severity. Sorting by severity alone doesn't guarantee that:
    Python's sort is stable, so two "critical" SOPs would keep whatever order they
    happened to load in from sops.yaml, which is an accident of file layout, not a
    decision. The `override` flag (set explicitly on SOP-012 in sops.yaml, not
    hardcoded here or anywhere in graph.py) is what makes the ordering a real,
    intentional rule instead of a side effect of file order.

    Example: if a Bhopal query matches both SOP-007 (Heavy Rainfall, critical,
    no override) and SOP-012 (Severe Weather System, critical, override: true),
    SOP-012 is guaranteed first -- not because it's "more critical" in the ranking
    table, but because its own policy text says it should be read first.
    """
    return sorted(
        matches,
        key=lambda m: (not m.get("override", False), -SEVERITY_RANK.get(m["severity"], 0)),
    )
