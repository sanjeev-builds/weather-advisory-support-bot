"""
Streamlit chat frontend for the Weather-Advisory Support Bot.

Backend behavior preserved exactly:
  - graph.invoke() call with same initial_state shape
  - session_state keys: graph, chat_history, resolved_location, activity_category
  - HumanMessage/AIMessage history management
  - Error handling (never exposes raw exceptions)
  - Reset clears chat_history, resolved_location, activity_category
"""

import html
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from graph import build_graph


def _flatten_html(s: str) -> str:
    """Normalize a multi-line HTML fragment to one tag per line, no leading
    whitespace, no blank lines.

    Why this exists: Streamlit's markdown renderer treats a blank (or
    whitespace-only) line as the end of a raw-HTML block. When a fragment
    that was dedented to column 0 (e.g. an SOP citation card) gets spliced
    into an outer f-string whose own lines still have their original source
    indentation, textwrap.dedent() can't help -- it looks for the common
    leading whitespace across the WHOLE combined string, and the presence of
    any zero-indented spliced line drags that common prefix down to zero, so
    nothing gets stripped. The outer wrapper's indented lines are then
    re-parsed as fresh Markdown after the blank-line break, and 4+ leading
    spaces there reads as an indented code block -- which is what showed
    literal "<div ...>" tags as visible text instead of rendering them.
    Stripping every line unconditionally sidesteps the whole failure mode.
    """
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Weather Advisory Assistant", page_icon="🌤️", layout="wide")

st.markdown("""
<style>
/* ---- Premium Dark Theme & Layout ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #070B14 0%, #0D1424 100%);
    color: #e2e8f0;
}

/* Hide top header bar */
[data-testid="stHeader"] {
    background: transparent;
    display: none;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0A0F1A !important;
    border-right: 1px solid rgba(255,255,255,0.05);
}
[data-testid="stSidebar"] * {
    color: #cbd5e1;
}

/* Increase main content area width to feel spacious but constrained */
.block-container {
    max-width: 68rem;
    padding-top: 3rem !important;
    padding-bottom: 7rem !important;
}

/* ---- Header Section ---- */
.main-header {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}
.main-header-icon {
    font-size: 3.5rem;
    filter: drop-shadow(0 0 20px rgba(56,189,248,0.2));
}
.main-header-text h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0;
    letter-spacing: -0.02em;
}
.main-header-text h1 span {
    color: #38bdf8;
}
.main-header-text p {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0.4rem 0;
}
.main-header-text .sub {
    font-size: 0.85rem;
    color: #64748b;
}

/* Status Pills */
.status-pills {
    display: flex;
    gap: 0.8rem;
    margin-top: 1rem;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 0.4rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: #94a3b8;
}
.pill-dot-green {
    width: 8px; height: 8px; border-radius: 50%; background: #34d399; box-shadow: 0 0 6px #34d399;
}
.pill-dot-blue {
    width: 8px; height: 8px; border-radius: 50%; background: #38bdf8; box-shadow: 0 0 6px #38bdf8;
}
.pill-dot-purple {
    width: 8px; height: 8px; border-radius: 50%; background: #c084fc; box-shadow: 0 0 6px #c084fc;
}

/* ---- Empty State Shortcuts ---- */
.shortcuts-row {
    display: flex;
    gap: 1rem;
    margin-top: 2rem;
    margin-bottom: 3rem;
    flex-wrap: nowrap;
    overflow-x: auto;
    padding-bottom: 1rem;
}
.shortcut-card {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    min-width: 200px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease;
}
.shortcut-card.active {
    border-color: rgba(56, 189, 248, 0.4);
    background: linear-gradient(145deg, rgba(56,189,248,0.1) 0%, #111827 100%);
}
.shortcut-icon {
    font-size: 1.5rem;
}
.shortcut-text h4 {
    margin: 0;
    color: #e2e8f0;
    font-size: 0.95rem;
    font-weight: 600;
}
.shortcut-text p {
    margin: 0;
    color: #64748b;
    font-size: 0.75rem;
    margin-top: 0.2rem;
}

/* ---- Chat Container ---- */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    margin-bottom: 2rem;
}
.chat-row {
    display: flex;
    width: 100%;
}
.chat-row.user {
    justify-content: flex-end;
}
.chat-row.assistant {
    justify-content: flex-start;
}

/* User Bubble */
.user-bubble-container {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    max-width: 70%;
}
.user-bubble {
    background: #1E3A8A; /* Darker blue */
    color: #f8fafc;
    padding: 1rem 1.2rem;
    border-radius: 16px 16px 4px 16px;
    font-size: 1rem;
    line-height: 1.5;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
}
.user-avatar {
    width: 38px;
    height: 38px;
    background: rgba(255,255,255,0.1);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    flex-shrink: 0;
}

/* Assistant Bubble */
.assistant-bubble-container {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    max-width: 85%;
}
.assistant-avatar {
    width: 48px;
    height: 48px;
    background: #111827;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    margin-top: 0.2rem;
}
.assistant-bubble {
    background: #151E2E;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 4px 16px 16px 16px;
    padding: 1.2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    width: 100%;
}
.assistant-text {
    color: #e2e8f0;
    font-size: 1rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}
.assistant-text p {
    margin-top: 0;
}

/* Inner Info Card (SOP / No-SOP) */
.inner-info-card {
    background: rgba(0,0,0,0.2);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 1rem;
}
.inner-info-card.sop {
    background: rgba(14, 165, 233, 0.05);
    border-color: rgba(14, 165, 233, 0.2);
}
.info-icon-large {
    font-size: 1.3rem;
    color: #38bdf8;
    margin-top: 0.1rem;
}
.info-content h5 {
    margin: 0 0 0.3rem 0;
    color: #f8fafc;
    font-size: 0.95rem;
    font-weight: 600;
}
.info-content p {
    margin: 0;
    color: #94a3b8;
    font-size: 0.85rem;
    line-height: 1.4;
}

/* Weather Facts Row */
.weather-facts-row {
    display: flex;
    flex-wrap: wrap;
    gap: 2rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    padding-top: 1.2rem;
    margin-top: 0.5rem;
}
.weather-fact {
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.fact-icon {
    font-size: 1.6rem;
    opacity: 0.9;
}
.fact-data {
    display: flex;
    flex-direction: column;
}
.fact-value {
    color: #f8fafc;
    font-weight: 600;
    font-size: 1rem;
}
.fact-label {
    color: #64748b;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---- Input Area ---- */
[data-testid="stChatInput"] {
    background-color: #111827;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 0.2rem 0.5rem;
    box-shadow: 0 -10px 40px rgba(0,0,0,0.5);
}
[data-testid="stChatInput"] textarea {
    background-color: #111827 !important;
    color: #f8fafc !important;
    caret-color: #f8fafc !important;
    font-size: 1rem;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;
}
[data-testid="stChatInput"] button {
    background-color: #3b82f6 !important;
    border-radius: 8px !important;
    color: white !important;
}

/* Sidebar Custom Details */
.sb-logo-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 2.5rem;
}
.sb-logo-icon {
    font-size: 2rem;
    color: #38bdf8;
}
.sb-logo-text h3 {
    margin: 0;
    color: #f8fafc;
    font-size: 1.2rem;
    font-weight: 700;
}
.sb-logo-text p {
    margin: 0;
    color: #94a3b8;
    font-size: 0.8rem;
}

.sb-section-title {
    color: #f8fafc;
    font-size: 0.95rem;
    font-weight: 600;
    margin: 1.8rem 0 1.2rem 0;
}
.sb-session-item {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    margin-bottom: 1.2rem;
}
.sb-session-icon {
    font-size: 1.3rem;
    margin-top: 0.1rem;
    color: #38bdf8;
}
.sb-session-data h4 {
    margin: 0;
    color: #e2e8f0;
    font-size: 0.9rem;
    font-weight: 500;
}
.sb-session-data p {
    margin: 0;
    color: #64748b;
    font-size: 0.8rem;
}

.sb-how-item {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 1.2rem;
}
.sb-how-number {
    width: 26px;
    height: 26px;
    border-radius: 4px;
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 700;
    flex-shrink: 0;
}
.sb-how-text h5 {
    margin: 0;
    color: #cbd5e1;
    font-size: 0.9rem;
    font-weight: 500;
}
.sb-how-text p {
    margin: 0;
    color: #64748b;
    font-size: 0.8rem;
}
.sb-promo-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 1.2rem;
    margin-top: 3rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.sb-promo-icon {
    color: #34d399;
    font-size: 1.6rem;
}

/* Clean up sidebar reset button */
.stButton>button {
    width: 100%;
    background-color: transparent !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #cbd5e1 !important;
    border-radius: 8px !important;
    padding: 0.7rem !important;
    font-weight: 500 !important;
    transition: all 0.2s;
}
.stButton>button:hover {
    background-color: rgba(255,255,255,0.05) !important;
    border-color: rgba(255,255,255,0.3) !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "resolved_location" not in st.session_state:
    st.session_state.resolved_location = None
if "activity_category" not in st.session_state:
    st.session_state.activity_category = None
if "message_metadata" not in st.session_state:
    # One entry per ASSISTANT message, in order -- {"matches": ..., "weather": ...}.
    # Kept as its own list (not just "the latest turn's data") so that scrolling
    # back through history still shows each reply's own SOP citation / weather
    # facts, not just the most recent one.
    st.session_state.message_metadata = []


# ---------------------------------------------------------------------------
# Helper definitions
# ---------------------------------------------------------------------------

CATEGORY_DISPLAY = {
    "outdoor_exercise": "Outdoor Exercise",
    "travel": "Travel & Commute",
    "vulnerable_groups": "Vulnerable Groups",
    "general": "General Activity",
    "off_topic": None,
}

SEVERITY_DISPLAY = {
    "critical": ("CRITICAL", "#f87171"),
    "high": ("HIGH", "#fb923c"),
    "moderate": ("MODERATE", "#fbbf24"),
    "low": ("LOW", "#34d399"),
}

def _extract_sop_citations(matches):
    """Only the concrete, already-decided decision metadata -- SOP id, name,
    category, severity. Never the LLM's free-text "reasoning" field (composite
    SOPs carry one internally) -- that's chain-of-thought, not a verified fact,
    and the assignment explicitly asks not to expose that, only the citation."""
    if not matches:
        return []
    citations = []
    for m in matches:
        citations.append({
            "id": m["id"],
            "name": m["name"],
            "category": CATEGORY_DISPLAY.get(m.get("category"), m.get("category", "")),
            "severity": m.get("severity", ""),
        })
    return citations

def render_weather_facts(weather_data):
    if not weather_data or "current" not in weather_data:
        return ""
    
    current = weather_data["current"]
    temp = current.get("temperature_2m", "--")
    wind = current.get("wind_speed_10m", "--")
    precip = current.get("precipitation", "--")
    uv = current.get("uv_index", "--")
    
    return _flatten_html(f"""
    <div class="weather-facts-row">
        <div class="weather-fact">
            <div class="fact-icon">🌡️</div>
            <div class="fact-data">
                <span class="fact-value">{temp}°C</span>
                <span class="fact-label">Temperature</span>
            </div>
        </div>
        <div class="weather-fact">
            <div class="fact-icon">💨</div>
            <div class="fact-data">
                <span class="fact-value">{wind} km/h</span>
                <span class="fact-label">Wind Speed</span>
            </div>
        </div>
        <div class="weather-fact">
            <div class="fact-icon">💧</div>
            <div class="fact-data">
                <span class="fact-value">{precip} mm</span>
                <span class="fact-label">Precipitation</span>
            </div>
        </div>
        <div class="weather-fact">
            <div class="fact-icon">☀️</div>
            <div class="fact-data">
                <span class="fact-value">{uv}</span>
                <span class="fact-label">UV Index</span>
            </div>
        </div>
    </div>
    """)


# ---------------------------------------------------------------------------
# Main UI rendering
# ---------------------------------------------------------------------------

# 1. Header
st.markdown(_flatten_html("""
<div class="main-header">
    <div class="main-header-icon">🌤️</div>
    <div class="main-header-text">
        <h1>Weather Advisory <span>Assistant</span></h1>
        <p>Live weather-based guidance for outdoor activities</p>
        <div class="sub">Advice is grounded in live weather data and written safety SOPs.</div>
        <div class="status-pills">
            <div class="status-pill"><div class="pill-dot-green"></div> Live weather data</div>
            <div class="status-pill"><div class="pill-dot-blue"></div> SOP grounded</div>
            <div class="status-pill"><div class="pill-dot-purple"></div> No guidance is invented</div>
        </div>
    </div>
</div>
"""), unsafe_allow_html=True)


# 2. Empty State / Shortcuts
if not st.session_state.chat_history:
    st.markdown(_flatten_html("""
    <div class="shortcuts-row">
        <div class="shortcut-card active">
            <div class="shortcut-icon">🚴</div>
            <div class="shortcut-text">
                <h4>Cycling</h4>
                <p>Is it safe to cycle today?</p>
            </div>
        </div>
        <div class="shortcut-card">
            <div class="shortcut-icon">🏃</div>
            <div class="shortcut-text">
                <h4>Running</h4>
                <p>Good for a run?</p>
            </div>
        </div>
        <div class="shortcut-card">
            <div class="shortcut-icon">✈️</div>
            <div class="shortcut-text">
                <h4>Travel</h4>
                <p>Safe to travel?</p>
            </div>
        </div>
        <div class="shortcut-card">
            <div class="shortcut-icon">👨‍👩‍👧</div>
            <div class="shortcut-text">
                <h4>Kids / Elderly</h4>
                <p>Is it safe for children?</p>
            </div>
        </div>
        <div class="shortcut-card">
            <div class="shortcut-icon">🌲</div>
            <div class="shortcut-text">
                <h4>Picnics</h4>
                <p>Plan a picnic?</p>
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)


# 3. Chat Messages Rendering (Custom HTML instead of st.chat_message)
if st.session_state.chat_history:
    chat_html = '<div class="chat-container">'
    assistant_index = 0  # counts assistant messages seen so far, to index message_metadata
    for i, msg in enumerate(st.session_state.chat_history):
        if msg.type == "human":
            safe_user_text = html.escape(msg.content)
            chat_html += _flatten_html(f"""
            <div class="chat-row user">
                <div class="user-bubble-container">
                    <div class="user-bubble">{safe_user_text}</div>
                    <div class="user-avatar">👤</div>
                </div>
            </div>
            """)
        else:
            safe_text = html.escape(msg.content).replace('\n', '<br>')
            metadata_html = ""
            weather_html = ""

            # Each assistant message gets ITS OWN matches/weather (not just the
            # most recent turn's) so evidence stays attached when you scroll
            # back through a longer conversation.
            meta = (
                st.session_state.message_metadata[assistant_index]
                if assistant_index < len(st.session_state.message_metadata)
                else None
            )
            assistant_index += 1

            if meta is not None:
                matches = meta.get("matches")
                if matches is not None:
                    if len(matches) > 0:
                        citations = _extract_sop_citations(matches)
                        c_html = ""
                        for c in citations:
                            sev_label, sev_color = SEVERITY_DISPLAY.get(c["severity"], (c["severity"].upper(), "#94a3b8"))
                            c_html += _flatten_html(f"""
                            <div class="inner-info-card sop">
                                <div class="info-icon-large">✓</div>
                                <div class="info-content">
                                    <h5>{html.escape(c['id'])} · {html.escape(c['name'])}</h5>
                                    <p>
                                        <span style="color:{sev_color}; font-weight:600;">{html.escape(sev_label)}</span>
                                        &nbsp;·&nbsp; {html.escape(c['category'])}
                                    </p>
                                </div>
                            </div>
                            """)
                        metadata_html = c_html
                    else:
                        metadata_html = _flatten_html("""
                        <div class="inner-info-card">
                            <div class="info-icon-large" style="color: #64748b;">ℹ️</div>
                            <div class="info-content">
                                <h5>No written guidance applies</h5>
                                <p>I checked the current weather, but none of the safety SOPs apply to these conditions.</p>
                            </div>
                        </div>
                        """)

                weather_html = render_weather_facts(meta.get("weather"))

            chat_html += _flatten_html(f"""
            <div class="chat-row assistant">
                <div class="assistant-bubble-container">
                    <div class="assistant-avatar">🌤️</div>
                    <div class="assistant-bubble">
                        <div class="assistant-text">{safe_text}</div>
                        {metadata_html}
                        {weather_html}
                    </div>
                </div>
            </div>
            """)
    chat_html += '</div>'
    # Belt-and-suspenders: flatten the fully-assembled string one more time.
    # Every piece above is already flattened, but this guarantees the whole
    # thing stays free of leading whitespace / blank lines no matter how the
    # pieces get concatenated.
    st.markdown(_flatten_html(chat_html), unsafe_allow_html=True)


# 4. Chat Input & Processing (UNCHANGED BACKEND)
user_input = st.chat_input("Ask about outdoor activity safety...")

if user_input:
    # Append to history and rerun to render immediately, then process
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # We use a spinner that spans the full width
    with st.spinner("Checking live weather and SOPs..."):
        try:
            initial_state = {
                "messages": st.session_state.chat_history,
                "resolved_location": st.session_state.resolved_location,
                "activity_category": st.session_state.activity_category,
                "activity_text": None,
                "weather": None,
                "matches": None,
                "error": None,
                "final_reply": None,
                "_location_query": None,
            }
            result = st.session_state.graph.invoke(initial_state)

            st.session_state.chat_history = result["messages"]
            st.session_state.resolved_location = result.get("resolved_location")
            st.session_state.activity_category = result.get("activity_category")
            # Exactly one assistant message was just added above -- append exactly
            # one metadata entry so message_metadata[i] keeps lining up with the
            # i-th assistant message in chat_history.
            st.session_state.message_metadata.append({
                "matches": result.get("matches"),
                "weather": result.get("weather"),
            })
        except Exception as exc:
            print(f"[app.py] graph.invoke failed: {type(exc).__name__}: {exc}")
            reply = "Something went wrong on my end and I can't reliably answer this turn. Please try again."
            st.session_state.chat_history.append(AIMessage(content=reply))
            st.session_state.message_metadata.append({"matches": None, "weather": None})

    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(_flatten_html("""
    <div class="sb-logo-row">
        <div class="sb-logo-icon">➕</div>
        <div class="sb-logo-text">
            <h3>MediBuddy</h3>
            <p>Weather Advisory</p>
            <p style="color: #64748b; font-size: 0.65rem;">Safer Days. Healthier Tomorrows.</p>
        </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title">Your Session</div>', unsafe_allow_html=True)

    if st.session_state.resolved_location:
        loc_name = html.escape(st.session_state.resolved_location['name'])
        st.markdown(_flatten_html(f"""
        <div class="sb-session-item">
            <div class="sb-session-icon">📍</div>
            <div class="sb-session-data">
                <h4>{loc_name}</h4>
                <p>Location Detected</p>
            </div>
        </div>
        """), unsafe_allow_html=True)

        if st.session_state.activity_category and st.session_state.activity_category != "off_topic":
            cat_name = html.escape(CATEGORY_DISPLAY.get(
                st.session_state.activity_category,
                st.session_state.activity_category.replace("_", " ").title()
            ))
            st.markdown(_flatten_html(f"""
        <div class="sb-session-item">
            <div class="sb-session-icon">🚲</div>
            <div class="sb-session-data">
                <p>Current Activity</p>
                <h4>{cat_name}</h4>
            </div>
        </div>
        """), unsafe_allow_html=True)
    else:
        st.markdown(_flatten_html("""
        <div class="sb-session-item">
            <div class="sb-session-icon" style="color:#64748b;">📍</div>
            <div class="sb-session-data">
                <h4 style="color:#64748b;">No location yet</h4>
            </div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown('<div style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)
    
    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.resolved_location = None
        st.session_state.activity_category = None
        st.session_state.message_metadata = []
        st.rerun()

    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section-title">How It Works</div>', unsafe_allow_html=True)
    
    st.markdown(_flatten_html("""
    <div class="sb-how-item">
        <div class="sb-how-number">1</div>
        <div class="sb-how-text">
            <h5>Fetches live weather data</h5>
            <p>Real-time data from Open-Meteo</p>
        </div>
    </div>
    <div class="sb-how-item">
        <div class="sb-how-number">2</div>
        <div class="sb-how-text">
            <h5>Matches written safety SOPs</h5>
            <p>Domain-specific rules & thresholds</p>
        </div>
    </div>
    <div class="sb-how-item">
        <div class="sb-how-number">3</div>
        <div class="sb-how-text">
            <h5>Returns grounded response</h5>
            <p>No invented advice</p>
        </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown(_flatten_html("""
    <div class="sb-promo-card">
        <div class="sb-promo-icon">🛡️</div>
        <div class="sb-how-text">
            <h5>Safety first, for a healthier you.</h5>
            <p style="margin-top:0.2rem;">MediBuddy Brainwave</p>
        </div>
    </div>
    """), unsafe_allow_html=True)
