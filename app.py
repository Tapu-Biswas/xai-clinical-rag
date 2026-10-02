"""
Streamlit UI for the Explainable AI in Clinical Diagnosis research assistant.

HOW ANSWERS ARE SERVED
----------------------
1. Ready-made answers: the "Starting points" questions (and close rewordings of
   them) are answered from precomputed_answers.json, made once by
   precompute_answers.py, so they load instantly and never touch the Gemini API.
2. Off-topic questions: if no paper is close enough to the question, the app
   says so without calling Gemini.
3. Live answers: anything else retrieves the relevant papers and asks Gemini.
   Results are remembered, and live use is capped to protect the free quota.

Settings, the prompt and the Starting points live in config.py.
If chroma_db/ doesn't exist, it is built automatically from corpus_raw.json.

SETUP
-----
pip install -r requirements.txt
Put your Gemini key in .streamlit/secrets.toml (only needed for live answers):
  GEMINI_API_KEY = "your-key-here"

RUN
---
streamlit run app.py
"""

import os
import json
import html
import streamlit as st

from config import (
    PRECOMPUTED_FILE, STARTING_POINTS, LIVE_LIMIT_PER_SESSION, LIVE_LIMIT_PER_HOUR,
    load_collection as _load_collection, retrieve, fingerprint, first_author, paper_count,
    find_ready, normalize_question, parse_sections, check_citations,
    make_client, generate_with_retry, is_rate_limit, error_code, get_api_key, LiveLimiter,
)

SECTION_META = {
    "Current Approaches": {"class": "sec-current", "mark": "01"},
    "Limitations": {"class": "sec-limits", "mark": "02"},
    "Suggested Future Work": {"class": "sec-future", "mark": "03"},
}

OFF_TOPIC_MSG = (
    "None of the papers in this database are close enough to that question to answer it. "
    "This assistant only covers <strong>explainable AI in clinical diagnosis</strong> — try asking "
    "about methods like SHAP or Grad-CAM, a disease or imaging type, evaluation, or clinical adoption."
)


# ---------- cached resources ----------

@st.cache_resource(show_spinner="Preparing the paper database (the first run takes a minute or two)...")
def load_collection():
    return _load_collection(log=lambda msg: None)


@st.cache_resource
def corpus_fingerprint():
    return fingerprint(load_collection())


@st.cache_resource
def load_precomputed(current_fp):
    """Saved answers, only if they were made for this exact corpus/model/prompt."""
    if not os.path.exists(PRECOMPUTED_FILE):
        return {}, "missing"
    with open(PRECOMPUTED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("fingerprint") != current_fp:
        return {}, "stale"
    return data.get("answers", {}), "ok"


@st.cache_resource
def answer_cache():
    """Shared across visitors, so a repeated question costs no API quota."""
    return {}


@st.cache_resource
def live_limiter():
    return LiveLimiter(LIVE_LIMIT_PER_HOUR)


# ---------- answering ----------

def answer_question(question, ready_answers, api_key):
    """Work out the answer for a question. Returns a result dict that is kept in
    session_state, so it stays on screen until the next question."""
    ready = find_ready(ready_answers, question)
    if ready:
        return {"kind": "answer", "question": ready["question"], "asked": question,
                "answer": ready["answer"], "sources": ready["sources"]}

    key = normalize_question(question)
    cache = answer_cache()
    if key in cache:
        return cache[key]

    with st.spinner("Finding relevant papers..."):
        retrieved = retrieve(load_collection(), question)
    if not retrieved:
        return {"kind": "offtopic"}

    if not api_key:
        return {"kind": "error", "message": "That question isn't one of the ready-made ones, and live answers "
                "aren't set up on this copy of the app. Pick a starting point from the sidebar."}
    if st.session_state.get("live_count", 0) >= LIVE_LIMIT_PER_SESSION:
        return {"kind": "error", "message": f"You've reached the limit of {LIVE_LIMIT_PER_SESSION} new questions "
                "for this session. The Starting points on the left still load instantly."}
    if not live_limiter().allow():
        return {"kind": "error", "message": "The app has reached its hourly limit for new questions. "
                "The Starting points on the left still load instantly — please try again later."}

    try:
        with st.spinner("Writing the answer (if the free-tier limit was hit, this can take up to a minute)..."):
            answer = generate_with_retry(make_client(api_key), question, retrieved, max_retries=1, max_wait=40)
    except Exception as e:
        msg, code = str(e), error_code(e)
        if is_rate_limit(e):
            text = "Live answers are rate-limited right now. Try a Starting point on the left, or wait a minute and search again."
        elif code == 404 or "not found" in msg.lower():
            text = f"The model name in config.py isn't available for this API key. Details: {msg[:300]}"
        elif code in (401, 403) or "api key" in msg.lower() or "permission" in msg.lower():
            text = f"API key problem. Details: {msg[:300]}"
        else:
            text = f"Something went wrong while writing the answer. Details: {msg[:300]}"
        return {"kind": "error", "message": text}

    st.session_state["live_count"] = st.session_state.get("live_count", 0) + 1
    result = {"kind": "answer", "question": question, "asked": question,
              "answer": answer, "sources": [meta for _, meta in retrieved]}
    cache[key] = result
    return result


# ---------- rendering ----------

def notice(kind, text):
    st.markdown(f'<div class="notice {kind}">{text}</div>', unsafe_allow_html=True)


def render_answer(result):
    answer, sources = result["answer"], result["sources"]
    if normalize_question(result["asked"]) != normalize_question(result["question"]):
        notice("info", f"Showing the saved answer for the closely matching question: "
                       f"<em>{html.escape(result['question'])}</em>")

    sections = parse_sections(answer)
    if not sections:
        st.markdown(answer)
    for heading, body in sections:
        meta = SECTION_META[heading]
        st.markdown(
            f'<div class="section-block {meta["class"]}"><div class="section-head">'
            f'<span class="section-mark">{meta["mark"]}</span>'
            f'<span class="section-title">{html.escape(heading)}</span></div></div>',
            unsafe_allow_html=True,
        )
        if heading == "Suggested Future Work":
            st.markdown('<div class="future-note">AI-generated suggestions based on the gaps above — '
                        'these are not findings from the papers.</div>', unsafe_allow_html=True)
        st.markdown(body)

    valid, invalid = check_citations(answer, len(sources))
    if invalid:
        nums = ", ".join(f"[{n}]" for n in sorted(invalid))
        notice("warn", f"<strong>Citation check:</strong> the answer cites {nums}, which doesn't match any "
                       "retrieved paper. Treat those claims with caution.")
    elif not valid:
        notice("warn", "<strong>Citation check:</strong> no numbered citations were found, so these claims "
                       "can't be traced to specific papers.")
    else:
        notice("ok", f"<strong>Citation check:</strong> every citation points to one of the {len(sources)} "
                     "retrieved papers below. Open a paper to confirm a specific claim.")

    st.markdown('<div class="sources-title">References</div>', unsafe_allow_html=True)
    for i, meta in enumerate(sources, 1):
        cite = html.escape(f"{first_author(meta)} et al. ({meta.get('year')})")
        title = html.escape(meta.get("title") or "Untitled")
        venue_line = html.escape(f"{meta.get('venue')} · {meta.get('quartile')}")
        if meta.get("openAccessPdf"):
            title = f'<a href="{html.escape(meta["openAccessPdf"])}" target="_blank">{title}</a>'
        tags = []
        if meta.get("relevance") is not None:
            tags.append(f"relevance {meta['relevance']:.2f}")
        if i not in valid:
            tags.append("retrieved, not cited")
        tag_html = f'<span class="source-tag">· {" · ".join(tags)}</span>' if tags else ""
        css_class = "source-item" if i in valid else "source-item uncited"
        st.markdown(
            f'<div class="{css_class}">[{i}] <span class="source-cite">{cite}</span> {title}. '
            f'<em>{venue_line}</em>{tag_html}</div>',
            unsafe_allow_html=True,
        )


def render_result(result):
    if result["kind"] == "answer":
        render_answer(result)
    elif result["kind"] == "offtopic":
        notice("info", OFF_TOPIC_MSG)
    else:
        st.error(result["message"])


# ---------- page setup ----------

st.set_page_config(page_title="Explainable AI in Clinical Diagnosis", page_icon="🩺", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --bg: #EEF0EC;
    --paper: #FBFBF9;
    --ink: #16232B;
    --ink-soft: #4C5A60;
    --line: #D8DAD3;
    --current: #2F7A6E;
    --limits: #9C3B2E;
    --future: #2C4A7C;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: var(--bg); }
.stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp div { color: var(--ink); }
header[data-testid="stHeader"] { background: var(--bg); }

.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 860px; margin: 0 auto; padding-top: 2.5rem;
}

h1, h2, h3, .journal-title { font-family: 'Source Serif 4', serif; color: var(--ink); }

.journal-header { border-bottom: 1px solid var(--line); padding-bottom: 1.2rem; margin-bottom: 1.4rem; }
.journal-title { font-size: 2.1rem; font-weight: 600; margin-bottom: 0.3rem; line-height: 1.15; }
.journal-sub { color: var(--ink-soft); font-size: 0.95rem; }

[data-testid="stSidebar"] { background-color: var(--paper); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] h3 { font-family: 'Source Serif 4', serif; font-size: 1.1rem; }
[data-testid="stSidebar"] .stButton>button {
    background: transparent; border: none; border-left: 2px solid var(--line);
    border-radius: 0; text-align: left; font-size: 0.88rem;
    padding: 0.5rem 0.8rem; width: 100%; transition: border-color 0.15s;
}
[data-testid="stSidebar"] .stButton>button p { color: var(--ink-soft); }
[data-testid="stSidebar"] .stButton>button:hover { border-left: 2px solid var(--current); background: transparent; }
[data-testid="stSidebar"] .stButton>button:hover p { color: var(--ink); }

.stTextInput input {
    font-family: 'Source Serif 4', serif; font-size: 1.05rem;
    border: 1px solid var(--line); border-radius: 2px; padding: 0.7rem 0.9rem;
    background: var(--paper); color: var(--ink) !important;
}
.stTextInput input::placeholder { color: var(--ink-soft) !important; opacity: 1; }
.stTextInput input:focus { border-color: var(--current); box-shadow: none; }

div.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button {
    background: var(--ink); border-radius: 2px; border: none;
    padding: 0.5rem 1.4rem; font-weight: 500;
}
div.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button:hover { background: var(--current); }
div.stButton > button[kind="primary"], div.stButton > button[kind="primary"] *,
[data-testid="stFormSubmitButton"] button, [data-testid="stFormSubmitButton"] button * { color: #FBFBF9 !important; }

.section-block { border-top: 1px solid var(--line); padding-top: 1.2rem; margin-top: 1.4rem; }
.section-head { display: flex; align-items: baseline; gap: 0.6rem; }
.section-mark { font-family: 'Source Serif 4', serif; font-size: 0.85rem; color: var(--ink-soft); }
.section-title { font-family: 'Source Serif 4', serif; font-size: 1.3rem; font-weight: 600; }
.sec-current .section-title { color: var(--current) !important; }
.sec-limits .section-title { color: var(--limits) !important; }
.sec-future .section-title { color: var(--future) !important; }

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { font-size: 0.98rem; line-height: 1.65; }

.sources-title {
    font-family: 'Source Serif 4', serif; font-size: 1.1rem; margin-top: 2rem;
    margin-bottom: 0.4rem; border-top: 1px solid var(--line); padding-top: 1.2rem;
}
.source-item { font-size: 0.88rem; color: var(--ink-soft); padding: 0.4rem 0; border-bottom: 1px solid var(--line); }
.source-cite { color: var(--ink); font-weight: 600; }
.source-item a { color: var(--ink); text-decoration: none; border-bottom: 1px solid var(--current); }
.source-cite { color: var(--ink); font-weight: 600; }
.source-item.uncited { opacity: 0.55; }
.source-tag { font-size: 0.78rem; color: var(--ink-soft); margin-left: 0.3rem; }
.future-note { font-size: 0.85rem; color: var(--ink-soft); font-style: italic; margin: 0.2rem 0 0.4rem 0; }
.notice { border-left: 3px solid var(--line); background: var(--paper); padding: 0.8rem 1rem; margin-top: 1.4rem; font-size: 0.95rem; }
.notice.ok { border-left-color: var(--current); }
.notice.warn { border-left-color: var(--limits); }
.notice.info { border-left-color: var(--future); }
.disclaimer { font-size: 0.8rem; color: var(--ink-soft); margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line); }
</style>
""", unsafe_allow_html=True)

collection = load_collection()
ready_answers, ready_status = load_precomputed(corpus_fingerprint())
api_key = get_api_key()  # only needed for questions outside the ready-made set

st.markdown(f"""
<div class="journal-header">
    <div class="journal-title">Explainable AI in Clinical Diagnosis</div>
    <div class="journal-sub">A research-gap assistant over {paper_count(collection)} papers from Q1–Q4 ranked journals — current approaches, stated limitations, and where the field could go next.</div>
</div>
""", unsafe_allow_html=True)


def pick_starting_point(q):
    st.session_state["question"] = q
    st.session_state["run"] = True


with st.sidebar:
    st.markdown("### Starting points")
    if ready_status == "stale":
        st.caption("⚠️ Saved answers are out of date — re-run precompute_answers.py.")
    for q in STARTING_POINTS:
        st.button(q, use_container_width=True, on_click=pick_starting_point, args=(q,))

# A form, so pressing Enter in the box searches just like clicking the button.
with st.form("ask", border=False):
    col1, col2 = st.columns([5, 1])
    with col1:
        st.text_input(
            "question", key="question",
            placeholder="Ask about current approaches, limitations, or future directions...",
            label_visibility="collapsed",
        )
    with col2:
        if st.form_submit_button("Search", type="primary", use_container_width=True):
            st.session_state["run"] = True

HISTORY_SIZE = 10


def remember(result):
    """Add an answer to this visitor's Recent searches (newest first, no duplicates)."""
    history = [r for r in st.session_state.get("history", []) if r["asked"] != result["asked"]]
    st.session_state["history"] = [result] + history[:HISTORY_SIZE - 1]


def reopen(i):
    result = st.session_state["history"][i]
    st.session_state["result"] = result
    st.session_state["question"] = result["asked"]


if st.session_state.pop("run", False):
    question = st.session_state.get("question", "").strip()
    if question:
        result = answer_question(question, ready_answers, api_key)
        st.session_state["result"] = result
        if result["kind"] == "answer":
            remember(result)

# Added after the question is handled, so the newest search shows up straight away.
with st.sidebar:
    if st.session_state.get("history"):
        st.markdown("### Recent searches")
        st.caption("Click to reopen. Cleared when you close or refresh the page.")
        for i, r in enumerate(st.session_state["history"]):
            label = r["asked"] if len(r["asked"]) <= 70 else r["asked"][:67] + "..."
            st.button(label, key=f"history_{i}", use_container_width=True, on_click=reopen, args=(i,))

if st.session_state.get("result"):
    render_result(st.session_state["result"])

st.markdown(
    '<div class="disclaimer">Research aid only — not medical advice. Answers are generated by an AI '
    'model from paper abstracts and can contain mistakes; always check the cited papers before relying '
    'on a claim.</div>',
    unsafe_allow_html=True,
)
