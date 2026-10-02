"""
Shared settings and logic for the Explainable AI in Clinical Diagnosis assistant.

Both app.py and precompute_answers.py import from this file, so the prompt,
model, retrieval rules and Starting points only ever change HERE.
"""

import os
import re
import time
import json
import hashlib
import difflib
import threading
from collections import deque

import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types

# ---------- settings ----------

DB_PATH = "./chroma_db"
COLLECTION_NAME = "explainable_ai_clinical_diagnosis"
PRECOMPUTED_FILE = "precomputed_answers.json"
CORPUS_FILE = "corpus_raw.json"
SECRETS_FILE = os.path.join(".streamlit", "secrets.toml")

# Bump this whenever the way papers are stored changes; an old database
# with a different version is rebuilt automatically.
SCHEMA_VERSION = 2

# Chunking: the embedding model only reads ~256 tokens, so long abstracts are
# split into overlapping word windows (title repeated on each) so no part of an
# abstract is ignored during search.
CHUNK_WORDS = 150
CHUNK_OVERLAP = 30

TOP_K = 10            # max papers sent to Gemini
MAX_DISTANCE = 1.5    # papers further than this are treated as irrelevant.
                      # Distance = 2 - 2*similarity, so 1.5 ~ similarity 0.25.
                      # Run calibrate.py to tune it for your data.

MODEL = "gemini-3.5-flash-lite"
TEMPERATURE = 0.2     # low = more consistent answers between runs

MATCH_THRESHOLD = 0.85         # how close a typed question must be to reuse a saved answer
LIVE_LIMIT_PER_SESSION = 5     # live Gemini questions per visitor session
LIVE_LIMIT_PER_HOUR = 40       # live Gemini questions per hour, all visitors combined

SYSTEM_PROMPT = """You are a research assistant helping a researcher understand
the current state of a topic before they write a paper on it. You will be given a
question and numbered paper excerpts ([1], [2], ...), each with its title, first
author, year, venue and abstract.

Using ONLY the information in the excerpts (no outside knowledge), write a
structured answer with exactly these three sections, using these exact
markdown headings:

## Current Approaches
What methods and approaches the excerpts describe.

## Limitations
Limitations or weaknesses the excerpts actually state. If you point out a gap
you inferred yourself rather than one a paper states, mark it "(inferred)".

## Suggested Future Work
Concrete research directions that follow from the limitations above and do not
seem to be covered by the excerpts. Say which limitation each one addresses.

CITATIONS: cite every claim with the excerpt number in square brackets, e.g. [3]
or [2][5]. Only use the numbers of excerpts you were given. Never mention a paper
without its number.

If the excerpts don't contain enough information for a section, say so plainly
in that section rather than guessing.
"""

STARTING_POINTS = [
    "What are the current limitations of explainable AI methods in clinical diagnosis?",
    "What approaches are being used to explain deep learning predictions in medical imaging?",
    "How is SHAP being applied in clinical diagnosis research?",
    "What are the main criticisms of Grad-CAM in medical applications?",
    "What gaps exist between explainable AI research and real clinical adoption?",
    "What ethical concerns are raised about explainable AI in clinical decision-making?",
    "What future research directions are suggested for explainable AI in healthcare?",
    "How do researchers evaluate whether an explanation is trustworthy to clinicians?",
]


# ---------- building + loading the paper database ----------

def _embed_fn():
    # Chroma's built-in all-MiniLM-L6-v2 (ONNX). Same model as before, but it
    # doesn't need PyTorch, so installs are much smaller and faster.
    return embedding_functions.DefaultEmbeddingFunction()


def chunk_text(title, abstract):
    words = abstract.split()
    step = CHUNK_WORDS - CHUNK_OVERLAP
    chunks = []
    start = 0
    while True:
        chunks.append(f"{title}. {' '.join(words[start:start + CHUNK_WORDS])}")
        if start + CHUNK_WORDS >= len(words):
            return chunks
        start += step


def build_collection(client=None, embed_fn=None, log=print):
    """(Re)build the vector database from corpus_raw.json."""
    client = client or chromadb.PersistentClient(path=DB_PATH)
    embed_fn = embed_fn or _embed_fn()

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        papers = json.load(f)
    log(f"Loaded {len(papers)} papers from {CORPUS_FILE}")

    ids, documents, metadatas = [], [], []
    kept = 0
    for i, paper in enumerate(papers):
        abstract = paper.get("abstract")
        title = paper.get("title") or "Untitled"
        if not abstract:
            continue
        kept += 1
        base = {
            "paper_id": f"paper_{i}",
            "title": title,
            "venue": paper.get("venue") or "",
            "year": paper.get("year") or 0,
            "quartile": paper.get("quartile") or "",
            "authors": ", ".join(paper.get("authors") or []),
            "citationCount": paper.get("citationCount") or 0,
            "openAccessPdf": paper.get("openAccessPdf") or "",
            "full_text": f"{title}. {abstract}",
        }
        for j, chunk in enumerate(chunk_text(title, abstract)):
            ids.append(f"paper_{i}_c{j}")
            documents.append(chunk)
            metadatas.append(base)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # didn't exist yet
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"schema": SCHEMA_VERSION, "papers": kept},
    )

    log(f"Embedding {kept} papers as {len(documents)} chunks "
        f"({len(papers) - kept} skipped -- no abstract)...")
    batch_size = 100
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        collection.add(ids=ids[start:end], documents=documents[start:end], metadatas=metadatas[start:end])
        log(f"  Added {min(end, len(documents))}/{len(documents)}")
    return collection


def load_collection(embed_fn=None, log=print):
    """Open the database; build or rebuild it from corpus_raw.json if missing or outdated."""
    client = chromadb.PersistentClient(path=DB_PATH)
    embed_fn = embed_fn or _embed_fn()
    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
        if (collection.metadata or {}).get("schema") == SCHEMA_VERSION and collection.count() > 0:
            return collection
        log("Paper database is from an older version -- rebuilding it...")
    except Exception:
        log("No paper database found -- building it from corpus_raw.json (first run only)...")
    return build_collection(client, embed_fn, log)


def paper_count(collection):
    return (collection.metadata or {}).get("papers") or collection.count()


# ---------- retrieval ----------

DISPLAY_FIELDS = ("title", "venue", "year", "quartile", "authors", "citationCount", "openAccessPdf")


def _ranked_papers(collection, question):
    """Every matching paper once (its best chunk), closest first, as (meta, distance)."""
    n = min(TOP_K * 4, collection.count())
    res = collection.query(query_texts=[question], n_results=n, include=["metadatas", "distances"])
    best = {}
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        pid = meta["paper_id"]
        if pid not in best or dist < best[pid][1]:
            best[pid] = (meta, dist)
    return sorted(best.values(), key=lambda x: x[1])


def retrieve(collection, question, apply_cutoff=True):
    """Up to TOP_K relevant papers as (full_text, display_meta) pairs.

    Returns an empty list when nothing is close enough -- i.e. the question is
    off-topic -- so the app can say so instead of calling Gemini.
    apply_cutoff=False skips that check (used for the hand-picked Starting points).
    """
    out = []
    for meta, dist in _ranked_papers(collection, question):
        if len(out) >= TOP_K or (apply_cutoff and dist > MAX_DISTANCE):
            break
        display = {k: meta.get(k) for k in DISPLAY_FIELDS}
        display["relevance"] = round(max(0.0, 1 - dist / 2), 3)
        out.append((meta["full_text"], display))
    return out


def closest_distance(collection, question):
    ranked = _ranked_papers(collection, question)
    return ranked[0][1] if ranked else None


def first_author(meta):
    name = (meta.get("authors") or "").split(",")[0].strip()
    return name or "Unknown"


def format_context(retrieved):
    blocks = []
    for i, (doc, meta) in enumerate(retrieved, 1):
        blocks.append(
            f"Excerpt [{i}]\nTitle: {meta.get('title')}\nFirst author: {first_author(meta)} et al.\n"
            f"Year: {meta.get('year')}\nVenue: {meta.get('venue')} ({meta.get('quartile')})\n"
            f"Abstract: {doc}\n"
        )
    return "\n".join(blocks)


# ---------- matching typed questions to saved answers ----------

def normalize_question(q):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", q.lower()).split())


FILLER_WORDS = set(
    "a an the of in on for to and or is are be being been was were do does did how what which who "
    "whom why when where can could should would there their its it this that these those with by from "
    "as at about into any some currently current main".split()
)


def _content_words(q):
    return {w for w in normalize_question(q).split() if w not in FILLER_WORDS}


def find_ready(answers, question):
    """A saved answer for this question, allowing small wording differences.

    Safe matching: every meaningful word typed must appear in the saved
    question, so "LIME" never matches a saved "SHAP" answer.
    """
    q = normalize_question(question)
    typed_words = _content_words(question)
    best, best_score = None, 0.0
    for entry in answers.values():
        saved = entry["question"]
        if not typed_words or not typed_words <= set(normalize_question(saved).split()):
            continue
        score = difflib.SequenceMatcher(None, q, normalize_question(saved)).ratio()
        if score > best_score:
            best, best_score = entry, score
    return best if best_score >= MATCH_THRESHOLD else None


# ---------- reading the answer ----------

SECTION_KEYS = [
    ("Current Approaches", ("current", "approach")),
    ("Limitations", ("limitation",)),
    ("Suggested Future Work", ("future",)),
]


def _canonical_heading(line):
    s = line.strip()
    if not s or len(s) > 60:
        return None
    is_heading = s.startswith("#") or (s.startswith("**") and s.rstrip(":").endswith("**"))
    if not is_heading:
        return None
    core = re.sub(r"[#*_:.\d)\s]+", " ", s).strip().lower()
    for name, keys in SECTION_KEYS:
        if any(k in core for k in keys):
            return name
    return None


def parse_sections(answer_text):
    """Split the answer into (heading, body) pairs, tolerating heading variations
    like '## 1. Current Approaches', '### Limitations:' or '**Suggested Future Work**'."""
    sections, current, buf = [], None, []
    for line in answer_text.splitlines():
        heading = _canonical_heading(line)
        if heading:
            if current:
                sections.append((current, "\n".join(buf).strip()))
            current, buf = heading, []
        else:
            buf.append(line)
    if current:
        sections.append((current, "\n".join(buf).strip()))
    return sections


_CITE_RE = re.compile(r"\[(\d+(?:\s*(?:,|;|-|–|\]\s*\[)\s*\d+)*)\]")


def extract_citations(text):
    cited = set()
    for group in _CITE_RE.findall(text):
        group = group.replace("–", "-")
        for part in re.split(r"\s*(?:,|;|\]\s*\[)\s*", group):
            if "-" in part:
                a, b = (int(x) for x in part.split("-")[:2])
                if 0 < b - a < 20:
                    cited.update(range(a, b + 1))
                    continue
                cited.update((a, b))
            elif part.strip().isdigit():
                cited.add(int(part))
    return cited


def check_citations(answer, n_sources):
    """(valid, invalid) citation numbers. Invalid = points to no retrieved paper."""
    cited = extract_citations(answer)
    valid = {c for c in cited if 1 <= c <= n_sources}
    return valid, cited - valid


# ---------- staleness check ----------

def fingerprint(collection):
    """Short ID for 'this corpus + model + prompt + retrieval settings'.

    If any of them change, saved answers are treated as out of date.
    """
    data = collection.get(include=["documents"])
    docs = sorted(zip(data["ids"], data["documents"]))
    settings = [MODEL, TEMPERATURE, TOP_K, MAX_DISTANCE, CHUNK_WORDS, CHUNK_OVERLAP, SCHEMA_VERSION, SYSTEM_PROMPT]
    payload = json.dumps([docs, settings])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------- API key ----------

def get_api_key():
    """Find the Gemini key: first an environment variable, then .streamlit/secrets.toml.

    Paste your key into .streamlit/secrets.toml once, like this:
        GEMINI_API_KEY = "your-key-here"
    That file is in .gitignore, so it never gets uploaded to GitHub.
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    if os.path.exists(SECRETS_FILE):
        with open(SECRETS_FILE, "r", encoding="utf-8-sig") as f:
            for line in f:
                m = re.match(r"""\s*GEMINI_API_KEY\s*=\s*["']?([^"'#\s]*)""", line)
                if m and m.group(1):
                    return m.group(1)
    return None


# ---------- Gemini ----------

def make_client(api_key):
    return genai.Client(api_key=api_key)


def generate(client, question, retrieved):
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Question: {question}\n\nRetrieved excerpts:\n\n{format_context(retrieved)}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            # no tools are used, so switch off automatic function calling (also silences its warning)
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty answer (it may have been blocked by a safety filter).")
    # normalise "[Excerpt 3]" -> "[3]" so citations can be checked
    return re.sub(r"\[Excerpts?\s+(\d+)\]", r"[\1]", response.text)


def error_code(e):
    return getattr(e, "code", None)


def is_rate_limit(e):
    msg = str(e)
    return error_code(e) == 429 or "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()


def generate_with_retry(client, question, retrieved, max_retries=2, on_wait=None, max_wait=65):
    """Call Gemini; if the free-tier limit is hit, wait the suggested time and retry."""
    for attempt in range(max_retries + 1):
        try:
            return generate(client, question, retrieved)
        except Exception as e:
            if is_rate_limit(e) and attempt < max_retries:
                m = re.search(r"retry in ([0-9.]+)s", str(e))
                wait = min(float(m.group(1)) + 2, max_wait) if m else min(30, max_wait)
                if on_wait:
                    on_wait(attempt + 1, max_retries, wait)
                time.sleep(wait)
                continue
            raise


class LiveLimiter:
    """Caps live Gemini calls per hour across all visitors (protects your free quota)."""

    def __init__(self, per_hour):
        self.per_hour = per_hour
        self.calls = deque()
        self.lock = threading.Lock()

    def allow(self):
        now = time.time()
        with self.lock:
            while self.calls and now - self.calls[0] > 3600:
                self.calls.popleft()
            if len(self.calls) >= self.per_hour:
                return False
            self.calls.append(now)
            return True
