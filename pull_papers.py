"""
Pull papers on "Explainable AI in Clinical Diagnosis" from Semantic Scholar,
keeping only papers published in SJR-ranked journals (Q1-Q4).

WHY THIS FILTERS THE WAY IT DOES
--------------------------------
"Q1 to Q4" means ANY ranked journal (Q1 = top 25% of journals in its field,
Q4 = bottom 25%). This excludes preprints (arXiv), unranked conferences,
and predatory/unindexed journals -- but it does NOT mean "top quality only".
If you want only strong journals, change MIN_QUARTILE below to "Q2".

SETUP
-----
1. pip install requests pandas
2. Get the Scimago Journal Rank list (free, no signup):
   https://www.scimagojr.com/journalrank.php -> "Download data" button
   Save the CSV as scimago_journals.csv in this same folder.
3. (Optional but recommended) Get a free Semantic Scholar API key for
   higher rate limits: https://www.semanticscholar.org/product/api
   Paste it into API_KEY below.

WHAT THIS SCRIPT DOES NOT DO YET
---------------------------------
- Full paper text: Semantic Scholar mostly gives abstracts. This script
  saves the `openAccessPdf` link when available. Extracting full text
  from those PDFs is a separate next step (e.g. with the `pdfplumber`
  or `PyMuPDF` library) -- only works for open-access papers.
- Fuzzy journal-name matching: venue names from Semantic Scholar don't
  always exactly match Scimago's journal titles (e.g. "IEEE Trans. on
  Medical Imaging" vs "IEEE Transactions on Medical Imaging"). This
  script does basic normalization; some real matches may still be missed.
  If your paper count looks low, that's the most likely reason -- worth
  spot-checking a few "dropped" venues manually.
"""

import requests
import time
import json
import re
import os
import pandas as pd

# Set your key as an environment variable instead of hardcoding it here,
# so it's never accidentally committed to GitHub. In VS Code's terminal:
#   Windows (PowerShell): $env:S2_API_KEY = "your-key-here"
#   Then run: python pull_papers.py
# (You'll need to set this each new terminal session, or add it to your
# system's environment variables permanently via Windows settings.)
API_KEY = os.environ.get("S2_API_KEY")
MIN_QUARTILE = "Q4"  # keep this quartile or better; use "Q2" for stricter filtering
QUARTILE_RANK = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,year,venue,authors,citationCount,publicationTypes,openAccessPdf,externalIds"

QUERIES = [
    "explainable AI clinical diagnosis",
    "SHAP medical diagnosis",
    "Grad-CAM healthcare",
    "interpretable deep learning disease prediction",
]

# Known strong Q1/Q2 journals in this field. Used for a second, targeted pass
# that directly restricts Semantic Scholar's search to these venues -- this
# catches good papers that the keyword search + Scimago name-matching might
# otherwise miss (e.g. because Semantic Scholar returns an abbreviated venue
# name like "Artif. Intell. Medicine" that doesn't match Scimago's full title).
# Quartiles below are manually verified as of Sept 2026 and used as a fallback
# ONLY when the Scimago name-match fails for one of these specific journals --
# if Scimago matches normally, that value is used instead (it's kept current).
TARGET_JOURNALS = [
    "Artificial Intelligence in Medicine",
    "Computerized Medical Imaging and Graphics",
    "IEEE Journal of Biomedical and Health Informatics",
    "Medical Image Analysis",
    "Computers in Biology and Medicine",
    "IEEE Transactions on Medical Imaging",
    "Journal of Biomedical Informatics",
    "Diagnostics",
    "Healthcare Analytics",
]
TARGET_JOURNAL_FALLBACK_QUARTILE = {
    "artificial intelligence in medicine": "Q1",
    "computerized medical imaging and graphics": "Q1",
    "ieee journal of biomedical and health informatics": "Q1",
    "medical image analysis": "Q1",
    "computers in biology and medicine": "Q1",
    "ieee transactions on medical imaging": "Q1",
    "journal of biomedical informatics": "Q2",
    "diagnostics": "Q2",
    "healthcare analytics": "Q2",
}


def search_papers(query, limit=100, max_retries=6, venue=None):
    headers = {"x-api-key": API_KEY} if API_KEY else {}
    params = {"query": query, "fields": FIELDS, "limit": limit}
    if venue:
        params["venue"] = venue  # comma-separated list of venue names

    wait = 5  # seconds; doubles after each failed attempt
    for attempt in range(1, max_retries + 1):
        resp = requests.get(S2_API, params=params, headers=headers)
        if resp.status_code == 429:
            print(f"  Rate limited (attempt {attempt}/{max_retries}) -- waiting {wait}s...")
            time.sleep(wait)
            wait = min(wait * 2, 60)
            continue
        resp.raise_for_status()
        return resp.json().get("data", [])

    print(f"  Gave up on '{query}' after {max_retries} rate-limited attempts.")
    return []


def load_scimago(path="scimago_journals.csv"):
    """Returns a dict: normalized journal name -> quartile (e.g. 'Q1')."""
    df = pd.read_csv(path, sep=";")
    df["Title_clean"] = df["Title"].apply(normalize_venue)
    # A journal can appear multiple times (once per subject category);
    # keep its best (lowest-numbered) quartile.
    best = {}
    for _, row in df.iterrows():
        q = row.get("SJR Best Quartile")
        if q not in QUARTILE_RANK:
            continue
        name = row["Title_clean"]
        if name not in best or QUARTILE_RANK[q] < QUARTILE_RANK[best[name]]:
            best[name] = q
    return best


def normalize_venue(venue):
    if not venue:
        return ""
    return re.sub(r"[^a-z0-9 ]", "", venue.lower()).strip()


def main():
    try:
        scimago = load_scimago()
        print(f"Loaded {len(scimago)} journals from Scimago data")
    except FileNotFoundError:
        print("scimago_journals.csv not found -- see SETUP step 2 in this file's docstring.")
        return

    min_rank = QUARTILE_RANK[MIN_QUARTILE]
    all_papers = {}
    dropped_venues = set()

    for q in QUERIES:
        print(f"Searching: {q}")
        papers = search_papers(q, limit=100)
        for p in papers:
            venue = p.get("venue") or ""
            pub_types = p.get("publicationTypes") or []

            if "JournalArticle" not in pub_types:
                continue  # drops arXiv preprints, conference-only papers

            quartile = scimago.get(normalize_venue(venue))
            if quartile is None or QUARTILE_RANK[quartile] > min_rank:
                if venue:
                    dropped_venues.add(venue)
                continue

            paper_id = p.get("paperId")
            all_papers[paper_id] = {
                "title": p.get("title"),
                "abstract": p.get("abstract"),
                "year": p.get("year"),
                "venue": venue,
                "quartile": quartile,
                "authors": [a.get("name") for a in p.get("authors", [])],
                "citationCount": p.get("citationCount"),
                "openAccessPdf": (p.get("openAccessPdf") or {}).get("url"),
            }
        time.sleep(5)  # be polite to the free API tier -- raise this if you keep getting rate limited

    # --- Second pass: targeted search restricted to known strong journals ---
    # This catches papers Semantic Scholar's venue field labels with an
    # abbreviated name that didn't match Scimago's title in the pass above.
    venue_filter = ",".join(TARGET_JOURNALS)
    for q in QUERIES:
        print(f"Searching (targeted journals): {q}")
        papers = search_papers(q, limit=100, venue=venue_filter)
        for p in papers:
            venue = p.get("venue") or ""
            pub_types = p.get("publicationTypes") or []
            if "JournalArticle" not in pub_types:
                continue

            quartile = scimago.get(normalize_venue(venue))
            if quartile is None:
                # fall back to our manually verified quartile for this known journal
                quartile = TARGET_JOURNAL_FALLBACK_QUARTILE.get(normalize_venue(venue))
            if quartile is None or QUARTILE_RANK[quartile] > min_rank:
                continue

            paper_id = p.get("paperId")
            if paper_id in all_papers:
                continue  # already captured in the first pass
            all_papers[paper_id] = {
                "title": p.get("title"),
                "abstract": p.get("abstract"),
                "year": p.get("year"),
                "venue": venue,
                "quartile": quartile,
                "authors": [a.get("name") for a in p.get("authors", [])],
                "citationCount": p.get("citationCount"),
                "openAccessPdf": (p.get("openAccessPdf") or {}).get("url"),
            }
        time.sleep(5)

    print(f"\nCollected {len(all_papers)} papers from {MIN_QUARTILE}-or-better ranked journals")
    print(f"(includes both the keyword search and the targeted pass over {len(TARGET_JOURNALS)} known strong journals)")
    with open("corpus_raw.json", "w", encoding="utf-8") as f:
        json.dump(list(all_papers.values()), f, indent=2)

    if dropped_venues:
        print(f"\n{len(dropped_venues)} distinct venues were dropped (unranked, below cutoff, or name mismatch).")
        print("Sample of dropped venues (worth spot-checking for name-matching misses):")
        for v in list(dropped_venues)[:15]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
