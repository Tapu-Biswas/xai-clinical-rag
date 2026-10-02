# Explainable AI in Clinical Diagnosis — Research-Gap Assistant

A RAG (retrieval-augmented generation) app over 488 papers on explainable AI in
clinical diagnosis, drawn from SJR-ranked (Q1–Q4) journals. Ask a question and get a
cited answer in three sections: Current Approaches, Limitations, and Suggested Future Work.

## How it works
1. `pull_papers.py` searches Semantic Scholar, keeps only journal articles in
   Scimago-ranked journals, and saves them to `corpus_raw.json` (510 papers, 488 with abstracts).
2. Each abstract is split into overlapping chunks (so no part is cut off by the embedding
   model's length limit), embedded with all-MiniLM-L6-v2, and stored in ChromaDB.
   This happens automatically on first run.
3. A question is embedded the same way; the closest papers are retrieved (one entry per paper).
   If no paper is close enough, the app says the question is off-topic instead of calling the LLM.
4. Gemini writes an answer using only those papers, citing each claim by number.
5. The app checks every citation points to a retrieved paper, marks which papers were
   actually cited, and labels Future Work as AI-generated.
6. The sidebar questions (and close rewordings) are answered ahead of time, so they load instantly.

## Safeguards
- Off-topic detection (no LLM call, no misleading references)
- Citation check on every answer
- Limitations the model inferred itself are marked "(inferred)"
- Low temperature for consistent answers
- Per-session and per-hour caps on live questions, to protect the free API quota
- Saved answers are ignored automatically if the corpus, model or prompt changes

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `config.py` | Shared settings, prompt, sidebar questions, database, retrieval, safeguards |
| `precompute_answers.py` | Generates the instant sidebar answers |
| `calibrate.py` | Tunes the off-topic cut-off for your data |
| `evaluate.py` | Scores retrieval and answer faithfulness; saves results to `eval/` |
| `eval/eval_set.json` | 58 test questions: 28 specific-paper, 20 topic, 10 off-topic |
| `embed_corpus.py` | Rebuilds the database by hand (only needed after pulling new papers) |
| `pull_papers.py` | Collects papers from Semantic Scholar |
| `corpus_raw.json` | The collected papers |
| `scimago_journals.csv` | Journal rankings used by `pull_papers.py` |
| `.streamlit/secrets.toml` | Your Gemini key (never uploaded to GitHub) |

## Run it
```
pip install -r requirements.txt
(open .streamlit/secrets.toml in Notepad, paste your Gemini key between the quotes, save)
python calibrate.py                     (first run also builds chroma_db/)
python precompute_answers.py
streamlit run app.py
```

## Evaluation
```
python evaluate.py --label baseline            (retrieval tests, no API key needed)
python evaluate.py --label baseline --judge    (also checks answer claims with Gemini)
```
- **Known-item retrieval:** 28 questions, each written about one specific paper. Hit@1/3/10 and MRR measure how highly that paper is ranked.
- **Topic precision@10:** 20 broad questions; share of the top 10 papers that are on-topic (keyword-based labels).
- **Off-topic handling:** share of off-topic questions refused, and of genuine questions wrongly refused.
- **Faithfulness:** every cited claim in the saved answers is checked by an LLM judge against the abstract it cites.

All runs are compared side by side in [`eval/RESULTS.md`](eval/RESULTS.md).

## Updating the papers
```
python pull_papers.py
python embed_corpus.py
python precompute_answers.py
```

## Known limitations
- Abstracts only — details in full texts (datasets, exact results) aren't searched.
- The corpus is a keyword-based sample of the field, not a systematic review.
