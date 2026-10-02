"""
Generate answers for the "Starting points" questions ONCE and save them to
precomputed_answers.json, so the app can show them instantly without calling
the Gemini API.

RUN
---
  1. Paste your Gemini key into .streamlit/secrets.toml (one time only)
  2. python precompute_answers.py

- Waits between questions to stay under the free-tier limit and retries if
  rate-limited. Saves after every question; if it stops halfway, run it again
  and it skips the ones already done.
- If the corpus, model or prompt has changed since the file was made, it
  throws the old answers away and regenerates them all.
- To add or change questions, edit STARTING_POINTS in config.py.
"""

import os
import json
import time

from config import (
    PRECOMPUTED_FILE, MODEL, STARTING_POINTS,
    load_collection, retrieve, fingerprint, make_client, generate_with_retry, get_api_key,
)

PAUSE_BETWEEN_QUESTIONS = 15  # seconds


def load_existing(current_fp):
    if not os.path.exists(PRECOMPUTED_FILE):
        return {}
    with open(PRECOMPUTED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("fingerprint") != current_fp:
        print("Saved answers are out of date (corpus, model or prompt changed) -- regenerating all.")
        return {}
    return data.get("answers", {})


def save(current_fp, corpus_size, answers):
    data = {"fingerprint": current_fp, "model": MODEL, "corpus_size": corpus_size, "answers": answers}
    with open(PRECOMPUTED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    api_key = get_api_key()
    if not api_key:
        print("No Gemini key found. Open .streamlit/secrets.toml in Notepad, paste your key")
        print('between the quotes:  GEMINI_API_KEY = "your-key-here"  and save.')
        return

    collection = load_collection()
    current_fp = fingerprint(collection)
    corpus_size = collection.count()
    client = make_client(api_key)
    answers = load_existing(current_fp)

    def on_wait(attempt, max_retries, wait):
        print(f"  Rate limited (retry {attempt}/{max_retries}) -- waiting {wait:.0f}s...")

    total = len(STARTING_POINTS)
    for n, question in enumerate(STARTING_POINTS, 1):
        key = question.strip().lower()
        if key in answers:
            print(f"[{n}/{total}] already done, skipping: {question}")
            continue

        print(f"[{n}/{total}] {question}")
        # Starting points are hand-picked on-topic questions, so the off-topic
        # cut-off isn't applied to them.
        retrieved = retrieve(collection, question, apply_cutoff=False)
        answer = generate_with_retry(client, question, retrieved, max_retries=5, on_wait=on_wait)
        answers[key] = {
            "question": question,
            "answer": answer,
            "sources": [meta for _, meta in retrieved],
        }
        save(current_fp, corpus_size, answers)
        print("  saved.")

        if n < total:
            time.sleep(PAUSE_BETWEEN_QUESTIONS)

    print(f"\nDone. {len(answers)} answers saved to {PRECOMPUTED_FILE}")


if __name__ == "__main__":
    main()
