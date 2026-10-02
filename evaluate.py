"""
Evaluate the research assistant and save the scores.

  python evaluate.py --label baseline            retrieval tests only (no API key needed)
  python evaluate.py --label baseline --judge    also checks the saved answers' claims with Gemini

WHAT IT MEASURES (questions are in eval/eval_set.json)
-----------------------------------------------------
Retrieval -- does search find the right papers?
  - Known-item questions: each was written about ONE specific paper.
      Hit@1 / Hit@3 / Hit@10 = how often that paper is ranked 1st / in the top 3 / in the top 10.
      MRR = mean reciprocal rank (1.0 = always first, 0.5 = typically second, ...).
  - Topic questions: Precision@10 = share of the top 10 papers that are actually on the topic
      (judged by keyword patterns, so treat it as an approximate, "silver-standard" score).
  - Off-topic handling: how many off-topic questions are correctly refused, and how many
      genuine questions are wrongly refused by the MAX_DISTANCE cut-off.

Answer faithfulness (--judge, uses your Gemini key):
  Every cited sentence in the saved Starting-point answers is checked against the abstract
  it cites: supported / partly supported / not supported. This measures red flags 1 and 2.

Each run is saved to eval/results/<label>.json, and eval/RESULTS.md is rebuilt as a
side-by-side table of all runs -- so you can show "before vs after" for every improvement.
"""

import os
import re
import sys
import json
import time
import argparse
from datetime import datetime

from config import (
    load_collection, _ranked_papers, MAX_DISTANCE, TOP_K, MODEL, PRECOMPUTED_FILE, CORPUS_FILE,
    get_api_key, make_client, is_rate_limit, extract_citations,
)

EVAL_FILE = os.path.join("eval", "eval_set.json")
RESULTS_DIR = os.path.join("eval", "results")
RESULTS_MD = os.path.join("eval", "RESULTS.md")
JUDGE_PAUSE = 10  # seconds between Gemini calls in --judge mode


# ---------- retrieval ----------

def evaluate_retrieval(collection, data):
    # Known-item: where does the target paper rank?
    ranks, misses = [], []
    for item in data["known_item"]:
        ranked = [m["paper_id"] for m, _ in _ranked_papers(collection, item["question"])][:TOP_K]
        rank = ranked.index(item["paper_id"]) + 1 if item["paper_id"] in ranked else None
        ranks.append(rank)
        if rank is None or rank > 3:
            misses.append({"question": item["question"], "expected": item["title"], "rank": rank})
    n = len(ranks)
    known = {
        "questions": n,
        "hit@1": sum(1 for r in ranks if r == 1) / n,
        "hit@3": sum(1 for r in ranks if r and r <= 3) / n,
        "hit@10": sum(1 for r in ranks if r) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }

    # Topic precision
    precisions, weak = [], []
    for item in data["topics"]:
        top = _ranked_papers(collection, item["question"])[:TOP_K]
        on = sum(1 for m, _ in top if re.search(item["pattern"], m["full_text"].lower()))
        p = on / len(top) if top else 0
        precisions.append(p)
        if p < 0.7:
            weak.append({"question": item["question"], "precision@10": p})
    topic = {"questions": len(precisions), "precision@10": sum(precisions) / len(precisions)}

    # Off-topic cut-off
    def closest(q):
        r = _ranked_papers(collection, q)
        return r[0][1] if r else 9.9
    off_refused = sum(1 for q in data["off_topic"] if closest(q) > MAX_DISTANCE)
    genuine = [i["question"] for i in data["known_item"]] + [i["question"] for i in data["topics"]]
    wrongly_refused = [q for q in genuine if closest(q) > MAX_DISTANCE]
    cutoff = {
        "max_distance": MAX_DISTANCE,
        "off_topic_refused": off_refused / len(data["off_topic"]),
        "genuine_wrongly_refused": len(wrongly_refused) / len(genuine),
    }
    return {"known_item": known, "topics": topic, "cutoff": cutoff,
            "details": {"known_item_misses": misses, "weak_topics": weak, "wrongly_refused": wrongly_refused}}


# ---------- faithfulness (LLM judge) ----------

JUDGE_PROMPT = """You are checking a research summary for accuracy. Below are numbered paper
abstracts, followed by numbered claims taken from the summary. Each claim lists the
abstract numbers it cites.

For each claim, decide whether the CITED abstracts support it:
- "supported": the cited abstracts clearly state or directly show this.
- "partial": partly supported, or stated more strongly / more generally than the abstracts.
- "unsupported": the cited abstracts do not say this.

Claims marked "(inferred)" are the summary's own reasoning: judge whether that inference is
reasonable given the abstracts ("supported" if reasonable, "unsupported" if not).

Respond with ONLY a JSON list, one object per claim, e.g.
[{"claim": 1, "verdict": "supported"}, {"claim": 2, "verdict": "partial"}]
"""


def split_claims(answer):
    """Sentences (or bullet points) that carry at least one [n] citation."""
    claims = []
    for line in answer.splitlines():
        line = re.sub(r"^\s*([-*+]|\d+\.)\s+", "", line).strip()
        if not line or line.startswith("#"):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line):
            cites = extract_citations(sentence)
            if cites:
                claims.append((sentence.strip(), sorted(cites)))
    return claims


def judge_answer(client, answer, sources, abstracts):
    claims = split_claims(answer)
    claims = [(c, [n for n in cites if 1 <= n <= len(sources)]) for c, cites in claims]
    claims = [(c, cites) for c, cites in claims if cites]
    if not claims:
        return []
    used = sorted({n for _, cites in claims for n in cites})
    papers = "\n\n".join(
        f"[{n}] {sources[n - 1].get('title')}\n{abstracts.get(sources[n - 1].get('title'), '(abstract unavailable)')}"
        for n in used
    )
    listed = "\n".join(f"Claim {i}: {c}   (cites {', '.join(map(str, cites))})" for i, (c, cites) in enumerate(claims, 1))
    from google.genai import types
    prompt = f"ABSTRACTS\n\n{papers}\n\nCLAIMS\n\n{listed}"
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=JUDGE_PROMPT, temperature=0,
                    response_mime_type="application/json",
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            text = re.sub(r"```(json)?", "", resp.text or "").strip()
            verdicts = {int(v["claim"]): v["verdict"].lower() for v in json.loads(text)}
            return [{"claim": c, "cites": cites, "verdict": verdicts.get(i, "unsupported")}
                    for i, (c, cites) in enumerate(claims, 1)]
        except Exception as e:
            if is_rate_limit(e) and attempt < 3:
                print("  rate limited -- waiting 40s...")
                time.sleep(40)
                continue
            raise


def evaluate_faithfulness(api_key):
    if not os.path.exists(PRECOMPUTED_FILE):
        print("No precomputed_answers.json -- run precompute_answers.py first.")
        return None
    answers = json.load(open(PRECOMPUTED_FILE, encoding="utf-8")).get("answers", {})
    corpus = json.load(open(CORPUS_FILE, encoding="utf-8"))
    abstracts = {p.get("title"): p.get("abstract") for p in corpus if p.get("abstract")}
    client = make_client(api_key)

    all_verdicts, per_answer = [], []
    for n, entry in enumerate(answers.values(), 1):
        print(f"  judging answer {n}/{len(answers)}: {entry['question'][:70]}")
        verdicts = judge_answer(client, entry["answer"], entry["sources"], abstracts)
        all_verdicts += verdicts
        per_answer.append({"question": entry["question"], "verdicts": verdicts})
        if n < len(answers):
            time.sleep(JUDGE_PAUSE)

    total = len(all_verdicts) or 1
    count = lambda v: sum(1 for x in all_verdicts if x["verdict"] == v)
    inferred = sum(1 for x in all_verdicts if "(inferred)" in x["claim"].lower())
    return {
        "claims_checked": len(all_verdicts),
        "supported": count("supported") / total,
        "partial": count("partial") / total,
        "unsupported": count("unsupported") / total,
        "claims_marked_inferred": inferred,
        "details": {"unsupported_claims": [
            {"question": a["question"], "claim": v["claim"], "cites": v["cites"]}
            for a in per_answer for v in a["verdicts"] if v["verdict"] == "unsupported"
        ]},
    }


# ---------- reporting ----------

def pct(x):
    return f"{x * 100:.0f}%"


def print_report(res):
    k, t, c = res["retrieval"]["known_item"], res["retrieval"]["topics"], res["retrieval"]["cutoff"]
    print(f"\nRETRIEVAL ({k['questions']} known-item, {t['questions']} topic questions)")
    print(f"  Hit@1  {pct(k['hit@1'])}    Hit@3  {pct(k['hit@3'])}    Hit@10  {pct(k['hit@10'])}    MRR  {k['mrr']:.3f}")
    print(f"  Topic precision@10  {pct(t['precision@10'])}")
    print(f"\nOFF-TOPIC CUT-OFF (MAX_DISTANCE = {c['max_distance']})")
    print(f"  Off-topic questions refused     {pct(c['off_topic_refused'])}   (higher is better)")
    print(f"  Genuine questions wrongly refused  {pct(c['genuine_wrongly_refused'])}   (lower is better)")
    d = res["retrieval"]["details"]
    if d["known_item_misses"]:
        print("\n  Known-item questions where the right paper wasn't in the top 3:")
        for m in d["known_item_misses"]:
            print(f"    rank {m['rank'] or '>10':>4}  {m['question'][:80]}")
    if d["weak_topics"]:
        print("\n  Topics with precision below 70%:")
        for w in d["weak_topics"]:
            print(f"    {pct(w['precision@10']):>4}  {w['question']}")
    f = res.get("faithfulness")
    if f:
        print(f"\nANSWER FAITHFULNESS ({f['claims_checked']} cited claims in the saved answers)")
        print(f"  Supported {pct(f['supported'])}   Partly {pct(f['partial'])}   Not supported {pct(f['unsupported'])}")
        print(f"  Claims marked (inferred): {f['claims_marked_inferred']}")


def write_results_md():
    runs = []
    for name in sorted(os.listdir(RESULTS_DIR)):
        if name.endswith(".json"):
            runs.append(json.load(open(os.path.join(RESULTS_DIR, name), encoding="utf-8")))
    runs.sort(key=lambda r: r["timestamp"])
    rows = [
        ("Known-item Hit@1", lambda r: pct(r["retrieval"]["known_item"]["hit@1"])),
        ("Known-item Hit@3", lambda r: pct(r["retrieval"]["known_item"]["hit@3"])),
        ("Known-item Hit@10", lambda r: pct(r["retrieval"]["known_item"]["hit@10"])),
        ("Known-item MRR", lambda r: f"{r['retrieval']['known_item']['mrr']:.3f}"),
        ("Topic precision@10", lambda r: pct(r["retrieval"]["topics"]["precision@10"])),
        ("Off-topic refused", lambda r: pct(r["retrieval"]["cutoff"]["off_topic_refused"])),
        ("Genuine wrongly refused", lambda r: pct(r["retrieval"]["cutoff"]["genuine_wrongly_refused"])),
        ("Claims supported", lambda r: pct(r["faithfulness"]["supported"]) if r.get("faithfulness") else "–"),
        ("Claims not supported", lambda r: pct(r["faithfulness"]["unsupported"]) if r.get("faithfulness") else "–"),
    ]
    lines = ["# Evaluation results", "",
             "Generated by `python evaluate.py`. Questions are in `eval/eval_set.json`.", "",
             "| Metric | " + " | ".join(r["label"] for r in runs) + " |",
             "|---|" + "---|" * len(runs)]
    for name, fn in rows:
        lines.append(f"| {name} | " + " | ".join(fn(r) for r in runs) + " |")
    lines += ["", "Hit@k: share of known-item questions whose target paper is in the top k. "
              "MRR: mean reciprocal rank. Topic precision uses keyword-based labels (approximate). "
              "Claims are judged by an LLM against the cited abstracts."]
    open(RESULTS_MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the research assistant.")
    parser.add_argument("--label", default="run", help="name for this run, e.g. baseline or hybrid-search")
    parser.add_argument("--judge", action="store_true", help="also judge saved answers with Gemini (uses API quota)")
    args = parser.parse_args()

    data = json.load(open(EVAL_FILE, encoding="utf-8"))
    print("Loading the paper database...")
    collection = load_collection(log=lambda m: print(" ", m))
    print("Running retrieval tests...")
    result = {"label": args.label, "timestamp": datetime.now().isoformat(timespec="seconds"),
              "retrieval": evaluate_retrieval(collection, data)}

    if args.judge:
        key = get_api_key()
        if not key:
            print("No Gemini key found (see .streamlit/secrets.toml) -- skipping --judge.")
        else:
            print("Judging answer faithfulness with Gemini (about 10 seconds per answer)...")
            result["faithfulness"] = evaluate_faithfulness(key)

    print_report(result)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", args.label)
    with open(os.path.join(RESULTS_DIR, f"{safe}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    write_results_md()
    print(f"\nSaved to {RESULTS_DIR}/{safe}.json and updated {RESULTS_MD}")


if __name__ == "__main__":
    sys.exit(main())
