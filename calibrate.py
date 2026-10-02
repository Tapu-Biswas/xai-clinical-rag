"""
Tune the off-topic cut-off (MAX_DISTANCE in config.py) for your data.

  python calibrate.py

For each test question it prints the distance to the closest paper
(lower = more relevant). On-topic questions should come out clearly lower
than off-topic ones; set MAX_DISTANCE between the two groups. Add your own
questions to the lists below to test more cases.
"""

from config import load_collection, closest_distance, MAX_DISTANCE

ON_TOPIC = [
    "How is SHAP applied in clinical diagnosis?",
    "Criticisms of Grad-CAM in medical imaging",
    "How is AI used to detect cancer?",
    "How are explanations evaluated with clinicians?",
]
OFF_TOPIC = [
    "What is the best pizza in London?",
    "How do rockets work?",
    "Who won the football World Cup?",
]
BORDERLINE = [
    "What is machine learning?",
]


def main():
    collection = load_collection()
    groups = {"ON-TOPIC": ON_TOPIC, "OFF-TOPIC": OFF_TOPIC, "BORDERLINE": BORDERLINE}
    results = {}
    for name, questions in groups.items():
        print(f"\n{name}")
        for q in questions:
            d = closest_distance(collection, q)
            results.setdefault(name, []).append(d)
            verdict = "answered" if d <= MAX_DISTANCE else "rejected as off-topic"
            print(f"  {d:.3f}  {verdict:22}  {q}")

    worst_on, best_off = max(results["ON-TOPIC"]), min(results["OFF-TOPIC"])
    print(f"\nCurrent MAX_DISTANCE = {MAX_DISTANCE}")
    if worst_on < best_off:
        print(f"Suggested MAX_DISTANCE = {(worst_on + best_off) / 2:.2f} "
              f"(halfway between {worst_on:.3f} and {best_off:.3f})")
    else:
        print("On-topic and off-topic questions overlap -- check the questions, or pick a value by hand.")


if __name__ == "__main__":
    main()
