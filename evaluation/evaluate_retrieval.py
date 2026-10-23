
import json
import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval import retrieve_documents


CASES_FILE = Path(__file__).with_name("rag_cases.json")
TOP_K = 3


def evaluate_retrieval():
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))

    if not cases:
        raise SystemExit("Evaluation dataset is empty.")

    passed = 0
    failed = 0
    recall_scores = []

    for case in cases:
        question = case["question"]
        expected_sources = set(case["expected_sources"])

        if not expected_sources:
            raise SystemExit(
                f"Evaluation case has no expected sources: {question}"
            )

        try:
            results = retrieve_documents(question, k=TOP_K)

            retrieved_sources = {
                document.metadata.get("source")
                for document, score in results
                if document.metadata.get("source")
            }

            matched_sources = expected_sources & retrieved_sources

            recall_at_k = len(matched_sources) / len(expected_sources)
            recall_scores.append(recall_at_k)

            success = recall_at_k == 1.0

            if success:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            print(f"\n[{status}] {question}")
            print(f"Expected sources: {sorted(expected_sources)}")
            print(f"Retrieved sources: {sorted(retrieved_sources)}")
            print(f"Matched sources: {sorted(matched_sources)}")
            print(f"Recall@{TOP_K}: {recall_at_k:.2%}")

        except Exception as exc:
            failed += 1
            print(f"\n[ERROR] {question}: {exc}")

    total = len(cases)
    average_recall = (
        sum(recall_scores) / len(recall_scores)
        if recall_scores
        else 0.0
    )

    print("\n" + "=" * 40)
    print(f"Cases passing all expected sources: {passed}/{total}")
    print(f"Cases failed or errored: {failed}/{total}")
    print(f"Average Recall@{TOP_K}: {average_recall:.2%}")
    print("=" * 40)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    evaluate_retrieval()
