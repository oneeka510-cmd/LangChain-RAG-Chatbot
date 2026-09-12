import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from rag.config import ROOT
from rag.service import AdvancedRAG, REFUSAL


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and optional answers.")
    parser.add_argument("--answers", action="store_true", help="Also call the configured LLM")
    args = parser.parse_args()
    load_dotenv()
    cases = json.loads((ROOT / "evaluation" / "questions.json").read_text(encoding="utf-8"))
    rag = AdvancedRAG()
    rows = []
    for case in cases:
        result = rag.ask(case["question"]) if args.answers else rag.inspect(case["question"])
        names = [source["source"] for source in result["sources"]]
        expected = case["expected_source"]
        hit = bool(expected in names) if expected else bool(
            result["confidence"] < rag.settings.confidence_threshold
        )
        row = {"question": case["question"], "retrieval_hit": hit, **result}
        if args.answers:
            answer_lower = result["answer"].lower()
            row["term_coverage"] = (
                sum(term.lower() in answer_lower for term in case["expected_terms"])
                / max(1, len(case["expected_terms"]))
            )
            row["correct_refusal"] = expected is not None or result["answer"] == REFUSAL
        rows.append(row)
    summary = {
        "cases": len(rows),
        "retrieval_hit_rate": sum(row["retrieval_hit"] for row in rows) / len(rows),
        "rows": rows,
    }
    output = ROOT / "evaluation" / "results.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    print(f"Detailed results: {output}")


if __name__ == "__main__":
    main()
