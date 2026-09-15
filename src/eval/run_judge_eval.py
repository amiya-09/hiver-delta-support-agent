"""Run the LLM judge over Phase 4's generated replies.

Small sample by construction (currently 10 examples, from Phase 4's
--sample-size default) -- expanding to more examples later is a separate,
quota-dependent step, not part of this script.
"""
import argparse
import json
from pathlib import Path

from src.eval.judge import judge_reply


def load_generation_results(path):
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def run(input_path, output_path):
    generation_results = load_generation_results(input_path)

    judged = []
    total = len(generation_results)
    for i, result in enumerate(generation_results, start=1):
        example_id = result.get("example_id")
        scores = judge_reply(
            result.get("customer_text_clean", ""),
            result.get("reply", ""),
            result.get("retrieved_examples", []),
        )

        judged.append(
            {
                "example_id": example_id,
                "customer_text_clean": result.get("customer_text_clean"),
                "reply": result.get("reply"),
                "grounding_source_indices": result.get("grounding_source_indices"),
                "scores": scores,
            }
        )

        print(f"{i}/{total}: {example_id} -- {scores}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({"n_judged": len(judged), "results": judged}, f, indent=2)

    print(f"Judged {len(judged)} replies. Full results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the LLM judge over Phase 4's generated replies.")
    parser.add_argument("--input", default="data/eval/phase4_generation_results.json")
    parser.add_argument("--output", default="data/eval/phase6_judge_results.json")
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
