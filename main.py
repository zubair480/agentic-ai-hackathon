from pathlib import Path

from agents.local_pipeline import run_pipeline


def main():
    result = run_pipeline(
        Path("model_input/Transfer-Model_original"),
        Path("reports/model_input_vitis_ai_compatibility.md"),
    )
    summary = result["compatibility"]["summary"]
    print(f"Report written to {result['report_path']}")
    print(
        "Operators: "
        f"{summary['total_operator_uses']} uses, "
        f"{summary['unique_operator_count']} unique, "
        f"{summary['conditional_unique_count']} conditional, "
        f"{summary['unsupported_unique_count']} unsupported"
    )


if __name__ == "__main__":
    main()
