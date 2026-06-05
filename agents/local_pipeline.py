"""Local model-portability pipeline with no RocketRide runtime dependency.

This script mirrors the stages in the .pipe workflow:
1. inspect a model source folder
2. extract model classes, functions, and operators
3. classify operators against a Vitis AI compatibility rule set
4. generate conservative and aggressive migration proposals
5. write a Markdown report
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


SUPPORTED_OPS = {
    "Conv2d",
    "BatchNorm2d",
    "ReLU",
    "ReLU6",
    "LeakyReLU",
    "MaxPool2d",
    "AvgPool2d",
    "AdaptiveAvgPool2d",
    "Linear",
    "Flatten",
    "Dropout",
    "Sequential",
    "ModuleList",
    "Identity",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "cat",
    "add",
    "mul",
    "reshape",
    "view",
    "permute",
    "transpose",
    "interpolate",
}

CONDITIONAL_OPS = {
    "ConvTranspose2d": "Often requires rewrite to resize plus Conv2d for DPU-friendly deployment.",
    "Upsample": "Supported only for limited static-size modes; prefer explicit resize plus Conv2d.",
    "GroupNorm": "Use BatchNorm2d where possible for quantized DPU execution.",
    "LayerNorm": "Usually better replaced or folded before DPU deployment.",
    "InstanceNorm2d": "Prefer BatchNorm2d for Vitis AI quantization.",
    "mean": "Reductions can fall back to CPU depending on axes and shape constraints.",
    "sum": "Reductions can fall back to CPU depending on axes and shape constraints.",
    "pow": "Elementwise powers are risky for quantized DPU graphs.",
    "abs": "Elementwise abs may require CPU fallback or graph rewrite.",
    "sub": "Elementwise subtraction is shape-sensitive in quantized graphs.",
    "chunk": "Dynamic tensor splitting is risky; use static slicing when possible.",
    "split": "Dynamic tensor splitting is risky; use static slicing when possible.",
    "getitem": "Slicing must be static and compiler-friendly.",
}

REPLACEMENTS = {
    "ConvTranspose2d": "Upsample/interpolate to a static size followed by Conv2d + BatchNorm2d + ReLU.",
    "Upsample": "Use interpolate with fixed scale/size, then Conv2d to refine features.",
    "GroupNorm": "Replace with BatchNorm2d or fold normalization into adjacent layers during export.",
    "LayerNorm": "Replace with BatchNorm2d for feature maps or move normalization outside the DPU graph.",
    "InstanceNorm2d": "Replace with BatchNorm2d and recalibrate statistics.",
    "mean": "Replace dynamic reductions with fixed pooling layers such as AvgPool2d or AdaptiveAvgPool2d.",
    "sum": "Replace dynamic reductions with fixed pooling or static accumulation paths.",
    "pow": "Replace with multiplication for fixed exponents or remove from the DPU subgraph.",
    "abs": "Approximate with ReLU(x) + ReLU(-x) only if accuracy validates, otherwise isolate on CPU.",
    "sub": "Prefer add with pre-negated tensors or ensure static broadcast shapes.",
    "chunk": "Replace with explicit static slicing in known channel ranges.",
    "split": "Replace with explicit static slicing in known channel ranges.",
    "getitem": "Keep only static index/slice expressions with fixed dimensions.",
}


@dataclass
class DiscoveredClass:
    name: str
    file: str
    line: int
    bases: list[str]
    methods: list[str]


@dataclass
class DiscoveredFunction:
    name: str
    file: str
    line: int


@dataclass
class OperatorUse:
    op: str
    file: str
    line: int
    context: str


@dataclass
class Compatibility:
    op: str
    status: str
    evidence: str
    count: int


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return dotted_name(node.func)
    return None


def short_op(name: str | None) -> str | None:
    if not name:
        return None
    if name.startswith("self."):
        return None
    return name.split(".")[-1]


def iter_python_files(root: Path) -> Iterable[Path]:
    ignored = {"__pycache__", ".git", ".venv", "venv"}
    for path in root.rglob("*.py"):
        if not any(part in ignored for part in path.parts):
            yield path


def inspect_model_folder(model_input_folder: Path) -> dict:
    python_files = sorted(iter_python_files(model_input_folder))
    return {
        "model_input_folder": str(model_input_folder),
        "python_file_count": len(python_files),
        "python_files": [str(path) for path in python_files],
    }


class ModelVisitor(ast.NodeVisitor):
    def __init__(self, file_path: Path, root: Path):
        self.file = str(file_path.relative_to(root.parent))
        self.classes: list[DiscoveredClass] = []
        self.functions: list[DiscoveredFunction] = []
        self.operators: list[OperatorUse] = []
        self.context: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = [dotted_name(base) or ast.unparse(base) for base in node.bases]
        methods = [item.name for item in node.body if isinstance(item, ast.FunctionDef)]
        self.classes.append(
            DiscoveredClass(
                name=node.name,
                file=self.file,
                line=node.lineno,
                bases=bases,
                methods=methods,
            )
        )
        self.context.append(node.name)
        self.generic_visit(node)
        self.context.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not self.context:
            self.functions.append(DiscoveredFunction(name=node.name, file=self.file, line=node.lineno))
        self.context.append(node.name)
        self.generic_visit(node)
        self.context.pop()

    def visit_Call(self, node: ast.Call) -> None:
        op = short_op(dotted_name(node.func))
        if op and is_operator_like(op):
            self.operators.append(
                OperatorUse(
                    op=op,
                    file=self.file,
                    line=node.lineno,
                    context=".".join(self.context) if self.context else "<module>",
                )
            )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.operators.append(
            OperatorUse(
                op="getitem",
                file=self.file,
                line=node.lineno,
                context=".".join(self.context) if self.context else "<module>",
            )
        )
        self.generic_visit(node)


def is_operator_like(op: str) -> bool:
    if op in SUPPORTED_OPS or op in CONDITIONAL_OPS:
        return True
    layer_suffixes = ("1d", "2d", "3d")
    layer_names = ("Conv", "Pool", "Norm", "ReLU", "Linear", "Dropout", "Softmax")
    return op.endswith(layer_suffixes) or any(token in op for token in layer_names)


def extract_model(model_input_folder: Path) -> dict:
    classes: list[DiscoveredClass] = []
    functions: list[DiscoveredFunction] = []
    operators: list[OperatorUse] = []
    parse_errors: list[dict] = []

    for file_path in iter_python_files(model_input_folder):
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            parse_errors.append({"file": str(file_path), "error": str(exc)})
            continue

        visitor = ModelVisitor(file_path, model_input_folder)
        visitor.visit(tree)
        classes.extend(visitor.classes)
        functions.extend(visitor.functions)
        operators.extend(visitor.operators)

    return {
        "candidate_files": inspect_model_folder(model_input_folder)["python_files"],
        "classes": [asdict(item) for item in classes],
        "functions": [asdict(item) for item in functions],
        "operators": [asdict(item) for item in operators],
        "parse_errors": parse_errors,
    }


def classify_operators(operators: list[dict]) -> dict:
    counts = Counter(item["op"] for item in operators)
    rows: list[Compatibility] = []
    unsupported: list[dict] = []

    for op, count in sorted(counts.items()):
        if op in SUPPORTED_OPS:
            status = "supported"
            evidence = "Common Vitis AI DPU-friendly operator when exported with static shapes and INT8 calibration."
        elif op in CONDITIONAL_OPS:
            status = "conditional"
            evidence = CONDITIONAL_OPS[op]
            unsupported.append({"op": op, "count": count, "reason": evidence})
        else:
            status = "unsupported"
            evidence = "No built-in local rule matched this operator; verify against AMD Vitis AI docs before deployment."
            unsupported.append({"op": op, "count": count, "reason": evidence})
        rows.append(Compatibility(op=op, status=status, evidence=evidence, count=count))

    return {
        "summary": {
            "total_operator_uses": sum(counts.values()),
            "unique_operator_count": len(counts),
            "supported_unique_count": sum(1 for item in rows if item.status == "supported"),
            "conditional_unique_count": sum(1 for item in rows if item.status == "conditional"),
            "unsupported_unique_count": sum(1 for item in rows if item.status == "unsupported"),
        },
        "compatibility_table": [asdict(item) for item in rows],
        "unsupported_ops": unsupported,
    }


def proposal_for_op(op: str, aggressive: bool) -> dict:
    replacement = REPLACEMENTS.get(op, "Confirm support, then rewrite to Conv2d/BatchNorm2d/ReLU/static-shape blocks if needed.")
    if aggressive:
        action = replacement
        reason = "Maximizes DPU partition size and reduces CPU fallback risk."
    else:
        action = replacement if op in REPLACEMENTS else "Keep unchanged until AMD documentation confirms it blocks DPU compilation."
        reason = "Minimizes code churn while addressing the highest-risk compatibility issue."
    return {"op": op, "replacement": action, "reason": reason}


def generate_proposals(compatibility: dict) -> dict:
    risky_ops = [item["op"] for item in compatibility["unsupported_ops"]]
    total_unique = compatibility["summary"]["unique_operator_count"] or 1
    supported = compatibility["summary"]["supported_unique_count"]
    current_pct = round((supported / total_unique) * 100)

    conservative_changes = [proposal_for_op(op, aggressive=False) for op in risky_ops[:8]]
    aggressive_changes = [proposal_for_op(op, aggressive=True) for op in risky_ops]

    return {
        "conservative": {
            "strategy": "Rewrite only operators most likely to block Vitis AI DPU compilation; keep validated model structure intact.",
            "changes": conservative_changes,
            "estimated_compatibility": f"{min(95, max(current_pct, current_pct + 15))}%",
            "accuracy_risk": "Low to medium; validate changed blocks with the original checkpoint and task metrics.",
            "effort": "Small targeted refactor plus export/quantization test.",
        },
        "aggressive": {
            "strategy": "Convert the model to a static-shape, quantization-friendly graph built from DPU-preferred blocks.",
            "changes": aggressive_changes,
            "estimated_compatibility": "95-100%",
            "quantization_plan": "Run post-training INT8 quantization with a representative calibration set, then inspect DPU/CPU graph partitioning.",
            "accuracy_risk": "Medium; requires calibration, regression testing, and possibly retraining.",
            "effort": "Broader architecture rewrite plus deployment validation.",
        },
    }


def write_report(result: dict, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    extraction = result["extraction"]
    compatibility = result["compatibility"]
    proposals = result["proposals"]

    by_file: dict[str, list[dict]] = defaultdict(list)
    for op in extraction["operators"]:
        by_file[op["file"]].append(op)

    lines = [
        "# Vitis AI Model Compatibility Report",
        "",
        f"- Generated: {result['timestamp']}",
        f"- Model input folder: `{result['model_input_folder']}`",
        f"- Python files scanned: {len(extraction['candidate_files'])}",
        f"- Classes discovered: {len(extraction['classes'])}",
        f"- Operator uses discovered: {len(extraction['operators'])}",
        "",
        "## Candidate Files",
        "",
    ]
    lines.extend(f"- `{path}`" for path in extraction["candidate_files"])

    lines.extend(["", "## Classes And Functions", ""])
    for cls in extraction["classes"]:
        bases = ", ".join(cls["bases"]) or "object"
        methods = ", ".join(cls["methods"]) or "none"
        lines.append(f"- `{cls['name']}` in `{cls['file']}:{cls['line']}` bases: {bases}; methods: {methods}")

    if extraction["functions"]:
        lines.extend(["", "Top-level functions:", ""])
        for fn in extraction["functions"]:
            lines.append(f"- `{fn['name']}` in `{fn['file']}:{fn['line']}`")

    lines.extend(["", "## Operator Inventory", ""])
    for file, ops in sorted(by_file.items()):
        counts = Counter(op["op"] for op in ops)
        inventory = ", ".join(f"{op} ({count})" for op, count in sorted(counts.items()))
        lines.append(f"- `{file}`: {inventory}")

    lines.extend(
        [
            "",
            "## Compatibility Table",
            "",
            "| Operator | Uses | Status | Evidence |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in compatibility["compatibility_table"]:
        lines.append(f"| `{row['op']}` | {row['count']} | {row['status']} | {row['evidence']} |")

    lines.extend(["", "## DPU Risks", ""])
    if compatibility["unsupported_ops"]:
        for item in compatibility["unsupported_ops"]:
            lines.append(f"- `{item['op']}` ({item['count']} uses): {item['reason']}")
    else:
        lines.append("- No unsupported or conditional operators found by the local rule set.")

    lines.extend(["", "## Conservative Plan", "", proposals["conservative"]["strategy"], ""])
    for change in proposals["conservative"]["changes"]:
        lines.append(f"- `{change['op']}`: {change['replacement']} Reason: {change['reason']}")
    lines.append(f"- Estimated compatibility: {proposals['conservative']['estimated_compatibility']}")
    lines.append(f"- Accuracy risk: {proposals['conservative']['accuracy_risk']}")
    lines.append(f"- Effort: {proposals['conservative']['effort']}")

    lines.extend(["", "## Aggressive Plan", "", proposals["aggressive"]["strategy"], ""])
    for change in proposals["aggressive"]["changes"]:
        lines.append(f"- `{change['op']}`: {change['replacement']} Reason: {change['reason']}")
    lines.append(f"- Estimated compatibility: {proposals['aggressive']['estimated_compatibility']}")
    lines.append(f"- Quantization plan: {proposals['aggressive']['quantization_plan']}")
    lines.append(f"- Accuracy risk: {proposals['aggressive']['accuracy_risk']}")
    lines.append(f"- Effort: {proposals['aggressive']['effort']}")

    lines.extend(
        [
            "",
            "## Migration Checklist",
            "",
            "- Export the model to ONNX with fixed input shapes.",
            "- Replace conditional or unsupported operators before quantization.",
            "- Run Vitis AI quantization with representative calibration data.",
            "- Inspect compiler graph partitioning for CPU fallback.",
            "- Compare task metrics against the original checkpoint.",
        ]
    )

    if extraction["parse_errors"]:
        lines.extend(["", "## Parse Errors", ""])
        for error in extraction["parse_errors"]:
            lines.append(f"- `{error['file']}`: {error['error']}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(model_input_folder: Path, report_path: Path) -> dict:
    model_input_folder = model_input_folder.resolve()
    if not model_input_folder.exists():
        raise FileNotFoundError(f"Model input folder does not exist: {model_input_folder}")
    if not model_input_folder.is_dir():
        raise NotADirectoryError(f"Model input path is not a folder: {model_input_folder}")

    extraction = extract_model(model_input_folder)
    compatibility = classify_operators(extraction["operators"])
    proposals = generate_proposals(compatibility)
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_input_folder": str(model_input_folder),
        "extraction": extraction,
        "compatibility": compatibility,
        "proposals": proposals,
        "report_path": str(report_path),
    }
    write_report(result, report_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Vitis AI compatibility pipeline.")
    parser.add_argument(
        "model_input_folder",
        nargs="?",
        default="model_input/Transfer-Model_original",
        help="Folder containing the model source code to analyze.",
    )
    parser.add_argument(
        "--report",
        default="reports/model_input_vitis_ai_compatibility.md",
        help="Markdown report path to write.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full JSON result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(Path(args.model_input_folder), Path(args.report))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
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
