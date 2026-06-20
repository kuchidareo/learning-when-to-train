from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


DIAGNOSIS_SYSTEM_PROMPT_TEMPLATE = """You are an edge-side diagnostic agent for federated learning.

Given a JSON metadata summary, identify the primary ROOT CAUSE of degraded FL/FCL performance.

Compare the current scenario against the included IID random-dropout reference when diagnostic_hints are present.
Do not classify normal IID random dropout as an anomaly unless the metrics clearly exceed the reference behavior.

Choose exactly one diagnosis from:
- system_heterogeneity_dropout_stragglers
- selection_aggregation_bias
- non_stationarity_distribution_shift
- continual_learning_temporal_forgetting
- statistical_heterogeneity_client_drift
- no_major_issue

Decision guide:
- system_heterogeneity_dropout_stragglers: Selected clients fail to finish or upload updates because of device, network, or runtime limits.
- selection_aggregation_bias: Client contribution is biased because selection or aggregation overrepresents certain available or completed clients.
- non_stationarity_distribution_shift: A large central shift in the current or evaluation feature distribution.
- continual_learning_temporal_forgetting: Old task accuracy or exposure decreases while new task accuracy or exposure increases.
- statistical_heterogeneity_client_drift: Clients locally learn well, but their updates point in conflicting directions.
- no_major_issue: no strong signal.

First decide the diagnosis. Then write concise metric-based evidence and one recommended cloud action.
This first stage does not need to be valid JSON.
""".strip()


DIAGNOSIS_USER_PROMPT_TEMPLATE = "[[summary_json]]"


REFORMAT_SYSTEM_PROMPT_TEMPLATE = """You convert a federated-learning diagnostic answer into strict JSON.

You must preserve the diagnosis intended by the diagnostic answer. Do not introduce a new diagnosis unless the answer is invalid or ambiguous.

Allowed diagnosis labels:
- system_heterogeneity_dropout_stragglers
- selection_aggregation_bias
- non_stationarity_distribution_shift
- continual_learning_temporal_forgetting
- statistical_heterogeneity_client_drift
- no_major_issue

Return only valid JSON:
{
  "diagnosis": "...",
  "confidence": 0.0,
  "evidence": ["metric-based reason 1", "metric-based reason 2"],
  "recommended_cloud_action": "..."
}
""".strip()


REFORMAT_USER_PROMPT_TEMPLATE = """Diagnostic answer to reformat:
[[diagnosis_output]]

Return only the fixed JSON object. No markdown, no explanation outside JSON.
""".strip()


def load_template(default: str, template_file: str | None) -> str:
    if template_file is None:
        return default
    return Path(template_file).read_text(encoding="utf-8").strip()


def validate_record(record: dict[str, Any], source_line: int) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"line {source_line}: expected messages with at least system and user entries")
    if not isinstance(messages[0], dict) or messages[0].get("role") != "system":
        raise ValueError(f"line {source_line}: expected messages[0].role == 'system'")
    if not isinstance(messages[1], dict) or messages[1].get("role") != "user":
        raise ValueError(f"line {source_line}: expected messages[1].role == 'user'")


def parse_user_payload(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return {"summary": record.get("summary", {})}
    user_content = str(messages[1].get("content", ""))
    try:
        parsed = json.loads(user_content)
    except json.JSONDecodeError:
        parsed = {}
    if isinstance(parsed, dict) and isinstance(parsed.get("summary"), dict):
        return parsed
    return {"summary": record.get("summary", {})}


def remove_comparison_z_scores(user_payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(user_payload)
    summary = cleaned.get("summary")
    if not isinstance(summary, dict):
        return cleaned
    hints = summary.get("diagnostic_hints")
    if not isinstance(hints, dict):
        return cleaned
    comparisons = hints.get("comparisons")
    if not isinstance(comparisons, dict):
        return cleaned
    for comparison in comparisons.values():
        if isinstance(comparison, dict):
            comparison.pop("z_score_vs_reference", None)
    return cleaned


def remove_misleading_diagnostic_hints(user_payload: dict[str, Any]) -> dict[str, Any]:
    """Drop generated prose hints that can conflict with the system prompt."""
    cleaned = copy.deepcopy(user_payload)
    summary = cleaned.get("summary")
    if not isinstance(summary, dict):
        return cleaned
    hints = summary.get("diagnostic_hints")
    if not isinstance(hints, dict):
        return cleaned
    hints.pop("reference_interpretation", None)
    hints.pop("decision_notes", None)
    return cleaned


def remove_label_prior_shift_signal(user_payload: dict[str, Any]) -> dict[str, Any]:
    """Remove label-prior shift signal that confounds the intended taxonomy."""
    cleaned = copy.deepcopy(user_payload)
    summary = cleaned.get("summary")
    if not isinstance(summary, dict):
        return cleaned
    shift_and_memory = summary.get("shift_and_memory")
    if isinstance(shift_and_memory, dict):
        shift_and_memory.pop("js_current_baseline_label", None)
    hints = summary.get("diagnostic_hints")
    if isinstance(hints, dict):
        comparisons = hints.get("comparisons")
        if isinstance(comparisons, dict):
            comparisons.pop("shift_and_memory.js_current_baseline_label", None)
        reference_stats = hints.get("reference_stats")
        if isinstance(reference_stats, dict):
            reference_stats.pop("shift_and_memory.js_current_baseline_label", None)
    return cleaned


def make_tokens(record: dict[str, Any], *, keep_z_scores: bool = False) -> dict[str, str]:
    messages = record.get("messages")
    messages = messages if isinstance(messages, list) else []
    original_system_prompt = str(messages[0].get("content", "")) if len(messages) >= 1 and isinstance(messages[0], dict) else ""
    original_user_prompt = str(messages[1].get("content", "")) if len(messages) >= 2 and isinstance(messages[1], dict) else ""
    user_payload = remove_label_prior_shift_signal(
        remove_misleading_diagnostic_hints(parse_user_payload(record))
    )
    user_payload_without_z_scores = remove_comparison_z_scores(user_payload)
    prompt_payload = user_payload if keep_z_scores else user_payload_without_z_scores
    return {
        "scenario_id": str(record.get("scenario_id", "")),
        "trial_id": str(record.get("trial_id", "")),
        "trial_index": str(record.get("trial_index", "")),
        "model": str(record.get("model", "")),
        "original_system_prompt": original_system_prompt,
        "original_user_prompt": original_user_prompt,
        "user_prompt_with_z_scores": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
        "user_prompt_without_z_scores": json.dumps(user_payload_without_z_scores, ensure_ascii=False, separators=(",", ":")),
        "summary_json_with_z_scores": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
        "summary_json": json.dumps(prompt_payload, ensure_ascii=False, separators=(",", ":")),
        "top_level_summary_json": json.dumps({"summary": record.get("summary", {})}, ensure_ascii=False, separators=(",", ":")),
    }


def render_template(template: str, tokens: dict[str, str]) -> str:
    rendered = template
    for name, value in tokens.items():
        rendered = rendered.replace(f"[[{name}]]", value)
    return rendered


def transform_record(
    record: dict[str, Any],
    *,
    source_line: int,
    diagnosis_system_template: str,
    diagnosis_user_template: str,
    reformat_system_template: str,
    reformat_user_template: str,
    keep_z_scores: bool = False,
) -> dict[str, Any]:
    validate_record(record, source_line)
    tokens = make_tokens(record, keep_z_scores=keep_z_scores)
    updated = dict(record)
    updated["modular_prompt"] = True
    updated["messages"] = [
        {"role": "system", "content": render_template(diagnosis_system_template, tokens)},
        {"role": "user", "content": render_template(diagnosis_user_template, tokens)},
    ]
    updated["reformat_messages"] = [
        {"role": "system", "content": render_template(reformat_system_template, tokens)},
        {"role": "user", "content": render_template(reformat_user_template, tokens)},
    ]
    updated["prompt_pipeline"] = {
        "stage_1": "diagnosis",
        "stage_2": "json_reformat",
        "stage_2_input_placeholder": "[[diagnosis_output]]",
    }
    return updated


def transform_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    diagnosis_system_template: str,
    diagnosis_user_template: str,
    reformat_system_template: str,
    reformat_user_template: str,
    keep_z_scores: bool = False,
) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for source_line, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            updated = transform_record(
                record,
                source_line=source_line,
                diagnosis_system_template=diagnosis_system_template,
                diagnosis_user_template=diagnosis_user_template,
                reformat_system_template=reformat_system_template,
                reformat_user_template=reformat_user_template,
                keep_z_scores=keep_z_scores,
            )
            dst.write(json.dumps(updated, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a two-stage modular SLM prompt JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("slm_prompts_for_ollama.jsonl"),
        help="Original JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("slm_prompts_modular.jsonl"),
        help="Output modular JSONL file.",
    )
    parser.add_argument("--diagnosis-system-template-file", default=None)
    parser.add_argument("--diagnosis-user-template-file", default=None)
    parser.add_argument("--reformat-system-template-file", default=None)
    parser.add_argument("--reformat-user-template-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-z-scores",
        action="store_true",
        help="Keep z_score_vs_reference fields in diagnostic_hints.comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnosis_system_template = load_template(DIAGNOSIS_SYSTEM_PROMPT_TEMPLATE, args.diagnosis_system_template_file)
    diagnosis_user_template = load_template(DIAGNOSIS_USER_PROMPT_TEMPLATE, args.diagnosis_user_template_file)
    reformat_system_template = load_template(REFORMAT_SYSTEM_PROMPT_TEMPLATE, args.reformat_system_template_file)
    reformat_user_template = load_template(REFORMAT_USER_PROMPT_TEMPLATE, args.reformat_user_template_file)

    if args.dry_run:
        with args.input.open("r", encoding="utf-8") as f:
            for source_line, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                updated = transform_record(
                    record,
                    source_line=source_line,
                    diagnosis_system_template=diagnosis_system_template,
                    diagnosis_user_template=diagnosis_user_template,
                    reformat_system_template=reformat_system_template,
                    reformat_user_template=reformat_user_template,
                    keep_z_scores=args.keep_z_scores,
                )
                print(json.dumps(updated, ensure_ascii=False, indent=2)[:10000])
                return
        raise ValueError(f"No records found in {args.input}")

    count = transform_jsonl(
        args.input,
        args.output,
        diagnosis_system_template=diagnosis_system_template,
        diagnosis_user_template=diagnosis_user_template,
        reformat_system_template=reformat_system_template,
        reformat_user_template=reformat_user_template,
        keep_z_scores=args.keep_z_scores,
    )
    print(f"Wrote {count} modular prompt records to {args.output}")


if __name__ == "__main__":
    main()
