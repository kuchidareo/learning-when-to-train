from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


# Edit these two templates when you want to change the prompt.
#
# Token replacement is intentionally simple: write [[token_name]] where a value
# should be inserted. JSON braces in the prompt do not need escaping.
#
# Available tokens:
#   [[scenario_id]]
#   [[trial_id]]
#   [[trial_index]]
#   [[model]]
#   [[original_system_prompt]]
#   [[original_user_prompt]]
#   [[user_prompt_with_z_scores]]
#   [[user_prompt_without_z_scores]]
#   [[summary_json_with_z_scores]]
#   [[summary_json]]
#   [[top_level_summary_json]]
#
# Ground-truth labels are preserved as metadata in the output JSONL, but they
# are intentionally not exposed as a replacement token to avoid label leakage.
SYSTEM_PROMPT_TEMPLATE = """You are an edge-side diagnostic agent for federated learning.

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

Return only valid JSON:
{
  "diagnosis": "...",
  "confidence": 0.0,
  "evidence": ["metric-based reason 1", "metric-based reason 2"],
  "recommended_cloud_action": "..."
}
""".strip()


# Keep this JSON-only by default. The Ollama/API runners can parse
# messages[1].content for diagnostic_hints when it remains a JSON object.
USER_PROMPT_TEMPLATE = "[[summary_json]]"


def load_template(default: str, template_file: str | None) -> str:
    if template_file is None:
        return default
    return Path(template_file).read_text(encoding="utf-8").strip()


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
    """Remove z_score_vs_reference from diagnostic_hints.comparisons only."""
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


def make_tokens(record: dict[str, Any], *, keep_z_scores: bool = False) -> dict[str, str]:
    messages = record.get("messages")
    messages = messages if isinstance(messages, list) else []
    system_prompt = ""
    user_prompt = ""
    if len(messages) >= 1 and isinstance(messages[0], dict):
        system_prompt = str(messages[0].get("content", ""))
    if len(messages) >= 2 and isinstance(messages[1], dict):
        user_prompt = str(messages[1].get("content", ""))

    user_payload = remove_misleading_diagnostic_hints(parse_user_payload(record))
    user_payload_without_z_scores = remove_comparison_z_scores(user_payload)
    prompt_payload = user_payload if keep_z_scores else user_payload_without_z_scores
    return {
        "scenario_id": str(record.get("scenario_id", "")),
        "trial_id": str(record.get("trial_id", "")),
        "trial_index": str(record.get("trial_index", "")),
        "model": str(record.get("model", "")),
        "original_system_prompt": system_prompt,
        "original_user_prompt": user_prompt,
        "user_prompt_with_z_scores": json.dumps(
            user_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "user_prompt_without_z_scores": json.dumps(
            user_payload_without_z_scores,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "summary_json_with_z_scores": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
        "summary_json": json.dumps(prompt_payload, ensure_ascii=False, separators=(",", ":")),
        "top_level_summary_json": json.dumps(
            {"summary": record.get("summary", {})},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


def render_template(template: str, tokens: dict[str, str]) -> str:
    rendered = template
    for name, value in tokens.items():
        rendered = rendered.replace(f"[[{name}]]", value)
    return rendered


def validate_record(record: dict[str, Any], source_line: int) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"line {source_line}: expected messages with at least system and user entries")
    if not isinstance(messages[0], dict) or messages[0].get("role") != "system":
        raise ValueError(f"line {source_line}: expected messages[0].role == 'system'")
    if not isinstance(messages[1], dict) or messages[1].get("role") != "user":
        raise ValueError(f"line {source_line}: expected messages[1].role == 'user'")


def transform_record(
    record: dict[str, Any],
    *,
    source_line: int,
    system_template: str,
    user_template: str,
    keep_z_scores: bool = False,
) -> dict[str, Any]:
    validate_record(record, source_line)
    updated = dict(record)
    updated_messages = [dict(message) for message in record["messages"]]
    tokens = make_tokens(record, keep_z_scores=keep_z_scores)
    updated_messages[0]["content"] = render_template(system_template, tokens)
    updated_messages[1]["content"] = render_template(user_template, tokens)
    updated["messages"] = updated_messages
    return updated


def transform_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    system_template: str,
    user_template: str,
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
                system_template=system_template,
                user_template=user_template,
                keep_z_scores=keep_z_scores,
            )
            dst.write(json.dumps(updated, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new SLM prompt JSONL by replacing messages[0].content and "
            "messages[1].content while preserving all top-level metadata."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("slm_prompts_for_ollama.jsonl"),
        help="Original JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("slm_prompts_changed.jsonl"),
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--system-template-file",
        default=None,
        help="Optional text file to use instead of SYSTEM_PROMPT_TEMPLATE.",
    )
    parser.add_argument(
        "--user-template-file",
        default=None,
        help="Optional text file to use instead of USER_PROMPT_TEMPLATE.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the first record and print a preview instead of writing output.",
    )
    parser.add_argument(
        "--keep-z-scores",
        action="store_true",
        help="Keep z_score_vs_reference fields in diagnostic_hints.comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    system_template = load_template(SYSTEM_PROMPT_TEMPLATE, args.system_template_file)
    user_template = load_template(USER_PROMPT_TEMPLATE, args.user_template_file)

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
                    system_template=system_template,
                    user_template=user_template,
                    keep_z_scores=args.keep_z_scores,
                )
                print(json.dumps(updated, ensure_ascii=False, indent=2)[:8000])
                return
        raise ValueError(f"No records found in {args.input}")

    count = transform_jsonl(
        args.input,
        args.output,
        system_template=system_template,
        user_template=user_template,
        keep_z_scores=args.keep_z_scores,
    )
    print(f"Wrote {count} prompt records to {args.output}")


if __name__ == "__main__":
    main()
