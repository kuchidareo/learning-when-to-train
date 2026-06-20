from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from socket import timeout as SocketTimeout
from typing import Any


@dataclass(frozen=True)
class PromptCase:
    prompt_name: str
    prompt_index: int
    repeat_index: int
    system_prompt: str
    user_prompt: str
    source_file: str
    source_line: int
    scenario_id: str = ""
    ground_truth: str = ""
    trial_id: str = ""
    trial_index: str = ""
    source_model: str = ""
    summary_json: str = ""
    top_level_summary_json: str = ""
    reference_scenario: str = ""
    reference_interpretation: str = ""
    above_reference_p95_metrics: str = ""
    above_reference_p95_count: int = 0
    comparison_metrics_json: str = ""
    reference_stats_json: str = ""
    decision_notes_json: str = ""
    reformat_system_prompt: str = ""
    reformat_user_prompt: str = ""

    @property
    def full_prompt(self) -> str:
        if self.system_prompt:
            return f"{self.system_prompt}\n\n{self.user_prompt}"
        return self.user_prompt

    @property
    def is_modular(self) -> bool:
        return bool(self.reformat_user_prompt)

    def render_reformat_user_prompt(self, diagnosis_output: str) -> str:
        return self.reformat_user_prompt.replace("[[diagnosis_output]]", diagnosis_output)


def parse_user_prompt_summary(user_prompt: str, fallback_summary: Any) -> dict[str, Any]:
    try:
        payload = json.loads(user_prompt)
    except json.JSONDecodeError:
        payload = {}
    if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
        return payload["summary"]
    return fallback_summary if isinstance(fallback_summary, dict) else {}


def extract_reference_fields(summary: dict[str, Any]) -> dict[str, Any]:
    hints = summary.get("diagnostic_hints")
    hints = hints if isinstance(hints, dict) else {}
    comparisons = hints.get("comparisons")
    comparisons = comparisons if isinstance(comparisons, dict) else {}
    reference_stats = hints.get("reference_stats")
    reference_stats = reference_stats if isinstance(reference_stats, dict) else {}
    decision_notes = hints.get("decision_notes")
    decision_notes = decision_notes if isinstance(decision_notes, list) else []

    above_p95 = sorted(
        metric
        for metric, comparison in comparisons.items()
        if isinstance(comparison, dict) and comparison.get("above_reference_p95") is True
    )

    return {
        "reference_scenario": str(hints.get("reference_scenario", "")),
        "reference_interpretation": str(hints.get("reference_interpretation", "")),
        "above_reference_p95_metrics": "|".join(above_p95),
        "above_reference_p95_count": len(above_p95),
        "comparison_metrics_json": json.dumps(comparisons, ensure_ascii=False),
        "reference_stats_json": json.dumps(reference_stats, ensure_ascii=False),
        "decision_notes_json": json.dumps(decision_notes, ensure_ascii=False),
    }


def duration_ns_to_sec(value: Any) -> float:
    try:
        return float(value) / 1_000_000_000.0
    except (TypeError, ValueError):
        return 0.0


def load_slm_jsonl_prompt_cases(jsonl_file: Path, runs: int) -> list[PromptCase]:
    cases: list[PromptCase] = []
    with jsonl_file.open("r", encoding="utf-8") as f:
        for source_line, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{jsonl_file}:{source_line} must contain at least two messages")

            system_message = messages[0]
            user_message = messages[1]
            if system_message.get("role") != "system" or user_message.get("role") != "user":
                raise ValueError(
                    f"{jsonl_file}:{source_line} expected messages[0].role=system "
                    "and messages[1].role=user"
                )
            reformat_system_prompt = ""
            reformat_user_prompt = ""
            reformat_messages = record.get("reformat_messages")
            if isinstance(reformat_messages, list) and len(reformat_messages) >= 2:
                if reformat_messages[0].get("role") != "system" or reformat_messages[1].get("role") != "user":
                    raise ValueError(
                        f"{jsonl_file}:{source_line} expected reformat_messages[0].role=system "
                        "and reformat_messages[1].role=user"
                    )
                reformat_system_prompt = str(reformat_messages[0].get("content", ""))
                reformat_user_prompt = str(reformat_messages[1].get("content", ""))

            scenario_id = str(record.get("scenario_id", ""))
            trial_index = str(record.get("trial_index", ""))
            prompt_name = f"{scenario_id}_trial_{trial_index}" if scenario_id else f"jsonl_line_{source_line}"
            top_level_summary = record.get("summary", {})
            prompt_summary = parse_user_prompt_summary(str(user_message.get("content", "")), top_level_summary)
            reference_fields = extract_reference_fields(prompt_summary)
            for repeat_index in range(1, runs + 1):
                cases.append(
                    PromptCase(
                        prompt_name=prompt_name,
                        prompt_index=len(cases) + 1,
                        repeat_index=repeat_index,
                        system_prompt=str(system_message.get("content", "")),
                        user_prompt=str(user_message.get("content", "")),
                        source_file=str(jsonl_file),
                        source_line=source_line,
                        scenario_id=scenario_id,
                        ground_truth=str(record.get("ground_truth", "")),
                        trial_id=str(record.get("trial_id", "")),
                        trial_index=trial_index,
                        source_model=str(record.get("model", "")),
                        summary_json=json.dumps(prompt_summary, ensure_ascii=False),
                        top_level_summary_json=json.dumps(top_level_summary, ensure_ascii=False),
                        reformat_system_prompt=reformat_system_prompt,
                        reformat_user_prompt=reformat_user_prompt,
                        **reference_fields,
                    )
                )

    if not cases:
        raise ValueError(f"No JSONL prompt records found in {jsonl_file}")
    return cases


def parse_response_object(raw_response: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_response)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw_response[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def call_ollama(
    *,
    case: PromptCase,
    model: str,
    host: str,
    timeout: float,
    temperature: float,
    num_predict: int | None,
    keep_alive: str,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    think: bool | None = None,
) -> tuple[dict[str, Any] | None, float, str]:
    effective_system_prompt = case.system_prompt if system_prompt is None else system_prompt
    effective_user_prompt = case.user_prompt if user_prompt is None else user_prompt
    payload: dict[str, Any] = {
        "model": model,
        "prompt": effective_user_prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"temperature": temperature},
    }
    if effective_system_prompt:
        payload["system"] = effective_system_prompt
    if think is not None:
        payload["think"] = think
    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
        elapsed = time.perf_counter() - start
        return json.loads(body.decode("utf-8")), elapsed, ""
    except (urllib.error.URLError, SocketTimeout, TimeoutError, json.JSONDecodeError, OSError) as exc:
        elapsed = time.perf_counter() - start
        return None, elapsed, f"{type(exc).__name__}: {exc}"


def make_row(
    *,
    run_started_at: str,
    case: PromptCase,
    model: str,
    host: str,
    response_json: dict[str, Any] | None,
    wall_time_sec: float,
    error: str,
    raw_response_max_chars: int,
    raw_ollama_max_chars: int,
    diagnosis_response_json: dict[str, Any] | None = None,
    diagnosis_wall_time_sec: float = 0.0,
    diagnosis_error: str = "",
    reformat_response_json: dict[str, Any] | None = None,
    reformat_wall_time_sec: float = 0.0,
    reformat_error: str = "",
) -> dict[str, Any]:
    response_json = response_json or {}
    diagnosis_response_json = diagnosis_response_json or {}
    reformat_response_json = reformat_response_json or {}
    raw_response = str(response_json.get("response") or "")
    raw_diagnosis_response = str(diagnosis_response_json.get("response") or "")
    raw_reformat_response = str(reformat_response_json.get("response") or "")
    parsed_response = parse_response_object(raw_response)
    response_diagnosis = str(parsed_response.get("diagnosis", ""))
    response_confidence = parsed_response.get("confidence", "")
    is_correct = ""
    if case.ground_truth and response_diagnosis:
        is_correct = response_diagnosis == case.ground_truth

    prompt_tokens = int(response_json.get("prompt_eval_count") or 0)
    completion_tokens = int(response_json.get("eval_count") or 0)
    eval_duration_sec = duration_ns_to_sec(response_json.get("eval_duration"))
    tokens_per_sec = completion_tokens / eval_duration_sec if eval_duration_sec > 0 else 0.0
    diagnosis_prompt_tokens = int(diagnosis_response_json.get("prompt_eval_count") or 0)
    diagnosis_completion_tokens = int(diagnosis_response_json.get("eval_count") or 0)
    reformat_prompt_tokens = int(reformat_response_json.get("prompt_eval_count") or 0)
    reformat_completion_tokens = int(reformat_response_json.get("eval_count") or 0)

    return {
        "run_started_at": run_started_at,
        "source_file": case.source_file,
        "source_line": case.source_line,
        "prompt_name": case.prompt_name,
        "prompt_index": case.prompt_index,
        "repeat_index": case.repeat_index,
        "scenario_id": case.scenario_id,
        "ground_truth": case.ground_truth,
        "trial_id": case.trial_id,
        "trial_index": case.trial_index,
        "source_model": case.source_model,
        "reference_scenario": case.reference_scenario,
        "reference_interpretation": case.reference_interpretation,
        "above_reference_p95_metrics": case.above_reference_p95_metrics,
        "above_reference_p95_count": case.above_reference_p95_count,
        "ollama_model": model,
        "host": host,
        "prompt_pipeline": "modular" if case.is_modular else "single",
        "success": error == "",
        "error": error,
        "wall_time_sec": wall_time_sec,
        "diagnosis_error": diagnosis_error,
        "reformat_error": reformat_error,
        "diagnosis_wall_time_sec": diagnosis_wall_time_sec,
        "reformat_wall_time_sec": reformat_wall_time_sec,
        "ollama_total_duration_sec": duration_ns_to_sec(response_json.get("total_duration")),
        "load_duration_sec": duration_ns_to_sec(response_json.get("load_duration")),
        "prompt_eval_duration_sec": duration_ns_to_sec(response_json.get("prompt_eval_duration")),
        "eval_duration_sec": eval_duration_sec,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "diagnosis_prompt_tokens": diagnosis_prompt_tokens,
        "diagnosis_completion_tokens": diagnosis_completion_tokens,
        "diagnosis_total_tokens": diagnosis_prompt_tokens + diagnosis_completion_tokens,
        "reformat_prompt_tokens": reformat_prompt_tokens,
        "reformat_completion_tokens": reformat_completion_tokens,
        "reformat_total_tokens": reformat_prompt_tokens + reformat_completion_tokens,
        "tokens_per_sec_eval": tokens_per_sec,
        "system_prompt_chars": len(case.system_prompt),
        "user_prompt_chars": len(case.user_prompt),
        "reformat_system_prompt_chars": len(case.reformat_system_prompt),
        "reformat_user_prompt_chars": len(case.reformat_user_prompt),
        "prompt_chars": len(case.full_prompt),
        "response_chars": len(raw_response),
        "done_reason": response_json.get("done_reason", ""),
        "response_diagnosis": response_diagnosis,
        "response_confidence": response_confidence,
        "is_correct": is_correct,
        "summary_json": case.summary_json,
        "top_level_summary_json": case.top_level_summary_json,
        "comparison_metrics_json": case.comparison_metrics_json,
        "reference_stats_json": case.reference_stats_json,
        "decision_notes_json": case.decision_notes_json,
        "raw_diagnosis_response": raw_diagnosis_response[:raw_response_max_chars],
        "raw_reformat_response": raw_reformat_response[:raw_response_max_chars],
        "raw_response": raw_response[:raw_response_max_chars],
        "raw_ollama_json": json.dumps(response_json, ensure_ascii=False)[:raw_ollama_max_chars],
    }


def write_rows_incrementally(output_csv: Path, rows: list[dict[str, Any]], append: bool) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_header = not append or not output_csv.exists() or output_csv.stat().st_size == 0
    with output_csv.open("a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SLM diagnostic JSONL prompts through local Ollama and log latency/token metrics."
    )
    parser.add_argument(
        "--jsonl_file",
        type=Path,
        default=Path(__file__).with_name("slm_prompts_for_ollama.jsonl"),
    )
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeats per JSONL prompt record.")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--num_predict",
        type=int,
        default=None,
        help="Optional maximum generated tokens. Omit to use the Ollama/model default.",
    )
    parser.add_argument("--keep_alive", default="10m")
    parser.add_argument(
        "--reformat_think",
        choices=["default", "true", "false"],
        default="default",
        help=(
            "Ollama thinking mode for modular JSON reformat stage only. "
            "Use false to send think=false, true to send think=true, default to omit."
        ),
    )
    parser.add_argument("--raw_response_max_chars", type=int, default=20000)
    parser.add_argument("--raw_ollama_max_chars", type=int, default=40000)
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_predict is not None and args.num_predict < 0:
        args.num_predict = None
    reformat_think = None
    if args.reformat_think == "true":
        reformat_think = True
    elif args.reformat_think == "false":
        reformat_think = False

    cases = load_slm_jsonl_prompt_cases(args.jsonl_file, args.runs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = args.output_csv or args.jsonl_file.parent / "logs" / f"slm_ollama_latency_{timestamp}.csv"
    run_started_at = datetime.now().isoformat(timespec="seconds")

    first_write = True
    total = len(cases)
    print(f"Loaded {total} prompt runs from {args.jsonl_file}")
    print(f"Using model={args.model} host={args.host}")

    for completed, case in enumerate(cases, start=1):
        diagnosis_response_json, diagnosis_wall_time_sec, diagnosis_error = call_ollama(
            case=case,
            model=args.model,
            host=args.host,
            timeout=args.timeout,
            temperature=args.temperature,
            num_predict=args.num_predict,
            keep_alive=args.keep_alive,
        )
        response_json = diagnosis_response_json
        wall_time_sec = diagnosis_wall_time_sec
        error = diagnosis_error
        reformat_response_json = None
        reformat_wall_time_sec = 0.0
        reformat_error = ""

        if case.is_modular and not diagnosis_error:
            diagnosis_output = str((diagnosis_response_json or {}).get("response") or "")
            reformat_user_prompt = case.render_reformat_user_prompt(diagnosis_output)
            reformat_response_json, reformat_wall_time_sec, reformat_error = call_ollama(
                case=case,
                model=args.model,
                host=args.host,
                timeout=args.timeout,
                temperature=args.temperature,
                num_predict=args.num_predict,
                keep_alive=args.keep_alive,
                system_prompt=case.reformat_system_prompt,
                user_prompt=reformat_user_prompt,
                think=reformat_think,
            )
            wall_time_sec = diagnosis_wall_time_sec + reformat_wall_time_sec
            response_json = reformat_response_json
            error = reformat_error

        row = make_row(
            run_started_at=run_started_at,
            case=case,
            model=args.model,
            host=args.host,
            response_json=response_json,
            wall_time_sec=wall_time_sec,
            error=error,
            raw_response_max_chars=args.raw_response_max_chars,
            raw_ollama_max_chars=args.raw_ollama_max_chars,
            diagnosis_response_json=diagnosis_response_json,
            diagnosis_wall_time_sec=diagnosis_wall_time_sec,
            diagnosis_error=diagnosis_error,
            reformat_response_json=reformat_response_json,
            reformat_wall_time_sec=reformat_wall_time_sec,
            reformat_error=reformat_error,
        )
        write_rows_incrementally(output_csv, [row], append=args.append or not first_write)
        first_write = False

        status = "ok" if not error else "error"
        diagnosis = f" diagnosis={row['response_diagnosis']}" if row["response_diagnosis"] else ""
        pipeline = " modular" if case.is_modular else ""
        message = f"[{completed}/{total}] {case.prompt_name} run={case.repeat_index}{pipeline} {status} wall={wall_time_sec:.3f}s{diagnosis}"
        if case.is_modular:
            message += f" diagnosis_wall={diagnosis_wall_time_sec:.3f}s reformat_wall={reformat_wall_time_sec:.3f}s"
        if error:
            message += f" error={error}"
        print(message)

    print(f"Wrote CSV: {output_csv}")


if __name__ == "__main__":
    main()
