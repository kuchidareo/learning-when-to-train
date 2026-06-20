from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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


def build_messages(case: PromptCase, system_prompt: str | None = None, user_prompt: str | None = None) -> list[dict[str, str]]:
    effective_system_prompt = case.system_prompt if system_prompt is None else system_prompt
    effective_user_prompt = case.user_prompt if user_prompt is None else user_prompt
    messages: list[dict[str, str]] = []
    if effective_system_prompt:
        messages.append({"role": "system", "content": effective_system_prompt})
    messages.append({"role": "user", "content": effective_user_prompt})
    return messages


def make_client(base_url: str, api_key_env: str, timeout: float):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install with `pip install openai`.") from exc

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {api_key_env}")
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def call_chat_api(
    *,
    client: Any,
    case: PromptCase,
    model: str,
    temperature: float,
    max_tokens: int | None,
    api_backend: str,
    reasoning_effort: str | None,
    deepseek_thinking: bool,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
) -> tuple[Any | None, float, str]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": build_messages(case, system_prompt=system_prompt, user_prompt=user_prompt),
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if api_backend == "deepseek":
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if deepseek_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    start = time.perf_counter()
    try:
        completion = client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - start
        return completion, elapsed, ""
    except Exception as exc:  # API clients raise several transport/provider exceptions.
        elapsed = time.perf_counter() - start
        return None, elapsed, f"{type(exc).__name__}: {exc}"


def extract_completion_fields(completion: Any | None) -> dict[str, Any]:
    if completion is None:
        return {
            "response_text": "",
            "finish_reason": "",
            "api_prompt_tokens": 0,
            "api_completion_tokens": 0,
            "api_total_tokens": 0,
            "response_id": "",
            "response_model": "",
            "reasoning_content": "",
            "raw_completion_json": "",
        }

    response_text = ""
    finish_reason = ""
    reasoning_content = ""
    if getattr(completion, "choices", None):
        choice = completion.choices[0]
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
        message = getattr(choice, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        response_text = str(content or "")
        if message is not None:
            reasoning = getattr(message, "reasoning_content", "") or ""
            if not reasoning and hasattr(message, "model_extra"):
                model_extra = getattr(message, "model_extra", None) or {}
                if isinstance(model_extra, dict):
                    reasoning = model_extra.get("reasoning_content", "") or ""
            reasoning_content = str(reasoning or "")

    usage = getattr(completion, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0

    if hasattr(completion, "model_dump_json"):
        raw_completion_json = completion.model_dump_json()
    else:
        raw_completion_json = str(completion)

    return {
        "response_text": response_text,
        "finish_reason": finish_reason,
        "api_prompt_tokens": prompt_tokens,
        "api_completion_tokens": completion_tokens,
        "api_total_tokens": total_tokens,
        "response_id": str(getattr(completion, "id", "") or ""),
        "response_model": str(getattr(completion, "model", "") or ""),
        "reasoning_content": reasoning_content,
        "raw_completion_json": raw_completion_json,
    }


def make_row(
    *,
    run_started_at: str,
    case: PromptCase,
    api_backend: str,
    model: str,
    base_url: str,
    reasoning_effort: str | None,
    deepseek_thinking: bool,
    completion: Any | None,
    wall_time_sec: float,
    error: str,
    raw_response_max_chars: int,
    raw_completion_max_chars: int,
    diagnosis_completion: Any | None = None,
    diagnosis_wall_time_sec: float = 0.0,
    diagnosis_error: str = "",
    reformat_completion: Any | None = None,
    reformat_wall_time_sec: float = 0.0,
    reformat_error: str = "",
) -> dict[str, Any]:
    fields = extract_completion_fields(completion)
    diagnosis_fields = extract_completion_fields(diagnosis_completion)
    reformat_fields = extract_completion_fields(reformat_completion)
    raw_response = fields["response_text"]
    raw_diagnosis_response = diagnosis_fields["response_text"]
    raw_reformat_response = reformat_fields["response_text"]
    parsed_response = parse_response_object(raw_response)
    response_diagnosis = str(parsed_response.get("diagnosis", ""))
    response_confidence = parsed_response.get("confidence", "")
    is_correct = ""
    if case.ground_truth and response_diagnosis:
        is_correct = response_diagnosis == case.ground_truth

    tokens_per_sec = 0.0
    if wall_time_sec > 0 and fields["api_completion_tokens"] > 0:
        tokens_per_sec = fields["api_completion_tokens"] / wall_time_sec

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
        "api_backend": api_backend,
        "api_model": model,
        "response_model": fields["response_model"],
        "base_url": base_url,
        "reasoning_effort": reasoning_effort or "",
        "deepseek_thinking": deepseek_thinking,
        "prompt_pipeline": "modular" if case.is_modular else "single",
        "success": error == "",
        "error": error,
        "wall_time_sec": wall_time_sec,
        "diagnosis_error": diagnosis_error,
        "reformat_error": reformat_error,
        "diagnosis_wall_time_sec": diagnosis_wall_time_sec,
        "reformat_wall_time_sec": reformat_wall_time_sec,
        "prompt_tokens": fields["api_prompt_tokens"],
        "completion_tokens": fields["api_completion_tokens"],
        "total_tokens": fields["api_total_tokens"],
        "diagnosis_prompt_tokens": diagnosis_fields["api_prompt_tokens"],
        "diagnosis_completion_tokens": diagnosis_fields["api_completion_tokens"],
        "diagnosis_total_tokens": diagnosis_fields["api_total_tokens"],
        "reformat_prompt_tokens": reformat_fields["api_prompt_tokens"],
        "reformat_completion_tokens": reformat_fields["api_completion_tokens"],
        "reformat_total_tokens": reformat_fields["api_total_tokens"],
        "tokens_per_sec_wall": tokens_per_sec,
        "system_prompt_chars": len(case.system_prompt),
        "user_prompt_chars": len(case.user_prompt),
        "reformat_system_prompt_chars": len(case.reformat_system_prompt),
        "reformat_user_prompt_chars": len(case.reformat_user_prompt),
        "prompt_chars": len(case.full_prompt),
        "response_chars": len(raw_response),
        "finish_reason": fields["finish_reason"],
        "response_id": fields["response_id"],
        "reasoning_content": fields["reasoning_content"][:raw_response_max_chars],
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
        "raw_completion_json": fields["raw_completion_json"][:raw_completion_max_chars],
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
        description="Run SLM diagnostic JSONL prompts through HF Router or DeepSeek chat APIs."
    )
    parser.add_argument(
        "--jsonl_file",
        type=Path,
        default=Path(__file__).with_name("slm_prompts_for_ollama.jsonl"),
    )
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument(
        "--api_backend",
        "--backend",
        dest="api_backend",
        choices=["hf", "deepseek"],
        default="hf",
        help=(
            "API provider defaults. hf uses model=Qwen/Qwen3.5-9B:together, "
            "base_url=https://router.huggingface.co/v1, api_key_env=HF_TOKEN. "
            "deepseek uses model=deepseek-v4-pro, base_url=https://api.deepseek.com, "
            "api_key_env=DEEPSEEK_API_KEY. Explicit --model/--base_url/--api_key_env override these."
        ),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key_env", default=None)
    parser.add_argument("--runs", type=int, default=1, help="Number of repeats per JSONL prompt record.")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--reasoning_effort",
        default="high",
        help="DeepSeek-only reasoning effort. Use an empty string to omit it.",
    )
    parser.add_argument(
        "--disable_deepseek_thinking",
        action="store_true",
        help="DeepSeek-only: do not send extra_body={'thinking': {'type': 'enabled'}}.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Optional maximum generated tokens. Omit to use provider/model default.",
    )
    parser.add_argument("--raw_response_max_chars", type=int, default=20000)
    parser.add_argument("--raw_completion_max_chars", type=int, default=40000)
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tokens is not None and args.max_tokens < 0:
        args.max_tokens = None
    defaults = {
        "hf": {
            "model": "Qwen/Qwen3.5-9B:together",
            "base_url": "https://router.huggingface.co/v1",
            "api_key_env": "HF_TOKEN",
        },
        "deepseek": {
            "model": "deepseek-v4-pro",
            "base_url": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
    }
    backend_defaults = defaults[args.api_backend]
    model = args.model or backend_defaults["model"]
    base_url = args.base_url or backend_defaults["base_url"]
    api_key_env = args.api_key_env or backend_defaults["api_key_env"]
    reasoning_effort = args.reasoning_effort.strip() if args.reasoning_effort else None
    deepseek_thinking = args.api_backend == "deepseek" and not args.disable_deepseek_thinking

    cases = load_slm_jsonl_prompt_cases(args.jsonl_file, args.runs)
    client = make_client(base_url, api_key_env, args.timeout)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = args.output_csv or args.jsonl_file.parent / "logs" / f"slm_api_latency_{timestamp}.csv"
    run_started_at = datetime.now().isoformat(timespec="seconds")

    first_write = True
    total = len(cases)
    print(f"Loaded {total} prompt runs from {args.jsonl_file}")
    print(f"Using backend={args.api_backend} model={model} base_url={base_url} api_key_env={api_key_env}")

    for completed, case in enumerate(cases, start=1):
        diagnosis_completion, diagnosis_wall_time_sec, diagnosis_error = call_chat_api(
            client=client,
            case=case,
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_backend=args.api_backend,
            reasoning_effort=reasoning_effort,
            deepseek_thinking=deepseek_thinking,
        )
        completion = diagnosis_completion
        wall_time_sec = diagnosis_wall_time_sec
        error = diagnosis_error
        reformat_completion = None
        reformat_wall_time_sec = 0.0
        reformat_error = ""

        if case.is_modular and not diagnosis_error:
            diagnosis_output = extract_completion_fields(diagnosis_completion)["response_text"]
            reformat_user_prompt = case.render_reformat_user_prompt(diagnosis_output)
            reformat_completion, reformat_wall_time_sec, reformat_error = call_chat_api(
                client=client,
                case=case,
                model=model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                api_backend=args.api_backend,
                reasoning_effort=reasoning_effort,
                deepseek_thinking=deepseek_thinking,
                system_prompt=case.reformat_system_prompt,
                user_prompt=reformat_user_prompt,
            )
            completion = reformat_completion
            wall_time_sec = diagnosis_wall_time_sec + reformat_wall_time_sec
            error = reformat_error

        row = make_row(
            run_started_at=run_started_at,
            case=case,
            api_backend=args.api_backend,
            model=model,
            base_url=base_url,
            reasoning_effort=reasoning_effort if args.api_backend == "deepseek" else None,
            deepseek_thinking=deepseek_thinking,
            completion=completion,
            wall_time_sec=wall_time_sec,
            error=error,
            raw_response_max_chars=args.raw_response_max_chars,
            raw_completion_max_chars=args.raw_completion_max_chars,
            diagnosis_completion=diagnosis_completion,
            diagnosis_wall_time_sec=diagnosis_wall_time_sec,
            diagnosis_error=diagnosis_error,
            reformat_completion=reformat_completion,
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
