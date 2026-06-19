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

    @property
    def full_prompt(self) -> str:
        if self.system_prompt:
            return f"{self.system_prompt}\n\n{self.user_prompt}"
        return self.user_prompt


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

            scenario_id = str(record.get("scenario_id", ""))
            trial_index = str(record.get("trial_index", ""))
            prompt_name = f"{scenario_id}_trial_{trial_index}" if scenario_id else f"jsonl_line_{source_line}"
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
                        summary_json=json.dumps(record.get("summary", {}), ensure_ascii=False),
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


def build_messages(case: PromptCase) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if case.system_prompt:
        messages.append({"role": "system", "content": case.system_prompt})
    messages.append({"role": "user", "content": case.user_prompt})
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
) -> tuple[Any | None, float, str]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": build_messages(case),
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

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
            "raw_completion_json": "",
        }

    response_text = ""
    finish_reason = ""
    if getattr(completion, "choices", None):
        choice = completion.choices[0]
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
        message = getattr(choice, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        response_text = str(content or "")

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
        "raw_completion_json": raw_completion_json,
    }


def make_row(
    *,
    run_started_at: str,
    case: PromptCase,
    model: str,
    base_url: str,
    completion: Any | None,
    wall_time_sec: float,
    error: str,
    raw_response_max_chars: int,
    raw_completion_max_chars: int,
) -> dict[str, Any]:
    fields = extract_completion_fields(completion)
    raw_response = fields["response_text"]
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
        "api_model": model,
        "response_model": fields["response_model"],
        "base_url": base_url,
        "success": error == "",
        "error": error,
        "wall_time_sec": wall_time_sec,
        "prompt_tokens": fields["api_prompt_tokens"],
        "completion_tokens": fields["api_completion_tokens"],
        "total_tokens": fields["api_total_tokens"],
        "tokens_per_sec_wall": tokens_per_sec,
        "system_prompt_chars": len(case.system_prompt),
        "user_prompt_chars": len(case.user_prompt),
        "prompt_chars": len(case.full_prompt),
        "response_chars": len(raw_response),
        "finish_reason": fields["finish_reason"],
        "response_id": fields["response_id"],
        "response_diagnosis": response_diagnosis,
        "response_confidence": response_confidence,
        "is_correct": is_correct,
        "summary_json": case.summary_json,
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
        description="Run SLM diagnostic JSONL prompts through the Hugging Face Router chat API."
    )
    parser.add_argument(
        "--jsonl_file",
        type=Path,
        default=Path(__file__).with_name("slm_prompts_for_ollama.jsonl"),
    )
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B:together")
    parser.add_argument("--base_url", default="https://router.huggingface.co/v1")
    parser.add_argument("--api_key_env", default="HF_TOKEN")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeats per JSONL prompt record.")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--temperature", type=float, default=0.2)
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

    cases = load_slm_jsonl_prompt_cases(args.jsonl_file, args.runs)
    client = make_client(args.base_url, args.api_key_env, args.timeout)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = args.output_csv or args.jsonl_file.parent / "logs" / f"slm_api_latency_{timestamp}.csv"
    run_started_at = datetime.now().isoformat(timespec="seconds")

    first_write = True
    total = len(cases)
    print(f"Loaded {total} prompt runs from {args.jsonl_file}")
    print(f"Using model={args.model} base_url={args.base_url}")

    for completed, case in enumerate(cases, start=1):
        completion, wall_time_sec, error = call_chat_api(
            client=client,
            case=case,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        row = make_row(
            run_started_at=run_started_at,
            case=case,
            model=args.model,
            base_url=args.base_url,
            completion=completion,
            wall_time_sec=wall_time_sec,
            error=error,
            raw_response_max_chars=args.raw_response_max_chars,
            raw_completion_max_chars=args.raw_completion_max_chars,
        )
        write_rows_incrementally(output_csv, [row], append=args.append or not first_write)
        first_write = False

        status = "ok" if not error else "error"
        diagnosis = f" diagnosis={row['response_diagnosis']}" if row["response_diagnosis"] else ""
        message = (
            f"[{completed}/{total}] {case.prompt_name} run={case.repeat_index} "
            f"{status} wall={wall_time_sec:.3f}s{diagnosis}"
        )
        if error:
            message += f" error={error}"
        print(message)

    print(f"Wrote CSV: {output_csv}")


if __name__ == "__main__":
    main()
