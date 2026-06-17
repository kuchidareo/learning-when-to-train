from __future__ import annotations

import argparse
import ast
import csv
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from socket import timeout as SocketTimeout
from typing import Any


def _duration_ns_to_sec(value: Any) -> float:
    try:
        return float(value) / 1_000_000_000.0
    except (TypeError, ValueError):
        return 0.0


def load_prompts(prompts_file: Path) -> list[tuple[str, str]]:
    """
    Read prompt strings from a Python prompt file without executing it.

    This supports files where `prompts = [prompt_1, ...]` appears before the
    actual prompt string assignments.
    """
    source = prompts_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(prompts_file))

    prompt_values: dict[str, str] = {}
    prompt_order: list[str] = []

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.startswith("prompt_"):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    prompt_values[target.id] = node.value.value

            if isinstance(target, ast.Name) and target.id == "prompts":
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    names = []
                    for item in node.value.elts:
                        if isinstance(item, ast.Name):
                            names.append(item.id)
                        elif isinstance(item, ast.Constant) and isinstance(item.value, str):
                            direct_name = f"prompt_{len(names) + 1}"
                            prompt_values[direct_name] = item.value
                            names.append(direct_name)
                    prompt_order = names

    if not prompt_order:
        prompt_order = sorted(
            prompt_values,
            key=lambda name: (
                0,
                int(name.rsplit("_", 1)[-1]),
            )
            if name.rsplit("_", 1)[-1].isdigit()
            else (1, name),
        )

    prompts: list[tuple[str, str]] = []
    missing: list[str] = []
    for name in prompt_order:
        prompt = prompt_values.get(name)
        if prompt is None:
            missing.append(name)
        else:
            prompts.append((name, prompt))

    if missing:
        raise ValueError(f"Missing prompt definitions: {', '.join(missing)}")
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_file}")
    return prompts


def call_ollama(
    prompt: str,
    model: str,
    host: str,
    timeout: float,
    temperature: float,
    num_predict: int | None,
    keep_alive: str,
) -> tuple[dict[str, Any] | None, float, str]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"temperature": temperature},
    }
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
    prompt_name: str,
    prompt_index: int,
    repeat_index: int,
    model: str,
    host: str,
    prompt: str,
    response_json: dict[str, Any] | None,
    wall_time_sec: float,
    error: str,
    raw_response_max_chars: int,
) -> dict[str, Any]:
    response_json = response_json or {}
    prompt_tokens = int(response_json.get("prompt_eval_count") or 0)
    completion_tokens = int(response_json.get("eval_count") or 0)
    eval_duration_sec = _duration_ns_to_sec(response_json.get("eval_duration"))
    tokens_per_sec = completion_tokens / eval_duration_sec if eval_duration_sec > 0 else 0.0
    raw_response = str(response_json.get("response") or "")

    return {
        "run_started_at": run_started_at,
        "prompt_name": prompt_name,
        "prompt_index": prompt_index,
        "repeat_index": repeat_index,
        "model": model,
        "host": host,
        "success": error == "",
        "error": error,
        "wall_time_sec": wall_time_sec,
        "ollama_total_duration_sec": _duration_ns_to_sec(response_json.get("total_duration")),
        "load_duration_sec": _duration_ns_to_sec(response_json.get("load_duration")),
        "prompt_eval_duration_sec": _duration_ns_to_sec(response_json.get("prompt_eval_duration")),
        "eval_duration_sec": eval_duration_sec,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "tokens_per_sec": tokens_per_sec,
        "prompt_chars": len(prompt),
        "response_chars": len(raw_response),
        "done_reason": response_json.get("done_reason", ""),
        "raw_response": raw_response[:raw_response_max_chars],
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
    parser = argparse.ArgumentParser(description="Run prompt latency/token benchmark against Ollama.")
    parser.add_argument("--prompts_file", type=Path, default=Path(__file__).with_name("prompts.py"))
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--runs", type=int, default=10, help="Number of repeats per prompt.")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--num_predict",
        type=int,
        default=None,
        help="Optional maximum generated tokens. Omit to use the Ollama/model default.",
    )
    parser.add_argument("--keep_alive", default="10m")
    parser.add_argument("--raw_response_max_chars", type=int, default=4000)
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_predict is not None and args.num_predict < 0:
        args.num_predict = None
    prompts = load_prompts(args.prompts_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = args.output_csv or args.prompts_file.parent / "logs" / f"ollama_latency_{timestamp}.csv"
    run_started_at = datetime.now().isoformat(timespec="seconds")

    first_write = True
    total = len(prompts) * args.runs
    completed = 0

    for prompt_index, (prompt_name, prompt) in enumerate(prompts, start=1):
        for repeat_index in range(1, args.runs + 1):
            response_json, wall_time_sec, error = call_ollama(
                prompt=prompt,
                model=args.model,
                host=args.host,
                timeout=args.timeout,
                temperature=args.temperature,
                num_predict=args.num_predict,
                keep_alive=args.keep_alive,
            )

            row = make_row(
                run_started_at=run_started_at,
                prompt_name=prompt_name,
                prompt_index=prompt_index,
                repeat_index=repeat_index,
                model=args.model,
                host=args.host,
                prompt=prompt,
                response_json=response_json,
                wall_time_sec=wall_time_sec,
                error=error,
                raw_response_max_chars=args.raw_response_max_chars,
            )
            write_rows_incrementally(output_csv, [row], append=args.append or not first_write)
            first_write = False
            completed += 1

            status = "ok" if not error else "error"
            message = f"[{completed}/{total}] {prompt_name} run={repeat_index} {status} wall={wall_time_sec:.3f}s"
            if error:
                message += f" error={error}"
            print(message)

    print(f"Wrote CSV: {output_csv}")


if __name__ == "__main__":
    main()
