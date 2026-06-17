from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Any

import numpy as np
import requests


def call_ollama(prompt: str, model: str = "qwen3.5:4b", host: str = "http://localhost:11434") -> tuple[str, float]:
    start = time.time()
    response = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
        timeout=120,
    )
    elapsed = time.time() - start
    response.raise_for_status()
    return str(response.json()["response"]), elapsed


def parse_llm_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        repaired = match.group(0).replace("```json", "").replace("```", "")
        return json.loads(repaired)


def _drift_key(report: dict) -> str:
    drift = report.get("data_drift", {})
    return str(drift.get("corruption_type") or drift.get("estimated_type") or "unknown")


def heuristic_select_clients(client_reports: list[dict], clients_per_round: int) -> list[int]:
    norms = [float(r.get("update_stats", {}).get("update_norm", 0.0)) for r in client_reports]
    median_norm = float(np.median(norms)) if norms else 0.0
    drift_counts: Counter[str] = Counter()
    selected: list[int] = []
    remaining = list(client_reports)
    while remaining and len(selected) < clients_per_round:
        scored = []
        for r in remaining:
            local_eval = r.get("local_eval", {})
            progress = float(local_eval.get("loss_before", 0.0)) - float(local_eval.get("loss_after", local_eval.get("loss_before", 0.0)))
            entropy = float(r.get("label_entropy", 0.0))
            norm = float(r.get("update_stats", {}).get("update_norm", 0.0))
            cosine = float(r.get("update_stats", {}).get("cosine_to_global_update", 0.0))
            drift = _drift_key(r)
            diversity_bonus = 0.35 / (1.0 + drift_counts[drift])
            stability_penalty = abs(norm - median_norm)
            score = 2.0 * entropy + progress + 0.2 * cosine + diversity_bonus - 0.1 * stability_penalty
            scored.append((score, int(r["cid"]), r))
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        _, cid, report = scored[0]
        selected.append(cid)
        drift_counts[_drift_key(report)] += 1
        remaining = [r for r in remaining if int(r["cid"]) != cid]
    return selected


def select_clients_with_llm(
    prompt: str,
    client_reports: list[dict],
    clients_per_round: int,
    model: str = "qwen3.5:4b",
    host: str = "http://localhost:11434",
) -> dict:
    try:
        raw, inference_time = call_ollama(prompt, model=model, host=host)
        parsed = parse_llm_json(raw)
        valid_cids = {int(r["cid"]) for r in client_reports}
        selected = [int(x) for x in parsed.get("selected_clients", []) if int(x) in valid_cids]
        if len(selected) != clients_per_round:
            selected = heuristic_select_clients(client_reports, clients_per_round)
            parsed["repair_reason"] = "invalid selected_clients length or cid"
        return {
            "selected_clients": selected,
            "raw_response": raw,
            "parsed": parsed,
            "llm_inference_time_sec": inference_time,
            "used_fallback": False,
            "reason": parsed.get("reason", ""),
        }
    except Exception as exc:
        selected = heuristic_select_clients(client_reports, clients_per_round)
        return {
            "selected_clients": selected,
            "raw_response": "",
            "parsed": {},
            "llm_inference_time_sec": 0.0,
            "used_fallback": True,
            "error": str(exc),
            "reason": "heuristic fallback",
        }
