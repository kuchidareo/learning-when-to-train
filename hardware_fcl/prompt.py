from __future__ import annotations

import json


SYSTEM_PROMPT = """
You are selecting clients for federated continual learning.
A good client subset should:
1. Cover the currently seen classes as evenly as possible.
2. Include clients containing recently introduced or frontier classes so the model adapts to new concepts.
3. Include some clients containing previously seen classes to reduce forgetting.
4. Avoid selecting many clients with the same dominant labels.
5. Cover diverse corruption conditions and severity levels.
6. Prefer clients with positive local learning progress.
7. Avoid selecting too many unstable clients with extreme update norms or conflicting update directions.
You must output strict JSON only.
Output schema:
{
  "selected_clients": [0, 1, 2],
  "reason": "short explanation",
  "selection_summary": {
    "class_coverage": "short text",
    "drift_diversity": "short text",
    "stability": "short text"
  }
}
""".strip()


def build_client_selection_prompt(
    round_id: int,
    global_task: int,
    clients_per_round: int,
    seen_classes: list[int],
    client_reports: list[dict],
) -> str:
    payload = {
        "round_id": round_id,
        "global_task": global_task,
        "clients_per_round": clients_per_round,
        "seen_classes": seen_classes,
        "candidate_client_reports": client_reports,
    }
    return SYSTEM_PROMPT + "\n\nInput:\n" + json.dumps(payload, indent=2, sort_keys=True)
