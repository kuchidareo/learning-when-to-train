from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable


class CSVLogger:
    def __init__(self, output_dir: str | Path, config: dict | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headers: dict[str, list[str]] = {}
        if config is not None:
            self.save_config(config)

    @classmethod
    def timestamped(cls, root: str | Path, config: dict | None = None) -> "CSVLogger":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls(Path(root) / stamp, config=config)

    def save_config(self, config: dict) -> None:
        with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)

    def log_row(self, filename: str, row: dict) -> None:
        path = self.output_dir / filename
        keys = list(row.keys())
        if filename not in self.headers:
            self.headers[filename] = keys
        header = self.headers[filename]
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({k: _stringify(row.get(k, "")) for k in header})

    def log_rows(self, filename: str, rows: Iterable[dict]) -> None:
        for row in rows:
            self.log_row(filename, row)


def _stringify(value) -> str | int | float | bool:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value
