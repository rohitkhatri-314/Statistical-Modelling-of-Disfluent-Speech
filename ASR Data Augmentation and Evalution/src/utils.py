import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def ensure_dir(path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(
    path: str,
    records: List[Dict[str, Any]],
    mode: str = "w",
) -> None:
    ensure_dir(Path(path).parent)
    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
