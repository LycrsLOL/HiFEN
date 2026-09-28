from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BEIJING_TZ = timezone(timedelta(hours=8))


class BeijingFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, BEIJING_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


def beijing_timestamp() -> str:
    return datetime.now(BEIJING_TZ).strftime("%Y%m%d_%H%M%S")


def safe_filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return name.strip("._") or "run"


def setup_logging(
    log_dir: str | os.PathLike[str] | None = None,
    name: str = "h_resannogat",
    *,
    timestamped: bool = True,
) -> Path | None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_path: Path | None = None
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_name = safe_filename(name)
        if timestamped:
            log_name = f"{log_name}_{beijing_timestamp()}"
        log_path = Path(log_dir) / f"{log_name}.log"
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    formatter = BeijingFormatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
    return log_path


def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def disk_free_gib(path: str | os.PathLike[str]) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def valid_text(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return text.lower() not in {"", "nan", "none", "null"}
