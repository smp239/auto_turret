# live_tuning.py  (robust version)
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple


class RuntimeParamWatcher:
    """Watch a JSON file and hot-reload its contents when it changes."""

    def __init__(self, path: str | Path = "runtime_params.json") -> None:
        self.path = Path(path).expanduser().resolve()
        self._stamp: Tuple[float, int] = (0.0, -1)  # (mtime, size)
        self.params: Dict[str, Any] = {}

        print(f"[Runtime] Watching: {self.path}")
        self._load(initial=True)

    # ------------------------------------------------------------------
    #   Internal helpers
    # ------------------------------------------------------------------
    def _load(self, *, initial: bool = False) -> None:
        try:
            with self.path.open("r", encoding="utf-8") as fp:
                self.params = json.load(fp)
            stat = self.path.stat()
            self._stamp = (stat.st_mtime, stat.st_size)
            if not initial:
                print(f"[Runtime] Reloaded parameters from {self.path}")
        except FileNotFoundError:
            if initial:
                print(
                    f"[Runtime] {self.path} not found – live-tuning disabled "
                    "(create the file to enable)."
                )
            else:
                print(f"[Runtime] {self.path} was deleted – keeping old params.")
        except json.JSONDecodeError as exc:
            print(f"[Runtime] JSON error in {self.path}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[Runtime] Failed to reload {self.path}: {exc}")

    # ------------------------------------------------------------------
    #   Public API
    # ------------------------------------------------------------------
    def maybe_reload(self) -> bool:
        """
        If the watched file changed since the last call reload it and
        return **True**, else return **False**.
        """
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return False

        mtime, fsize = self._stamp
        # Some filesystems only update timestamps in 1- or 2-second ticks,
        # so we treat any change >=1 s *or* size change as “modified”.
        if stat.st_size != fsize or stat.st_mtime - mtime >= 1.0:
            self._load()
            return True
        return False

    def get(self, key: str, default: Any | None = None) -> Any:
        return self.params.get(key, default)
