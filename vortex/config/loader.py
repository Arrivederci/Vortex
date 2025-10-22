from __future__ import annotations

import pathlib
from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


class ConfigLoader:
    """Utility for reading YAML configuration files.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.
    """

    def __init__(self, path: str | pathlib.Path) -> None:
        self.path = pathlib.Path(path)

    def load(self) -> Dict[str, Any]:
        """Load the YAML configuration file.

        Returns
        -------
        dict
            Parsed configuration dictionary.
        """

        with self.path.open("r", encoding="utf-8") as fp:
            content = fp.read()
        if yaml is None:
            return self._minimal_yaml_parse(content)
        return yaml.safe_load(content)

    def _minimal_yaml_parse(self, text: str) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'")
            if value.isdigit():
                data[key] = int(value)
            elif value.lower() in {"true", "false"}:
                data[key] = value.lower() == "true"
            else:
                data[key] = value
        return data
