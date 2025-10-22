"""配置加载模块，封装 YAML 文件解析逻辑并提供后备方案。"""

from __future__ import annotations

import pathlib
from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


class ConfigLoader:
    """Utility for reading YAML configuration files.

    中文说明
    -------
    该类用于读取 YAML 配置文件，如果缺少 `yaml` 依赖，则使用最小解析器。

    Parameters
    ----------
    path:
        Path to the YAML configuration file.
    """

    def __init__(self, path: str | pathlib.Path) -> None:
        """Initialize the loader with a path.

        中文说明：接受字符串或 ``Path`` 对象并转换为 ``Path`` 实例。
        """
        self.path = pathlib.Path(path)

    def load(self) -> Dict[str, Any]:
        """Load the YAML configuration file.

        中文说明
        -------
        读取 YAML 文本并根据环境选择解析方式。

        Returns
        -------
        dict
            Parsed configuration dictionary.
        """

        with self.path.open("r", encoding="utf-8") as fp:
            content = fp.read()
        if yaml is None:
            # 如果未安装 PyYAML，则回退到最小解析逻辑。
            return self._minimal_yaml_parse(content)
        return yaml.safe_load(content)

    def _minimal_yaml_parse(self, text: str) -> Dict[str, Any]:
        """Parse a minimal subset of YAML syntax.

        中文说明：提供在缺少 PyYAML 时的简单键值对解析能力。
        """
        data: Dict[str, Any] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                # 跳过空行与注释行。
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
