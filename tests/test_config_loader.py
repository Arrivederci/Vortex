"""ConfigLoader 单元测试，验证 YAML 解析逻辑。"""

from pathlib import Path

from vortex.config.loader import ConfigLoader


def test_config_loader_reads_yaml(tmp_path: Path):
    """Ensure YAML content is parsed correctly.

    中文说明：检查 YAML 文件能被准确读取为字典。
    """

    config_content = {"project_name": "demo", "value": 42}
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project_name: demo\nvalue: 42\n", encoding="utf-8")
    loader = ConfigLoader(config_path)
    data = loader.load()
    assert data == config_content
