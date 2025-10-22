from pathlib import Path

from vortex.config.loader import ConfigLoader


def test_config_loader_reads_yaml(tmp_path: Path):
    config_content = {"project_name": "demo", "value": 42}
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project_name: demo\nvalue: 42\n", encoding="utf-8")
    loader = ConfigLoader(config_path)
    data = loader.load()
    assert data == config_content
