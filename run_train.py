"""训练入口脚本，通过命令行参数加载配置并执行完整工作流。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vortex.orchestrator import run_workflow


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="传入 YAML 配置文件路径并启动 Vortex 模型训练流程。"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="必填：配置文件路径，支持相对或绝对路径。",
    )
    parser.add_argument(
        "--output-dir",
        default="./artifacts",
        help="选填：指标及模型输出目录，默认 ./artifacts。",
    )
    return parser.parse_args()


def main() -> None:
    """脚本主函数，负责触发训练并输出评估指标。"""

    args = _parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        # 中文说明：在正式执行前先检查配置文件是否存在，避免训练过程运行到一半才失败。
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    metrics = run_workflow(str(config_path), output_dir=args.output_dir)
    # 中文说明：将结果格式化为 JSON 方便后续自动化系统解析。
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
