"""预测入口脚本，通过命令行参数加载配置并执行样本外推理。"""

from __future__ import annotations

import argparse
from pathlib import Path

from vortex.orchestrator import run_prediction_workflow


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="传入 YAML 配置文件路径并启动 Vortex 模型预测流程。"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="必填：预测配置文件路径，支持相对或绝对路径。",
    )
    return parser.parse_args()


def main() -> None:
    """脚本主函数，负责触发预测并提示输出位置。"""

    args = _parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    output_path = run_prediction_workflow(str(config_path))
    # 中文说明：控制台打印预测结果的保存路径，方便集成系统进一步处理。
    print(str(output_path.resolve()))


if __name__ == "__main__":
    main()
