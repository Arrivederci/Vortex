"""Pytest 配置文件，用于调整导入路径便于测试。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # 将项目根目录加入 sys.path，确保测试能够导入包。
    sys.path.insert(0, str(ROOT))
