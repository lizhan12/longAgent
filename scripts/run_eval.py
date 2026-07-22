#!/usr/bin/env python3
"""CI 评估入口

用法:
    python scripts/run_eval.py [--dataset PATH] [--output PATH] [--strict] [--category CATEGORY]

功能:
    评估模块暂未实现，此脚本保留作为占位。
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Long Agent EvalOps CI Runner")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to golden dataset JSONL file",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to output report JSON file",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Strict mode (higher pass threshold)",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Only run cases in this category",
    )
    args = parser.parse_args()

    print("EvalOps 评估模块暂未实现，请使用其他验证方式。")
    sys.exit(0)


if __name__ == "__main__":
    main()