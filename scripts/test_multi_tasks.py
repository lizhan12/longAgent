"""多任务真实测试 — 通过 CLI stdin 管道模拟"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

TASKS = [
    {"id": "T1", "name": "简单搜索", "prompt": "今天杭州天气怎么样"},
    {"id": "T2", "name": "代码生成", "prompt": "用Python实现归并排序，并测试"},
    {"id": "T3", "name": "图表+中文(默认md)", "prompt": "搜索杭州未来一周天气，生成一份包含气温趋势图的Markdown报告"},
    {"id": "T4", "name": "指定docx格式", "prompt": "帮我生成一份Python排序算法对比的Word文档报告"},
]

OUTPUT_DIR = Path("./workspace/output")


def get_output_files():
    if not OUTPUT_DIR.exists():
        return {}
    files = {}
    for f in OUTPUT_DIR.iterdir():
        if f.is_file() and not f.name.startswith("."):
            files[f.name] = f.stat().st_mtime
    return files


def run_task(prompt, timeout=300):
    try:
        result = subprocess.run(
            [sys.executable, "main.py"],
            input=prompt + "\n/exit\n",
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(__file__).parent.parent),
            env={
                **__import__("os").environ,
                "PYTHONIOENCODING": "utf-8",
            },
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        return stdout, stderr, 0
    except subprocess.TimeoutExpired as e:
        return e.stdout or "", e.stderr or "", 1
    except Exception as e:
        return "", str(e), 2


def main():
    print("=" * 60)
    print("Long 多任务真实测试")
    print("=" * 60)

    before_files = get_output_files()
    results = []

    for task in TASKS:
        print(f"\n{'='*60}")
        print(f"[{task['id']}] {task['name']}: {task['prompt']}")
        print(f"{'='*60}")

        start = time.time()
        stdout, stderr, exit_code = run_task(task["prompt"])
        elapsed = time.time() - start

        output_lines = stdout.strip().split("\n")
        meaningful = [l for l in output_lines if l.strip() and not l.strip().startswith(("⏳", "🔧", "📋"))]
        output_text = "\n".join(meaningful[:30])

        passed = len(output_text) > 50
        print(f"\n输出 ({elapsed:.1f}s, exit={exit_code}):")
        print(output_text[:800])

        if stderr and "error" in stderr.lower():
            print(f"\nstderr: {stderr[:300]}")

        results.append({
            "id": task["id"],
            "name": task["name"],
            "passed": passed,
            "elapsed": elapsed,
            "output_len": len(output_text),
        })

    after_files = get_output_files()
    new_files = set(after_files.keys()) - set(before_files.keys())
    updated_files = {k for k in after_files if k in before_files and after_files[k] > before_files[k]}

    print(f"\n\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")

    for r in results:
        status = "✅" if r["passed"] else "❌"
        print(f"  {status} [{r['id']}] {r['name']}: {r['elapsed']:.1f}s, {r['output_len']}字符")

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\n通过率: {passed}/{total} ({passed/total:.0%})")

    print(f"\n--- 新增/更新文件 ---")
    for fname in sorted(new_files | updated_files):
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            size = fpath.stat().st_size
            print(f"  {size:>8}B  {fname}")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
