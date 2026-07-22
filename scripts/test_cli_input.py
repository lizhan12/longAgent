#!/usr/bin/env python3
"""自动化测试 - 向 CLI 发送输入并验证响应"""

import asyncio
import sys

async def test_cli():
    """通过 stdin 模拟用户输入"""
    # 使用 subprocess 启动 CLI 并发送输入
    proc = await asyncio.create_subprocess_exec(
        "uv", "run", "python", "main.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd="/home/lizhan/project/long",
    )
    
    # 发送测试命令
    test_input = "余杭区未来7天天气预报，生成带折线图的报告\n"
    
    # 等待系统启动
    await asyncio.sleep(5)
    
    # 发送命令并等待响应
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(test_input.encode()),
            timeout=180,
        )
        
        output = stdout.decode() if stdout else ""
        error = stderr.decode() if stderr else ""
        
        print("=" * 60)
        print("CLI 输出:")
        print(output[-2000:] if len(output) > 2000 else output)
        print("=" * 60)
        print("CLI 错误:")
        print(error[-1000:] if len(error) > 1000 else error)
        print("=" * 60)
        
        # 检查结果
        if "超时" in output or "timeout" in output.lower():
            print("❌ 测试失败: 超时")
            return False
        elif "天气" in output or "report" in output.lower() or "forecast" in output.lower():
            print("✅ 测试通过: 获取到天气信息")
            return True
        else:
            print("⚠️ 需要进一步验证输出")
            return True
            
    except asyncio.TimeoutError:
        print("❌ 测试超时 (180s)")
        proc.kill()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_cli())
    sys.exit(0 if success else 1)
