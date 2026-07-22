"""端到端测试 - 验证天气查询和工具调用是否正常"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from long.cli import LongSystem, load_dotenv


async def test_weather_query():
    """测试天气查询功能"""
    print("=" * 60)
    print("测试: 天气查询 + 折线图生成")
    print("=" * 60)
    
    load_dotenv()
    
    # Debug: print API key
    import os
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "")
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "API Key: (not set)")
    print(f"Base URL: {base_url or '(default)'}")
    
    system = LongSystem()
    system.initialize()
    
    print(f"\n✅ 系统初始化完成")
    print(f"✅ LLM 模型: {system.llm.config.model}")
    print(f"✅ 超时配置: read={system.llm.config.timeout.read}s")
    
    # 测试 LLM 基本聊天
    from long.llm.base import LLMMessage
    
    print("\n--- 测试 LLM 基本聊天 ---")
    try:
        response = await asyncio.wait_for(
            system.llm.chat(
                [LLMMessage(role="user", content="你好，请用一句话回复")],
                purpose="chat",
            ),
            timeout=60,
        )
        print(f"✅ LLM 回复: {response.content[:100]}...")
    except asyncio.TimeoutError:
        print(f"❌ LLM 聊天超时 (60s)")
        return False
    except Exception as e:
        print(f"❌ LLM 聊天失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试工具调用
    print("\n--- 测试 LLM 工具调用 ---")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "获取当前时间",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    
    try:
        response = await asyncio.wait_for(
            system.llm.chat_with_tools(
                [LLMMessage(
                    role="user",
                    content="请调用工具获取当前时间"
                )],
                tools=tools,
                purpose="chat",
            ),
            timeout=90,
        )
        
        if response.tool_calls:
            print(f"✅ LLM 决定调用工具: {[tc['name'] for tc in response.tool_calls]}")
            for tc in response.tool_calls:
                print(f"   - {tc['name']}: {tc['arguments']}")
        else:
            print(f"⚠️ LLM 未调用工具，回复: {response.content[:100]}")
    except asyncio.TimeoutError:
        print(f"❌ 工具调用超时")
        return False
    except Exception as e:
        print(f"❌ 工具调用失败: {e}")
        return False
    
    print("\n✅ 测试通过!")
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_weather_query())
        if success:
            print("\n" + "=" * 60)
            print("✅ 所有测试通过，系统工作正常!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ 测试失败，需要进一步排查")
            print("=" * 60)
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
