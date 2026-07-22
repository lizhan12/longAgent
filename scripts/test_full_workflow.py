#!/usr/bin/env python3
"""直接测试 LLM 的完整工具调用流程"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from long.cli import LongSystem, load_dotenv
from long.llm.base import LLMMessage


async def test_full_weather_workflow():
    """测试完整的天气查询工作流"""
    print("=" * 60)
    print("测试: 完整天气查询工作流")
    print("=" * 60)
    
    load_dotenv()
    
    import os
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "")
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "API Key: (not set)")
    print(f"Base URL: {base_url or '(default)'}")
    print(f"LLM Model: deepseek-ai/DeepSeek-V4-Flash")
    
    system = LongSystem()
    system.initialize()
    
    print(f"\n✅ 系统初始化完成")
    print(f"✅ LLM 模型: {system.llm.config.model}")
    
    # 构建工具列表
    tools = system._gather_tools()
    print(f"✅ 可用工具数: {len(tools)}")
    
    # 模拟用户输入
    user_message = "余杭区未来7天天气预报，生成带折线图的报告"
    
    print(f"\n--- 模拟用户输入: {user_message} ---")
    
    # 构建消息
    system_prompt = system._build_system_prompt()
    history_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    
    llm_messages = [
        LLMMessage(role=m["role"], content=m.get("content", ""))
        for m in history_msgs
    ]
    
    print(f"\n发送请求到 LLM...")
    
    try:
        response = await asyncio.wait_for(
            system.llm.chat_with_tools(
                llm_messages,
                system._clean_tools_for_api(tools),
                purpose="chat",
            ),
            timeout=150,
        )
        
        print(f"\n✅ LLM 响应成功!")
        print(f"   Finish reason: {response.finish_reason}")
        
        if response.tool_calls:
            print(f"\n🔧 LLM 决定调用 {len(response.tool_calls)} 个工具:")
            for tc in response.tool_calls:
                print(f"   - {tc['name']}: {tc['arguments']}")
                
            # 执行工具调用
            print(f"\n--- 执行工具调用 ---")
            for tc in response.tool_calls:
                tool_name = tc["name"]
                arguments = tc["arguments"]
                
                print(f"🔧 执行 {tool_name}({arguments})...")
                tool_result = await system._execute_tool(tool_name, arguments)
                
                # 显示结果预览
                preview = tool_result[:150].replace("\n", " ")
                if len(tool_result) > 150:
                    preview += "..."
                print(f"   ✅ 结果: {preview}")
                
                # 添加工具结果到消息历史
                history_msgs.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": str(arguments),
                        },
                    }],
                })
                history_msgs.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tc["id"],
                })
            
            # 再次调用 LLM 获取最终回复
            print(f"\n--- 获取 LLM 最终回复 ---")
            llm_messages2 = [
                LLMMessage(
                    role=m["role"],
                    content=m.get("content", ""),
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                )
                for m in history_msgs
            ]
            
            final_response = await asyncio.wait_for(
                system.llm.chat_with_tools(
                    llm_messages2,
                    system._clean_tools_for_api(tools),
                    purpose="chat",
                ),
                timeout=150,
            )
            
            print(f"\n✅ 最终回复:")
            if final_response.content:
                print(f"   {final_response.content[:500]}")
            if final_response.tool_calls:
                print(f"\n🔧 LLM 还要调用工具:")
                for tc in final_response.tool_calls:
                    print(f"   - {tc['name']}: {tc['arguments']}")
        else:
            print(f"\n💬 LLM 直接回复:")
            print(f"   {response.content[:300]}")
        
        print(f"\n" + "=" * 60)
        print(f"✅ 测试通过!")
        print("=" * 60)
        return True
        
    except asyncio.TimeoutError:
        print(f"\n❌ 请求超时 (150s)")
        return False
    except Exception as e:
        print(f"\n❌ 请求失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(test_full_weather_workflow())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
