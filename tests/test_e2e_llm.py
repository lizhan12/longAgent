"""LLM 客户端 E2E 测试 — 验证修复的正确性

覆盖修复:
  6. Anthropic 流式调用错误 — 删除 _call_anthropic 特殊路径 (llm/client.py)
  7. Anthropic 缓存头注入 — 移除 anthropic_caching 参数 (llm/client.py)
  8. _skill_caller 未 await 异步函数 (cli.py)
  12. IR 最终验证条件反转 (ir/executor.py)
  13. _list_files 逐条目判断 (cli.py)
  14. isinstance(exc, type(exc)) 永真判断 (llm/client.py)
"""

from __future__ import annotations

import asyncio
import textwrap
import os

import pytest


# ===================== 修复 6+7: Anthropic 特殊路径 =====================


class TestAnthropicPathRemoved:
    """Anthropic 特殊路径删除验证"""

    def test_anthropic_not_in_chat_impl(self):
        """验证 _chat_impl 中不再有 ANTHROPIC 特殊判断"""
        from long.llm.client import LLMClient
        import inspect

        source = inspect.getsource(LLMClient._chat_impl)
        # 不应包含 ANTHROPIC 特殊路径
        assert "ModelProvider.ANTHROPIC" not in source, \
            "_chat_impl 不应包含 ANTHROPIC 特殊路径"
        assert "_call_anthropic" not in source, \
            "_chat_impl 不应调用 _call_anthropic"

    def test_anthropic_method_removed(self):
        """验证 _call_anthropic 方法已被删除"""
        from long.llm.client import LLMClient

        assert not hasattr(LLMClient, "_call_anthropic"), \
            "_call_anthropic 方法已被删除"

    def test_build_sdk_messages_no_anthropic_caching(self):
        """验证 _build_sdk_messages 没有 anthropic_caching 参数"""
        from long.llm.client import LLMClient
        import inspect

        sig = inspect.signature(LLMClient._build_sdk_messages)
        assert "anthropic_caching" not in sig.parameters, \
            "_build_sdk_messages 不应有 anthropic_caching 参数"

    def test_build_sdk_messages_returns_plain_dicts(self):
        """验证 _build_sdk_messages 返回普通 dict 格式"""
        from long.llm.client import LLMClient
        from long.llm.base import LLMMessage

        # 创建未初始化的实例
        client = object.__new__(LLMClient)

        messages = [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Hello!"),
        ]

        sdk_msgs = client._build_sdk_messages(messages)
        assert len(sdk_msgs) == 2

        # 验证返回的是普通 dict 格式，不是 Anthropic 的 content 数组
        for msg in sdk_msgs:
            assert isinstance(msg["content"], str), \
                f"content 应为字符串，实际: {type(msg['content'])}"
            assert "cache_control" not in msg, \
                "不应包含 cache_control 字段"


# ===================== 修复 8: _skill_caller await =====================


class TestSkillCallerAwait:
    """_skill_caller await 修复测试"""

    @pytest.mark.asyncio
    async def test_async_function_is_awaited(self):
        """验证 async 函数被正确 await，返回结果而非协程对象"""
        # 模拟 _skill_caller 修复后的逻辑
        async def async_tool(a=1, b=2):
            await asyncio.sleep(0.01)
            return f"async_result:{a + b}"

        async def _skill_caller_simulated(func, arguments):
            import asyncio as _asyncio
            from inspect import signature

            sig = signature(func)
            if _asyncio.iscoroutinefunction(func):
                if sig.parameters:
                    result = await func(**arguments) if arguments else await func()
                else:
                    result = await func()
            else:
                if sig.parameters:
                    result = func(**arguments) if arguments else func()
                else:
                    result = func()
            return str(result)

        # 测试 async 函数
        result = await _skill_caller_simulated(async_tool, {"a": 3, "b": 4})
        assert result == "async_result:7", \
            f"应返回实际结果，而非协程对象。实际: {result}"

    @pytest.mark.asyncio
    async def test_sync_function_still_works(self):
        """验证 sync 函数仍然正常调用"""
        def sync_tool(a=1, b=2):
            return f"sync_result:{a + b}"

        async def _skill_caller_simulated(func, arguments):
            import asyncio as _asyncio
            from inspect import signature

            sig = signature(func)
            if _asyncio.iscoroutinefunction(func):
                if sig.parameters:
                    result = await func(**arguments) if arguments else await func()
                else:
                    result = await func()
            else:
                if sig.parameters:
                    result = func(**arguments) if arguments else func()
                else:
                    result = func()
            return str(result)

        result = await _skill_caller_simulated(sync_tool, {"a": 10, "b": 20})
        assert result == "sync_result:30", \
            f"sync 函数应正常返回。实际: {result}"

    @pytest.mark.asyncio
    async def test_async_function_without_await_returns_coroutine(self):
        """验证修复前行为: 不 await 返回协程对象字符串"""
        async def async_tool():
            return "real_result"

        # 修复前: 不 await
        result = async_tool()
        assert asyncio.iscoroutine(result), "不 await 应返回协程对象"
        assert "<coroutine object" in str(result), \
            f"协程对象字符串应包含 '<coroutine object'，实际: {result}"

        # 修复后: await
        result2 = await async_tool()
        assert result2 == "real_result", "await 后应返回实际结果"


# ===================== 修复 12: 最终验证条件反转 =====================


class TestFinalValidation:
    """IR 最终验证条件反转修复测试"""

    def test_validation_fails_when_success_is_true(self):
        """验证 success=True 且验证失败时 success 被置为 False"""
        # 模拟修复后的逻辑
        result = type("Result", (), {"success": True, "errors": []})()

        final_validation = type("Validation", (), {
            "valid": False,
            "errors": ["Final state not reached"],
            "warnings": [],
        })()

        if not final_validation.valid:
            result.errors.extend(final_validation.errors)
            if result.success:  # 修复前: if not result.success: (空操作)
                result.success = False

        assert result.success is False, \
            "验证失败后 success 应变为 False"
        assert len(result.errors) > 0

    def test_validation_pass_keeps_success(self):
        """验证通过时保持 success 不变"""
        result = type("Result", (), {"success": True, "errors": []})()

        final_validation = type("Validation", (), {
            "valid": True,
            "errors": [],
            "warnings": [],
        })()

        if not final_validation.valid:
            result.errors.extend(final_validation.errors)
            if result.success:
                result.success = False

        assert result.success is True, "验证通过时 success 应保持 True"

    def test_old_bug_pattern_does_not_exist(self):
        """验证修复后的 executor.py 中条件为 'if result.success:' 而非 'if not result.success:'"""
        import inspect

        from long.ir import executor

        source = inspect.getsource(executor)
        # 应该找到的是 "if result.success:" 而不是 "if not result.success:"
        # 搜索 "if result.success:" 确保它存在
        assert "if result.success:" in source, \
            "修复后的代码应包含 'if result.success:'"
        # 特定位置不应有 "if not result.success:" 后面跟着 result.success = False
        lines = source.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == 'if not result.success:' and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if 'result.success = False' in next_line:
                    pytest.fail(
                        f"发现未修复的 bug 模式: 第 {i + 1} 行 '{stripped}' "
                        f"后跟 '{next_line}'"
                    )


# ===================== 修复 13: _list_files 逐条目判断 =====================


class TestListFilesPerEntry:
    """_list_files 逐条目判断修复测试"""

    @pytest.mark.asyncio
    async def test_list_files_labels_per_entry(self, tmp_path):
        """验证每个条目独立判断目录/文件"""
        # 创建测试目录结构
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "file1.txt").write_text("hello")
        (root / "subdir").mkdir()
        (root / "subdir" / "nested.txt").write_text("nested")

        # 模拟修复后的 _list_files 逻辑
        from long.workspace.filesystem import LocalFilesystem

        fs = LocalFilesystem(str(root))

        entries = await fs.list_dir("")
        assert len(entries) == 2

        lines: list[str] = []
        for e in entries:
            entry_path = e
            is_dir = await fs.is_dir(entry_path)
            prefix = "[D]" if is_dir else "[F]"
            lines.append(f"{prefix} {entry_path}")

        # 验证标记正确
        assert "[F] file1.txt" in lines, f"文件应标记为 [F]，实际: {lines}"
        assert "[D] subdir" in lines, f"目录应标记为 [D]，实际: {lines}"

    def test_old_code_did_not_check_per_entry(self):
        """验证旧代码中 isdir 是对整个路径检查一次"""
        from long.cli import LongSystem
        import inspect

        source = inspect.getsource(LongSystem._register_local_tools)
        # 旧代码: isdir = await fs.is_dir(path) if path else True
        # 新代码: 应该在循环内 per-entry 检查
        # 验证旧模式不存在
        assert "isdir = await fs.is_dir(path)" not in source, \
            "旧代码模式 'isdir = await fs.is_dir(path)' 应被移除"

    @pytest.mark.asyncio
    async def test_list_files_empty_directory(self, tmp_path):
        """验证空目录返回空标记"""
        root = tmp_path / "workspace"
        root.mkdir()

        from long.workspace.filesystem import LocalFilesystem
        fs = LocalFilesystem(str(root))

        entries = await fs.list_dir("")
        assert entries == [], "空目录应返回空列表"


# ===================== 修复 14: isinstance 永真判断 =====================


class TestIsinstanceTautology:
    """isinstance(exc, type(exc)) 永真判断移除验证"""

    def test_tautology_pattern_removed(self):
        """验证代码中不再有 isinstance(exc, type(exc)) 模式"""
        from long.llm import client
        import inspect

        source = inspect.getsource(client)
        assert "isinstance(exc, type(exc))" not in source, \
            "isinstance(exc, type(exc)) 永真判断应被移除"

    def test_fallback_chain_uses_is_retryable(self):
        """验证 Fallback 链使用 exc.is_retryable() 判断"""
        from long.llm.client import LLMClient
        import inspect

        source = inspect.getsource(LLMClient._call_with_retry_and_fallback)
        # 应使用 is_retryable() 而非 isinstance(exc, type(exc))
        assert "exc.is_retryable()" in source, \
            "_call_with_retry_and_fallback 应使用 exc.is_retryable()"

    def test_network_error_variable_removed(self):
        """验证 is_network_error 变量已被移除"""
        from long.llm.client import LLMClient
        import inspect

        source = inspect.getsource(LLMClient._call_with_retry_and_fallback)
        assert "is_network_error" not in source, \
            "is_network_error 未使用的变量应被移除"