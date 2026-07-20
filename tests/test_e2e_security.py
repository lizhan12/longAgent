"""安全相关 E2E 测试 — 验证修复的正确性

覆盖修复:
  1. Workspace 路径穿越 (manager.py + filesystem.py)
  2. ErrorBoundaryGuard 抑制 KeyboardInterrupt (context_isolation.py)
  3. IR 条件表达式 fail-closed (executor.py)
  4. OutputGuard.mask_text 使用掩码值 (output_guard.py)
  9. 权限清单 fail-closed (permission_manifest.py)
"""

from __future__ import annotations

import pytest


# ===================== 修复 1: 路径穿越 =====================


class TestWorkspacePathTraversal:
    """Workspace 路径穿越修复测试"""

    def test_traversal_blocked_by_prefix_match(self, tmp_path):
        """验证无法通过前缀匹配绕过路径穿越检查"""
        root = tmp_path / "workspace"
        root.mkdir()
        (tmp_path / "workspace2").mkdir()

        from long.workspace.manager import WorkspaceManager
        from long.workspace.exceptions import PathTraversalError

        ws = WorkspaceManager(str(root))
        # 旧 bug: root=/workspace, path=../workspace2/secret
        # resolved=/workspace2/secret, startswith('/workspace') = True (绕过!)
        with pytest.raises(PathTraversalError):
            ws.resolve("../workspace2/secret")

    def test_traversal_blocked_by_dotdot(self, tmp_path):
        """验证标准 ../ 路径穿越被拒绝"""
        root = tmp_path / "workspace"
        root.mkdir()

        from long.workspace.manager import WorkspaceManager
        from long.workspace.exceptions import PathTraversalError

        ws = WorkspaceManager(str(root))
        with pytest.raises(PathTraversalError):
            ws.resolve("../../etc/passwd")

    def test_legitimate_path_allowed(self, tmp_path):
        """验证合法路径正常工作"""
        root = tmp_path / "workspace"
        root.mkdir()

        from long.workspace.manager import WorkspaceManager

        ws = WorkspaceManager(str(root))
        path = ws.resolve("output")
        assert path == (root / "output").resolve()

    def test_filesystem_traversal_blocked(self, tmp_path):
        """验证 filesystem.py 的路径穿越修复"""
        root = tmp_path / "workspace"
        root.mkdir()
        (tmp_path / "workspace2").mkdir()

        from long.workspace.filesystem import WorkspacePath

        wsp = WorkspacePath(str(root))
        with pytest.raises(ValueError):
            wsp.resolve("../workspace2/secret")

    def test_filesystem_legitimate_path_allowed(self, tmp_path):
        """验证 filesystem.py 合法路径正常"""
        root = tmp_path / "workspace"
        root.mkdir()

        from long.workspace.filesystem import WorkspacePath

        wsp = WorkspacePath(str(root))
        path = wsp.resolve("output")
        assert path == (root / "output").resolve()

    def test_absolute_path_rejected(self, tmp_path):
        """验证绝对路径被拒绝"""
        root = tmp_path / "workspace"
        root.mkdir()

        from long.workspace.manager import WorkspaceManager
        from long.workspace.exceptions import AbsolutePathError

        ws = WorkspaceManager(str(root))
        with pytest.raises(AbsolutePathError):
            ws.resolve("/etc/passwd")


# ===================== 修复 2: ErrorBoundaryGuard =====================


class TestErrorBoundaryGuard:
    """ErrorBoundaryGuard 修复测试"""

    def test_keyboard_interrupt_propagates(self):
        """验证 KeyboardInterrupt 不被抑制"""
        from long.harness.context_isolation import ErrorBoundary, ErrorBoundaryGuard

        boundary = ErrorBoundary()
        guard = ErrorBoundaryGuard("test", boundary)

        with pytest.raises(KeyboardInterrupt):
            with guard:
                raise KeyboardInterrupt()

    def test_system_exit_propagates(self):
        """验证 SystemExit 不被抑制"""
        from long.harness.context_isolation import ErrorBoundary, ErrorBoundaryGuard

        boundary = ErrorBoundary()
        guard = ErrorBoundaryGuard("test", boundary)

        with pytest.raises(SystemExit):
            with guard:
                raise SystemExit(1)

    def test_regular_exception_suppressed(self):
        """验证普通异常仍被抑制"""
        from long.harness.context_isolation import ErrorBoundary, ErrorBoundaryGuard

        boundary = ErrorBoundary()
        guard = ErrorBoundaryGuard("test", boundary)

        with guard:
            raise ValueError("test error")
        # 如果不抛出异常，测试通过
        assert boundary.has_failed("test")
        assert "ValueError" in boundary.get_error("test")


# ===================== 修复 3: IR 条件 fail-closed =====================


class TestConditionEvaluation:
    """IR 条件表达式 fail-closed 修复测试"""

    def test_condition_fail_closed_on_exception(self):
        """验证异常条件返回 False（fail-closed）"""
        from long.ir.executor import PlanExecutor
        from long.ir.constraint_validator import RuntimeCheckContext

        # RuntimeCheckContext 需要一个 history 参数
        class MockHistory:
            def has_state(self, state):
                return False

        context = RuntimeCheckContext(history=MockHistory())

        # 直接调用静态方法 _evaluate_condition
        # 注意: _evaluate_condition 是实例方法，需要通过实例调用
        # 但它是 protected 方法，我们可以通过 PlanExecutor 实例访问
        executor = PlanExecutor.__new__(PlanExecutor)

        # 测试条件异常: 1/0 导致 ZeroDivisionError
        result = executor._evaluate_condition("1/0", context)
        assert result is False, "异常条件应返回 False"

        # 测试未定义变量
        result = executor._evaluate_condition("undefined_var", context)
        assert result is False, "未定义变量应返回 False"

        # 测试语法错误
        result = executor._evaluate_condition("syntax error {{{", context)
        assert result is False, "语法错误应返回 False"

    def test_condition_normal_evaluation(self):
        """验证正常条件评估工作"""
        from long.ir.executor import PlanExecutor
        from long.ir.constraint_validator import RuntimeCheckContext

        class MockHistory:
            def has_state(self, state):
                return False

        context = RuntimeCheckContext(history=MockHistory())

        executor = PlanExecutor.__new__(PlanExecutor)

        # 测试 True 条件
        result = executor._evaluate_condition("True", context)
        assert result is True, "True 条件应返回 True"

        # 测试 False 条件
        result = executor._evaluate_condition("False", context)
        assert result is False, "False 条件应返回 False"

        # 测试 AST 检查: 禁止的 ast.Call 应返回 False
        result = executor._evaluate_condition("has_data", context)
        # has_data 调用 MockHistory.has_state("HAS_DATA")，返回 False
        assert result is False


# ===================== 修复 4: OutputGuard mask_text =====================


class TestOutputGuardMaskText:
    """OutputGuard mask_text 修复测试"""

    def test_phone_number_masked(self):
        """验证手机号被正确掩码"""
        from long.harness.output_guard import OutputGuard, OutputGuardConfig

        guard = OutputGuard(OutputGuardConfig(enabled=True))

        result = guard.mask_text("请联系 13800138000")
        assert "138****8000" in result, f"应包含掩码号码，实际: {result}"
        assert "13800138000" not in result, "原始号码不应出现在结果中"

    def test_china_id_masked(self):
        """验证身份证号被正确掩码"""
        from long.harness.output_guard import OutputGuard, OutputGuardConfig

        guard = OutputGuard(OutputGuardConfig(enabled=True))

        # 使用合法的身份证号格式
        result = guard.mask_text("身份证: 110101199001011234")
        assert "110101" not in result or "**" in result, \
            "身份证号应被掩码"

    def test_email_masked(self):
        """验证邮箱被正确掩码"""
        from long.harness.output_guard import OutputGuard, OutputGuardConfig

        guard = OutputGuard(OutputGuardConfig(enabled=True))

        result = guard.mask_text("邮箱: testuser@example.com")
        assert "testuser@example.com" not in result, \
            "原始邮箱不应出现在结果中"

    def test_mixed_content_masked(self):
        """验证混合内容中的 PII 被正确掩码"""
        from long.harness.output_guard import OutputGuard, OutputGuardConfig

        guard = OutputGuard(OutputGuardConfig(enabled=True))

        text = "姓名: 张三, 电话: 13912345678, 地址: 北京市"
        result = guard.mask_text(text)
        assert "139****5678" in result, f"应包含掩码号码，实际: {result}"
        assert "13912345678" not in result, "原始号码不应出现"

    def test_disabled_guard_returns_original(self):
        """验证禁用时返回原文"""
        from long.harness.output_guard import OutputGuard, OutputGuardConfig

        guard = OutputGuard(OutputGuardConfig(enabled=False))
        text = "电话: 13800138000"
        result = guard.mask_text(text)
        assert result == text, "禁用时应返回原文"

    def test_empty_text_returns_empty(self):
        """验证空文本返回空"""
        from long.harness.output_guard import OutputGuard, OutputGuardConfig

        guard = OutputGuard(OutputGuardConfig(enabled=True))
        assert guard.mask_text("") == ""
        assert guard.mask_text(None) is None


# ===================== 修复 9: 权限 fail-closed =====================


class TestPermissionManifest:
    """权限清单 fail-closed 修复测试"""

    def test_undeclared_tool_denied(self):
        """验证未声明工具默认拒绝"""
        from long.harness.permission_manifest import PermissionManifest

        manifest = PermissionManifest()
        assert manifest.is_allowed("undeclared_tool") is False, \
            "未声明的工具应返回 False"

    def test_declared_tool_allowed(self):
        """验证声明工具正常"""
        from long.harness.permission_manifest import PermissionManifest, ToolPermission

        manifest = PermissionManifest(
            tools=[ToolPermission(name="read_file", allowed=True)]
        )
        assert manifest.is_allowed("read_file") is True

    def test_declared_tool_denied_by_flag(self):
        """验证声明但禁止的工具返回 False"""
        from long.harness.permission_manifest import PermissionManifest, ToolPermission

        manifest = PermissionManifest(
            tools=[ToolPermission(name="delete_file", allowed=False)]
        )
        assert manifest.is_allowed("delete_file") is False

    def test_mode_based_denial(self):
        """验证基于模式的禁止"""
        from long.harness.permission_manifest import PermissionManifest, ToolPermission

        manifest = PermissionManifest(
            tools=[ToolPermission(
                name="execute_code", allowed=True, forbidden_in=["service"],
            )]
        )
        assert manifest.is_allowed("execute_code", mode="development") is True
        assert manifest.is_allowed("execute_code", mode="service") is False