"""进程沙箱 (L1)

使用 subprocess + setrlimit 实现进程级隔离。
"""

from __future__ import annotations

import asyncio
import os
try:
    import resource
except ImportError:
    resource = None  # Windows: resource module not available
import signal
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .base import (
    ExecutionResult,
    ExecutionSpec,
    ExecutionStatus,
    ResourceLimits,
    Sandbox,
)


class ProcessSandbox(Sandbox):
    """进程沙箱 (L1)

    使用 subprocess 启动子进程，通过 setrlimit 限制资源。

    Attributes:
        workspace_dir: 工作区根目录（用于创建临时文件）
    """

    def __init__(self, workspace_dir: str | Path | None = None) -> None:
        self._workspace_dir = Path(workspace_dir) if workspace_dir else None
        self._sandboxes: dict[str, dict[str, Any]] = {}

    async def create(self, spec: ExecutionSpec) -> str:
        """创建沙箱环境

        流程: 校验 spec → 创建临时目录 → 写入代码 → 记录元数据
        """
        sandbox_id = str(uuid.uuid4())[:8]

        # 创建临时目录
        if self._workspace_dir:
            base_dir = self._workspace_dir / "sandbox"
            base_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = tempfile.mkdtemp(prefix=f"sandbox_{sandbox_id}_", dir=str(base_dir))
        else:
            temp_dir = tempfile.mkdtemp(prefix=f"sandbox_{sandbox_id}_")

        # 写入代码文件
        code_file = Path(temp_dir) / self._get_filename(spec.language)
        code_file.write_text(spec.code)

        self._sandboxes[sandbox_id] = {
            "spec": spec,
            "temp_dir": temp_dir,
            "code_file": str(code_file),
            "pid": None,
            "created_at": asyncio.get_event_loop().time(),
        }

        return sandbox_id

    async def run(self, sandbox_id: str) -> ExecutionResult:
        """执行沙箱中的代码

        流程: 构建 setrlimit 前缀 → subprocess 启动 → 带超时等待 → 收集输出
        """
        info = self._sandboxes.get(sandbox_id)
        if info is None:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"Sandbox {sandbox_id} not found",
            )

        spec: ExecutionSpec = info["spec"]
        code_file: str = info["code_file"]
        temp_dir: str = info["temp_dir"]

        # 构建命令
        cmd = self._build_command(spec, code_file)
        env = self._build_env(spec, temp_dir)

        # 记录开始时间
        start_time = asyncio.get_event_loop().time()

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=spec.working_dir or temp_dir,
                preexec_fn=self._make_preexec_fn(spec.resource_limits),
            )

            info["pid"] = process.pid

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=spec.timeout,
                )
            except asyncio.TimeoutError:
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()

                duration = asyncio.get_event_loop().time() - start_time
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    exit_code=-9,
                    stdout="",
                    stderr="执行超时",
                    duration=duration,
                )

            duration = asyncio.get_event_loop().time() - start_time
            exit_code = process.returncode or 0

            # 判断执行状态
            status = ExecutionStatus.SUCCESS
            if exit_code == -9:
                status = ExecutionStatus.OOM
            elif exit_code != 0:
                status = ExecutionStatus.ERROR

            return ExecutionResult(
                status=status,
                exit_code=exit_code,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                duration=duration,
            )

        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=str(e),
                duration=duration,
            )

    async def kill(self, sandbox_id: str) -> bool:
        """终止沙箱中的进程"""
        info = self._sandboxes.get(sandbox_id)
        if info is None:
            return False

        pid = info.get("pid")
        if pid is None:
            return False

        try:
            os.kill(pid, signal.SIGKILL)
            return True
        except ProcessLookupError:
            return False

    async def cleanup(self, sandbox_id: str) -> None:
        """清理沙箱资源"""
        import shutil

        info = self._sandboxes.get(sandbox_id)
        if info is None:
            return

        temp_dir = info.get("temp_dir")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

        self._sandboxes.pop(sandbox_id, None)

    def _get_filename(self, language: str) -> str:
        """获取代码文件名"""
        extensions = {
            "python": "script.py",
            "javascript": "script.js",
            "shell": "script.sh",
            "bash": "script.sh",
        }
        return extensions.get(language, "script")

    def _build_command(self, spec: ExecutionSpec, code_file: str) -> list[str]:
        """构建执行命令"""
        commands = {
            "python": ["python3", code_file],
            "javascript": ["node", code_file],
            "shell": ["bash", code_file],
            "bash": ["bash", code_file],
        }

        cmd = commands.get(spec.language, ["python3", code_file])
        cmd.extend(spec.args)
        return cmd

    def _build_env(self, spec: ExecutionSpec, temp_dir: str) -> dict[str, str]:
        """构建安全环境变量"""
        work_dir = spec.working_dir or temp_dir
        output_dir_path = str(Path(work_dir) / "output")
        mpl_dir_path = str(Path(temp_dir) / ".matplotlib")

        # 复制系统 matplotlib 字体缓存到沙箱，避免重新扫描
        mpl_dir = Path(mpl_dir_path)
        mpl_dir.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib
            sys_cache = Path(matplotlib.get_cachedir())
            for cache_file in sys_cache.glob("fontlist-*.json"):
                import shutil
                shutil.copy2(str(cache_file), str(mpl_dir / cache_file.name))
        except Exception:
            pass

        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": temp_dir,
            "TMPDIR": temp_dir,
            "TEMP": temp_dir,
            "TMP": temp_dir,
            "LANG": "en_US.UTF-8",
            "PYTHONIOENCODING": "utf-8",
            "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OMP_NUM_THREADS": "2",
            "MPLCONFIGDIR": mpl_dir_path,
            "MPLBACKEND": "Agg",
            "OUTPUT_DIR": output_dir_path,
            "XDG_DATA_DIRS": "/usr/share:/usr/local/share",
            "FONTCONFIG_FILE": os.environ.get("FONTCONFIG_FILE", "/etc/fonts/fonts.conf"),
        }

        Path(output_dir_path).mkdir(parents=True, exist_ok=True)

        python_path = os.environ.get("PYTHONPATH", "")
        if python_path:
            env["PYTHONPATH"] = python_path

        # 继承 SSL 证书路径，确保 requests/urllib3 能正常发起 HTTPS 请求
        for ssl_key in ("SSL_CERT_FILE", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE",
                        "SSL_CERT_DIR", "CA_BUNDLE"):
            ssl_val = os.environ.get(ssl_key, "")
            if ssl_val:
                env[ssl_key] = ssl_val

        # 如果没有显式设置 SSL 证书，尝试自动检测常见路径
        if "SSL_CERT_FILE" not in env:
            _cert_candidates = [
                "/etc/ssl/certs/ca-certificates.crt",       # Debian/Ubuntu
                "/etc/pki/tls/certs/ca-bundle.crt",         # RHEL/CentOS
                "/etc/ssl/ca-bundle.pem",                    # OpenSUSE
                "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",  # Fedora
                "/usr/local/etc/openssl/cert.pem",           # macOS
            ]
            for cert_path in _cert_candidates:
                if os.path.isfile(cert_path):
                    env["SSL_CERT_FILE"] = cert_path
                    env["CURL_CA_BUNDLE"] = cert_path
                    env["REQUESTS_CA_BUNDLE"] = cert_path
                    break

        if not spec.resource_limits.network:
            env["http_proxy"] = ""
            env["https_proxy"] = ""
            env["HTTP_PROXY"] = ""
            env["HTTPS_PROXY"] = ""
            env["no_proxy"] = "*"
            env["NO_PROXY"] = "*"

        _SENSITIVE_ENV_KEYS = frozenset({
            "OPENAI_API_KEY", "LLM_API_KEY", "ANTHROPIC_API_KEY",
            "TAVILY_API_KEY", "SERPER_API_KEY",
        })

        # spec.env 是调用方显式请求的环境变量，直接放行
        for key, value in spec.env.items():
            env[key] = value

        return env

    @staticmethod
    def _is_api_key_env(key: str) -> bool:
        """检查环境变量名是否可能是 API Key"""
        key_upper = key.upper()
        sensitive_suffixes = ("_KEY", "_SECRET", "_TOKEN", "_PASSWORD")
        return any(key_upper.endswith(suffix) for suffix in sensitive_suffixes)

    @staticmethod
    def _make_preexec_fn(limits: ResourceLimits):
        """创建 preexec_fn，在子进程中设置资源限制
        Windows 不支持 resource 模块和 preexec_fn，返回 None。
        """
        if resource is None:
            return None

        def preexec():
            try:
                # CPU 时间限制（秒）
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (int(limits.cpu_time), int(limits.cpu_time)),
                )
            except (ValueError, OSError):
                pass

            try:
                # 内存限制（字节）
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (limits.memory, limits.memory),
                )
            except (ValueError, OSError):
                pass

            try:
                # 文件大小限制（字节）
                resource.setrlimit(
                    resource.RLIMIT_FSIZE,
                    (limits.disk, limits.disk),
                )
            except (ValueError, OSError):
                pass

            try:
                # 文件描述符限制
                resource.setrlimit(
                    resource.RLIMIT_NOFILE,
                    (limits.file_descriptors, limits.file_descriptors),
                )
            except (ValueError, OSError):
                pass

        return preexec
