"""进程沙箱 (L1)

使用 subprocess + setrlimit 实现进程级隔离。
"""

from __future__ import annotations

import asyncio
import functools
import os
try:
    import resource
except ImportError:
    resource = None  # Windows: resource module not available
import signal
import sys
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

# Windows 没有 SIGKILL；os.kill(pid, SIGTERM) 在 Windows 上走 TerminateProcess，
# 效果等价。没有这个兜底，超时和 kill() 路径会直接抛 AttributeError。
_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)

# matplotlib 默认字体 DejaVu Sans 没有中文字形，中文标签会渲染成方块。
# 按优先级挑一个系统里真实存在的中文字体写进沙箱 matplotlibrc。
_CJK_FONT_CANDIDATES = (
    "Microsoft YaHei",      # Windows
    "SimHei",               # Windows
    "Noto Sans CJK SC",     # Linux
    "Source Han Sans SC",   # Linux
    "WenQuanYi Zen Hei",    # Linux
    "PingFang SC",          # macOS
    "Hiragino Sans GB",     # macOS
    "SimSun",
)

@functools.lru_cache(maxsize=1)
def _detect_cjk_font() -> str | None:
    """探测系统可用的中文字体名（扫描字体较慢，缓存一次）"""
    try:
        from matplotlib import font_manager

        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        return None
    return next((name for name in _CJK_FONT_CANDIDATES if name in available), None)


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
        # 显式 UTF-8：默认按 locale 编码写入，Windows(GBK) 下代码里的中文/emoji
        # 会直接抛 UnicodeEncodeError，执行还没开始就失败。
        code_file.write_text(spec.code, encoding="utf-8")

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
                    os.kill(process.pid, _KILL_SIGNAL)
                except (ProcessLookupError, OSError):
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
            os.kill(pid, _KILL_SIGNAL)
            return True
        except (ProcessLookupError, OSError):
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
        # Windows 上没有 python3，直接执行会以 9009 (command not found) 失败。
        # 用 sys.executable 还能保证沙箱和宿主用同一个解释器/依赖环境。
        python_exe = sys.executable or ("python" if os.name == "nt" else "python3")

        commands = {
            "python": [python_exe, code_file],
            "javascript": ["node", code_file],
            "shell": ["bash", code_file],
            "bash": ["bash", code_file],
        }

        cmd = commands.get(spec.language, [python_exe, code_file])
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

        # 预置 matplotlibrc，让沙箱里的图表默认就能正确渲染中文。
        # 否则 matplotlib 回落到 DejaVu Sans（无中文字形），中文标签全是方块，
        # 模型只能自己去猜字体路径 / 写 FontProperties，反而更容易出错。
        cjk_font = _detect_cjk_font()
        if cjk_font:
            try:
                (mpl_dir / "matplotlibrc").write_text(
                    "font.family: sans-serif\n"
                    f"font.sans-serif: {cjk_font}, DejaVu Sans\n"
                    # 中文字体的减号字形常缺失，关掉 unicode 减号避免负数刻度变方块
                    "axes.unicode_minus: False\n",
                    encoding="utf-8",
                )
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
            # 让 open() 等默认走 UTF-8，避免生成的代码在 GBK 环境写中文文件时失败
            "PYTHONUTF8": "1",
            "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OMP_NUM_THREADS": "2",
            "MPLCONFIGDIR": mpl_dir_path,
            "MPLBACKEND": "Agg",
            "OUTPUT_DIR": output_dir_path,
            "XDG_DATA_DIRS": "/usr/share:/usr/local/share",
        }

        # fontconfig 只在 POSIX 上有意义，Windows 上指向 /etc/fonts 会让 matplotlib 报错
        _fontconfig = os.environ.get("FONTCONFIG_FILE") or (
            "" if os.name == "nt" else "/etc/fonts/fonts.conf"
        )
        if _fontconfig:
            env["FONTCONFIG_FILE"] = _fontconfig

        # Windows: 子进程缺少 SystemRoot/COMSPEC 时 Python 无法初始化 socket/ssl，
        # 缺少 PATHEXT 时找不到 .exe。这些必须从宿主继承。
        if os.name == "nt":
            for _win_key in (
                "SystemRoot", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
                "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
            ):
                _win_val = os.environ.get(_win_key)
                if _win_val:
                    env[_win_key] = _win_val
            # 与 HOME 保持一致，指向沙箱临时目录而不是真实用户目录
            for _home_key in ("USERPROFILE", "APPDATA", "LOCALAPPDATA"):
                env[_home_key] = temp_dir

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
