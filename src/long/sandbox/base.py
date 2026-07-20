"""沙箱抽象

定义沙箱隔离级别、资源限制、安全策略和执行结果模型。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class IsolationLevel(str, Enum):
    """隔离级别"""

    NONE = "none"
    PROCESS = "process"
    CONTAINER = "container"
    MICROVM = "microvm"


class ResourceLimits(BaseModel):
    """资源限制

    Attributes:
        cpu_time: CPU 时间限制（秒）
        memory: 内存限制（字节）
        disk: 磁盘限制（字节）
        network: 是否允许网络访问
        processes: 最大进程数
        file_descriptors: 最大文件描述符数
    """

    cpu_time: float = 60.0
    memory: int = 1024 * 1024 * 1024  # 1 GB
    disk: int = 200 * 1024 * 1024  # 200 MB
    network: bool = False
    processes: int = 64
    file_descriptors: int = 128


class FilesystemPolicy(BaseModel):
    """文件系统策略"""

    read_only_paths: list[str] = Field(default_factory=list)
    read_write_paths: list[str] = Field(default_factory=list)
    deny_paths: list[str] = Field(default_factory=list)
    allow_tmp: bool = True


class NetworkPolicy(BaseModel):
    """网络策略"""

    allowed_hosts: list[str] = Field(default_factory=list)
    allowed_ports: list[int] = Field(default_factory=list)
    deny_all: bool = True


class SecurityPolicy(BaseModel):
    """安全策略

    Attributes:
        filesystem: 文件系统策略
        network: 网络策略
        syscalls: 允许的系统调用列表（空=全部允许）
        capabilities: 允许的 Linux capabilities
    """

    filesystem: FilesystemPolicy = Field(default_factory=FilesystemPolicy)
    network: NetworkPolicy = Field(default_factory=NetworkPolicy)
    syscalls: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)


class ExecutionSpec(BaseModel):
    """执行规格

    Attributes:
        code: 要执行的代码
        language: 编程语言
        args: 命令行参数
        env: 环境变量
        timeout: 超时时间（秒）
        resource_limits: 资源限制
        security_policy: 安全策略
        working_dir: 工作目录（相对路径）
    """

    code: str
    language: str = "python"
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    timeout: float = 30.0
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    security_policy: SecurityPolicy = Field(default_factory=SecurityPolicy)
    working_dir: str | None = None


class ExecutionStatus(str, Enum):
    """执行状态"""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    OOM = "oom"
    ERROR = "error"
    KILLED = "killed"
    SECURITY_VIOLATION = "security_violation"


class ExecutionResult(BaseModel):
    """执行结果

    Attributes:
        status: 执行状态
        exit_code: 退出码
        stdout: 标准输出
        stderr: 标准错误
        duration: 执行时长（秒）
        peak_memory: 峰值内存（字节）
        peak_cpu: 峰值 CPU 使用率
        error: 错误信息
    """

    status: ExecutionStatus = ExecutionStatus.SUCCESS
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    peak_memory: int = 0
    peak_cpu: float = 0.0
    error: str | None = None


class Sandbox(ABC):
    """沙箱抽象基类

    所有沙箱实现都必须遵循此接口。
    """

    @abstractmethod
    async def create(self, spec: ExecutionSpec) -> str:
        """创建沙箱环境

        Args:
            spec: 执行规格

        Returns:
            沙箱 ID
        """
        ...

    @abstractmethod
    async def run(self, sandbox_id: str) -> ExecutionResult:
        """执行沙箱中的代码

        Args:
            sandbox_id: 沙箱 ID

        Returns:
            执行结果
        """
        ...

    @abstractmethod
    async def kill(self, sandbox_id: str) -> bool:
        """终止沙箱

        Args:
            sandbox_id: 沙箱 ID

        Returns:
            是否成功终止
        """
        ...

    @abstractmethod
    async def cleanup(self, sandbox_id: str) -> None:
        """清理沙箱资源

        Args:
            sandbox_id: 沙箱 ID
        """
        ...

    async def execute(self, spec: ExecutionSpec) -> ExecutionResult:
        """完整执行流程: 创建 -> 运行 -> 清理

        Args:
            spec: 执行规格

        Returns:
            执行结果
        """
        sandbox_id = await self.create(spec)
        try:
            result = await self.run(sandbox_id)
            return result
        finally:
            await self.cleanup(sandbox_id)
