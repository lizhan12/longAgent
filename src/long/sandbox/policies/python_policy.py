"""Python 执行安全策略

专为 Python 代码执行优化的安全配置。
允许更宽松的资源限制，但禁止网络和文件系统写操作。
"""

from __future__ import annotations

from ..base import (
    ExecutionSpec,
    FilesystemPolicy,
    NetworkPolicy,
    ResourceLimits,
    SecurityPolicy,
)


class PythonPolicy:
    """Python 执行安全策略

    适用于 Python 脚本/代码的受控执行。
    - 允许读取标准库路径
    - 禁止网络访问
    - 限制文件写入到工作区
    """

    NAME = "python"

    @staticmethod
    def get_resource_limits() -> ResourceLimits:
        """获取 Python 执行的资源限制"""
        return ResourceLimits(
            cpu_time=60.0,
            memory=1024 * 1024 * 1024,  # 1 GB
            disk=200 * 1024 * 1024,  # 200 MB
            network=False,
            processes=64,
            file_descriptors=128,
        )

    @staticmethod
    def get_security_policy() -> SecurityPolicy:
        """获取 Python 执行的安全策略"""
        return SecurityPolicy(
            filesystem=FilesystemPolicy(
                read_only_paths=[
                    "/usr/lib/python3",
                    "/usr/local/lib",
                ],
                read_write_paths=["/tmp", "/workspace/output"],
                deny_paths=[
                    "/etc/passwd", "/etc/shadow",
                    "/root", "/home",
                ],
                allow_tmp=True,
            ),
            network=NetworkPolicy(
                allowed_hosts=[],
                allowed_ports=[],
                deny_all=True,
            ),
            syscalls=[
                "read", "write", "open", "close", "fstat", "lseek",
                "mmap", "munmap", "brk", "access", "exit_group",
                "clone", "fork", "execve",
            ],
            capabilities=[],
        )

    @staticmethod
    def apply(spec: ExecutionSpec) -> ExecutionSpec:
        """应用 Python 策略到执行规格

        Args:
            spec: 执行规格

        Returns:
            应用策略后的执行规格
        """
        if spec.resource_limits is None:
            spec.resource_limits = PythonPolicy.get_resource_limits()
        if spec.security_policy is None:
            spec.security_policy = PythonPolicy.get_security_policy()
        spec.timeout = min(spec.timeout, 60.0)
        spec.language = "python"
        return spec