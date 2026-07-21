"""默认安全策略

适用于未知或通用代码执行的默认安全配置。
安全优先：默认拒绝所有，按需开放。
"""

from __future__ import annotations

from ..base import (
    ExecutionSpec,
    FilesystemPolicy,
    NetworkPolicy,
    ResourceLimits,
    SecurityPolicy,
)


class DefaultPolicy:
    """默认安全策略

    提供最保守的安全配置，适用于不可信代码执行。
    """

    NAME = "default"

    @staticmethod
    def get_resource_limits() -> ResourceLimits:
        """获取默认资源限制"""
        return ResourceLimits(
            cpu_time=30.0,
            memory=512 * 1024 * 1024,  # 512 MB
            disk=50 * 1024 * 1024,  # 50 MB
            network=False,
            processes=32,
            file_descriptors=64,
        )

    @staticmethod
    def get_security_policy() -> SecurityPolicy:
        """获取默认安全策略"""
        return SecurityPolicy(
            filesystem=FilesystemPolicy(
                read_only_paths=["/usr", "/etc", "/lib"],
                read_write_paths=["/tmp", "/workspace"],
                deny_paths=["/etc/passwd", "/etc/shadow", "/root"],
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
            ],
            capabilities=[],
        )

    @staticmethod
    def apply(spec: ExecutionSpec) -> ExecutionSpec:
        """应用默认策略到执行规格

        Args:
            spec: 执行规格

        Returns:
            应用策略后的执行规格
        """
        if spec.resource_limits is None:
            spec.resource_limits = DefaultPolicy.get_resource_limits()
        if spec.security_policy is None:
            spec.security_policy = DefaultPolicy.get_security_policy()
        spec.timeout = min(spec.timeout, 30.0)
        return spec