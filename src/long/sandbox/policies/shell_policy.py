"""Shell 执行安全策略

专为 Shell 命令/脚本执行优化的安全配置。
最严格的限制：禁止网络、禁止文件写操作、限制系统调用。
"""

from __future__ import annotations

from ..base import (
    ExecutionSpec,
    FilesystemPolicy,
    NetworkPolicy,
    ResourceLimits,
    SecurityPolicy,
)


class ShellPolicy:
    """Shell 执行安全策略

    适用于 Shell 命令/脚本的受控执行。
    - 严格限制 CPU 和内存
    - 禁止网络和文件写操作
    - 最小系统调用集
    """

    NAME = "shell"

    @staticmethod
    def get_resource_limits() -> ResourceLimits:
        """获取 Shell 执行的资源限制"""
        return ResourceLimits(
            cpu_time=15.0,
            memory=256 * 1024 * 1024,  # 256 MB
            disk=10 * 1024 * 1024,  # 10 MB
            network=False,
            processes=16,
            file_descriptors=32,
        )

    @staticmethod
    def get_security_policy() -> SecurityPolicy:
        """获取 Shell 执行的安全策略"""
        return SecurityPolicy(
            filesystem=FilesystemPolicy(
                read_only_paths=["/usr/bin", "/bin"],
                read_write_paths=[],
                deny_paths=[
                    "/etc", "/root", "/home",
                    "/var", "/opt",
                ],
                allow_tmp=False,
            ),
            network=NetworkPolicy(
                allowed_hosts=[],
                allowed_ports=[],
                deny_all=True,
            ),
            syscalls=[
                "read", "write", "open", "close",
                "exit_group", "brk",
            ],
            capabilities=[],
        )

    @staticmethod
    def apply(spec: ExecutionSpec) -> ExecutionSpec:
        """应用 Shell 策略到执行规格

        Args:
            spec: 执行规格

        Returns:
            应用策略后的执行规格
        """
        if spec.resource_limits is None:
            spec.resource_limits = ShellPolicy.get_resource_limits()
        if spec.security_policy is None:
            spec.security_policy = ShellPolicy.get_security_policy()
        spec.timeout = min(spec.timeout, 15.0)
        spec.language = "sh"
        return spec