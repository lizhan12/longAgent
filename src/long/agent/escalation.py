"""Escalation Controller — HAR 核心失败路由层 (Phase 2: Learnable Policy)

分层失败处理策略（Local Recovery vs Global Replanning）：

            ┌────────────────────┐
            │   Node Execution    │
            │   (Worker Loop)     │
            └────────┬───────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
  Execution Err  Data Err   Semantic Err
       │            │            │
       ▼            ▼            ▼
  ┌──────────────────────────────────────┐
  │   Escalation Controller (Policy)     │
  │  ┌─────────┐ ┌────────┐ ┌─────────┐ │
  │  │Feature  │→│Policy  │→│Outcome  │ │
  │  │Extractor│ │(Bandit)│ │Eval     │ │
  │  └─────────┘ └────────┘ └─────────┘ │
  └──────────────────┬───────────────────┘
                     │ decision
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   RETRY_LOCAL   RETRY_REFINE  REPLAN_DAG
                  SWITCH_TOOL   REPLAN_NODE
                                   HITL

三层架构：
  Level 1: Feature Extractor  — failure → vector state
  Level 2: Escalation Policy  — learning-based decision (Contextual Bandit)
  Level 3: Outcome Evaluator  — reward signal generator

核心理念：
  - Escalation Controller 的本质不是"判断错误"，而是"学习如何以最低成本修复错误"
  - 它是 failure → action 的策略模型: (state, failure_trace, context) → escalation_action
  - 必须是无状态 + 可训练：状态在 memory/trace system，policy 是 stateless inference
  - 所有 learning 来自 trace，不是 runtime hook
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Level 1: Feature Extractor — EscalationState
# =============================================================================


class FailureType(str, Enum):
    """失败类型三层分类"""

    EXECUTION = "execution"
    DATA_VALIDITY = "data_validity"
    SEMANTIC = "semantic"
    SAFETY = "safety"


class EscalationAction(str, Enum):
    """升级动作

    动作空间 A = { retry_worker, refine_prompt, switch_tool,
                    partial_replan, full_replan, HITL }
    """

    RETRY_LOCAL = "retry_local"
    RETRY_REFINE = "retry_refine"
    SWITCH_TOOL = "switch_tool"
    REPLAN_NODE = "replan_node"
    REPLAN_DAG = "replan_dag"
    HITL = "hitl"


@dataclass
class EscalationState:
    """结构化特征向量 — Escalation Controller 的感知层

    不能是 raw log，必须是结构化 feature space 才能学习。

    Attributes:
        failure_type: 失败类型
        retry_count: 已重试次数
        tool_failure_rate: 工具调用失败比例 (0.0~1.0)
        cost_spent: 已消耗 token 成本估算
        node_depth: Worker 执行深度 (轮次)
        dag_position: DAG 中的位置
        time_elapsed: 已用时间 (ms)
        critic_scores: Critic 评分摘要
        recent_actions: 最近 N 次动作 (用于时间序列建模)
        sandbox_signals: 沙箱信号 (crash, timeout 等)
    """

    failure_type: FailureType = FailureType.EXECUTION
    retry_count: int = 0
    tool_failure_rate: float = 0.0
    cost_spent: float = 0.0
    node_depth: int = 0
    dag_position: str = ""
    time_elapsed: float = 0.0
    critic_scores: dict[str, float] = field(default_factory=dict)
    recent_actions: list[str] = field(default_factory=list)
    sandbox_signals: dict[str, Any] = field(default_factory=dict)

    def to_feature_vector(self) -> list[float]:
        """将状态转换为学习用的特征向量

        Returns:
            [failure_type_onehot(4), retry_norm, tool_fail_rate, cost_norm,
             node_depth_norm, time_norm, critic_avg, recent_act_onehot(6)]
        """
        ft_idx = {"execution": 0, "data_validity": 1, "semantic": 2, "safety": 3}
        ft_onehot = [0.0, 0.0, 0.0, 0.0]
        ft_onehot[ft_idx.get(self.failure_type.value, 0)] = 1.0

        retry_norm = min(self.retry_count / 10.0, 1.0)
        cost_norm = min(self.cost_spent / 10000.0, 1.0)
        depth_norm = min(self.node_depth / 10.0, 1.0)
        time_norm = min(self.time_elapsed / 300000.0, 1.0)

        critic_avg = 0.0
        if self.critic_scores:
            critic_avg = sum(self.critic_scores.values()) / len(self.critic_scores)
            critic_avg = min(max(critic_avg, 0.0), 1.0)

        return (
            ft_onehot
            + [retry_norm, self.tool_failure_rate, cost_norm, depth_norm, time_norm, critic_avg]
        )

    @classmethod
    def from_signal(
        cls,
        signal: "FailureSignal",
        tool_history: list[dict[str, Any]] | None = None,
    ) -> "EscalationState":
        """从 FailureSignal 构建结构化状态"""
        tool_history = tool_history or signal.context.get("tool_history", []) or []

        tool_fails = [t for t in tool_history if t.get("error") or not t.get("success", True)]
        tool_fail_rate = min(len(tool_fails) / max(len(tool_history), 1), 1.0)

        total_tokens = signal.context.get("tokens_used", 0)
        cost_spent = total_tokens * 0.000002  # ~$2/M tokens

        return cls(
            failure_type=signal.failure_type,
            retry_count=signal.retry_count,
            tool_failure_rate=tool_fail_rate,
            cost_spent=cost_spent,
            node_depth=signal.context.get("rounds", signal.retry_count),
            dag_position=signal.context.get("sub_agent_name", ""),
            time_elapsed=signal.context.get("elapsed_ms", 0.0),
            critic_scores=signal.context.get("critic_scores", {}),
            recent_actions=signal.context.get("recent_actions", []),
            sandbox_signals=signal.context.get("sandbox_signals", {}),
        )

    def make_key(self) -> str:
        """生成状态的轻量级 hash key（用于 bucket 化）"""
        ft = self.to_feature_vector()
        buckets = tuple(int(f * 4) for f in ft[:4])
        return f"{self.failure_type.value}:{buckets}:r{self.retry_count}"


# =============================================================================
# Level 2: Escalation Policy — Contextual Bandit
# =============================================================================


@dataclass
class PolicyActionRecord:
    """策略动作记录 — 用于 Bandit 学习"""

    state_key: str
    action: str
    reward: float = 0.0
    probability: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_key": self.state_key,
            "action": self.action,
            "reward": self.reward,
            "probability": self.probability,
            "timestamp": self.timestamp,
        }


class EscalationPolicy:
    """Contextual Bandit 升级策略

    从规则驱动（冷启动）逐步过渡到数据驱动（学习）。
    使用 ε-greedy 探索策略 + 指数加权移动平均 Q 值估计。

    设计原则:
      - 无状态：状态在 trace system，policy 是 stateless inference
      - 可复现：随机种子用于探索
      - 渐进式：exploration 随数据增加而减少

    用法:
        policy = EscalationPolicy()
        action, prob = policy.select_action(state, fallback=rule_based_action)
        policy.update(state, action, reward=0.85)
    """

    # 动作空间
    ALL_ACTIONS = [
        EscalationAction.RETRY_LOCAL,
        EscalationAction.RETRY_REFINE,
        EscalationAction.SWITCH_TOOL,
        EscalationAction.REPLAN_NODE,
        EscalationAction.REPLAN_DAG,
        EscalationAction.HITL,
    ]

    def __init__(
        self,
        epsilon: float = 0.25,
        learning_rate: float = 0.1,
        min_samples_for_learning: int = 30,
        seed: int = 42,
    ) -> None:
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.min_samples = min_samples_for_learning

        self._rng = random.Random(seed)
        self._q_values: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._action_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._total_samples: int = 0
        self._records: list[PolicyActionRecord] = []

    @property
    def samples(self) -> int:
        return self._total_samples

    def select_action(
        self,
        state: EscalationState,
        fallback: EscalationAction | None = None,
    ) -> tuple[EscalationAction, float]:
        """选择升级动作

        Args:
            state: 结构化特征向量
            fallback: 规则决策（冷启动回退）

        Returns:
            (选择的动作, 选择概率)
        """
        state_key = state.make_key()

        if self._total_samples < self.min_samples:
            if fallback is not None:
                return fallback, 1.0
            return EscalationAction.RETRY_LOCAL, 1.0

        q_for_state = self._q_values[state_key]
        if not q_for_state:
            if fallback is not None:
                return fallback, 1.0
            return EscalationAction.RETRY_LOCAL, 1.0

        if self._rng.random() < self.epsilon:
            valid = [
                a for a in self.ALL_ACTIONS
                if self._is_valid_for(a, state)
            ]
            if not valid:
                return EscalationAction.HITL, 1.0
            action = self._rng.choice(valid)
            prob = self.epsilon / len(valid)
            return action, prob

        best_action = max(q_for_state.items(), key=lambda x: x[1])
        prob = 1.0 - self.epsilon + self.epsilon / len(self.ALL_ACTIONS)
        return EscalationAction(best_action[0]), prob

    def update(self, state: EscalationState, action: EscalationAction, reward: float) -> None:
        """更新 Q 值（指数加权移动平均）

        Args:
            state: 决策时的状态
            action: 执行的动作
            reward: 获得的回报
        """
        state_key = state.make_key()
        action_key = action.value

        self._total_samples += 1
        self._action_counts[state_key][action_key] += 1

        old_q = self._q_values[state_key][action_key]
        self._q_values[state_key][action_key] = (
            old_q + self.learning_rate * (reward - old_q)
        )

        record = PolicyActionRecord(
            state_key=state_key,
            action=action_key,
            reward=reward,
        )
        self._records.append(record)

        if len(self._records) > 10000:
            self._records = self._records[-5000:]

        self._adapt_epsilon()

    def _adapt_epsilon(self) -> None:
        """自适应探索率：数据越多，探索越少"""
        if self._total_samples > 1000:
            self.epsilon = max(0.05, self.epsilon * 0.999)
        elif self._total_samples > 100:
            self.epsilon = max(0.10, self.epsilon * 0.995)

    def _is_valid_for(self, action: EscalationAction, state: EscalationState) -> bool:
        """检查动作是否对该状态有效"""
        if state.failure_type == FailureType.SEMANTIC:
            return action in {EscalationAction.REPLAN_DAG, EscalationAction.REPLAN_NODE}
        if state.failure_type == FailureType.EXECUTION:
            return action in {
                EscalationAction.RETRY_LOCAL,
                EscalationAction.RETRY_REFINE,
                EscalationAction.SWITCH_TOOL,
                EscalationAction.HITL,
            }
        return True

    def get_action_distribution(self, state: EscalationState) -> dict[str, float]:
        """获取动作的 Q 值分布（用于调试和可解释性）"""
        state_key = state.make_key()
        q = self._q_values[state_key]
        if not q:
            return {}
        total = sum(max(v, 0) for v in q.values()) or 1.0
        return {k: max(v, 0) / total for k, v in q.items()}

    def export_q_table(self) -> dict[str, Any]:
        """导出 Q 表用于持久化和离线分析"""
        return {
            "total_samples": self._total_samples,
            "epsilon": self.epsilon,
            "q_values": {
                k: dict(v) for k, v in self._q_values.items()
            },
            "action_counts": {
                k: dict(v) for k, v in self._action_counts.items()
            },
        }

    def save(self, path: str | Path) -> None:
        """持久化 Q 表"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.export_q_table()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("EscalationPolicy Q 表已保存到 %s (%d samples)", path, self._total_samples)

    def load(self, path: str | Path) -> None:
        """加载 Q 表"""
        path = Path(path)
        if not path.exists():
            logger.info("Q 表文件不存在，使用冷启动: %s", path)
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._total_samples = data.get("total_samples", 0)
        self.epsilon = data.get("epsilon", 0.25)
        for state_key, actions in data.get("q_values", {}).items():
            for action, q_val in actions.items():
                self._q_values[state_key][action] = q_val
        for state_key, counts in data.get("action_counts", {}).items():
            for action, count in counts.items():
                self._action_counts[state_key][action] = count
        logger.info("EscalationPolicy Q 表已加载 (%d samples)", self._total_samples)


# =============================================================================
# Level 3: Outcome Evaluator — Reward Generator
# =============================================================================


@dataclass
class EscalationOutcome:
    """升级动作的结果反馈"""

    action: EscalationAction
    success: bool = False
    tokens_spent: int = 0
    latency_ms: float = 0.0
    hitl_triggered: bool = False
    replan_triggered: bool = False
    description: str = ""


class RewardCalculator:
    """回报计算器

    奖励设计:
      +1.0   成功完成
      -0.1   每次重试
      -0.002 每 1K tokens
      -0.2   HITL 惩罚
      -0.5   Replan 惩罚
      +0.3   避免 HITL 时的成功
    """

    REWARD_SUCCESS = 1.0
    PENALTY_RETRY = 0.1
    PENALTY_TOKENS_PER_1K = 0.002
    PENALTY_HITL = 0.2
    PENALTY_REPLAN = 0.5
    BONUS_AUTO_RESOLVE = 0.3

    def calculate(self, outcome: EscalationOutcome) -> float:
        """计算奖励值

        reward =
          + success_completion
          - cost_tokens
          - latency
          - HITL frequency
          - replan frequency penalty
        """
        reward = 0.0

        if outcome.success:
            reward += self.REWARD_SUCCESS
            if not outcome.hitl_triggered and not outcome.replan_triggered:
                reward += self.BONUS_AUTO_RESOLVE

        reward -= outcome.tokens_spent / 1000 * self.PENALTY_TOKENS_PER_1K

        if outcome.hitl_triggered:
            reward -= self.PENALTY_HITL

        if outcome.replan_triggered:
            reward -= self.PENALTY_REPLAN

        return reward


# =============================================================================
# Trace Record — EscalationTraceRecord
# =============================================================================


@dataclass
class EscalationTraceRecord:
    """Escalation 追踪记录 — 写入 AgentTrace 供离线 RL 使用"""

    trace_id: str = field(default_factory=lambda: f"esc_{int(time.time() * 1000) % 100000}")
    failure_type: str = ""
    retry_count: int = 0
    state_features: list[float] = field(default_factory=list)
    action_taken: str = ""
    reward: float = 0.0
    policy_source: str = "rule"
    epsilon: float = 0.0
    success: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "failure_type": self.failure_type,
            "retry_count": self.retry_count,
            "state_features": self.state_features,
            "action_taken": self.action_taken,
            "reward": self.reward,
            "policy_source": self.policy_source,
            "epsilon": self.epsilon,
            "success": self.success,
            "timestamp": self.timestamp,
        }


# =============================================================================
# EscalationController v2 — Policy-Driven
# =============================================================================


@dataclass
class FailureSignal:
    """失败信号 — 从 Worker/Critic 上报给 EscalationController"""

    failure_type: FailureType
    source: str
    description: str = ""
    retry_count: int = 0
    failure_pattern: bool = False
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"FailureSignal(type={self.failure_type.value}, "
            f"source={self.source}, retry={self.retry_count}, "
            f"pattern={self.failure_pattern})"
        )


@dataclass
class EscalationDecision:
    """升级决策"""

    action: EscalationAction
    reason: str = ""
    repair_hint: str = ""
    needs_human_review: bool = False
    policy_source: str = "rule"


class EscalationController:
    """升级控制器 v2 — Policy-Driven (Contextual Bandit)

    三层架构:
      Level 1: Feature Extractor  — failure → EscalationState
      Level 2: EscalationPolicy   — Bandit 决策
      Level 3: RewardCalculator   — 回报评估

    冷启动时使用规则决策，数据充足后切换到 Bandit 学习。

    用法:
        controller = EscalationController(policy=EscalationPolicy())
        decision = controller.decide(signal, tool_history, rule_based_action)
        outcome = EscalationOutcome(success=True, tokens_spent=2000)
        reward = controller.evaluate(decision, outcome)
        controller.record_trace(state, decision, outcome, reward)
    """

    def __init__(
        self,
        max_retries: int = 3,
        max_refine_attempts: int = 2,
        hitl_enabled: bool = True,
        hitl_on_financial: bool = True,
        policy: EscalationPolicy | None = None,
        q_table_path: str | Path | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.max_refine_attempts = max_refine_attempts
        self.hitl_enabled = hitl_enabled
        self.hitl_on_financial = hitl_on_financial

        self._policy = policy or EscalationPolicy()
        self._reward_calc = RewardCalculator()
        self._failure_history: list[FailureSignal] = []
        self._trace_records: list[EscalationTraceRecord] = []
        self._MAX_HISTORY = 100
        self._last_state: EscalationState | None = None
        self._last_decision: EscalationDecision | None = None

        if q_table_path is not None:
            self._policy.load(q_table_path)

    @property
    def policy(self) -> EscalationPolicy:
        return self._policy

    def decide(
        self,
        signal: FailureSignal,
        tool_history: list[dict[str, Any]] | None = None,
    ) -> EscalationDecision:
        """核心路由决策（含策略学习）

        流程:
          1. failure → EscalationState (feature extraction)
          2. 规则推理 → rule_based_action (冷启动基准)
          3. Policy.select_action(state, fallback=rule_based_action)
          4. 输出 EscalationDecision

        Args:
            signal: 失败信号
            tool_history: 工具调用历史（用于结构化状态）

        Returns:
            EscalationDecision
        """
        self._record(signal)

        state = EscalationState.from_signal(signal, tool_history)
        self._last_state = state

        rule_action, rule_reason = self._rule_based_action(signal)

        action, prob = self._policy.select_action(state, fallback=rule_action)

        policy_source = "learned" if self._policy.samples >= self._policy.min_samples else "rule"

        self._last_decision = EscalationDecision(
            action=action,
            reason=rule_reason if action == rule_action else f"[Bandit] {action.value} (prob={prob:.3f})",
            repair_hint=signal.description,
            needs_human_review=(action == EscalationAction.HITL),
            policy_source=policy_source,
        )
        return self._last_decision

    def evaluate(
        self,
        decision: EscalationDecision,
        outcome: EscalationOutcome,
    ) -> float:
        """评估决策结果并更新策略

        Args:
            decision: 已执行的决策
            outcome: 执行结果

        Returns:
            计算出的 reward 值
        """
        reward = self._reward_calc.calculate(outcome)

        if self._last_state is not None:
            self._policy.update(self._last_state, decision.action, reward)

        trace = EscalationTraceRecord(
            failure_type=self._last_state.failure_type.value if self._last_state else "",
            retry_count=self._last_state.retry_count if self._last_state else 0,
            state_features=self._last_state.to_feature_vector() if self._last_state else [],
            action_taken=decision.action.value,
            reward=reward,
            policy_source=decision.policy_source,
            epsilon=self._policy.epsilon,
            success=outcome.success,
        )
        self._trace_records.append(trace)

        if len(self._trace_records) > 500:
            self._trace_records = self._trace_records[-300:]

        return reward

    def get_trace_records(self, limit: int = 50) -> list[dict[str, Any]]:
        """获取最近的 trace records（供 AgentTrace 集成）"""
        return [r.to_dict() for r in self._trace_records[-limit:]]

    def save_policy(self, path: str | Path | None = None) -> None:
        """持久化策略模型"""
        if path is None:
            path = Path("workspace") / "knowledge" / "escalation_policy.json"
        self._policy.save(path)

    def _rule_based_action(self, signal: FailureSignal) -> tuple[EscalationAction, str]:
        """规则基准决策（冷启动 fallback）

        专家定义的规则策略，用于初始阶段和 Bandit 冷启动。
        """
        if signal.failure_type == FailureType.EXECUTION:
            if signal.retry_count < self.max_retries:
                return EscalationAction.RETRY_LOCAL, "执行性失败，Worker 本地重试"
            return EscalationAction.HITL, f"执行性失败重试 {signal.retry_count} 次仍未解决"

        if signal.failure_type == FailureType.DATA_VALIDITY:
            if signal.retry_count < self.max_refine_attempts:
                return EscalationAction.RETRY_REFINE, "数据有效性失败，Worker 带修复策略重试"
            if signal.failure_pattern:
                return EscalationAction.REPLAN_NODE, "数据失败模式持续，局部子图重规划"
            return EscalationAction.HITL, "数据有效性失败超过阈值"

        if signal.failure_type == FailureType.SEMANTIC:
            return EscalationAction.REPLAN_DAG, "结构性/语义失败，DAG 设计需重规划"

        if signal.failure_type == FailureType.SAFETY:
            return EscalationAction.HITL, "安全相关失败，需人工介入"

        return EscalationAction.HITL, f"未知失败类型: {signal.failure_type}"

    def _record(self, signal: FailureSignal) -> None:
        self._failure_history.append(signal)
        if len(self._failure_history) > self._MAX_HISTORY:
            self._failure_history = self._failure_history[-self._MAX_HISTORY:]

    def detect_pattern(self, source: str, window: int = 5) -> bool:
        recent = [s for s in self._failure_history[-window:] if s.source == source]
        if len(recent) < 2:
            return False
        same_types = set(s.failure_type for s in recent)
        return len(same_types) == 1 and len(recent) >= 3

    def get_recent_failures(self, limit: int = 10) -> list[FailureSignal]:
        return self._failure_history[-limit:]

    def clear_history(self) -> None:
        self._failure_history.clear()

    def get_stats(self) -> dict[str, Any]:
        """获取学习和使用统计"""
        return {
            "policy_samples": self._policy.samples,
            "epsilon": self._policy.epsilon,
            "trace_count": len(self._trace_records),
            "failure_history_size": len(self._failure_history),
            "recent_rewards": [
                r["reward"] for r in self.get_trace_records(20)
            ],
            "avg_reward": (
                sum(r["reward"] for r in self.get_trace_records(50)) / max(len(self.get_trace_records(50)), 1)
            ),
        }