from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class MemoryBridge:

    def __init__(
        self,
        output_guard: Any,
        alert_manager: Any,
        eval_pipeline: Any,
        optimizer: Any,
        feedback_loop: Any,
        conversation_turn_getter: Callable[[], list[dict[str, Any]]],
        auto_eval_interval: int = 5,
        feature_flags: Any = None,
        llm_budget_tokens: int = 200000,
    ) -> None:
        self._output_guard = output_guard
        self._alert_manager = alert_manager
        self._eval_pipeline = eval_pipeline
        self._optimizer = optimizer
        self._feedback_loop = feedback_loop
        self._conversation_turn_getter = conversation_turn_getter
        self._auto_eval_interval = auto_eval_interval
        self._feature_flags = feature_flags
        self._llm_budget_tokens = llm_budget_tokens

        self._background_tasks: list[asyncio.Task] = []
        self._max_background_tasks: int = 3
        self._conversation_turn_count: int = 0
        self._llm_call_total: int = 0
        self._llm_call_timeout: int = 0
        self._llm_call_fail: int = 0
        self._llm_total_tokens: int = 0

    @property
    def llm_call_total(self) -> int:
        return self._llm_call_total

    @property
    def llm_call_timeout(self) -> int:
        return self._llm_call_timeout

    @property
    def llm_call_fail(self) -> int:
        return self._llm_call_fail

    @property
    def llm_total_tokens(self) -> int:
        return self._llm_total_tokens

    def submit_background_task(self, coro: Any) -> None:
        """提交后台任务（并发上限3个）"""
        self._background_tasks = [t for t in self._background_tasks if not t.done()]
        if len(self._background_tasks) >= self._max_background_tasks:
            logger.debug("后台任务已达上限 %d，跳过", self._max_background_tasks)
            return
        task = asyncio.ensure_future(coro)
        task.add_done_callback(self.on_background_task_done)
        self._background_tasks.append(task)

    def on_background_task_done(self, task: asyncio.Task) -> None:
        """后台任务完成回调"""
        if task.exception():
            logger.warning("后台任务异常: %s", task.exception())

    async def auto_eval_conversation(self) -> None:
        """每隔若干轮对话自动运行轻量级评估，记录到优化器"""
        self._conversation_turn_count += 1
        if self._conversation_turn_count % self._auto_eval_interval != 0:
            return
        if self._eval_pipeline is None or self._optimizer is None:
            return

        recent = self._conversation_turn_getter()
        user_msgs = [m for m in recent if m["role"] == "user"]
        assistant_msgs = [m for m in recent if m["role"] == "assistant"]

        if not user_msgs or not assistant_msgs:
            return

        try:
            from long.eval.report import EvalTask

            last_user = user_msgs[-1]["content"]
            last_assistant = assistant_msgs[-1]["content"]

            task = EvalTask(
                name=f"auto_eval_turn_{self._conversation_turn_count}",
                input=last_user,
                expected=None,
            )

            report = self._eval_pipeline.run(task, output=last_assistant)

            self._optimizer.collector.record_eval_result(
                task_name=task.name,
                score=report.score,
                category=task.category.value,
            )
            self._optimizer.collector.record_execution_metrics(
                step_count=len(recent),
                duration=0.0,
                success=report.score >= 0.5,
            )

            logger.info(
                "自动评估完成: turn=%d, score=%.2f, needs_review=%s",
                self._conversation_turn_count,
                report.score,
                report.needs_human_review,
            )

            if self._optimizer is not None:
                self._optimizer.on_conversation_complete()

            if self._feature_flags is not None and self._feature_flags.is_enabled("auto_eval_feedback") and report.score < 0.7:
                self._feedback_loop.generate_from_eval({
                    "scores": {"efficiency": report.score},
                    "turn": self._conversation_turn_count,
                })

        except Exception as e:
            logger.warning("自动评估失败: %s", e)

    def schedule_auto_eval(self) -> None:
        """将自动评估提交为后台任务"""
        self.submit_background_task(self.auto_eval_conversation())

    def record_llm_stats(self, response: Any) -> None:
        """记录 LLM 调用统计（用于告警和监控）"""
        self._llm_call_total += 1
        if response.usage:
            self._llm_total_tokens += getattr(response.usage, "total_tokens", 0) or 0

    def record_llm_timeout(self) -> None:
        self._llm_call_timeout += 1
        self._llm_call_total += 1

    def record_llm_fail(self) -> None:
        self._llm_call_fail += 1
        self._llm_call_total += 1

    def check_output_safety(self, text: str) -> None:
        """检查输出安全（PII过滤 + 敏感词）"""
        if self._output_guard is None or self._feature_flags is None or not self._feature_flags.is_enabled("output_pii_filter"):
            return
        if not text:
            return
        result = self._output_guard.scan(text)
        if result.pii_matches:
            self._alert_manager.trigger(
                "pii_detected",
                f"LLM 输出包含 PII: {result.warning}",
            )
        if result.sensitive_matches:
            self._alert_manager.trigger(
                "sensitive_word",
                f"LLM 输出包含敏感词: {result.warning}",
            )

    def check_alerts(self) -> None:
        """在每轮对话后检查告警条件"""
        if self._alert_manager is None:
            return
        self._alert_manager.collect_metrics_alert(
            llm_call_total=self._llm_call_total,
            llm_call_timeout=self._llm_call_timeout,
            total_tokens=self._llm_total_tokens,
            budget_tokens=self._llm_budget_tokens,
        )
        if self._llm_call_fail > 0:
            self._alert_manager.check("consecutive_failures", float(self._llm_call_fail))
