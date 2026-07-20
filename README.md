# Long

可控 AI 智能系统框架 —— 面向复杂任务的形式化约束与自主执行引擎。

## 概述

Long 是一个 Python AI Agent 框架，核心思路是通过**形式化中间表示（PlanIR）** 和**状态机**对 LLM 的输出进行约束，将 Agent 的执行流程从开放式生成收敛到可验证、可审计的受控路径上。

与常见的 AutoGPT-style 自由循环不同，Long 在每次 LLM 调用后引入多层结构化的校验链：类型检查 → 约束验证 → LTL 时序逻辑校验 → 状态转移。这一设计使其更适用于对可靠性和可审计性有要求的场景，如自动化开发、数据处理流水线、合规操作等。

## 架构

```
User Input
    │
    ▼
┌──────────────────────────────────────┐
│        Interaction Controller        │  ← 会话管理 / 事件总线 / HITL / 流式输出
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│         Cognitive Runtime            │  ← StateGraph 驱动执行 / 复杂度评估
│  (Think → Act → Observe → Reflect)  │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│          Plan Executor               │
│  ┌──────────┐  ┌──────────────────┐  │
│  │ PlanIR   │  │ Constraint       │  │  ← 形式化计划 / 约束验证
│  │ Parser   │  │ Validator        │  │
│  ├──────────┤  ├──────────────────┤  │
│  │ State    │  │ LTL              │  │
│  │ Machine  │  │ Validator        │  │
│  └──────────┘  └──────────────────┘  │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│          LLM Client                  │  ← 统一模型入口 / 重试 / 降级 / 缓存 / 中间件
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│        Unified Tool Registry         │  ← 本地工具 / MCP 工具 / Skill 能力
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│           Sandbox Manager            │  ← 代码执行 / 文件操作 / 进程隔离
└──────────────────────────────────────┘

横向支撑层:
┌──────────┐ ┌──────────┐ ┌────────────┐ ┌──────────────┐
│  Memory  │ │Tracing & │ │   Auto     │ │  Eval        │
│  (三栖)   │ │Logging   │ │ Optimizer  │ │  Pipeline    │
└──────────┘ └──────────┘ └────────────┘ └──────────────┘
```

## 核心模块

| 模块 | 路径 | 说明 |
|------|------|------|
| **Cognitive Runtime** | `cognitive/` | StateGraph 驱动的执行引擎，将 Agent 循环抽象为 Think → Act → Observe → Reflect 的有向图 |
| **PlanIR** | `ir/` | 结构化中间表示：计划解析器、状态机、LTL 校验器、类型检查器、约束验证器 |
| **LLM Client** | `llm/` | 封装 OpenAI SDK，内置重试策略、Fallback 模型降级、响应缓存和安全边界中间件 |
| **Memory** | `memory/` | 三栖记忆架构：滑动窗口（短期/工作） + 向量检索（语义/情景/过程） + 语义压缩 |
| **Unified Tool Registry** | `capabilities/` | 统一注册与调用本地工具、MCP 工具和 Skill 能力 |
| **Sandbox** | `sandbox/` | 提供代码沙箱执行环境，支持进程隔离、安全策略和资源监控 |
| **Interaction** | `interaction/` | 会话管理、事件总线、HITL 交互、CLI/WebUI 适配器、流式输出 |
| **Observability** | `observability/` | 全链路 Trace/Span 追踪、结构化日志、健康仪表盘 |
| **Optimization** | `optimization/` | OODA 自动优化循环（Observe → Orient → Decide → Act），指标收集、模式分析、人工审批 |
| **Eval Pipeline** | `eval/` | 多层评估体系：结果层、过程层、系统层、对抗性测试 |
| **Agent Subsystem** | `agent/` | 子 Agent 调度：Worker（执行）、Planner（规划）、Critic（审查）、Runner（编排）、Escalation（升级） |

## 技术要点

### 计划验证链

每个 LLM 输出的计划在执行前经过四级校验：

1. **Type Checking** — 校验 Action 参数类型是否匹配 Schema
2. **Constraint Validation** — 校验步骤间依赖、变量引用的完整性
3. **LTL Validation** — 基于线性时序逻辑检查执行序列是否满足安全/活性规约
4. **State Machine** — 状态转移合法性检查，仅允许预定义的迁移路径

### 多层评估体系

```
结果层 Outcome  ← 最终产出质量（运行/测试验证）
过程层 Process  ← 计划结构、步骤效率、资源使用
系统层 System   ← 规则应用、约束遵守、安全性
对抗层 Adversarial ← 边界案例、异常输入模糊测试
```

### 自动优化闭环

基于 OODA 循环，系统在运行中持续收集指标（延迟、成功率、预算消耗），通过 PatternAnalyzer 识别可优化点，生成 OptimizationProposal，经 HumanApprovalGate 审批后由 ChangeApplier 执行变更。

### 配置驱动

系统通过 `configs/` 目录下的 YAML 文件进行配置，覆盖 LLM、执行、记忆、交互、功能开关、可观测性等维度，支持动态加载。

## 快速开始

```bash
# 安装依赖
pip install -e .

# 配置模型
cp .env.example .env
# 编辑 .env 填入 API Key 和模型配置

# 启动交互
python main.py
```

## 项目结构

```
long/
├── main.py                 # 入口
├── configs/                # YAML 配置文件
├── src/long/
│   ├── cli.py              # CLI 入口与 LongSystem 集成
│   ├── cognitive/          # 认知运行时（StateGraph、Planner、Reflection）
│   ├── ir/                 # 形式化中间表示（PlanIR、StateMachine、LTL）
│   ├── llm/                # LLM 客户端（调用、缓存、中间件）
│   ├── memory/             # 三栖记忆系统
│   ├── capabilities/       # 工具注册与 MCP/Skill 支持
│   ├── sandbox/            # 沙箱执行环境
│   ├── interaction/        # 交互控制器与适配器
│   ├── observability/      # 追踪、日志、仪表盘
│   ├── optimization/       # 自动优化引擎
│   ├── eval/               # 评估流水线
│   ├── agent/              # 子 Agent 调度
│   └── session/            # 会话持久化
├── tests/                  # 测试用例
└── scripts/                # 辅助脚本
```

## 依赖

- Python >= 3.12
- openai — LLM API 调用
- pydantic — 数据校验
- rich / prompt-toolkit — 终端交互
- fastapi / uvicorn — Web 服务
- pyyaml — 配置加载

## License

MIT