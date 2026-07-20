# Long 可控AI智能系统 — 技术架构与功能实现文档

> 版本：0.1.0 | Python >=3.12 | MIT License
>
> 生成日期：2026-05-17

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 系统架构](#2-系统架构)
  - [2.1 整体架构图](#21-整体架构图)
  - [2.2 三级执行策略](#22-三级执行策略)
  - [2.3 模块依赖关系](#23-模块依赖关系)
- [3. 核心模块详解](#3-核心模块详解)
  - [3.1 入口与编排层](#31-入口与编排层)
  - [3.2 IR 中间表示与约束系统](#32-ir-中间表示与约束系统)
  - [3.3 认知运行时](#33-认知运行时)
  - [3.4 LLM 客户端](#34-llm-客户端)
  - [3.5 多智能体协同](#35-多智能体协同)
  - [3.6 记忆系统](#36-记忆系统)
  - [3.7 沙箱执行系统](#37-沙箱执行系统)
  - [3.8 交互系统](#38-交互系统)
  - [3.9 评估系统](#39-评估系统)
  - [3.10 优化系统](#310-优化系统)
  - [3.11 能力系统](#311-能力系统)
  - [3.12 Harness 治理层](#312-harness-治理层)
  - [3.13 可观测性](#313-可观测性)
  - [3.14 错误体系](#314-错误体系)
- [4. 配置体系](#4-配置体系)
- [5. 前端界面](#5-前端界面)
- [6. 测试体系](#6-测试体系)
- [7. 设计模式汇总](#7-设计模式汇总)
- [8. 已知问题与待改进项](#8-已知问题与待改进项)

---

## 1. 项目概述

**Long** 是一个具有形式化约束的可控 AI 智能系统框架。其核心设计理念是：**LLM 不直接执行，只生成结构化计划（PlanIR），经过约束验证后才受控执行**。系统通过三层防御体系（状态机 + LTL 时序逻辑 + Runtime Check）确保 AI 行为在可证明的安全边界内。

### 核心特性

| 特性 | 说明 |
|------|------|
| 三级执行策略 | 计划模式 > 认知运行时 > 降级模式，按任务复杂度自动选择 |
| 形式化约束 | 状态机（路径合法）+ LTL（时序合规）+ Runtime（Schema/白名单/预算） |
| 认知运行时 | Think-Act-Observe-Reflect-Plan-Output 状态图循环 |
| 三栖记忆 | 短期记忆 + 工作记忆 + 语义记忆，含衰减与晋升机制 |
| 沙箱执行 | 进程级隔离，12 种恶意代码模式预扫描，资源限制 |
| 多智能体 | P-W-E 三栖拓扑（Planner/Worker/Escalation），声明式子 Agent |
| 自动优化 | OODA 循环（Observe→Orient→Decide→Act），人工审批安全阀 |
| 评估流水线 | 结果层 + 过程层 + 系统层三层评估，含对抗性测试 |
| 灰度发布 | Feature Flag + Session 粘性分流 |
| 全链路追踪 | OpenTelemetry 风格 Trace/Span 模型 |

### 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.12+ |
| LLM SDK | OpenAI SDK (>=2.36.0) |
| 数据模型 | Pydantic V2 |
| 异步框架 | asyncio + aiofiles + aiosqlite |
| Web 框架 | FastAPI + Uvicorn |
| HTTP 客户端 | httpx |
| CLI | prompt-toolkit + Rich |
| 数据序列化 | PyYAML |
| 测试 | pytest + pytest-asyncio + pytest-cov |
| 类型检查 | mypy (strict mode) |
| 代码规范 | ruff |

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         用户入口层                                   │
│                    CLI (prompt-toolkit)                              │
│                    WebUI (FastAPI + WebSocket)                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     LongSystem 编排器 (cli.py)                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              ExecutionOrchestrator (三级执行)                  │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌─────────────────────┐  │   │
│  │  │ Plan模式    │  │ Cognitive模式 │  │ Fallback降级模式     │  │   │
│  │  │ PlanExecutor│  │ CognitiveRT  │  │ FallbackLoop        │  │   │
│  │  └─────┬──────┘  └──────┬───────┘  └──────────┬──────────┘  │   │
│  └────────┼────────────────┼──────────────────────┼─────────────┘   │
│           │                │                      │                  │
│  ┌────────▼────────────────▼──────────────────────▼─────────────┐   │
│  │                    ToolManager (工具管理)                      │   │
│  │   Local Tools │ Skills │ MCP Servers │ SubAgent Tools        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────┐  │
│  │PromptBuilder │ │SessionManager│ │ MemoryBridge │ │OutputGuard│  │
│  └─────────────┘ └──────────────┘ └──────────────┘ └───────────┘  │
└─────────────────────────────────────────────────────────────────────┘
           │                │                │               │
           ▼                ▼                ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  IR 约束系统  │ │  认知运行时   │ │  记忆系统     │ │  沙箱系统    │
│  PlanIR      │ │  StateGraph  │ │  三栖记忆     │ │  CodeScanner │
│  StateMachine│ │  Reflector   │ │  衰减/晋升    │ │  ProcessSbox │
│  LTL         │ │  ToolRouter  │ │  检索策略     │ │  ResourceMon │
│  TypeChecker │ │  TaskIR      │ │  Controller  │ │  SessionSbox │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
           │                │                │               │
           ▼                ▼                ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  LLM 客户端  │ │  多智能体     │ │  评估系统     │ │  优化系统    │
│  重试+降级   │ │  P-W-E拓扑   │ │  三层评估     │ │  OODA循环    │
│  缓存+预算   │ │  TaskOrchestr│ │  对抗测试     │ │  审批门      │
│  中间件管道  │ │  CriticAgent │ │  数据集管理   │ │  变更应用    │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
           │                                                     │
           ▼                                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Harness 治理层                                │
│   FeatureFlag (灰度) │ AlertManager (告警) │ FeedbackLoop (飞轮)    │
└─────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       可观测性层                                     │
│   Tracer (Trace/Span) │ StructuredLogger │ Dashboard │ Analyzer    │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 三级执行策略

系统根据任务复杂度自动选择执行策略：

```
用户输入
    │
    ▼
TaskComplexityClassifier.classify()
    │
    ├── SIMPLE (score < 1.5) ──────→ Fallback 降级模式
    │                                  直接 while 循环工具调用
    │                                  最多 8 轮
    │
    ├── MODERATE (1.5 ≤ score < 4.0) → Cognitive 认知运行时
    │                                  StateGraph 循环
    │                                  Think→Act→Observe→Reflect→Plan→Output
    │
    └── COMPLEX (score ≥ 4.0) ──────→ Plan 计划模式
                                       LLM 生成 PlanIR
                                       约束验证 → 受控执行
                                       最多 2 次重试
```

**降级规则**：Plan 模式失败 → 降级到 Cognitive 模式 → 再失败 → 降级到 Fallback 模式。

### 2.3 模块依赖关系

```
main.py
  └── cli.py (LongSystem)
        ├── llm/client.py (LLMClient) ← llm/base.py
        │     └── errors.py
        ├── components/
        │     ├── execution_orchestrator.py
        │     │     ├── cognitive/runtime.py (CognitiveRuntime)
        │     │     │     ├── cognitive/reflection.py
        │     │     │     └── cognitive/compression.py
        │     │     ├── execution/plan_execution.py
        │     │     ├── execution/cognitive_bridge.py
        │     │     └── execution/fallback_loop.py
        │     ├── tool_manager.py
        │     ├── prompt_builder.py
        │     ├── session_manager.py
        │     └── memory_bridge.py
        ├── ir/
        │     ├── plan_ir.py (PlanIR, StepIR)
        │     ├── ir_parser.py (IRParser)
        │     ├── constraint_validator.py
        │     ├── state_machine.py
        │     ├── ltl.py (LTLValidator)
        │     ├── type_checker.py
        │     ├── repair_strategies.py
        │     └── types.py (参数模型)
        ├── agent/
        │     ├── __init__.py (TaskOrchestrator, SubAgentRegistry)
        │     ├── critic.py
        │     ├── escalation.py
        │     ├── planner.py
        │     ├── runner.py
        │     └── worker.py
        ├── memory/ (三栖记忆)
        ├── sandbox/ (沙箱执行)
        ├── interaction/ (交互协议)
        ├── eval/ (评估流水线)
        ├── optimization/ (OODA 优化)
        ├── capabilities/ (工具/MCP/Skill)
        ├── harness/ (治理层)
        ├── observability/ (可观测性)
        ├── context/compressor.py
        └── errors.py
```

---

## 3. 核心模块详解

### 3.1 入口与编排层

#### 3.1.1 main.py

极简入口，5 行代码，委托给 `long.cli:main`。

#### 3.1.2 cli.py — LongSystem 核心编排器

| 属性 | 值 |
|------|-----|
| 行数 | ~3505 |
| 核心类 | `LongSystem`, `_PromptCache` |

**LongSystem** 是整个系统的门面（Facade），负责：

1. **初始化 20+ 子模块**（`initialize()`, L149-340），按依赖顺序：
   - Workspace → LLM → StateMachine → ToolRegistry → Memory → MCP → Skill → Sandbox → SubAgent → FeatureFlag → OutputGuard → AlertManager → EvalPipeline → Optimizer → PlanExecutor → Components

2. **三级执行策略**（`_chat_with_tools_loop()`）：
   - `_try_plan_execution()` — 结构化计划模式
   - `_cognitive_runtime_loop()` — 认知运行时模式
   - `_fallback_tool_call_loop()` — 降级模式

3. **9 个内置工具注册**（`_register_local_tools()`）：

| 工具名 | 功能 | 安全级别 |
|--------|------|----------|
| `list_files` | 列出目录内容 | 安全 |
| `read_file` | 读取文件 | 安全 |
| `write_file` | 写入文件 | 中风险 |
| `delete_file` | 删除文件 | 高风险（需确认） |
| `read_skill_md` | 读取 Skill 描述 | 安全 |
| `execute_code` | 执行代码 | 高风险（沙箱+扫描） |
| `execute_file` | 执行文件 | 高风险（沙箱+扫描） |
| `get_current_time` | 获取当前时间 | 安全 |
| `tavily_search` | 网络搜索 | 受限（最多2次） |

4. **子 Agent 工具**：`delegate_task`, `check_task`, `cancel_task`

5. **CLI 命令**：`/status`, `/skill`, `/mcp`, `/history`, `/eval`, `/health`, `/traces`, `/optimization`

#### 3.1.3 Components 组件层

从 `LongSystem` 单体拆分出的 5 个专职组件：

| 组件 | 文件 | 行数 | 职责 |
|------|------|------|------|
| `ExecutionOrchestrator` | execution_orchestrator.py | 909 | 三级执行策略编排 |
| `ToolManager` | tool_manager.py | 970 | 工具注册/发现/执行/缓存 |
| `PromptBuilder` | prompt_builder.py | 354 | 分层缓存提示词构建 |
| `SessionManager` | session_manager.py | 176 | 会话/偏好/摘要持久化 |
| `MemoryBridge` | memory_bridge.py | 179 | 记忆↔评估/优化/安全桥接 |

**ExecutionOrchestrator** 进一步拆分为 3 个执行子组件：

| 子组件 | 文件 | 行数 | 职责 |
|--------|------|------|------|
| `PlanExecution` | execution/plan_execution.py | 250 | 计划模式执行 |
| `FallbackLoop` | execution/fallback_loop.py | 678 | 降级模式循环 |
| `CognitiveBridge` | execution/cognitive_bridge.py | 121 | 认知运行时桥接 |

**PromptBuilder 分层缓存设计**：

```
┌─────────────────────────────────────────┐
│  静态层 (初始化时构建，不变)              │
│  - 责任归属准则                          │
│  - 禁止幻觉规则                          │
│  - 工具效率约束                          │
│  - AGENTS.md (支持灰度版本)              │
├─────────────────────────────────────────┤
│  半静态层 (变更时标记 dirty)              │
│  - 用户偏好                             │
│  - 用户画像                             │
│  - 日终摘要                             │
│  - Skill 列表                           │
├─────────────────────────────────────────┤
│  动态层 (每次请求时更新)                  │
│  - 当前时间 (UTC+8)                     │
└─────────────────────────────────────────┘
```

**ToolManager 核心功能**：

- `register_local_tools()` — 注册 9 个内置工具处理器
- `register_subagent_tools()` — 注册子 Agent 编排工具
- `auto_discover_skills()` — 自动发现并加载 Skills
- `connect_mcp_servers()` — 连接 MCP 服务器
- `gather_tools()` — 收集所有可用工具（local + skill + mcp），去重（local 优先）
- `execute_tool()` — 执行工具调用，含逻辑时钟追踪、缓存检查、Trace 记录

**TaskTimeline 逻辑时钟**：跟踪任务生命周期（created_at / deadline / ttl / retry_count / step_count / status），支持 `touch()` / `inc_retry()` / `is_expired()` / `checkpoint()`。

---

### 3.2 IR 中间表示与约束系统

IR 模块是系统安全性的核心，实现了"LLM 输出不可直接执行"的设计理念。

#### 3.2.1 PlanIR — 计划中间表示

| 属性 | 值 |
|------|-----|
| 文件 | ir/plan_ir.py |
| 行数 | 364 |

**ActionType 枚举**（9 种动作类型）：

| 类型 | 说明 | 目标状态 |
|------|------|----------|
| `SEARCH` | 搜索信息 | HAS_DATA |
| `CALL_API` | 调用 API | HAS_DATA |
| `CALL_TOOL` | 调用工具 | HAS_DATA |
| `CALL_MCP` | 调用 MCP 工具 | HAS_DATA |
| `CALL_SKILL` | 调用 Skill | HAS_DATA |
| `REASON` | 推理分析 | VERIFIED |
| `SUMMARIZE` | 摘要生成 | GENERATED |
| `OUTPUT` | 输出结果 | DONE |
| `WAIT_APPROVAL` | 等待审批 | APPROVED |

**StepIR 数据模型**：

```python
class StepIR(BaseModel):
    step_id: str                    # 步骤唯一标识
    action: ActionType              # 动作类型
    args: dict = {}                 # 动作参数
    depends_on: list[str] = []      # 依赖步骤
    condition: str | None = None    # 执行条件表达式
    fallback_step: str | None = None # 失败回退步骤
    expected_state: str | None = None # 预期状态
    risk_level: RiskLevel = RiskLevel.LOW
    description: str = ""
```

**PlanIR 关键方法**：

| 方法 | 说明 |
|------|------|
| `_coerce_and_normalize()` | model_validator，自动归一化 action 类型、映射参数别名 |
| `build_structured_output_schema()` | 生成兼容 OpenAI Structured Outputs 的扁平 JSON Schema |
| `validate_dependencies()` | 检查无效依赖引用 |
| `auto_fix_dependencies()` | 自动移除无效的 depends_on 和 fallback_step |
| `get_execution_order()` | **Kahn 拓扑排序算法**（BFS 入度法），返回合法执行顺序 |

**别名映射**：约 30 条 action 别名（如 `tavily_search → search`、`think → reason`）和 12 条参数字段别名（如 `tool → tool_name`、`q → query`）。

#### 3.2.2 IRParser — IR 解析器

| 属性 | 值 |
|------|-----|
| 文件 | ir/ir_parser.py |
| 行数 | 366 |

**三轮解析流程**：

```
LLM 输出文本
    │
    ▼
┌─────────────────────────────────────────┐
│ 第1轮：快速路径 (structured_output=True) │
│   直接 json.loads + Pydantic 验证        │
│   ✅ 成功 → 返回                         │
│   ❌ 失败 ↓                              │
├─────────────────────────────────────────┤
│ 第2轮：标准路径                           │
│   _extract_json() 提取 JSON              │
│   ├── Markdown 代码块正则提取             │
│   ├── 纯 JSON 检测                       │
│   └── 混合文本括号匹配                   │
│   + Pydantic 验证                        │
│   ✅ 成功 → 返回                         │
│   ❌ 失败 ↓                              │
├─────────────────────────────────────────┤
│ 第3轮：修复路径                           │
│   遍历修复策略，尝试修复后重新解析         │
│   ✅ 成功 → 返回                         │
│   ❌ 失败 ↓                              │
├─────────────────────────────────────────┤
│ 重试：build_retry_prompt()               │
│   构建修正 Prompt（含错误列表+指引）      │
│   重新调用 LLM，最多 max_retries 次       │
└─────────────────────────────────────────┘
```

**ParseMetrics** 统计：total / success / repairable / unparseable / fast_path_hits / strategies_applied。

#### 3.2.3 AgentStateMachine — 形式化状态机

| 属性 | 值 |
|------|-----|
| 文件 | ir/state_machine.py |
| 行数 | 350 |

**AgentState 枚举**（9 种状态）：

```
INIT ──search/call_api/call_tool/call_mcp/call_skill──→ HAS_DATA
HAS_DATA ──reason──→ VERIFIED
VERIFIED ──summarize──→ GENERATED
GENERATED ──wait_approval──→ APPROVED
APPROVED ──output──→ DONE
VERIFIED ──output──→ DONE
HAS_DATA ──output──→ DONE

任意状态 ──abort──→ ABORTED
任意状态 ──cancel──→ CANCELLED
任意状态 ──budget_exceeded──→ BUDGET_EXCEEDED
```

**终态集合**：DONE / ABORTED / CANCELLED / BUDGET_EXCEEDED

**设计特点**：使用条件路由函数 `_resolve_target_state()` 替代硬编码转移表，支持自定义配置和语义路由。

**关键方法**：

| 方法 | 说明 |
|------|------|
| `can_transition(state, action)` | 检查转移是否合法 |
| `get_allowed_actions(state)` | 获取当前状态允许的动作 |
| `validate_plan_path(steps)` | 验证计划路径的合法性 |
| `check_transition(state, action)` | 返回 (bool, transition, error_msg) |

#### 3.2.4 LTLValidator — LTL 时序逻辑验证器

| 属性 | 值 |
|------|-----|
| 文件 | ir/ltl.py |
| 行数 | 425 |

**LTL 公式类层次**：

```
LTLFormula (抽象基类)
├── Globally(inner)          — G(φ): φ 必须始终满足
├── Eventually(inner)        — F(φ): φ 最终必须满足
├── Implies(antecedent, con) — φ → ψ: 前件满足则后件必须满足
├── ActionOccurred(action)   — 原子命题：某动作已发生
├── StateReached(state)      — 原子命题：某状态已到达
├── TerminalStateReached()   — 原子命题：终态已到达
├── And(left, right)         — φ ∧ ψ
├── Or(left, right)          — φ ∨ ψ
└── Not(inner)               — ¬φ
```

**6 条默认规则**：

| 规则名 | 公式 | 语义 |
|--------|------|------|
| `output_requires_verified` | G(output → VERIFIED) | 输出前必须验证 |
| `approval_requires_verified` | G(wait_approval → VERIFIED) | 审批前必须验证 |
| `must_reach_terminal` | F(Terminal) | 必须到达终态 |
| `aborted_is_terminal` | G(ABORTED → Terminal) | ABORTED 是终态 |
| `cancelled_is_terminal` | G(CANCELLED → Terminal) | CANCELLED 是终态 |
| `done_requires_verified` | G(DONE → (VERIFIED ∨ APPROVED)) | 完成需验证或审批 |

**性能优化**：`check_runtime()` 跳过 `Eventually` 规则避免 O(N²) 开销，仅在 `check_final()` 终态时全量检查。

#### 3.2.5 ConstraintValidator — 三层约束验证器

| 属性 | 值 |
|------|-----|
| 文件 | ir/constraint_validator.py |
| 行数 | 288 |

```
┌─────────────────────────────────────────────────────────┐
│                  三层防御体系                              │
│                                                         │
│  第1层：状态机 (AgentStateMachine)                       │
│    - 编译时：validate_plan_path()                        │
│    - 运行时：can_transition()                            │
│    - 限制哪些状态转移是合法的                              │
│                                                         │
│  第2层：LTL 时序逻辑 (LTLValidator)                      │
│    - 编译时：不检查（运行时才有历史）                      │
│    - 运行时：check_runtime()（跳过 Eventually）           │
│    - 终态：check_final()（含 Eventually）                │
│    - 限制跨时间维度的规则                                  │
│                                                         │
│  第3层：Runtime Check (TypeChecker + 白名单 + 预算)       │
│    - 编译时：Schema + 依赖 + 白名单                      │
│    - 运行时：Schema + 白名单 + 预算                      │
│    - 终态：预算超限检查                                   │
│    - 每步执行前后的安全检查                                │
└─────────────────────────────────────────────────────────┘
```

**编译时验证**（`validate_plan`，5 项检查）：
1. 状态机路径检查
2. 类型检查（Schema + 依赖 + 白名单）
3. 安全约束（CRITICAL 步骤需 wait_approval）
4. 预算约束（estimated_steps ≤ max_steps）
5. DAG 环检测

**运行时验证**（`validate_step_runtime`，5 项检查）：
1. 状态机转移检查
2. LTL 时序检查（降级为终态检查）
3. Schema/类型检查
4. 白名单检查
5. 预算检查

**终态验证**（`validate_final`，3 项检查）：
1. 终态合法性
2. LTL 终态规则（含 Eventually）
3. 预算超限

#### 3.2.6 TypeChecker — 类型检查器

| 属性 | 值 |
|------|-----|
| 文件 | ir/type_checker.py |
| 行数 | 282 |

**检查项**：
- `check_plan`：plan_id/goal 非空、步骤非空、依赖验证、step_id 唯一性
- `check_step`：step_id 非空、ActionType 枚举校验、参数 Pydantic 校验、条件表达式语法
- `_check_condition`：AST 解析，仅允许 `has_data/verified/approved/error_count/tokens_used/True/False/and/or/not`
- `check_whitelist`：工具/API/MCP/Skill 白名单校验

#### 3.2.7 PlanExecutor — 计划执行器

| 属性 | 值 |
|------|-----|
| 文件 | ir/executor.py |
| 行数 | 1211 |

**核心流程**：

```
generate_plan()
    │
    ├── 构建 Prompt（含工具描述 + 历史教训回避提示）
    ├── 尝试 Structured Outputs 模式
    ├── 失败降级为 json_object
    ├── 最多 max_plan_retries 次重试
    ├── 解析成功后 validate_plan() 编译时验证
    └── 最终尝试 CascadeRouter 兜底

execute_plan()
    │
    ├── 拓扑排序获取执行顺序
    ├── 逐步 _execute_step_with_checks()
    │     ├── 运行时约束验证
    │     ├── _dispatch_action() 动作分发
    │     ├── 失败回退（fallback_step）
    │     └── _try_repair_state_violation() 自动修复
    ├── 终态检测 + 最大步数限制
    └── validate_final() 终态验证
```

**TaskComplexityClassifier**：多维评分（简单模式 -2、复杂模式 +3、多步骤 +1.5~+3、高风险 +1~+2.5、创造性 +1~+2、消息长度 +0.5~+1.5、句子数 +0.5~+1.5、问题数 +0.5），阈值 ≥4.0 COMPLEX / ≥1.5 MODERATE / <1.5 SIMPLE。

**验证教训学习**：`_record_validation_lesson()` / `_get_avoidance_hints()` — 记录验证失败模式到 JSONL，增强后续计划生成。

---

### 3.3 认知运行时

| 属性 | 值 |
|------|-----|
| 文件 | cognitive/runtime.py |
| 行数 | 1742 |

认知运行时是系统最核心的创新组件，基于状态图实现认知循环。

#### StateGraph 状态图引擎

```
┌──────────────────────────────────────────────────────┐
│                  StateGraph 拓扑                      │
│                                                      │
│  think ──has_tool_calls──→ act                       │
│  think ──has_final_text──→ output                    │
│  think ──_error──────────→ error                     │
│                                                      │
│  act ──→ observe                                     │
│  act ──→ error                                       │
│                                                      │
│  observe ──→ reflect                                 │
│                                                      │
│  reflect ──needs_retry──→ act                        │
│  reflect ──→ plan                                    │
│                                                      │
│  plan ──should_continue──→ think                     │
│  plan ──is_complete──────→ output                    │
│                                                      │
│  error ──recovered──→ think                          │
│  error ──→ output                                    │
└──────────────────────────────────────────────────────┘
```

**GraphNode**：name, kind, handler, transitions, max_visits
**GraphEdge**：from_node, to_node, condition, priority

支持循环检测（max_visits）、错误恢复、checkpoint/restore。

#### CognitiveContext 认知上下文

包含完整执行状态：user_message, intent, round_count, tool_history, search_count, messages, reflections, task_ir, plan_result, strategy_critique, errors 等。

#### Reflector 三层反思系统

```
Layer 1: Execution Validation
    └── 工具执行成功/失败判断

Layer 2: Strategy Critique (委托 StrategyCritique)
    ├── _check_redundant_search()    — 重复搜索检测
    ├── _check_tool_mismatch()       — 工具选择不合理检测
    ├── _check_stalled_progress()    — 进度停滞检测（3轮无完成/5轮进度<30%）
    └── _check_search_waste()        — 搜索结果浪费检测

Layer 3: Plan Repair (委托 PlanRepair)
    ├── _handle_redundant_search()   — 移除搜索建议
    ├── _handle_stalled_progress()   — 简化依赖
    ├── _handle_tool_mismatch()      — 标记优先工具
    └── _handle_search_waste()       — 更新子任务描述
```

#### ToolRouter 工具路由

- 搜索次数限制
- 连续搜索拦截
- execute_file 依赖检查
- delete_file 安全检查

#### CognitiveRuntime 核心处理器

**_handle_think()**（最复杂，L698-943）：
- TaskIR 解析与注入
- 记忆上下文注入
- 搜索耗尽提示
- LLM 调用（含超时控制）
- **防幻觉检测**：编造执行结果 / 代码任务未调用工具 / 代码已写未执行
- 自动执行代码块

**_handle_act()**（L993-1080）：
- 搜索优先执行
- 并行/串行工具执行

**_handle_output()**（L1203-1330）：
- 防幻觉检查
- 降级输出（LLM 超时率高时）
- LLM 摘要生成

**_autonomous_execution()**（L1508-1617）：
- 连续超时时触发自主执行模式
- 搜索 + 代码生成 + 执行

---

### 3.4 LLM 客户端

| 属性 | 值 |
|------|-----|
| 文件 | llm/client.py + llm/base.py |
| 行数 | 830 + 169 |

#### LLMClient 核心功能

| 方法 | 说明 |
|------|------|
| `chat()` | 普通聊天（缓存 + 重试 + Fallback） |
| `chat_with_tools()` | 带工具定义的聊天（流式 tool_calls 解析） |
| `stream_chat()` | 逐 token 流式输出 |
| `_call_with_retry_and_fallback()` | 重试 + Fallback 降级链 |
| `_check_budget()` | 单任务/日预算检查 |
| `get_judge_fn()` / `get_repair_fn()` | 裁判/修复函数工厂 |

**重试 + Fallback 降级链**：

```
主模型调用
    │
    ├── 成功 → 返回
    │
    └── 失败 → 重试 (max_retries=1, base_delay=2s)
                │
                ├── 成功 → 返回
                │
                └── 失败 → Fallback 模型链
                            │
                            ├── 降级模型1 (DeepSeek-R1)
                            │     ├── 成功 → 返回
                            │     └── 失败 → 降级模型2 ...
                            │
                            └── 全部失败 → 抛出 LLMError
```

**超时控制**：
- 首 token 超时：120s
- 空闲超时：60s
- 连接超时：15s
- 读取超时：180s

**预算控制**：
- 单任务上限：200,000 tokens
- 日预算上限：1,000,000 tokens

**LLMConfig 类型系统**（Pydantic V2）：

| 模型 | 说明 |
|------|------|
| `ModelProvider` | OPENAI / ANTHROPIC / CUSTOM |
| `RetryConfig` | max_retries, base_delay, backoff_factor |
| `FallbackConfig` | 模型降级链 |
| `TimeoutConfig` | connect=15, read=180, write=30 |
| `BudgetConfig` | max_tokens_per_task, daily_token_limit |
| `ModelConfig` | model, temperature, max_tokens, top_p |
| `LLMConfig` | 完整配置，含 `resolve_api_key()` 和 `resolve_base_url()` |

**级联路由**（CascadeRouter）：三级模型路由
- 旗舰级（planning 用途）
- 快速级（chat 用途）
- 边缘级（简单任务）

---

### 3.5 多智能体协同

| 属性 | 值 |
|------|-----|
| 文件 | agent/ |
| 核心文件 | agent/__init__.py (255行) |

**P-W-E 三栖拓扑**：

```
         ┌──────────────┐
         │  PlannerAgent │ ← 规划层
         └──────┬───────┘
                │ 分配任务
         ┌──────▼───────┐
         │  WorkerAgent  │ ← 执行层
         └──────┬───────┘
                │ 结果上报
         ┌──────▼───────┐
         │ EscalationCtrl│ ← 升级层（异常处理）
         └──────┬───────┘
                │ 质量审查
         ┌──────▼───────┐
         │  CriticAgent  │ ← 评审层
         └──────────────┘
```

**SubAgentSpec**：声明式子 Agent 规格，支持从 YAML/JSON 文件加载（`from_file()`），包含 name / description / tools / prompt / model / timeout / isolation_scope。

**TaskOrchestrator**：异步任务编排器，管理任务完整生命周期（submit → execute → result → timeout），使用 `asyncio.Semaphore` 控制并发上限（默认 5）。

**SubAgentRegistry**：从 `workspace/subagents/` 目录自动加载声明文件。

---

### 3.6 记忆系统

| 属性 | 值 |
|------|-----|
| 文件 | memory/ |
| 配置 | configs/memory.yaml |

**三栖记忆架构**：

```
┌─────────────────────────────────────────────────────────┐
│                    MemoryController                      │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ ShortTermMem │  │ WorkingMem   │  │ SemanticMem  │  │
│  │ (窗口记忆)    │  │ (任务记忆)    │  │ (语义记忆)    │  │
│  │ 128消息/50任务│  │ 任务隔离      │  │ 向量检索      │  │
│  │ 最近优先      │  │ 容量限制      │  │ ChromaDB     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                  │          │
│         └────────┬────────┴──────────┬──────┘          │
│                  │                   │                  │
│         ┌────────▼────────┐  ┌───────▼───────┐         │
│         │  Decay 衰减引擎  │  │ 检索策略       │         │
│         │  短期: 2h衰减    │  │ - 时效性       │         │
│         │  长期: 72h衰减   │  │ - 相关性       │         │
│         │  晋升阈值        │  │ - 重要性       │         │
│         └─────────────────┘  │ - 混合策略     │         │
│                              └───────────────┘         │
└─────────────────────────────────────────────────────────┘
```

**InMemoryBackend**：内存后端，支持存储/遗忘/搜索/类型搜索/计数/最大容量淘汰。

**检索策略**：
- 时效性（Recency）：按时间衰减评分
- 相关性（Relevance）：按查询相似度评分
- 重要性（Importance）：按访问频率评分
- 混合策略（Hybrid）：加权组合

**衰减与晋升**：
- 短期记忆：2 小时衰减
- 长期记忆：72 小时衰减
- 晋升条件：访问次数 ≥ 阈值 且 重要性评分 ≥ 阈值

---

### 3.7 沙箱执行系统

| 属性 | 值 |
|------|-----|
| 文件 | sandbox/ |
| 配置 | configs/sandbox.yaml |

**沙箱架构**：

```
┌─────────────────────────────────────────────────────────┐
│                    SandboxManager                        │
│                                                         │
│  1. 代码预扫描 (CodeScanner)                             │
│     ├── 12 种恶意模式正则检测                             │
│     ├── DANGEROUS → 阻止执行                             │
│     └── WARNING → 标记                                   │
│                                                         │
│  2. 创建沙箱实例                                         │
│     ├── PROCESS (已实现)                                 │
│     ├── CONTAINER (降级为 PROCESS)                       │
│     └── MICROVM (降级为 PROCESS)                         │
│                                                         │
│  3. 执行 (ProcessSandbox)                                │
│     ├── 创建临时目录                                     │
│     ├── 写入代码文件                                     │
│     ├── setrlimit 资源限制                               │
│     ├── subprocess 启动                                  │
│     ├── 带超时等待                                       │
│     └── 收集输出                                         │
│                                                         │
│  4. 资源监控 (ResourceMonitor)                           │
│     ├── 异步轮询进程资源                                  │
│     ├── 阈值告警 (80% 警告, 100% 终止)                   │
│     └── 依赖 psutil (可选)                               │
│                                                         │
│  5. 会话级沙箱 (SessionSandbox)                          │
│     ├── EPHEMERAL: 每次销毁                              │
│     ├── SESSION: 会话复用                                │
│     └── PERSISTENT: 跨会话复用                           │
└─────────────────────────────────────────────────────────┘
```

**CodeScanner 12 种恶意模式**：

| 模式 | 说明 |
|------|------|
| fork_bomb | Fork 炸弹 |
| reverse_shell | 反向 Shell |
| system_exec | 系统命令执行 |
| filesystem_destruction | 文件系统破坏 |
| dangerous_import | 危险导入 (ctypes) |
| env_tampering | 环境变量篡改 |
| dynamic_exec | 动态执行 (eval/exec/compile) |
| privilege_escalation | 权限提升 |
| signal_manipulation | 信号操纵 |
| mmap_usage | 内存映射 |
| network_listen | 网络监听 |
| resource_exhaustion | 资源耗尽 |

**资源限制**（默认策略）：

| 资源 | 默认值 | Python 策略 | Shell 策略 |
|------|--------|-------------|------------|
| CPU 时间 | 30s | 60s | 15s |
| 内存 | 512MB | 1GB | 256MB |
| 磁盘 | 100MB | 200MB | 50MB |
| 网络 | 禁止 | 禁止 | 禁止 |
| 进程数 | 10 | 5 | 3 |
| 文件描述符 | 64 | 128 | 32 |

**安全措施**：
- API Key 环境变量过滤（`_KEY` / `_SECRET` / `_TOKEN` / `_PASSWORD` 后缀）
- 网络代理禁用
- matplotlib 字体缓存复制
- setrlimit 资源限制（CPU / 内存 / 文件大小 / 文件描述符）

---

### 3.8 交互系统

| 属性 | 值 |
|------|-----|
| 文件 | interaction/ |
| 配置 | configs/interaction.yaml |

**交互协议**：

```
┌─────────────────────────────────────────────────────────┐
│              InteractionController                       │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ CLIAdapter   │  │ WebUIAdapter │  │ StreamManager│  │
│  │ prompt-toolkit│  │ FastAPI+WS   │  │ 流式输出管理  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  事件类型 (12 种):                                       │
│  message, error, warning, info, progress,               │
│  stream_token, stream_end, hitl_request, system,        │
│  tool_call, tool_result, reflection                     │
│                                                         │
│  HITL (Human-in-the-Loop):                              │
│  ├── 低风险 → 自动通过                                   │
│  ├── 中风险 → 通知用户                                   │
│  └── 高风险 → 需用户审批                                 │
└─────────────────────────────────────────────────────────┘
```

**Session 管理**：
- 最多 10 个并发会话
- 30 分钟空闲超时
- 1000 条历史记录

**流式输出**：
- 10K 缓冲区
- 5 token 刷新间隔

---

### 3.9 评估系统

| 属性 | 值 |
|------|-----|
| 文件 | eval/ |
| 配置 | configs/eval.yaml |

**三层评估架构**：

```
┌─────────────────────────────────────────────────────────┐
│                    EvalPipeline                          │
│                                                         │
│  第1层：OutcomeEvaluator (权重 0.3)                      │
│    ├── 精确匹配                                          │
│    ├── 部分匹配                                          │
│    ├── 字典准确率                                        │
│    └── JSON Schema 验证                                  │
│                                                         │
│  第2层：ProcessEvaluator (权重 0.5)                      │
│    ├── 轨迹合法性 (状态机 + LTL)                         │
│    ├── 效率评分 (步骤数 / 时长)                           │
│    └── MultiJudgeVoting (3 裁判多数投票)                  │
│                                                         │
│  第3层：SystemEvaluator (权重 0.2)                       │
│    ├── 稳定性 (一致性)                                   │
│    ├── 收敛性 (改善趋势)                                  │
│    └── 失败模式分析                                      │
│                                                         │
│  对抗性测试 (AdversarialTestSuite):                      │
│    ├── 正常任务                                          │
│    ├── 对抗任务 (注入/绕过)                               │
│    └── 边界任务                                          │
│                                                         │
│  数据集管理 (EvalDatasetManager):                        │
│    ├── 公开集 / 隐藏集                                   │
│    ├── 30 天轮换                                         │
│    └── 任务哈希                                          │
└─────────────────────────────────────────────────────────┘
```

**人工审核**：80% 自动 + 20% 人工

---

### 3.10 优化系统

| 属性 | 值 |
|------|-----|
| 文件 | optimization/ |
| 配置 | configs/optimization.yaml |

**OODA 循环架构**：

```
┌─────────────────────────────────────────────────────────┐
│                  AutoOptimizer                           │
│                                                         │
│  Observe (观察)                                         │
│    └── MetricsCollector 收集所有指标快照                  │
│        ├── LLM 调用指标                                  │
│        ├── 工具调用指标                                  │
│        ├── 执行指标                                      │
│        └── 评估结果                                      │
│                                                         │
│  Orient (分析)                                          │
│    └── PatternAnalyzer 分析模式                          │
│        ├── 成功率 < 0.5 → PROMPT 优化                    │
│        ├── 平均步骤 > 8 → ROUTING 优化                   │
│        ├── 平均时长 > 60s → BUDGET 优化                  │
│        └── 评估分数 < 0.6 → TOOL 优化                    │
│                                                         │
│  Decide (决策)                                          │
│    └── HumanApprovalGate 审批                            │
│        ├── LOW 风险 → 可自动审批 (confidence ≥ 0.9)      │
│        ├── MEDIUM/HIGH → 需人工审批                      │
│        ├── CRITICAL → 永不自动                           │
│        └── 安全目标 → 永不被建议变更                      │
│                                                         │
│  Act (执行)                                             │
│    └── ChangeApplier 应用变更                            │
│        ├── 保存配置快照                                   │
│        ├── 应用变更                                      │
│        ├── 检测回归 (10% 阈值)                           │
│        └── 自动回滚                                      │
└─────────────────────────────────────────────────────────┘
```

**安全目标（永不被建议变更）**：state_machine, ltl_rules, security_policy

**5 个优化目标 Tuner**：

| Tuner | 触发条件 | 优化动作 |
|-------|----------|----------|
| PromptTuner | Prompt 版本平均分 < 0.6 | 优化措辞/结构 |
| RoutingTuner | 路由成功率 < 50% | 调整路由规则 |
| BudgetTuner | 利用率 > 90% / < 30% | 增加/减少预算 |
| ToolTuner | 工具成功率 < 50% / 耗时 > 10s | 检查/替换/缓存 |
| RetryTuner | 低成功率+高重试 / 429频繁 | 快速失败/增加delay/Fallback |

**指标收集器**：支持 SQLite 持久化（自动建表+索引）、启动时加载最近 24 小时历史数据、聚合缓存。

**审计日志**：不可变追加日志（Append-Only Log），记录提案/审批/拒绝/应用/回滚全生命周期。

---

### 3.11 能力系统

| 属性 | 值 |
|------|-----|
| 文件 | capabilities/ |
| 配置 | configs/skills.yaml, configs/mcp.yaml |

**UnifiedToolRegistry**：统一工具注册表，管理三种工具来源：

```
┌─────────────────────────────────────────────────────────┐
│               UnifiedToolRegistry                        │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Local Tools  │  │ MCP Tools    │  │ Skill Tools  │  │
│  │ (内置9个)     │  │ (MCP Server) │  │ (自动发现)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  优先级：Local > Skill > MCP (去重)                      │
└─────────────────────────────────────────────────────────┘
```

**MCP Client**：Model Context Protocol 客户端，支持：
- 工具发现与调用
- 资源读取
- Prompt 模板获取

**SystemMCPServer**：内置 MCP Server，提供 4 个工具 / 2 个资源 / 2 个 Prompt。

**SkillManager**：技能管理器，支持：
- 自动发现（从 `./skills/` 目录）
- 加载/卸载/重载
- 启用/禁用
- 安全扫描 + 受限导入

**SkillLoader**：技能加载器，安全措施：
- 代码预扫描（12 种恶意模式）
- 受限 globals（禁止 `open`/`exec`/`eval`/`__import__`）
- 安全内置函数白名单

**已实现 Skills**：

| Skill | 工具 | 说明 |
|-------|------|------|
| calculator | calculate, math_add, math_multiply | 数学计算（递归下降解析器） |
| filesystem | list_files, read_file, write_file, delete_file | 文件操作（沙箱化） |
| tavily-search | tavily_search | 网络搜索（Tavily API） |

**已实现 MCP Servers**：

| Server | 工具 | 协议 |
|--------|------|------|
| calculator_server | calc_add, calc_sqrt, calc_pow | JSON-RPC (stdio) |
| filesystem_server | read_file, write_file, list_files | JSON-RPC (stdio) |

---

### 3.12 Harness 治理层

| 属性 | 值 |
|------|-----|
| 文件 | harness/ |

#### FeatureFlag — 灰度发布

| 属性 | 值 |
|------|-----|
| 文件 | harness/feature_flag.py |
| 行数 | 205 |

**默认 Feature Flag**：

| Flag | 分流策略 | 说明 |
|------|----------|------|
| prompt_version | 80% stable / 20% canary | Prompt 版本灰度 |
| model_version | 70% primary / 30% fallback | 模型版本分流 |
| output_pii_filter | 全局开关 | PII 过滤 |
| memory_consolidator | 全局开关 | 记忆整合器 |
| auto_eval_feedback | 全局开关 | 自动评估反馈 |

**分流算法**：MD5(session_id) → bucket，Session 粘性保证同一用户始终命中同一变体。

#### AlertManager — 告警管理

| 属性 | 值 |
|------|-----|
| 文件 | harness/alert.py |
| 行数 | 220 |

**默认告警规则**：

| 告警类型 | 阈值 | 级别 | 冷却期 |
|----------|------|------|--------|
| 超时率 | > 20% | ERROR | 120s |
| Token 预算 | > 80% | WARNING | 300s |
| 连续失败 | ≥ 3 次 | CRITICAL | 60s |
| 沙箱失败 | ≥ 3 次 | WARNING | 120s |

**防抖机制**：cooldown 冷却期防止告警洪水。

#### FeedbackLoop — 反馈飞轮

| 属性 | 值 |
|------|-----|
| 文件 | harness/feedback_loop.py |
| 行数 | 225 |

**数据飞轮闭环**：

```
评估结果 → generate_from_eval()
    │
    ├── efficiency < 0.7 → Prompt 优化提案
    ├── accuracy < 0.7 → Tool 优化提案
    └── safety < 0.7 → 安全增强提案
    │
    ▼
审批 (approve/reject)
    │
    ▼
应用 (mark_applied)
    │
    ▼
回退 (revert) — 如果效果不佳
```

#### OutputGuard — 输出安全治理

| 属性 | 值 |
|------|-----|
| 文件 | harness/output_guard.py |
| 行数 | 159 |

**PII 检测模式**（5 种）：

| 模式 | 说明 |
|------|------|
| china_id | 中国身份证号 |
| china_phone | 中国手机号 |
| email | 电子邮箱 |
| ip | IP 地址 |
| bank_card | 银行卡号 |

**敏感词过滤**：可配置敏感词列表。

**处理策略**：检测 + 标记，不自动删除（保留用户知情权）。`mask_text()` 提供遮蔽功能。

---

### 3.13 可观测性

| 属性 | 值 |
|------|-----|
| 文件 | observability/ |

#### Tracer — 全链路追踪

| 属性 | 值 |
|------|-----|
| 文件 | observability/tracing.py |
| 行数 | 271 |

**OpenTelemetry 风格模型**：

```
Trace (请求级追踪)
  └── Span (操作级追踪)
        ├── set_attribute()
        ├── add_event()
        └── finish()
```

- 基于 ContextVar 实现上下文传播
- 支持异步 + 同步双上下文管理器
- 最大保留 1000 条 Trace
- 全局函数：`current_trace_id()`, `current_span_id()`, `current_trace()`

#### StructuredLogger — 结构化日志

| 属性 | 值 |
|------|-----|
| 文件 | observability/structured_logging.py |
| 行数 | 189 |

- JSON 格式化，自动注入 trace_id / span_id
- 敏感信息脱敏（`api_key` / `secret` / `password` / `token` → `***REDACTED***`）
- `sk-` / `key-` / `token-` 前缀字符串部分遮蔽
- RotatingFileHandler + 控制台双输出

#### TraceAnalyzer — 追踪分析器

| 属性 | 值 |
|------|-----|
| 文件 | observability/analyzer.py |
| 行数 | 237 |

| 分析方法 | 说明 |
|----------|------|
| `analyze_failures()` | 按类型/工具/模型聚合失败模式 |
| `analyze_latency()` | P50/P95/P99/Max/Std 延迟趋势 |
| `analyze_success_rates()` | LLM/工具/执行成功率 |
| `analyze_cascade_failures()` | 基于 Trace parent_span_id 的级联失败分析 |
| `generate_report()` | 完整分析报告 |

#### HealthDashboard — 健康仪表盘

| 属性 | 值 |
|------|-----|
| 文件 | observability/dashboard.py |
| 行数 | 170 |

- LLM 健康状态（healthy / degraded / critical）
- 工具调用统计
- 重试统计
- 熔断器状态

---

### 3.14 错误体系

| 属性 | 值 |
|------|-----|
| 文件 | errors.py |
| 行数 | 442 |

**异常层次结构**：

```
LongError
├── LLMError
│   ├── LLMRateLimitError      (429 - 可重试)
│   ├── LLMServerError         (5xx - 可重试)
│   ├── LLMTimeoutError        (超时 - 可重试)
│   ├── LLMConnectionError     (网络 - 可重试)
│   ├── LLMBudgetExceededError (预算 - 不可重试)
│   └── LLMInvalidRequestError (4xx - 不可重试)
├── ToolError
│   ├── ToolExecutionError     (执行失败)
│   ├── ToolTimeoutError       (超时 - 可重试)
│   └── ToolNotFoundError      (不存在 - 不可重试)
├── MemoryError
│   ├── MemoryStorageError     (存储失败)
│   └── MemoryRetrievalError   (检索失败)
└── PlanError
    ├── PlanGenerationError    (生成失败)
    └── PlanValidationError    (验证失败)
```

**每个异常**携带结构化上下文（trace_id, span_id, cause），支持 `is_retryable` 标记。

**`classify_openai_error()`**：将 OpenAI SDK 异常自动分类为 Long 异常体系。

---

## 4. 配置体系

所有配置通过 YAML 文件管理，位于 `configs/` 目录：

| 配置文件 | 关键配置项 |
|----------|-----------|
| `llm.yaml` | 默认 DeepSeek-V4-Flash，重试(1次/2s)，降级链(DeepSeek-R1)，三级级联，4用途模型，预算(20万/任务, 100万/天) |
| `ir.yaml` | 类型检查(strict=false)，白名单，解析器(max_retries=3, auto_repair=true) |
| `sandbox.yaml` | 默认隔离 process，scanner 开启，资源限制(512MB/30s)，语言覆盖(python 1GB/60s, shell 256MB/15s) |
| `memory.yaml` | 三栖记忆：windowed(128消息/50任务)，vector_rag(chromadb/3集合/top5)，compressor(4000 tokens)，衰减配置 |
| `workspace.yaml` | 根目录 ./workspace，12个子目录，清理策略(24h)，审计(10000条) |
| `execution.yaml` | 三种模式: controlled(2轮/严格), balanced(4轮/自纠正/10轮), exploratory(6轮/12轮) |
| `security.yaml` | 开发/服务双模式，服务模式禁用 execute_code/execute_file/delete_file |
| `optimization.yaml` | OODA 循环(1h间隔)，指标保留90天，审批矩阵，回归阈值10% |
| `eval.yaml` | 三层评估: outcome(0.3), process(0.5/3裁判), system(0.2)，30天轮换 |
| `interaction.yaml` | CLI(1000历史/自动建议)，流式(10K缓冲/5token刷新)，HITL(低风险自动/高风险审批) |
| `skills.yaml` | 搜索路径 ./skills，自动发现，安全扫描+受限导入+热重载 |
| `mcp.yaml` | 2个 MCP Server(filesystem/calculator, stdio传输) |
| `benchmark.yaml` | 3级任务集: simple(5个), moderate(5个), complex(5个) |
| `feature_flags.yaml` | prompt_version(80/20灰度), model_version(70/30分流), PII过滤, 记忆整合, 自动评估 |

---

## 5. 前端界面

### WebUI

| 文件 | 行数 | 说明 |
|------|------|------|
| static/index.html | 69 | 主页面（header + sidebar + chat-area + HITL 面板） |
| static/app.js | 340 | 前端逻辑（WebSocket 通信 + 消息路由 + HITL 交互 + 流式渲染） |
| static/styles.css | 416 | 暗色主题样式（CSS 变量 + 消息气泡 + HITL 面板 + 数据表格） |

**WebSocket 消息类型**：message, error, warning, info, progress, stream_token, stream_end, hitl_request, system

**HITL 审核面板**：显示风险等级（low/medium/high/critical），支持批准/拒绝操作。

---

## 6. 测试体系

### 单元测试（tests/）

| 测试文件 | 行数 | 覆盖模块 |
|----------|------|----------|
| tests/ir/test_ir.py | 423 | PlanIR, StepIR, ActionType, TypeChecker, IRParser, RepairStrategies |
| tests/ir/test_state_machine.py | 605 | StateMachine, LTL, ConstraintValidator, 三层防御 |
| tests/cognitive/test_cognitive.py | 760 | TaskIR, TaskPlanner, StrategyCritique, PlanRepair, SemanticCompressor, CognitiveRuntime |
| tests/cognitive/test_e2e_simulation.py | 509 | 端到端模拟（十五五规划任务） |
| tests/llm/test_llm.py | 262 | LLM 配置、客户端、预算控制 |
| tests/sandbox/test_sandbox.py | 447 | 沙箱资源限制、代码扫描、进程沙箱、安全策略 |
| tests/interaction/test_interaction.py | 544 | 交互协议、会话、流式输出、CLI 适配器 |
| tests/interaction/test_webui.py | 425 | WebUI 适配器、WebSocket、REST API、HITL |
| tests/eval/test_eval.py | 425 | 三层评估、对抗测试、数据集管理 |
| tests/capabilities/test_capabilities.py | 485 | 工具注册表、MCP Client/Server、Skill Manager/Loader |
| tests/optimization/test_optimization.py | 510 | 指标收集、模式分析、审批门、变更应用、审计日志 |
| tests/memory/test_memory.py | 448 | 记忆项、内存后端、各 MemoryStore、衰减、检索策略 |

### 集成测试脚本（scripts/）

| 脚本 | 行数 | 说明 |
|------|------|------|
| test_real_task.py | 206 | 真实 LLM + 工具执行，3 个任务 |
| test_business_tasks.py | 545 | 5 阶段业务验证（模块完整性 + 5 业务任务 + 级联路由 + 记忆 + EvalOps） |
| test_cognitive_runtime.py | 465 | Cognitive Runtime 单元测试（Mock LLM） |
| test_search_limits.py | 198 | 搜索限制机制测试 |
| test_e2e_full.py | 160 | 端到端天气查询流程 |
| test_full_workflow.py | 165 | LLM 工具调用完整流程 |
| test_e2e_weather.py | 115 | 端到端天气查询 |
| test_cli_input.py | 60 | CLI 自动化输入测试 |
| test_tool_call_round2.py | 574 | 框架式 vs 直接 SDK 调用对比 |
| run_benchmark.py | 27 | Benchmark 运行器 |
| run_eval.py | 244 | CI 评估入口 |

---

## 7. 设计模式汇总

| 设计模式 | 应用位置 |
|----------|----------|
| **三级降级策略** | ExecutionOrchestrator: Plan > Cognitive > Fallback |
| **状态图 (State Graph)** | CognitiveRuntime: Think→Act→Observe→Reflect→Plan→Output |
| **三层反思** | Reflector: Execution Validation > Strategy Critique > Plan Repair |
| **三层防御** | ConstraintValidator: 状态机 + LTL + Runtime Check |
| **Fail-Closed** | 条件评估异常→条件不满足；IR 解析失败→ABORTED |
| **规则优先，LLM 辅助** | StrategyCritique: 规则引擎检测 + LLM 处理复杂场景 |
| **分层缓存** | PromptCache: 静态 > 半静态 > 动态 |
| **重试 + Fallback 降级链** | LLMClient: 主模型重试 > Fallback 模型链 |
| **拦截器链** | FallbackLoop: 搜索限制 > 安全检查 > 预算检查 |
| **注册表模式** | SubAgentRegistry, UnifiedToolRegistry |
| **桥接模式** | MemoryBridge, CognitiveBridge |
| **门面模式** | LongSystem, SandboxManager |
| **模板方法** | Sandbox.execute(): create→run→cleanup |
| **策略模式** | 隔离级别→不同 Sandbox；优化目标→不同 Tuner |
| **OODA 循环** | AutoOptimizer: Observe→Orient→Decide→Act |
| **HITL (Human-in-the-Loop)** | HumanApprovalGate, FeedbackLoop |
| **灰度发布** | FeatureFlag: session_hash 粘性分流 |
| **数据飞轮** | FeedbackLoop: 评估→提案→审批→应用→回退 |
| **快照/回滚** | ChangeApplier: ConfigSnapshot + 配置版本栈 |
| **防抖** | AlertManager: cooldown 冷却期 |
| **后置校验** | OutputGuard: PII 检测 + 敏感词过滤 |
| **对象池** | SandboxSessionManager: 会话级沙箱复用 |
| **不可变审计日志** | AuditLog: Append-Only Log |
| **逻辑时钟** | TaskTimeline: 任务生命周期追踪 |
| **声明式子 Agent** | SubAgentSpec: YAML/JSON 声明 + 自动发现 |
| **中间件管道** | LLMClient: MiddlewarePipeline (pre/post process) |
| **Trace/Span 模型** | Tracer: OpenTelemetry 风格分布式追踪 |
| **防幻觉** | 多层检测: 代码任务未调用工具 / 代码已写未执行 / 编造执行结果 |

---

## 8. 已知问题与待改进项

### 代码重复

`cli.py`（3505行）中的 `_register_local_tools()`, `_fallback_tool_call_loop()`, `_build_system_prompt()` 等方法与 `components/` 下的组件高度重复。系统正处于从单体向组件化拆分的过渡期。

### 部分实现

| 问题 | 位置 | 说明 |
|------|------|------|
| `_instances` 未使用 | SandboxManager | `execute()` 每次创建新 Sandbox 而非复用，`kill_all()` 遍历空字典 |
| `killed/kill_reason` 未设置 | ResourceMonitor | 缺少超限自动终止逻辑 |
| 硬编码状态 | HealthDashboard | memory/optimization 状态返回 `{"status": "active"}` |
| `mask_text` bug | OutputGuard | 遮蔽逻辑设为空字符串而非遮蔽字符 |
| `_find_record` 匹配不精确 | AuditLog | 通过 `proposal.change` 内容匹配而非 `proposal_id` |
| CONTAINER/MICROVM 未实现 | ProcessSandbox | 降级为 PROCESS |
| REPL 持久进程未使用 | SessionSandbox | `process`/`installed_packages` 字段已定义但未使用 |

### 安全边界声明

系统**不能保证**以下事项：
- 输出语义一定正确（LLM 边界）
- 条件表达式一定正确（安全回退）
- MCP 远程工具返回安全数据（外部不可控）
- LLM 一定输出有效 PlanIR（修复/重试兜底）

**不适用场景**：创意写作、超低延迟、高并发多租户、Windows 部署。
