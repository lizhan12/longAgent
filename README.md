# Long — 可控 AI 智能系统框架

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-560%20passing-brightgreen)](tests/)

**Long** 是一个面向复杂任务的 AI Agent 框架，核心思路是通过**形式化中间表示（PlanIR）** 和**状态机**对 LLM 的输出进行约束，将 Agent 的执行流程从开放式生成收敛到可验证、可审计的受控路径上。

---

## 特性

### 🛡️ 三层防御体系
| 层级 | 技术 | 作用 |
|------|------|------|
| L1 | **AgentStateMachine** | 编译时检查 PlanIR 路径合法性，运行时检查每步状态转移 |
| L2 | **LTLValidator** | 6 条默认时序逻辑规则（可扩展），运行时 + 终态双重校验 |
| L3 | **ConstraintValidator** | 每步执行前 Schema/权限/预算检查，执行后状态更新 + 日志 |

### 🧠 五层记忆架构
```
短期记忆 (RingBuffer) → 工作记忆 (Dict) → 语义记忆 (ChromaDB) 
                                     → 情景记忆 (SQLite) 
                                     → 过程记忆 (SQLite)
```
- 准入评分 → 衰减策略 → 冲突检测 → 自动提升
- 衰减公式: `strength(t) = initial × exp(-λ × hours)`

### 🔧 Harness 工程八大支柱
| 支柱 | 说明 |
|------|------|
| **执行引擎** | LLMClient(流式+重试+熔断) + ProcessSandbox |
| **工具层** | UnifiedToolRegistry(LOCAL/MCP/SKILL) + 安全扫描 |
| **记忆系统** | 五层记忆 + 准入/提升/衰减/冲突检测 |
| **编排引擎** | TaskComplexityClassifier + PlanIR DAG + 状态机 |
| **输出治理** | ConstraintValidator + TypeChecker + OutputGuard(PII/敏感词) |
| **安全层** | security.yaml + ProcessSandbox + WorkspaceManager |
| **可观测层** | Trace/Span + AlertManager + ResourceMonitor |
| **反馈回路** | EvalPipeline → AutoOptimizer(OODA) → FeedbackLoop |

### 🔒 安全特性
- **进程沙箱** — `setrlimit` 资源限制 + 代码预扫描
- **路径边界** — 所有文件操作经 WorkspaceManager 校验
- **PII 检测** — 身份证/手机号/邮箱/银行卡正则匹配 + 自动掩码
- **权限清单** — 未声明工具默认拒绝（fail-closed）
- **熔断器** — CLOSED → OPEN → HALF_OPEN 自动恢复
- **预算控制** — 日预算 + 任务预算 + 自动重置

---

## 快速开始

### 1. 安装
```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 2. 配置 LLM
```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key 和 Base URL
```

### 3. 启动
```bash
python main.py
```

### 4. 可用命令
| 命令 | 说明 |
|------|------|
| `/status` | 查看系统状态 |
| `/skill` | 管理 Skill |
| `/mcp` | 管理 MCP 服务器 |
| `/history` | 查看对话历史 |
| `/eval` | 运行评估 |
| `/health` | 系统健康报告 |
| `/traces` | 查看 Trace 记录 |
| `/optimization` | 优化器状态 |
| `exit` / `/exit` | 退出 |

---

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
│  (五层)   │ │Logging   │ │ Optimizer  │ │  Pipeline    │
└──────────┘ └──────────┘ └────────────┘ └──────────────┘
```

---

## 配置

### LLM 配置 (`configs/llm.yaml`)

```yaml
llm:
  provider: openai          # 兼容 OpenAI 协议
  model: sensenova-6.7-flash-lite
  api_key: ${LLM_API_KEY}   # 从 .env 读取
  base_url: ${LLM_BASE_URL:}
  fallback:
    chain:
      - deepseek-v4-flash    # 主模型失败时兜底
```

### 多模型路由
| 用途 | 模型 |
|------|------|
| 规划 (planning) | sensenova-6.7-flash-lite |
| 修复 (repair) | deepseek-v4-flash |
| 评估 (judge) | deepseek-v4-flash |
| 对话 (chat) | sensenova-6.7-flash-lite |

### 环境变量 (`.env`)
```bash
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-endpoint/v1
# TAVILY_API_KEY=your-tavily-key   # 可选：网络搜索
```

---

## 项目结构

```
long/
├── main.py                 # 入口
├── configs/                # YAML 配置文件
│   ├── llm.yaml            # LLM 模型配置
│   ├── security.yaml       # 安全与部署配置
│   ├── feature_flags.yaml  # 渐进式发布
│   └── ...
├── src/long/
│   ├── cli.py              # CLI 入口与 LongSystem 集成
│   ├── agent/              # 子 Agent 调度 (Worker/Planner/Critic/Runner/Escalation)
│   ├── capabilities/       # 工具注册与 MCP/Skill 支持
│   ├── cognitive/          # 认知运行时 (StateGraph/Planner/Reflection)
│   ├── components/         # 组件层 (ToolManager/PromptBuilder/SessionManager/Orchestrator)
│   ├── eval/               # 评估流水线 (结果/过程/系统/对抗)
│   ├── harness/            # 治理层 (FeatureFlag/Alert/OutputGuard/Permission/FeedbackLoop)
│   ├── interaction/        # 交互控制器与适配器 (CLI/WebUI)
│   ├── ir/                 # 形式化中间表示 (PlanIR/StateMachine/LTL/TypeChecker)
│   ├── llm/                # LLM 客户端 (调用/缓存/中间件/降级)
│   ├── memory/             # 五层记忆系统 (短期/工作/语义/情景/过程)
│   ├── observability/      # 追踪/日志/仪表盘
│   ├── optimization/       # 自动优化引擎 (OODA 循环)
│   ├── sandbox/            # 沙箱执行环境 (进程隔离/资源限制/代码扫描)
│   └── session/            # 会话持久化/偏好/画像
├── tests/                  # 560+ 测试用例
│   ├── test_e2e_security.py    # 安全 E2E 测试 (21)
│   ├── test_e2e_harness.py     # 治理 E2E 测试 (14)
│   ├── test_e2e_llm.py         # LLM E2E 测试 (16)
│   └── test_e2e_real_llm.py    # 真实 LLM 调用测试 (12)
└── .env.example            # 环境变量模板
```

---

## 测试

```bash
# 运行所有测试
uv run pytest tests/ -v

# 运行真实 LLM 调用测试（需要配置 API Key）
uv run pytest tests/test_e2e_real_llm.py -v -s

# 运行安全测试
uv run pytest tests/test_e2e_security.py -v

# 查看测试覆盖率
uv run pytest tests/ --cov=src/long --cov-report=html
```

### 测试统计
| 测试类型 | 数量 | 说明 |
|----------|------|------|
| 原有单元测试 | ~500 | 各模块独立测试 |
| 安全 E2E 测试 | 21 | 路径穿越/异常抑制/条件 fail-closed/PII 掩码/权限 |
| 治理 E2E 测试 | 14 | 并行工具/日预算/熔断器/YAML 回退 |
| LLM E2E 测试 | 16 | Anthropic 路径/skill caller/验证反转/list_files |
| 真实 LLM 测试 | 12 | 实际调用 SenseNova API 验证完整链路 |

---

## 许可

MIT License

---

## 技术栈

- **Python** ≥ 3.12
- **OpenAI SDK** — LLM API 调用（兼容 OpenAI/SenseNova/DeepSeek 等）
- **Pydantic** — 数据校验与 Schema 定义
- **Rich / prompt-toolkit** — 终端交互
- **FastAPI / uvicorn** — Web 服务
- **ChromaDB** — 向量存储
- **Matplotlib** — 图表生成