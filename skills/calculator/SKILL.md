---
name: calculator
description: |
  执行数学计算和运算。支持基本四则运算、幂运算、括号优先级等。
  TRIGGER when: 用户需要进行数学计算、算术运算、加减乘除、数值计算时。
  包括表达式计算、两数相加、两数相乘等操作。
  DO NOT TRIGGER when: 用户只是讨论数学概念而非实际计算。
metadata:
  category: utility
  version: "1.0.0"
  tools:
    - calculate
    - math_add
    - math_multiply
---

# Calculator Skill

执行安全的数学计算，支持表达式解析和基本运算。

## 工具列表

### calculate
计算数学表达式，支持 `+`, `-`, `*`, `/`, `**`, `//`, `%` 和括号。
示例: `3 + 4 * 2` → `11`

### math_add
两个数字相加。
示例: `10, 20` → `30`

### math_multiply
两个数字相乘。
示例: `5, 6` → `30`