---
name: filesystem
description: |
  文件系统操作工具集。支持在工作区内读取文件、写入文件、删除文件、列出目录。
  TRIGGER when: 用户需要读写文件、查看目录内容、列出文件(ls)、删除文件、创建文件、
  查看文件内容(cat)、编辑文件、保存内容到文件时。
  DO NOT TRIGGER when: 用户只是讨论文件概念、文件格式等非实际操作场景。
metadata:
  category: builtin
  version: "1.0.0"
  tools:
    - list_files
    - read_file
    - write_file
    - delete_file
---

# Filesystem Skill (内置工具)

工作区文件系统操作，所有路径均相对于工作区根目录，具有路径穿越和符号链接防护。

## 工具列表

### list_files
列出目录下的文件和子目录。
- 参数: `path` - 目录路径（留空=根目录）
- 输出: `[D]` 目录 / `[F]` 文件

### read_file
读取文件内容。
- 参数: `path` - 文件路径（必填）

### write_file
写入内容到文件，自动创建父目录。
- 参数: `path` - 文件路径（必填）, `content` - 内容（必填）

### delete_file
删除文件或目录。
- 参数: `path` - 路径（必填）

## 安全机制
- 所有路径限制在工作区 (`./workspace/`) 内
- 禁止绝对路径和 `..` 路径穿越
- 符号链接逃逸检测