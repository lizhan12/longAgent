"""文件系统 MCP Server — 提供文件读写工具

用法: python mcp_servers/filesystem_server.py
通过 MCPClient 的 stdio transport 连接。
"""

import json
import sys
from pathlib import Path


SANDBOX_DIR = Path("./workspace/sandbox")


def handle_initialize(params: dict) -> dict:
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "filesystem", "version": "1.0.0"},
    }


def handle_tools_list() -> list[dict]:
    return [
        {
            "name": "read_file",
            "description": "读取文件内容",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"}
                },
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "写入文件内容",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "文件内容"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "list_files",
            "description": "列出目录下的文件",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径"}
                },
            },
        },
    ]


def handle_tools_call(name: str, arguments: dict) -> str:
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

    if name == "read_file":
        file_path = SANDBOX_DIR / arguments["path"]
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        return f"文件不存在: {arguments['path']}"

    elif name == "write_file":
        file_path = SANDBOX_DIR / arguments["path"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(arguments["content"], encoding="utf-8")
        return f"写入成功: {arguments['path']}"

    elif name == "list_files":
        dir_path = SANDBOX_DIR / arguments.get("path", "")
        if dir_path.exists() and dir_path.is_dir():
            files = [str(p.relative_to(SANDBOX_DIR)) for p in dir_path.iterdir()]
            return "\n".join(files) if files else "(空目录)"
        return "(目录不存在)"

    return f"未知工具: {name}"


def main() -> None:
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            request = json.loads(line.strip())
            request_id = request.get("id")
            method = request.get("method")

            if method == "initialize":
                result = handle_initialize(request.get("params", {}))
                response = {"jsonrpc": "2.0", "id": request_id, "result": result}
            elif method == "tools/list":
                result = handle_tools_list()
                response = {"jsonrpc": "2.0", "id": request_id, "result": {"tools": result}}
            elif method == "tools/call":
                params = request.get("params", {})
                result = handle_tools_call(params.get("name", ""), params.get("arguments", {}))
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": result}]},
                }
            else:
                response = {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"未知方法: {method}"}}

            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()

        except json.JSONDecodeError:
            continue
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()