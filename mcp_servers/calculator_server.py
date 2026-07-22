"""计算器 MCP Server — 提供数学计算工具

用法: python mcp_servers/calculator_server.py
"""

import json
import math
import sys


def handle_initialize(params: dict) -> dict:
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "calculator", "version": "1.0.0"},
    }


def handle_tools_list() -> list[dict]:
    return [
        {
            "name": "calc_add",
            "description": "加法运算",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个数"},
                    "b": {"type": "number", "description": "第二个数"},
                },
                "required": ["a", "b"],
            },
        },
        {
            "name": "calc_sqrt",
            "description": "平方根",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "数值"},
                },
                "required": ["x"],
            },
        },
        {
            "name": "calc_pow",
            "description": "幂运算",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "base": {"type": "number", "description": "底数"},
                    "exp": {"type": "number", "description": "指数"},
                },
                "required": ["base", "exp"],
            },
        },
    ]


def handle_tools_call(name: str, arguments: dict) -> str:
    if name == "calc_add":
        return str(arguments["a"] + arguments["b"])
    elif name == "calc_sqrt":
        return str(math.sqrt(arguments["x"]))
    elif name == "calc_pow":
        return str(math.pow(arguments["base"], arguments["exp"]))
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