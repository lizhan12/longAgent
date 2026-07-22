SKILL_NAME = "calculator"
SKILL_VERSION = "1.0.0"
SKILL_DESCRIPTION = "执行基本数学运算"
SKILL_PERMISSIONS = ["compute.cpu"]
SKILL_TOOLS = ["calculate", "math_add", "math_multiply"]
SKILL_DEPENDENCIES = []
SKILL_ENTRY_POINT = "calculate"

_OPERATORS = {"+": (lambda a, b: a + b), "-": (lambda a, b: a - b),
              "*": (lambda a, b: a * b), "/": (lambda a, b: a / b),
              "**": (lambda a, b: a ** b), "//": (lambda a, b: a // b),
              "%": (lambda a, b: a % b)}


def _tokenize(expr: str) -> list:
    tokens, current = [], ""
    for ch in expr:
        if ch in " \t":
            continue
        if ch in "+-*/%()":
            if current:
                tokens.append(current)
                current = ""
            if ch == "*" and tokens and tokens[-1] == "*":
                tokens[-1] = "**"
            else:
                tokens.append(ch)
        else:
            current += ch
    if current:
        tokens.append(current)
    return tokens


def _parse_calc(tokens: list) -> float:
    def parse_expression(idx):
        left, idx = parse_term(idx)
        while idx < len(tokens) and tokens[idx] in ("+", "-"):
            op = tokens[idx]
            right, idx = parse_term(idx + 1)
            left = _OPERATORS[op](left, right)
        return left, idx

    def parse_term(idx):
        left, idx = parse_factor(idx)
        while idx < len(tokens) and tokens[idx] in ("*", "/", "//", "%"):
            op = tokens[idx]
            right, idx = parse_factor(idx + 1)
            left = _OPERATORS[op](left, right)
        return left, idx

    def parse_factor(idx):
        if idx < len(tokens) and tokens[idx] == "(":
            val, idx = parse_expression(idx + 1)
            if idx < len(tokens) and tokens[idx] == ")":
                return val, idx + 1
            return val, idx
        if idx < len(tokens) and tokens[idx] == "-":
            val, idx = parse_factor(idx + 1)
            return -val, idx
        token = tokens[idx]
        try:
            return float(token), idx + 1
        except (ValueError, TypeError):
            return 0.0, idx + 1

    result, _ = parse_expression(0)
    return result


def calculate(text: str) -> str:
    try:
        result = _parse_calc(_tokenize(text))
        if result == int(result):
            return str(int(result))
        return str(result)
    except Exception:
        return f"无法计算: {text}"


def math_add(text: str) -> str:
    parts = text.replace(",", " ").split()
    a = float(parts[0]) if parts else 0
    b = float(parts[1]) if len(parts) > 1 else 0
    return str(a + b)


def math_multiply(text: str) -> str:
    parts = text.replace(",", " ").split()
    a = float(parts[0]) if parts else 0
    b = float(parts[1]) if len(parts) > 1 else 0
    return str(a * b)