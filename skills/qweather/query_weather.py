#!/usr/bin/env python3
"""查询指定城市天气 - 按照 qweather skill 的规范执行"""

import os
import sys
import time
import json
import jwt
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 1. 读取 API 凭证
# ============================================================

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))

# PROJECT_ID 和 KEY_ID（JWT 必需，有默认值），兼容带/不带 QWEATHER_ 前缀
PROJECT_ID = os.environ.get('QWEATHER_PROJECT_ID') or os.environ.get('PROJECT_ID', '282GDW3P2B')
KEY_ID = os.environ.get('QWEATHER_KEY_ID') or os.environ.get('KEY_ID', 'CHWF88AXFP')

# API 地址
API_HOST = os.environ.get('QWEATHER_API_HOST', 'k9436gc9br.re.qweatherapi.com')
# 如果 API_HOST 不含协议，自动补上 https://
if not API_HOST.startswith('http'):
    API_HOST = f'https://{API_HOST}'

# 私钥：优先环境变量，其次 skill 目录下的文件，最后尝试项目标准路径
private_key = os.environ.get('QWEATHER_PRIVATE_KEY')
if not private_key:
    # 尝试从脚本所在目录查找
    key_path = os.path.join(SKILL_DIR, 'ed25519-private.pem')
    if os.path.exists(key_path):
        with open(key_path, 'r') as f:
            private_key = f.read()
    else:
        # 回退：从多个标准位置查找（兼容 execute_code 在临时目录运行的情况）
        fallback_paths = [
            'skills/qweather/ed25519-private.pem',
            'workspace/skills/qweather/ed25519-private.pem',
            os.path.join(os.getcwd(), 'skills/qweather/ed25519-private.pem'),
            os.path.join(os.getcwd(), 'workspace/skills/qweather/ed25519-private.pem'),
        ]
        for p in fallback_paths:
            if os.path.exists(p):
                with open(p, 'r') as f:
                    private_key = f.read()
                break
if not private_key:
    print("错误: 未找到和风天气私钥文件", file=sys.stderr)
    print(json.dumps({'error': '未找到私钥。请设置环境变量 QWEATHER_PRIVATE_KEY 或将 ed25519-private.pem 放到 skills/qweather/ 目录下'}, ensure_ascii=False))
    sys.exit(1)

# ============================================================
# 2. 生成 JWT Token
# ============================================================

def generate_token(project_id, key_id, private_key):
    payload = {
        'iat': int(time.time()) - 30,  # 提前30秒，防止时钟偏差
        'exp': int(time.time()) + 900,  # 15分钟有效期
        'sub': project_id
    }
    headers = {'kid': key_id}
    return jwt.encode(payload, private_key, algorithm='EdDSA', headers=headers)


def api_get(url, headers, timeout=8, retries=2):
    """带重试的 API 请求，超时和重试次数较低以避免沙箱总超时"""
    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            return resp
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(1)
        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(1)
    raise last_error  # type: ignore[misc]


def validate_range(value, min_val, max_val):
    """校验数值是否在合理范围内，超出返回 None"""
    if value is None:
        return None
    try:
        v = float(value)
        if min_val <= v <= max_val:
            return str(value)
        return None
    except (ValueError, TypeError):
        return None


def query_city_id(city_name, token):
    """查询城市 ID"""
    url = f'{API_HOST}/geo/v2/city/lookup?location={city_name}'
    headers = {'Authorization': f'Bearer {token}'}
    resp = api_get(url, headers=headers)
    data = resp.json()

    if data.get('code') != '200':
        return None, data

    locations = data.get('location', [])
    if not locations:
        return None, {'code': '404', 'error': f'未找到城市: {city_name}'}

    return locations[0], data


def query_now_weather(city_id, token):
    """查询实时天气"""
    url = f'{API_HOST}/v7/weather/now?location={city_id}'
    headers = {'Authorization': f'Bearer {token}'}
    resp = api_get(url, headers=headers)
    data = resp.json()

    if data.get('code') != '200':
        return None, data

    now = data.get('now', {})
    result = {
        'temp': now.get('temp'),
        'feelsLike': now.get('feelsLike'),
        'text': now.get('text'),
        'windDir': now.get('windDir'),
        'windScale': now.get('windScale'),
        'humidity': now.get('humidity'),
        'pressure': None,
        'vis': now.get('vis'),
    }

    if now.get('pressure'):
        try:
            pressure_hpa = float(now['pressure'])
            pressure_kpa = pressure_hpa / 10
            if 800 <= pressure_hpa <= 1200:
                result['pressure'] = str(pressure_kpa)
        except (ValueError, TypeError):
            pass

    result['temp'] = validate_range(result['temp'], -50, 50)
    result['feelsLike'] = validate_range(result['feelsLike'], -50, 50)
    result['humidity'] = validate_range(result['humidity'], 0, 100)
    result['vis'] = validate_range(result['vis'], 0, 50)

    return result, data


def query_7d_weather(city_id, token):
    """查询7天预报"""
    url = f'{API_HOST}/v7/weather/7d?location={city_id}'
    headers = {'Authorization': f'Bearer {token}'}
    resp = api_get(url, headers=headers)
    data = resp.json()

    if data.get('code') != '200':
        return None, data

    daily_list = []
    for day in data.get('daily', []):
        entry = {
            'fxDate': day.get('fxDate'),
            'tempMax': day.get('tempMax'),
            'tempMin': day.get('tempMin'),
            'textDay': day.get('textDay'),
            'precip': day.get('precip'),
            'humidity': day.get('humidity'),
            'windDirDay': day.get('windDirDay'),
            'windScaleDay': day.get('windScaleDay'),
            'uvIndex': day.get('uvIndex'),
        }

        entry['tempMax'] = validate_range(entry['tempMax'], -50, 50)
        entry['tempMin'] = validate_range(entry['tempMin'], -50, 50)
        entry['humidity'] = validate_range(entry['humidity'], 0, 100)
        entry['precip'] = validate_range(entry['precip'], 0, 500)

        daily_list.append(entry)

    return daily_list, data


# ============================================================
# 执行查询
# ============================================================

def query_city(city_name: str, token: str) -> dict | None:
    """查询单个城市天气，返回结果字典或 None。
    部分数据获取失败时仍返回可用数据，而非整体返回 None。"""
    try:
        location_info, city_resp = query_city_id(city_name, token)
        if location_info is None:
            return None

        city_id = location_info['id']

        result = {
            'city': location_info.get('name', city_name),
            'adm1': location_info.get('adm1', ''),
            'adm2': location_info.get('adm2', ''),
        }

        now_data, now_resp = query_now_weather(city_id, token)
        if now_data is None:
            # 实况获取失败，仍返回城市信息
            result['now'] = None
        else:
            result['now'] = now_data

        daily_data, daily_resp = query_7d_weather(city_id, token)
        if daily_data is not None:
            result['daily'] = daily_data

        # 至少有实况或预报数据才算成功
        if result.get('now') is None and 'daily' not in result:
            return None

        return result
    except Exception as e:
        print(f"查询 {city_name} 异常: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def format_weather_brief(result: dict) -> str:
    """将天气结果格式化为简洁文本，便于 LLM 理解"""
    lines = []
    city = result.get('city', '')
    adm1 = result.get('adm1', '')
    lines.append(f"【{city}（{adm1}）】")

    now = result.get('now')
    if now:
        parts = [f"{now.get('text', '?')}"]
        if now.get('temp'): parts.append(f"气温{now['temp']}℃")
        if now.get('feelsLike'): parts.append(f"体感{now['feelsLike']}℃")
        if now.get('windDir'): parts.append(f"{now['windDir']}{now.get('windScale', '')}级")
        if now.get('humidity'): parts.append(f"湿度{now['humidity']}%")
        lines.append("  实况: " + "，".join(parts))
    else:
        lines.append("  实况: 数据获取失败")

    daily = result.get('daily', [])
    if daily:
        lines.append("  预报:")
        for d in daily[:3]:
            day_parts = [d.get('fxDate', '')]
            if d.get('textDay'): day_parts.append(d['textDay'])
            if d.get('tempMax') and d.get('tempMin'):
                day_parts.append(f"{d['tempMin']}~{d['tempMax']}℃")
            if d.get('windDirDay'): day_parts.append(f"{d['windDirDay']}{d.get('windScaleDay', '')}级")
            lines.append("    " + " ".join(day_parts))
        if len(daily) > 3:
            lines.append(f"    ...共{len(daily)}天预报")

    return "\n".join(lines)


def main():
    # 支持多城市：用逗号、空格、顿号分隔
    raw_args = sys.argv[1:] if len(sys.argv) > 1 else ['杭州']
    cities = []
    for arg in raw_args:
        for sep in [',', '，', '、', ' ']:
            arg = arg.replace(sep, '|')
        cities.extend(c for c in arg.split('|') if c.strip())
    cities = [c.strip() for c in cities if c.strip()]

    if not cities:
        print(json.dumps({'error': '请提供城市名'}, ensure_ascii=False))
        sys.exit(1)

    token = generate_token(PROJECT_ID, KEY_ID, private_key)

    if len(cities) == 1:
        result = query_city(cities[0], token)
        if result is None:
            print(json.dumps({'error': f'查询城市 {cities[0]} 失败'}, ensure_ascii=False))
            sys.exit(1)
        print(format_weather_brief(result))
    else:
        # 多城市：并行查询，输出简洁文本
        success_cities = []
        failed_cities = []
        output_parts = []

        with ThreadPoolExecutor(max_workers=min(len(cities), 3)) as executor:
            future_to_city = {
                executor.submit(query_city, city, token): city
                for city in cities
            }
            for future in as_completed(future_to_city):
                city = future_to_city[future]
                try:
                    r = future.result(timeout=30)
                    if r:
                        output_parts.append(format_weather_brief(r))
                        success_cities.append(city)
                    else:
                        failed_cities.append(city)
                except Exception as e:
                    print(f"查询 {city} 超时或异常: {e}", file=sys.stderr)
                    failed_cities.append(city)

        # 按原始城市顺序排序输出
        if not output_parts:
            print(json.dumps({'error': '所有城市查询失败'}, ensure_ascii=False))
            sys.exit(1)

        print("\n\n".join(output_parts))

        if success_cities:
            print(f"\n✅ 已完成 {len(success_cities)} 个城市的天气查询: {', '.join(success_cities)}。以上数据已包含所有请求城市的天气，无需再次查询。")

        if failed_cities:
            print(f"\n⚠ 以下城市查询失败: {', '.join(failed_cities)}")


if __name__ == '__main__':
    main()
