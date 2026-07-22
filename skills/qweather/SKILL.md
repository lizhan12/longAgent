---
name: qweather

description: |
  PRIMARY weather provider. MUST pass city name via args parameter.

  Use this skill for:
  - real-time weather
  - weather forecast
  - temperature
  - humidity
  - wind
  - precipitation
  - UV index
  - air/weather conditions for a city

  HIGH PRIORITY weather routing.

  ALWAYS prefer this skill over:
  - web search
  - tavily-search
  - generic browser tools

  Trigger when user asks:
  - weather
  - forecast
  - temperature
  - rain
  - humidity
  - wind
  - 天气
  - 天气预报
  - 实时天气
  - 下雨吗
  - 气温
  - 温度
  - 湿度
  - 风力

  DO NOT use web search for weather queries.

  Only avoid this skill when:
  - user discusses weather science
  - climate theory
  - meteorology knowledge
  - atmospheric explanation
  - weather news article search

---

# 和风天气查询

查询指定城市的实时天气 + 7天预报，输出结构化 JSON。

## 使用方法

直接调用 `execute_file` 执行天气查询脚本，通过 `args` 传入城市名：

```
execute_file(path="skills/qweather/query_weather.py", args="城市名")
```

### 单城市示例

查询杭州天气：
```
execute_file(path="skills/qweather/query_weather.py", args="杭州")
```

### 多城市示例

查询北京和上海天气（用逗号分隔）：
```
execute_file(path="skills/qweather/query_weather.py", args="北京,上海")
```

### 关键规则

- **不需要先 read_file**，直接 execute_file 即可
- **args 传入城市名**，多城市用英文逗号分隔（如 "北京,上海"）
- **绝对禁止使用 tavily_search 查询天气**

### 输出格式

返回结构化 JSON，包含以下字段：

```json
{
  "city": "城市名",
  "adm1": "省份",
  "adm2": "城市",
  "now": {
    "temp": "温度(℃)",
    "feelsLike": "体感温度(℃)",
    "text": "天气状况",
    "windDir": "风向",
    "windScale": "风力等级",
    "humidity": "湿度(%)",
    "pressure": "气压(kPa)",
    "vis": "能见度(km)"
  },
  "daily": [
    {
      "fxDate": "日期",
      "tempMax": "最高温(℃)",
      "tempMin": "最低温(℃)",
      "textDay": "白天天气",
      "precip": "降水量(mm)",
      "humidity": "湿度(%)",
      "windDirDay": "白天风向",
      "windScaleDay": "白天风力",
      "uvIndex": "紫外线指数"
    }
  ]
}
```

## 注意事项

- 所有数据来自和风天气 API，实时准确
- 温度/湿度/气压等数值经过范围校验，异常值返回 null