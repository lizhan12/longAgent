#!/usr/bin/env python3
"""
天气报告生成脚本 - 增强版（优化版）
包含：实时天气、7天预报、图表、穿衣指数、出行指南

优化：
1. 气压显示为 kPa（从 hPa 转换）
2. 数据校验，异常数据用 - 代替
3. 穿衣指数去掉温度描述
"""
import os
import sys
import time
import requests
import json
import jwt
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 切换到qweather目录
os.chdir('/root/.openclaw/workspace/qweather')

# 和风天气API配置
PROJECT_ID = os.environ.get('QWEATHER_PROJECT_ID', '282GDW3P2B')
KEY_ID = 'CHWF88AXFP'

# Private Key
with open('ed25519-private.pem', 'r') as f:
    private_key = f.read()

# 天气代码映射
WEATHER_EMOJIS = {
    '100': '☀️', '101': '🌤️', '102': '⛅', '103': '🌥️', '104': '☁️',
    '150': '🌞', '153': '🌙', '200': '⛈️', '201': '🌌️',
    '210': '🌧️', '211': '🌧️', '212': '🌧️', '213': '🌧️',
    '215': '🌨️', '220': '🌨️',
    '300': '🌨️', '301': '🌧️', '302': '🌧️', '303': '🌧️',
    '305': '🌧️', '306': '🌧️', '307': '🌧️',
    '310': '🌧️', '311': '🌧️', '312': '🌧️', '313': '🌧️',
    '400': '🌫️', '401': '🌫️', '402': '🌫️', '403': '🌫️',
    '500': '🌫️', '501': '🌫️', '502': '🌫️',
    '900': '🌡', '901': '🌡', '999': '🌡'
}

def validate_value(value, data_type):
    """数据校验，如果数据明显错误则用-代替"""
    try:
        if value is None or value == '':
            return '-'
        
        num_value = float(value)
        
        if data_type == 'temperature':
            # 温度合理范围：-50℃ 到 50℃
            if -50 <= num_value <= 50:
                return str(num_value)
            return '-'
        elif data_type == 'humidity':
            # 湿度合理范围：0% 到 100%
            if 0 <= num_value <= 100:
                return str(int(num_value))
            return '-'
        elif data_type == 'pressure':
            # 气压合理范围：800 hPa 到 1200 hPa (转换为 kPa 后：80-120 kPa)
            if 800 <= num_value <= 1200:
                return str(num_value / 10)  # 转换为 kPa
            return '-'
        elif data_type == 'visibility':
            # 能见度合理范围：0 km 到 50 km
            if 0 <= num_value <= 50:
                return str(num_value)
            return '-'
        elif data_type == 'precipitation':
            # 降水量合理范围：0 mm 到 500 mm
            if 0 <= num_value <= 500:
                return str(num_value)
            return
        else:
            return str(value)
    except (ValueError, TypeError):
        return '-'

def generate_jwt_token(project_id):
    """生成 JWT 令牌"""
    payload = {
        'iat': int(time.time()) - 30,
        'exp': int(time.time()) + 900,
        'sub': project_id
    }
    headers = {'kid': KEY_ID}
    return jwt.encode(payload, private_key, algorithm='EdDSA', headers=headers)

def get_city_location(city_name, token):
    """获取城市位置信息"""
    headers = {'Authorization': f'Bearer {token}'}
    url = f'https://k9436gc9br.re.qweatherapi.com/geo/v2/city/lookup?location={city_name}'
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200' and data.get('location'):
                return data['location'][0]
        print(f"❌ 无法找到城市：{city_name}")
        return None
    except Exception as e:
        print(f"查询城市失败：{e}")
        return None

def get_current_weather(city_id, token):
    """获取实时天气"""
    headers = {'Authorization': f'Bearer {token}'}
    url = f'https://k9436gc9br.re.qweatherapi.com/v7/weather/now?location={city_id}'
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200':
                return data.get('now', {})
        print(f"❌ 实时天气获取失败")
        return None
    except Exception as e:
        print(f"实时天气查询失败：{e}")
        return None

def get_7day_weather(city_id, token):
    """获取7天天气预报"""
    headers = {'Authorization': f'Bearer {token}'}
    url = f'https://k9436gc9br.re.qweatherapi.com/v7/weather/7d?location={city_id}'
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '200':
                return data.get('daily', [])
        print(f"❌ 获取天气数据失败")
        return []
    except Exception as e:
        print(f"查询天气失败：{e}")
        return []

def weather_code_to_emoji(code):
    """天气代码转emoji"""
    return WEATHER_EMOJIS.get(str(code), '🌡')

def to_camel_case(text):
    """将文本转换为驼峰命名"""
    city_mapping = {
        '杭州': 'hangzhou', '北京': 'beijing', '上海': 'shanghai',
        '广州': 'guangzhou', '深圳': 'shenzhen', '成都': 'chengdu',
        '重庆': 'chongqing', '西安': 'xian', '武汉': 'wuhan',
        '南京': 'nanjing', '天津': 'tianjin', '苏州': 'suzhou',
        '阜阳': 'fuyang', '临泉县': 'linquan'
    }
    return city_mapping.get(text, text.lower())

def get_clothing_index(temp, uv_index=None):
    """获取穿衣指数（包含防晒建议）"""
    clothing_data = {
        'uv_advice': ''
    }

    if temp < 5:
        clothing_data.update({
            'level': '冬季',
            'clothing': '羽绒服、棉裤、保暖内衣、雪地靴、围巾、手套',
            'tips': '注意防寒保暖，外出建议佩戴口罩和帽子'
        })
    elif temp < 10.0:
        clothing_data.update({
            'level': '深秋',
            'clothing': '薄羽绒服、毛衣、牛仔裤、保暖鞋',
            'tips': '天气较冷，注意保暖'
        })
    elif temp < 15.0:
        clothing_data.update({
            'level': '初秋',
            'clothing': '风衣、薄毛衣、长裤、运动鞋',
            'tips': '早晚温差大，适当增减衣物'
        })
    elif temp < 20.0:
        clothing_data.update({
            'level': '春秋',
            'clothing': '夹克、长袖T恤、牛仔裤、运动鞋',
            'tips': '天气舒适'
        })
    elif temp < 25.0:
        clothing_data.update({
            'level': '春夏',
            'clothing': '薄外套、短袖T恤、牛仔裤/短裤、运动鞋',
            'tips': '天气温暖，适合轻装出行'
        })
    elif temp < 30.0:
        clothing_data.update({
            'level': '初夏',
            'clothing': '短袖T恤、短裤、凉鞋',
            'tips': '天气较热'
        })
    else:
        clothing_data.update({
            'level': '盛夏',
            'clothing': '背心、短裤、凉鞋、遮阳帽',
            'tips': '天气炎热，减少多喝水'
        })

    # 添加防晒建议
    if uv_index is not None:
        if uv_index >= 11:
            clothing_data['uv_advice'] = "🌞 紫外线极强（UV指数{}），外出必须做好防晒：防晒霜、墨镜、遮阳帽、太阳伞".format(uv_index)
        elif uv_index >= 8:
            clothing_data['uv_advice'] = "☀️ 紫外线很强（UV指数{}），外出请做好防晒：防晒霜、墨镜、遮阳帽".format(uv_index)
        elif uv_index >= 6:
            clothing_data['uv_advice'] = "☀️ 紫外线较强（UV指数{}），外出建议防晒".format(uv_index)
        elif uv_index >= 3:
            clothing_data['uv_advice'] = "🧢 紫外线中等（UV指数{}），适当防晒".format(uv_index)

    return clothing_data

def get_travel_guide(weather_data):
    """获取出行指南"""
    guides = []
    
    temp_avg = (float(weather_data.get('tempMin', 20)) + float(weather_data.get('tempMax', 20))) / 2
    if temp_avg < 5:
        guides.append('❄️ 天气寒冷，建议减少户外活动，注意防滑')
    elif temp_avg < 15:
        guides.append('🧥 天气较凉，建议携带外套')
    elif temp_avg > 30:
        guides.append('☀️ 天气炎热，避免长时间暴晒，多补充水分')
    elif 15 <= temp_avg <= 25:
        guides.append('🌡️ 温度舒适，适合户外活动和出行')
    
    precip = float(weather_data.get('precip', 0))
    weather_code = str(weather_data.get('iconDay', '100'))
    if precip > 0 or weather_code.startswith(('2', '3')):
        guides.append(f'☔ 有降雨（预估降水量{precip}mm），外出请携带雨具，注意路面湿滑')
    else:
        guides.append('☀️ 无降雨，户外活动条件良好')
    
    wind_scale_str = str(weather_data.get('windScaleDay', '0'))
    wind_scale = int(wind_scale_str.split('-')[0]) if '-' in wind_scale_str else int(wind_scale_str)
    if wind_scale >= 6:
        guides.append(f'💨 风力较大（{weather_data.get("windScaleDay", "0")}级），注意防风，避免高空作业')
    else:
        guides.append('💨 风力较小，适合户外活动')
    
    humidity = int(weather_data.get('humidity', 50))
    if humidity > 80:
        guides.append('💧 空气湿度大，体感闷热，注意通风')
    elif humidity < 30:
        guides.append('🌵 空气干燥，注意补水和皮肤保湿')
    else:
        guides.append(f'💧 空气湿度适宜（{humidity}%），体感舒适')
    
    # 紫外线提示已移到穿衣指数段落
    vis = float(weather_data.get('vis', 10))
    if vis < 5:
        guides.append('🌫️ 能见度较低，驾车注意安全，减速慢行')
    else:
        guides.append(f'👁️ 能见度良好（{vis} km），出行安全')
    
    return guides if guides else ['🌤️ 天气良好，适合出行']

def find_nice_ticks(min_val, max_val, num_ticks=6):
    """找到合适的刻度范围"""
    range_val = max_val - min_val
    ideal_step = range_val / (num_ticks - 1) if range_val > 0 else 1
    
    nice_steps = [1, 2, 5, 10, 0.5, 0.2, 0.1, 20, 25, 50]
    nice_steps.sort(key=lambda x: abs(x - ideal_step))
    
    for step in nice_steps[:8]:
        start = (min_val // step) * step
        if start > min_val:
            start -= step
        
        ticks = [start + i * step for i in range(num_ticks)]
        lower_bound_ok = ticks[0] <= min_val + ideal_step * 0.5
        upper_bound_ok = ticks[-1] >= max_val - ideal_step * 0.5
        
        if lower_bound_ok and upper_bound_ok:
            all_near_int = all(abs(t - round(t)) < 0.01 for t in ticks)
            if all_near_int:
                return ticks, True
    
    return np.linspace(min_val, max_val, num_ticks), False

def generate_chart(days, city_name):
    """生成双图表 - 优化版"""
    # X轴：只显示日期（月日），不显示星期
    dates = [f"{d['fxDate'][5:7]}月{d['fxDate'][8:]}日" for d in days]
    
    min_temps = [float(d['tempMin']) for d in days]
    max_temps = [float(d['tempMax']) for d in days]
    precips = [float(d['precip']) if d.get('precip') else 0 for d in days]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.suptitle(f'{city_name}未来7天天气预报', fontsize=14, fontweight='bold')

    # 左边Y轴：降水柱状图（宽度0.8，优化3）
    # 创建实际的bar高度（给小数值设置最小显示高度）
    min_display_height = 0.3  # 最小显示高度，确保小数值也能看到bar
    display_precips = [max(p, min_display_height) if p > 0 else 0 for p in precips]
    
    bars = ax1.bar(dates, display_precips, color='#45B7D1', alpha=0.7, edgecolor='white', linewidth=1, label='降水量', width=0.8)
    ax1.set_ylabel('降水量 (mm)', fontsize=11, color='#45B7D1')
    ax1.tick_params(axis='y', labelcolor='#45B7D1')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(bottom=0)

    # 优化1：降水的Y轴刻度从0开始
    max_precip = max(precips) if precips else 0
    precip_min = 0
    if max_precip > 0:
        precip_max = max_precip * 1.2 if max_precip > 1 else 2
    else:
        precip_max = 1

    precip_ticks, precip_int = find_nice_ticks(precip_min, precip_max, 6)
    ax1.set_yticks(precip_ticks)
    ax1.set_ylim(precip_min, precip_max)

    # 添加降水值标签（只显示数值大于0的）
    for bar, precip in zip(bars, precips):
        if precip > 0:
            if precip == int(precip):
                label = f'{int(precip)}mm'
            else:
                label = f'{precip:.1f}mm'
            # 标签位置使用实际数值的显示高度
            label_y = max(precip, min_display_height) + 0.1
            ax1.text(bar.get_x() + bar.get_width()/2, label_y,
                    label, ha='center', va='bottom', fontsize=9, color='#45B7D1')

    # 右边Y轴：温度折线图
    ax2 = ax1.twinx()
    ax2.plot(dates, max_temps, marker='o', linewidth=2.5, markersize=8, label='最高温 (°C)', color='#FF6B6B')
    ax2.plot(dates, min_temps, marker='o', linewidth=2.5, markersize=8, label='最低温 (°C)', color='#4ECDC4')
    ax2.set_ylabel('温度 (°C)', fontsize=11, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.grid(False)

    # 优化2：温度的刻度最大刻度一定要比最大温度值大
    min_temp = min(min_temps) if min_temps else 0
    max_temp = max(max_temps) if max_temps else 0
    
    # 计算温度刻度范围，确保最大刻度 > 最大温度值至少3度
    if max_temp > 0:
        temp_min = min_temp - 2
        temp_max = max_temp + 3
    else:
        temp_min = 0
        temp_max = 10
    
    temp_ticks, temp_int = find_nice_ticks(temp_min, temp_max, 6)
    ax2.set_yticks(temp_ticks)
    ax2.set_ylim(temp_min, temp_max)

    # 格式化刻度标签
    precip_labels = []
    for tick in precip_ticks:
        if tick == int(tick):
            precip_labels.append(f'{int(tick)}')
        else:
            precip_labels.append(f'{tick:.1f}')
    ax1.set_yticklabels(precip_labels)

    temp_labels = []
    for tick in temp_ticks:
        if tick == int(tick):
            temp_labels.append(f'{int(tick)}')
        else:
            temp_labels.append(f'{tick:.1f}')
    ax2.set_yticklabels(temp_labels)

    # 添加温度值标签
    for i, (max_t, min_t) in enumerate(zip(max_temps, min_temps)):
        ax2.text(i, max_t + 0.5, f'{int(max_t)}°', ha='center', va='bottom', fontsize=9, color='#FF6B6B')
        ax2.text(i, min_t - 0.5, f'{int(min_t)}°', ha='center', va='top', fontsize=9, color='#4ECDC4')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=True)

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')
    plt.tight_layout()

    # 保存图表到 /tmp 目录
    target_dir = '/tmp'
    os.makedirs(target_dir, exist_ok=True)
    city_safe = to_camel_case(city_name)
    chart_filename = f'{city_safe}WeatherChart.png'
    chart_path = os.path.join(target_dir, chart_filename)
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return chart_path

def generate_html_report(now_weather, days, chart_path, city_name, city_info):
    """生成HTML报告"""
    chart_url = f"http://106.12.82.2/static/{os.path.basename(chart_path)}"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    weekday_map_cn = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    
    html = f'<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n'
    html += '    <meta charset="UTF-8">\n'
    html += '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
    html += f'    <title>{city_name}天气报告</title>\n'
    html += '    <style>\n'
    html += '        * { margin: 0; padding: 0; box-sizing: border-box; }\n'
    html += '        body { font-family: "Microsoft YaHei", "SimHei", sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; color: #333; }\n'
    html += '        h1 { color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; font-size: 28px; }\n'
    html += '        h2 { color: #34495e; margin-top: 30px; margin-bottom: 15px; border-left: 4px solid #3498db; padding-left: 10px; font-size: 24px; }\n'
    html += '        .section { background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin: 20px 0; }\n'
    html += '        .table-wrapper { overflow-x: auto; margin: 20px 0; }\n'
    html += '        table { width: 100%; min-width: 720px; border-collapse: collapse; }\n'
    html += '        th, td { padding: 12px; text-align: center; border-bottom: 1px solid #ecf0f1; white-space: nowrap; }\n'
    html += '        th { background-color: #3498db; color: white; font-weight: bold; }\n'
    html += '        .info-box { background-color: #d1ecf1; border-left: 4px solid #17a2b8; padding: 10px 15px; margin: 10px 0; }\n'
    html += '        ul { padding-left: 20px; }\n'
    html += '        li { margin: 5px 0; }\n'
    html += '        img { max-width: 100%; height: auto; }\n'
    html += '    </style>\n'
    html += '</head>\n<body>\n'
    html += f'    <h1>🌤️ {city_name}天气报告</h1>\n'
    html += '    <div class="info-box">\n'
    html += f'        <p><strong>生成时间：</strong> {timestamp}</p>\n'
    # 地点显示格式：市级=城市，省；县级=县，市，省
    if city_info.get('level') == 'city' or not city_info.get('adm2') or city_info.get('name') == city_info.get('adm2'):
        # 市级
        location_str = f'{city_info.get("name", city_name)}，{city_info.get("adm1", "")}'
    else:
        # 县级
        location_str = f'{city_info.get("name", city_name)}，{city_info.get("adm2", "")}，{city_info.get("adm1", "")}'
    html += f'        <p><strong>地点：</strong> {location_str}</p>\n'
    html += '    </div>\n'
    
    # 1. 实时天气信息 - 优化版（数据校验 + 气压转换为 kPa）
    html += '    <h2>1. 实时天气信息</h2>\n'
    html += '    <div class="section">\n'
    if now_weather:
        html += '        <ul>\n'
        # 数据校验
        temp = validate_value(now_weather.get("temp"), 'temperature')
        feels_like = validate_value(now_weather.get("feelsLike"), 'temperature')
        humidity = validate_value(now_weather.get("humidity"), 'humidity')
        pressure = validate_value(now_weather.get("pressure"), 'pressure')  # 自动转换为 kPa
        visibility = validate_value(now_weather.get("vis"), 'visibility')
        
        html += f'            <li>🌡️ <strong>当前温度：</strong>{temp}°C (体感：{feels_like}°C)</li>\n'
        html += f'            <li>{weather_code_to_emoji(now_weather.get("icon", "999"))} <strong>天气状况：</strong>{now_weather.get("text", "-")}</li>\n'
        html += f'            <li>💨 <strong>风向风力：</strong>{now_weather.get("windDir", "-")} {now_weather.get("windScale", "-")}级</li>\n'
        html += f'            <li>💧 <strong>湿度：</strong>{humidity}%</li>\n'
        html += f'            <li>📊 <strong>气压：</strong>{pressure} kPa</li>\n'
        html += f'            <li>👁️ <strong>能见度：</strong>{visibility} km</li>\n'
        html += '        </ul>\n'
    else:
        html += '        <p>实时天气数据获取失败</p>\n'
    html += '    </div>\n'
    
    # 2. 天气预报
    html += '    <h2>2. 天气预报（未来7天）</h2>\n'
    html += '    <div class="section">\n'
    html += '        <div class="table-wrapper">\n'
    html += '            <table>\n'
    html += '                <thead><tr><th>日期</th><th>星期</th><th>天气</th><th>温度</th><th>降水</th><th>湿度</th><th>风力</th></tr></thead>\n'
    html += '                <tbody>\n'

    for day in days:
        fx_date = day.get('fxDate', '')
        date_obj = datetime.strptime(fx_date, '%Y-%m-%d')
        week = weekday_map_cn[date_obj.weekday()]
        icon = weather_code_to_emoji(day.get('iconDay', '100'))
        wind = f"{day.get('windDirDay', '')} {day.get('windScaleDay', '')}级"
        
        # 数据校验
        temp_min = validate_value(day.get('tempMin'), 'temperature')
        temp_max = validate_value(day.get('tempMax'), 'temperature')
        precip = validate_value(day.get('precip'), 'precipitation')
        humidity = validate_value(day.get('humidity'), 'humidity')
        
        html += f'                    <tr><td>{fx_date}</td><td>{week}</td><td>{icon} {day.get("textDay", "-")}</td><td>{temp_min}°C ~ {temp_max}°C</td><td>{precip}mm</td><td>{humidity}%</td><td>{wind}</td></tr>\n'

    html += '                </tbody></table></div></div>\n'
    
    # 3. 图表展示
    html += '    <h2>3. 温度与降水趋势图</h2>\n'
    html += '    <div class="section">\n'
    html += f'        <img src="{chart_url}" alt="天气图表">\n'
    html += '        <p style="margin-top: 15px;"><strong>图表说明：</strong>柱状图显示降水量（左Y轴），折线图显示温度（右Y轴）</p>\n'
    html += '    </div>\n'
    
    # 4. 穿衣指数 - 优化版（去掉温度描述，添加防晒建议）
    html += '    <h2>4. 穿衣指数</h2>\n'
    html += '    <div class="section">\n'
    if days:
        today = days[0]
        temp_avg = (float(today.get('tempMin', 20)) + float(today.get('tempMax', 20))) / 2
        uv_index = int(today.get('uvIndex', 0))
        clothing = get_clothing_index(temp_avg, uv_index)
        html += '        <ul>\n'
        # 不再显示"当前温度"这一行
        html += f'            <li>👕 <strong>穿衣建议：</strong>{clothing.get("clothing", "")}</li>\n'
        html += f'            <li>💡 <strong>温馨提示：</strong>{clothing.get("tips", "")}</li>\n'
        # 添加防晒建议
        if clothing.get('uv_advice'):
            html += f'            <li>{clothing.get("uv_advice")}</li>\n'
        html += '        </ul>\n'
    html += '    </div>\n'
    
    # 5. 出行指南
    html += '    <h2>5. 出行指南</h2>\n'
    html += '    <div class="section">\n'
    if days:
        today = days[0]
        guides = get_travel_guide(today)
        html += '        <ul>\n'
        for guide in guides:
            html += f'            <li>{guide}</li>\n'
        html += '        </ul>\n'
    html += '    </div>\n'
    
    html += '</body>\n</html>\n'
    return html

def generate_feishu_markdown(now_weather, days, city_name, city_info):
    """生成飞书文档的Markdown内容（分段版本）"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 地点显示
    if city_info.get('level') == 'city' or not city_info.get('adm') or city_info.get('name') == city_info.get('adm2'):
        location_str = f'{city_info.get("name", city_name)}，{city_info.get("adm1", "")}'
    else:
        location_str = f'{city_info.get("name", city_name)}，{city_info.get("adm2", "")}，{city_info.get("adm1", "")}'
    
    # 第1-3章内容
    markdown = f"# 🌤️ {city_name}天气报告\n\n"
    markdown += f"**生成时间：** {timestamp}\n\n"
    markdown += f"**地点：** {location_str}\n\n"
    
    # 1. 实时天气
    markdown += "## 1. 实时天气信息\n\n"
    if now_weather:
        temp = validate_value(now_weather.get("temp"), 'temperature')
        feels_like = validate_value(now_weather.get("feelsLike"), 'temperature')
        humidity = validate_value(now_weather.get("humidity"), 'humidity')
        pressure = validate_value(now_weather.get("pressure"), 'pressure')
        visibility = validate_value(now_weather.get("vis"), 'visibility')
        
        markdown += f"- 🌡️ 当前温度：{temp}°C (体感：{feels_like}°C)\n"
        markdown += f"- {weather_code_to_emoji(now_weather.get('icon', '999'))} 天气状况：{now_weather.get('text', '-')}\n"
        markdown += f"- 💨 风向风力：{now_weather.get('windDir', '-')} {now_weather.get('windScale', '-')}级\n"
        markdown += f"- 💧 湿度：{humidity}%\n"
        markdown += f"- 📊 气压：{pressure} kPa\n"
        markdown += f"- 👁️ 能见度：{visibility} km\n"
    markdown += "\n"
    
    # 2. 天气预报
    markdown += "## 2. 天气预报（未来7天）\n\n"
    markdown += "| 日期 | 星期 | 天气 | 温度 | 降水 | 湿度 | 风力 |\n"
    markdown += "|------|------|------|------|------|------|------|\n"
    
    weekday_map_cn = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    for day in days:
        fx_date = day.get('fxDate', '')
        date_obj = datetime.strptime(fx_date, '%Y-%m-%d')
        week = weekday_map_cn[date_obj.weekday()]
        icon = weather_code_to_emoji(day.get('iconDay', '100'))
        wind = f"{day.get('windDirDay', '')} {day.get('windScaleDay', '')}级"
        
        temp_min = validate_value(day.get('tempMin'), 'temperature')
        temp_max = validate_value(day.get('tempMax'), 'temperature')
        precip = validate_value(day.get('precip'), 'precipitation')
        humidity = validate_value(day.get('humidity'), 'humidity')
        
        markdown += f"| {fx_date} | {week} | {icon} {day.get('textDay', '-')} | {temp_min}°C ~ {temp_max}°C | {precip}mm | {humidity}% | {wind} |\n"
    markdown += "\n"
    
    # 3. 图表说明
    markdown += "## 3. 温度与降水趋势图\n\n"
    
    # 生成第4章内容（用于 append）
    part4_lines = []
    part4_lines.append("## 4. 穿衣指数\n\n")
    if days:
        today = days[0]
        temp_avg = (float(today.get('tempMin', 20)) + float(today.get('tempMax', 20))) / 2
        uv_index = int(today.get('uvIndex', 0))
        clothing = get_clothing_index(temp_avg, uv_index)
        part4_lines.append(f"- 👕 穿衣建议：{clothing.get('clothing', '')}\n")
        part4_lines.append(f"- 💡 温馨提示：{clothing.get('tips', '')}\n")
        # 添加防晒建议
        if clothing.get('uv_advice'):
            part4_lines.append(f"- {clothing.get('uv_advice')}\n")
    part4_lines.append("\n")
    
    # 生成第5章内容（用于 append）- 优化版：使用实时天气数据生成出行指南
    part5_lines = []
    part5_lines.append("## 5. 出行指南\n\n")
    
    # 使用实时天气数据生成出行指南（更准确）
    if now_weather:
        # 从实时天气数据构造出行指南所需的字段
        now_weather_for_guide = {
            'tempMin': now_weather.get('temp', 20),
            'tempMax': now_weather.get('temp', 20),
            'precip': 0,  # 实时天气没有precip字段，暂时设为0
            'iconDay': now_weather.get('icon', '100'),
            'windScaleDay': now_weather.get('windScale', '0'),
            'humidity': now_weather.get('humidity', 50),
            'vis': now_weather.get('vis', 10)
        }
        guides = get_travel_guide(now_weather_for_guide)
        for guide in guides:
            part5_lines.append(f"- {guide}\n")
    elif days:
        # 如果没有实时天气，回退到使用预报数据
        today = days[0]
        guides = get_travel_guide(today)
        for guide in guides:
            part5_lines.append(f"- {guide}\n")
    part5_lines.append("\n")
    
    return markdown, part4_lines, part5_lines

def validate_report_content(temp_dir):
    """验证报告内容是否完整"""
    errors = []
    warnings = []
    
    # 检查必要文件是否存在
    required_files = {
        'part1_3.md': '第1-3章内容（实时天气、7天预报、趋势图标题）',
        'part4.md': '第4章内容（穿衣指数）',
        'part5.md': '第5章内容（出行指南）',
        'chart_path.txt': '图表路径'
    }
    
    for filename, description in required_files.items():
        file_path = os.path.join(temp_dir, filename)
        if not os.path.exists(file_path):
            errors.append(f"❌ 缺少文件：{filename} ({description})")
    
    # 检查 part1_3.md 内容是否包含必要的章节
    part1_3_path = os.path.join(temp_dir, 'part1_3.md')
    if os.path.exists(part1_3_path):
        with open(part1_3_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_sections = {
            '## 1. 实时天气信息': '实时天气信息章节',
            '## 2. 天气预报': '天气预报章节',
            '## 3. 温度与降水趋势图': '趋势图章节标题',
            '| 日期 | 星期 | 天气 | 温度 | 降水 | 湿度 | 风力 |': '天气预报表格表头'
        }
        
        for section, description in required_sections.items():
            if section not in content:
                errors.append(f"❌ part1_3.md 缺少：{description} ({section})")
        
        # 检查实时天气数据关键字段
        required_fields = ['当前温度', '天气状况', '风向风力', '湿度', '气压', '能见度']
        for field in required_fields:
            if field not in content:
                warnings.append(f"⚠️ part1_3.md 可能缺少：{field}")
    
    # 检查 part4.md 内容
    part4_path = os.path.join(temp_dir, 'part4.md')
    if os.path.exists(part4_path):
        with open(part4_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if '## 4. 穿衣指数' not in content:
            errors.append("❌ part4.md 缺少：穿衣指数章节")
        if '穿衣建议' not in content and '温馨提示' not in content:
            warnings.append("⚠️ part4.md 可能缺少：穿衣建议或温馨提示")
    
    # 检查 part5.md 内容
    part5_path = os.path.join(temp_dir, 'part5.md')
    if os.path.exists(part5_path):
        with open(part5_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if '## 5. 出行指南' not in content:
            errors.append("❌ part5.md 缺少：出行指南章节")
    
    # 检查图表路径
    chart_path_file = os.path.join(temp_dir, 'chart_path.txt')
    if os.path.exists(chart_path_file):
        with open(chart_path_file, 'r', encoding='utf-8') as f:
            chart_path = f.read().strip()
        if not os.path.exists(chart_path):
            errors.append(f"❌ 图表文件不存在：{chart_path}")
    else:
        errors.append("❌ 缺少图表路径文件")
    
    # 输出验证结果
    print("\n" + "="*50)
    print("📋 报告内容完整性验证")
    print("="*50)
    
    if not errors and not warnings:
        print("✅ 所有检查通过！报告内容完整。")
        return True
    else:
        if errors:
            print("\n❌ 发现以下问题：")
            for error in errors:
                print(f"   {error}")
        if warnings:
            print("\n⚠️ 发现以下警告：")
            for warning in warnings:
                print(f"   {warning}")
        print(f"\n❌ 验证失败！发现 {len(errors)} 个错误，{len(warnings)} 个警告。")
        return False

if __name__ == "__main__":
    city_name = sys.argv[1] if len(sys.argv) > 1 else "杭州"
    
    print(f"🔍 正在生成{city_name}天气报告（飞书文档版）...")
    
    token = generate_jwt_token(PROJECT_ID)
    
    city_info = get_city_location(city_name, token)
    if not city_info:
        print(f"❌ 无法找到城市：{city_name}")
        sys.exit(1)
    
    city_id = city_info.get('id')
    print(f"✅ 找到城市：{city_info.get('name')}（{city_id}）")
    
    print("📊 获取实时天气...")
    now_weather = get_current_weather(city_id, token)
    
    print("📊 获取7天天气预报...")
    days = get_7day_weather(city_id, token)
    if not days:
        print("❌ 获取天气数据失败")
        sys.exit(1)
    
    print(f"✅ 获取到{len(days)}天天气数据")
    
    # 生成图表
    chart_path = generate_chart(days, city_name)
    print(f"✅ 图表已生成：{chart_path}")
    
    # 生成飞书文档内容
    print("📝 生成飞书文档内容...")
    part1_3, part4_lines, part5_lines = generate_feishu_markdown(now_weather, days, city_name, city_info)
    print("✅ Markdown内容已生成")
    
    # 保存到临时文件，供后续使用
    import tempfile
    temp_dir = '/tmp/weather_feishu'
    os.makedirs(temp_dir, exist_ok=True)
    
    with open(f'{temp_dir}/part1_3.md', 'w', encoding='utf-8') as f:
        f.write(part1_3)
    
    with open(f'{temp_dir}/part4.md', 'w', encoding='utf-8') as f:
        f.write(''.join(part4_lines))
    
    with open(f'{temp_dir}/part5.md', 'w', encoding='utf-8') as f:
        f.write(''.join(part5_lines))
    
    with open(f'{temp_dir}/chart_path.txt', 'w', encoding='utf-8') as f:
        f.write(chart_path)
    
    with open(f'{temp_dir}/city_name.txt', 'w', encoding='utf-8') as f:
        f.write(city_name)
    
    print(f"✅ 飞书文档内容已保存到临时目录：{temp_dir}")
    print(f"   - 第1-3章：{temp_dir}/part1_3.md")
    print(f"   - 第4章：{temp_dir}/part4.md")
    print(f"   - 第5章：{temp_dir}/part5.md")
    print(f"   - 图表路径：{temp_dir}/chart_path.txt")
    
    # 验证报告内容完整性
    is_valid = validate_report_content(temp_dir)
    
    if not is_valid:
        print(f"\n❌ 报告生成失败，内容不完整！")
        sys.exit(1)
    
    # 最终报告
    print("\n" + "="*50)
    print("🎉 天气报告生成完成！")
    print("="*50)
    print(f"📊 图表路径：{chart_path}")
    print(f"📁 临时目录：{temp_dir}")
    print(f"✅ 所有步骤执行成功，飞书文档将由 Chandler 继续创建")
