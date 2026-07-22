/**
 * Long WebUI - Frontend Logic
 * 注意：所有 JS 动态创建的元素使用 styles.css 中的 CSS class，
 * 不使用 Tailwind 工具类（Tailwind CDN 不会扫描 JS 中的 class）。
 */

let ws = null;
let sessionId = null;
let reconnectTimer = null;
let isProcessing = false;
let currentStreamDiv = null;
var _streamRenderTimer = null;
var _simulateTimer = null;
var _lastFinishedStreamDiv = null;  // 记录最近一次完成的流式消息，用于最终下载按钮
var _finalMsgTimer = null;          // 延迟判定最终消息的定时器
var _isSimStreaming = false;        // 是否正在执行 simulateStreaming（message 事件的打字机动画）
const RECONNECT_DELAY = 3000;

// --- Init ---

function init() {
    sessionId = generateSessionId();
    updateSessionInfo();
    connect();
    document.getElementById('user-input').focus();
    // 初始化 Mermaid
    if (typeof mermaid !== 'undefined') {
        mermaid.initialize({
            startOnLoad: false,
            theme: 'dark',
            securityLevel: 'loose',
            fontFamily: 'Plus Jakarta Sans, sans-serif',
        });
    }
}

function generateSessionId() {
    return 'ws_' + Math.random().toString(36).substring(2, 10);
}

var _processingTimeout = null;

// --- Processing State ---

function setProcessing(processing) {
    isProcessing = processing;
    var input = document.getElementById('user-input');
    var sendBtn = document.getElementById('send-btn');
    var sendIcon = document.getElementById('send-icon');
    var spinnerIcon = document.getElementById('spinner-icon');

    console.log('setProcessing:', processing, 'input:', !!input, 'btn:', !!sendBtn, 'sendIcon:', !!sendIcon, 'spinner:', !!spinnerIcon);

    if (!input || !sendBtn) {
        console.error('setProcessing: input or sendBtn not found');
        return;
    }

    input.disabled = processing;
    sendBtn.disabled = processing;

    if (processing) {
        if (sendIcon) sendIcon.style.display = 'none';
        if (spinnerIcon) {
            spinnerIcon.style.display = '';
            spinnerIcon.removeAttribute('hidden');
        }
        sendBtn.classList.add('loading');
        input.placeholder = '正在处理中...';
        updateSystemStatus('active', '处理中');
        console.log('setProcessing: LOADING state set');
        // 安全超时：120 秒后如果还没结束，自动恢复（防止短回复等边缘情况卡住）
        if (_processingTimeout) clearTimeout(_processingTimeout);
        _processingTimeout = setTimeout(function () {
            console.log('setProcessing: safety timeout triggered');
            finishStream();
            setProcessing(false);
        }, 120000);
    } else {
        if (_processingTimeout) { clearTimeout(_processingTimeout); _processingTimeout = null; }
        if (_finalMsgTimer) { clearTimeout(_finalMsgTimer); _finalMsgTimer = null; }
        if (sendIcon) sendIcon.style.display = '';
        if (spinnerIcon) {
            spinnerIcon.style.display = 'none';
            spinnerIcon.setAttribute('hidden', '');
        }
        sendBtn.classList.remove('loading');
        input.placeholder = '输入任务或命令...';
        input.focus();
        updateSystemStatus('active', '已连接');
        // 在对话真正结束时，给最后的 bot 消息添加下载按钮（仅限报告类内容）
        if (_lastFinishedStreamDiv && _lastFinishedStreamDiv._rawText && !_lastFinishedStreamDiv._hasDownload) {
            var raw = _lastFinishedStreamDiv._rawText;
            // 仅对包含标题或表格的 Markdown 报告添加下载按钮
            if (/#{1,3}\s|\|.*\|.*\|/.test(raw)) {
                addDownloadBtn(_lastFinishedStreamDiv, raw);
                _lastFinishedStreamDiv._hasDownload = true;
            }
            _lastFinishedStreamDiv = null;
        }
        // 兜底：如果 _lastFinishedStreamDiv 为空，查找最后一条 msg-bot 消息
        if (!_lastFinishedStreamDiv) {
            var botMsgs = document.querySelectorAll('.msg-bot');
            var lastBot = botMsgs[botMsgs.length - 1];
            if (lastBot && lastBot._rawText && !lastBot._hasDownload) {
                var raw2 = lastBot._rawText;
                if (/#{1,3}\s|\|.*\|.*\|/.test(raw2)) {
                    addDownloadBtn(lastBot, raw2);
                    lastBot._hasDownload = true;
                }
            }
        }
        console.log('setProcessing: IDLE state set');
    }
}

// --- Sidebar ---

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('collapsed');
}

function clearChat() {
    document.getElementById('messages').innerHTML =
        '<div class="welcome-block">' +
            '<div class="welcome-icon">L</div>' +
            '<h2 class="welcome-title">欢迎使用 Long</h2>' +
            '<p class="welcome-desc">可控AI智能系统，输入任务开始对话</p>' +
            '<div class="welcome-hints">' +
                '<button class="hint-chip" onclick="sendCommand(\'今天杭州天气怎么样\')">查询天气</button>' +
                '<button class="hint-chip" onclick="sendCommand(\'用Python实现归并排序\')">编写代码</button>' +
                '<button class="hint-chip" onclick="sendCommand(\'生成一份AI发展趋势PPT\')">生成报告</button>' +
            '</div>' +
        '</div>';
}

// --- WebSocket ---

function connect() {
    var protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    var url = protocol + '//' + location.host + '/ws/' + sessionId;
    ws = new WebSocket(url);

    ws.onopen = function () {
        updateConnectionStatus(true);
        updateSystemStatus('active', '已连接');
        addSystemMessage('连接已建立');
    };

    ws.onmessage = function (event) {
        try {
            handleServerMessage(JSON.parse(event.data));
        } catch (e) {
            console.error('Message parse error:', e);
        }
    };

    ws.onclose = function () {
        updateConnectionStatus(false);
        updateSystemStatus('idle', '等待连接');
        addSystemMessage('连接已断开，尝试重连...');
        setProcessing(false);
        scheduleReconnect();
    };

    ws.onerror = function (error) {
        console.error('WebSocket error:', error);
    };
}

function scheduleReconnect() {
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connect, RECONNECT_DELAY);
}

function updateConnectionStatus(connected) {
    var el = document.getElementById('connection-status');
    if (!el) return;
    if (connected) {
        el.style.background = 'rgba(56,201,122,0.12)';
        el.style.color = '#38c97a';
        el.innerHTML = '<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:currentColor;margin-right:8px;"></span><span>已连接</span>';
    } else {
        el.style.background = 'rgba(232,64,87,0.12)';
        el.style.color = '#e84057';
        el.innerHTML = '<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:currentColor;margin-right:8px;"></span><span>未连接</span>';
    }

    var sessionCard = document.getElementById('session-info');
    if (sessionCard) {
        if (connected) sessionCard.classList.add('active');
        else sessionCard.classList.remove('active');
    }
}

// --- Message Handling ---

function handleServerMessage(data) {
    var type = data.type;
    var content = data.content || '';
    var metadata = data.metadata || {};

    // 所有事件都取消安全超时定时器（只要后端还在发消息就说明还没结束）
    if (_finalMsgTimer && type !== 'stream_end' && type !== 'error' && type !== 'turn_complete') {
        clearTimeout(_finalMsgTimer);
        _finalMsgTimer = null;
    }

    switch (type) {
        case 'message':
            // 过滤无意义的验证消息
            if (_isStatusOnlyMsg(content)) break;
            // 步骤进度消息以紧凑样式展示
            if (_isStepProgressMsg(content)) {
                addStepProgress(content);
                break;
            }
            finishStream();
            if (content && content.length > 40) {
                // 长消息 = 报告/工具结果，模拟打字机效果
                simulateStreaming(content, metadata);
            } else if (content) {
                // 短消息 = 中间状态信息（如"任务复杂度:..."），显示但不结束 processing
                addBotMessage(content, metadata);
            }
            break;
        case 'error':
            finishStream();
            addErrorMessage(content);
            setProcessing(false);
            break;
        case 'warning':
            addWarningMessage(content);
            break;
        case 'info':
            break;
        case 'progress':
            addProgressMessage(content, metadata);
            break;
        case 'stream_token':
            appendStreamToken(content);
            break;
        case 'stream_end':
            // 只在非 simulateStreaming 模式下才调用 finishStream
            // 如果正在执行 message 事件的打字机动画，不要打断它
            if (!_isSimStreaming) {
                finishStream();
            }
            // stream_end 可能只是中间流的结束，不直接结束 processing。
            // 等待 turn_complete 事件确认对话真正结束。
            // 安全超时：如果 15 秒内没有收到 turn_complete，强制结束 processing
            if (_finalMsgTimer) clearTimeout(_finalMsgTimer);
            _finalMsgTimer = setTimeout(function () {
                _finalMsgTimer = null;
                console.warn('Safety timeout: turn_complete not received within 15s, forcing processing=false');
                setProcessing(false);
            }, 15000);
            break;
        case 'turn_complete':
            // 等待模拟打字机完成后再结束 processing
            if (_simulateTimer) {
                // 模拟打字机还在运行，等它完成
                if (_finalMsgTimer) clearTimeout(_finalMsgTimer);
                _finalMsgTimer = setTimeout(function () {
                    _finalMsgTimer = null;
                    finishStream();
                    setProcessing(false);
                }, 500);
            } else {
                finishStream();
                if (_finalMsgTimer) {
                    clearTimeout(_finalMsgTimer);
                    _finalMsgTimer = null;
                }
                setProcessing(false);
            }
            break;
        case 'trace':
            renderTracePanel(metadata.trace || {});
            break;
        case 'hitl_request':
            showHITLPanel(content, metadata);
            break;
        case 'system':
            addSystemMessage(content);
            break;
        default:
            finishStream();
            addBotMessage(content, metadata);
            setProcessing(false);
    }
}

// --- Send ---

function sendMessage() {
    var input = document.getElementById('user-input');
    var text = input.value.trim();
    if (!text || isProcessing) return;

    var welcome = document.querySelector('.welcome-block');
    if (welcome) welcome.remove();

    // 清除上一次的步骤进度容器
    var oldProgress = document.getElementById('step-progress-container');
    if (oldProgress) oldProgress.remove();

    addUserMessage(text);
    input.value = '';
    setProcessing(true);

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'user_input',
            content: text,
            session_id: sessionId,
        }));
    }
}

function sendCommand(command) {
    var welcome = document.querySelector('.welcome-block');
    if (welcome) welcome.remove();
    document.getElementById('user-input').value = command;
    sendMessage();
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// --- HITL ---

function showHITLPanel(title, metadata) {
    var panel = document.getElementById('hitl-panel');
    document.getElementById('hitl-title').textContent = title;
    var riskEl = document.getElementById('hitl-risk');
    var level = metadata.risk_level || 'medium';
    riskEl.textContent = level;
    riskEl.className = 'hitl-risk-' + level;
    document.getElementById('hitl-description').textContent = metadata.description || '';

    var optionsEl = document.getElementById('hitl-options');
    optionsEl.innerHTML = '';
    var options = metadata.options || ['approve', 'reject'];
    options.forEach(function (opt) {
        var btn = document.createElement('button');
        btn.textContent = opt;
        btn.className = 'hitl-btn-' + opt;
        btn.onclick = function () { respondHITL(metadata.request_id, opt); };
        optionsEl.appendChild(btn);
    });
    panel.classList.remove('hidden');
}

function respondHITL(requestId, decision) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'hitl_response', request_id: requestId, decision: decision }));
    }
    hideHITLPanel();
    addSystemMessage('审核结果: ' + decision);
}

function hideHITLPanel() {
    document.getElementById('hitl-panel').classList.add('hidden');
}

// --- Trace Visualization ---

var _tracePanelVisible = false;

function toggleTracePanel() {
    var panel = document.getElementById('trace-panel');
    if (!panel) return;
    _tracePanelVisible = !_tracePanelVisible;
    if (_tracePanelVisible) {
        panel.classList.remove('hidden');
        loadTraceHistory();
    } else {
        panel.classList.add('hidden');
    }
}

function loadTraceHistory() {
    fetch('/api/traces?limit=10')
        .then(function (r) { return r.json(); })
        .then(function (data) {
            var container = document.getElementById('trace-list');
            if (!container) return;
            container.innerHTML = '';
            var traces = data.traces || [];
            if (traces.length === 0) {
                container.innerHTML = '<div class="trace-empty">暂无追踪数据</div>';
                return;
            }
            traces.forEach(function (trace) {
                var item = createTraceItem(trace);
                container.appendChild(item);
            });
        })
        .catch(function (err) {
            console.error('Failed to load traces:', err);
        });
}

function createTraceItem(trace) {
    var div = document.createElement('div');
    div.className = 'trace-item';
    div.setAttribute('data-trace-id', trace.trace_id || trace.id || '');

    var duration = trace.duration_ms ? (trace.duration_ms / 1000).toFixed(1) + 's' : '-';
    var spanCount = (trace.spans || []).length;
    var statusClass = trace.status === 'ok' ? 'trace-status-ok' : 'trace-status-error';
    var statusText = trace.status === 'ok' ? 'OK' : (trace.status || '-');

    var header = document.createElement('div');
    header.className = 'trace-item-header';
    header.innerHTML =
        '<span class="trace-name">' + escapeHtml(trace.name || 'unknown') + '</span>' +
        '<span class="trace-duration">' + duration + '</span>' +
        '<span class="trace-spans">' + spanCount + ' spans</span>' +
        '<span class="trace-status ' + statusClass + '">' + statusText + '</span>';

    var timeline = document.createElement('div');
    timeline.className = 'trace-timeline';

    if (trace.spans && trace.spans.length > 0) {
        // 后端返回 Unix 秒级时间戳（浮点数），直接做数值计算
        var traceStart = trace.start_time || 0;
        var traceEnd = trace.end_time || traceStart;
        var totalDuration = Math.max(traceEnd - traceStart, 0.001);

        // 构建 span 深度映射（根据 parent_span_id 计算嵌套层级）
        var spanDepthMap = {};
        var spanIdSet = {};
        trace.spans.forEach(function (span) {
            if (span.span_id) spanIdSet[span.span_id] = true;
        });
        function getDepth(span) {
            if (!span.parent_span_id || !spanIdSet[span.parent_span_id]) return 0;
            if (spanDepthMap[span.parent_span_id] !== undefined) {
                return spanDepthMap[span.parent_span_id] + 1;
            }
            // 递归计算父 span 深度
            var parent = null;
            for (var i = 0; i < trace.spans.length; i++) {
                if (trace.spans[i].span_id === span.parent_span_id) {
                    parent = trace.spans[i];
                    break;
                }
            }
            if (!parent) return 0;
            return getDepth(parent) + 1;
        }
        trace.spans.forEach(function (span) {
            spanDepthMap[span.span_id] = getDepth(span);
        });

        trace.spans.forEach(function (span) {
            var depth = spanDepthMap[span.span_id] || 0;
            var spanEl = createSpanBar(span, traceStart, totalDuration, depth);
            timeline.appendChild(spanEl);
        });
    }

    div.appendChild(header);
    div.appendChild(timeline);

    // 点击展开/折叠详情
    var detailDiv = document.createElement('div');
    detailDiv.className = 'trace-detail hidden';
    if (trace.spans) {
        trace.spans.forEach(function (span) {
            var spanDetail = createSpanDetail(span);
            detailDiv.appendChild(spanDetail);
        });
    }
    div.appendChild(detailDiv);

    header.onclick = function () {
        detailDiv.classList.toggle('hidden');
    };

    return div;
}

function createSpanBar(span, traceStart, totalDuration, depth) {
    // 后端返回 Unix 秒级时间戳（浮点数），直接做数值差值计算
    var spanStart = span.start_time || traceStart;
    var spanEnd = span.end_time || spanStart;
    var offset = ((spanStart - traceStart) / totalDuration) * 100;
    var width = Math.max(((spanEnd - spanStart) / totalDuration) * 100, 0.5);

    depth = depth || 0;
    var indentPx = depth * 16;

    var bar = document.createElement('div');
    bar.className = 'trace-span-bar';
    bar.style.marginLeft = 'calc(' + offset + '% + ' + indentPx + 'px)';
    bar.style.width = 'calc(' + width + '% - ' + indentPx + 'px)';

    var kind = (span.name || '').toLowerCase();
    if (kind.indexOf('tool') >= 0 || kind.indexOf('execute') >= 0 || kind.indexOf('search') >= 0) {
        bar.classList.add('span-tool');
    } else if (kind.indexOf('think') >= 0 || kind.indexOf('llm') >= 0 || kind.indexOf('chat') >= 0) {
        bar.classList.add('span-llm');
    } else if (kind.indexOf('plan') >= 0) {
        bar.classList.add('span-plan');
    } else if (kind.indexOf('observe') >= 0 || kind.indexOf('output') >= 0) {
        bar.classList.add('span-io');
    }

    // 使用 duration_ms 属性（后端已计算好），或用秒级时间戳差值计算
    var durationMs = span.duration_ms || ((spanEnd - spanStart) * 1000);
    bar.title = (span.name || 'span') + ' (' + durationMs.toFixed(0) + 'ms)';

    var label = document.createElement('span');
    label.className = 'span-label';
    label.textContent = span.name || 'span';
    bar.appendChild(label);

    return bar;
}

function createSpanDetail(span) {
    var div = document.createElement('div');
    div.className = 'span-detail';

    // 使用后端已计算好的 duration_ms，或用秒级时间戳差值计算
    var durationMs = '-';
    if (span.duration_ms) {
        durationMs = span.duration_ms.toFixed(0) + 'ms';
    } else if (span.start_time && span.end_time) {
        durationMs = ((span.end_time - span.start_time) * 1000).toFixed(0) + 'ms';
    }

    var html = '<div class="span-detail-header">' +
        '<span class="span-detail-name">' + escapeHtml(span.name || 'span') + '</span>' +
        '<span class="span-detail-duration">' + durationMs + '</span>' +
        '</div>';

    if (span.attributes && Object.keys(span.attributes).length > 0) {
        html += '<div class="span-detail-attrs">';
        for (var key in span.attributes) {
            html += '<div class="span-attr"><span class="span-attr-key">' +
                escapeHtml(key) + '</span><span class="span-attr-val">' +
                escapeHtml(String(span.attributes[key])) + '</span></div>';
        }
        html += '</div>';
    }

    if (span.events && span.events.length > 0) {
        html += '<div class="span-detail-events">';
        span.events.forEach(function (ev) {
            html += '<div class="span-event">' + escapeHtml(ev.name || String(ev)) + '</div>';
        });
        html += '</div>';
    }

    div.innerHTML = html;
    return div;
}

function renderTracePanel(traceData) {
    // 实时更新：将新的 trace 追加或更新到面板
    if (!_tracePanelVisible) {
        // 自动展开追踪面板
        _tracePanelVisible = true;
        var panel = document.getElementById('trace-panel');
        if (panel) panel.classList.remove('hidden');
    }

    var container = document.getElementById('trace-list');
    if (!container) return;

    // 移除空状态提示
    var emptyEl = container.querySelector('.trace-empty');
    if (emptyEl) emptyEl.remove();

    // 检查是否已有相同 trace_id 的元素，如果有则更新
    var traceId = traceData.trace_id;
    var existingItem = null;
    if (traceId) {
        existingItem = container.querySelector('[data-trace-id="' + traceId + '"]');
    }

    if (existingItem) {
        // 增量更新：替换已有 trace 元素
        var newItem = createTraceItem(traceData);
        existingItem.replaceWith(newItem);
    } else {
        // 新 trace：插入到列表顶部
        var item = createTraceItem(traceData);
        container.insertBefore(item, container.firstChild);
    }
}

function escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// --- Message Rendering ---

function addUserMessage(text) {
    var container = document.getElementById('messages');
    var div = document.createElement('div');
    div.className = 'msg-bubble msg-user';
    div.textContent = text;
    container.appendChild(div);
    scrollToBottom();
}

/**
 * 判断代码内容是否为 Mermaid 图表语法
 */
function _isMermaidContent(code) {
    var trimmed = code.trim().toLowerCase();
    if (/^(xychart-beta|xychart|pie\s|pie\s+showData|flowchart|graph\s+(td|tb|bt|rl|lr)|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|journey|mindmap|timeline|quadrantChart|sankey)/.test(trimmed)) {
        return true;
    }
    return false;
}

/**
 * 渲染页面中所有未处理的 Mermaid 图表
 */
function _renderMermaidCharts() {
    if (typeof mermaid === 'undefined') return;
    var charts = document.querySelectorAll('.mermaid-chart:not([data-processed])');
    if (charts.length === 0) return;
    charts.forEach(function (el) {
        el.setAttribute('data-processed', 'true');
        var code = el.textContent;
        var id = 'mermaid-' + Date.now() + '-' + Math.random().toString(36).substr(2, 6);
        try {
            mermaid.render(id, code).then(function (result) {
                el.innerHTML = result.svg;
                el.classList.add('mermaid-rendered');
                // 添加下载按钮
                var dlBtn = document.createElement('a');
                dlBtn.className = 'chart-download-btn';
                dlBtn.href = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(result.svg);
                dlBtn.download = 'chart_' + Date.now() + '.svg';
                dlBtn.innerHTML = '⬇ 下载图表';
                dlBtn.target = '_blank';
                el.appendChild(dlBtn);
            }).catch(function (err) {
                console.warn('Mermaid render error:', err);
                el.innerHTML = '<pre class="code-block">' + code + '</pre><p class="mermaid-error">图表渲染失败，原始语法如上</p>';
            });
        } catch (e) {
            console.warn('Mermaid init error:', e);
        }
    });
}

function addStepProgress(content) {
    var container = document.getElementById('messages');
    // 复用或创建步骤进度容器
    var progressContainer = document.getElementById('step-progress-container');
    if (!progressContainer) {
        progressContainer = document.createElement('div');
        progressContainer.id = 'step-progress-container';
        progressContainer.className = 'step-progress-container';
        container.appendChild(progressContainer);
    }
    var step = document.createElement('div');
    step.className = 'step-progress-item';
    step.textContent = content.trim();
    progressContainer.appendChild(step);
    scrollToBottom();
}

function addBotMessage(content, metadata) {
    var container = document.getElementById('messages');
    var div = document.createElement('div');
    div.className = 'msg-bubble msg-bot';

    var format = metadata.format || 'markdown';
    if (format === 'code') {
        var pre = document.createElement('pre');
        pre.className = 'code-block';
        pre.textContent = content;
        div.appendChild(pre);
    } else if (format === 'table' && metadata.headers) {
        div.appendChild(createTable(metadata.headers, metadata.rows));
    } else {
        div.innerHTML = renderMarkdown(content);
        // 自动检测并渲染文件下载链接
        _renderFileLinks(div, content);
    }
    container.appendChild(div);
    scrollToBottom();
}

function addErrorMessage(content) {
    appendMessage('msg-error', content);
}

function addWarningMessage(content) {
    appendMessage('msg-warning', content);
}

function addSystemMessage(content) {
    appendMessage('msg-system', content);
}

function addProgressMessage(content, metadata) {
    var container = document.getElementById('messages');
    var div = document.createElement('div');
    div.className = 'msg-bubble msg-progress';
    if (metadata.percent !== undefined) {
        var bar = document.createElement('div');
        bar.className = 'progress-bar-wrap';
        bar.innerHTML = '<div class="progress-bar-fill" style="width:' + metadata.percent + '%"></div>';
        div.appendChild(bar);
        div.appendChild(document.createTextNode(metadata.percent + '% ' + content));
    } else {
        div.textContent = content;
    }
    container.appendChild(div);
    scrollToBottom();
}

function appendMessage(cls, content) {
    var container = document.getElementById('messages');
    var div = document.createElement('div');
    div.className = 'msg-bubble ' + cls;
    div.textContent = content;
    container.appendChild(div);
    scrollToBottom();
}

// --- Streaming (Typewriter Effect) ---

var _streamRenderTimer = null;
var _simulateTimer = null;

/**
 * 模拟流式输出：将完整文本逐字符输出，实现打字机效果。
 * 用于非流式代码路径（_cognitive_runtime_loop 等），
 * 让用户在视觉上获得和真流式一样的体验。
 */
function simulateStreaming(text, metadata) {
    // 如果已有模拟流在进行，先完整结束旧流
    if (_simulateTimer) { clearTimeout(_simulateTimer); _simulateTimer = null; }
    // 临时关闭 _isSimStreaming 标志让 finishStream 能完成旧 div
    _isSimStreaming = false;
    finishStream();
    _isSimStreaming = true;

    var container = document.getElementById('messages');
    currentStreamDiv = document.createElement('div');
    currentStreamDiv.className = 'msg-bubble msg-streaming';
    currentStreamDiv._rawText = '';
    currentStreamDiv._simIdx = 0;
    currentStreamDiv._simFull = text;
    currentStreamDiv._simMeta = metadata;
    container.appendChild(currentStreamDiv);
    scrollToBottom();

    _simulateTick();
}

function _simulateTick() {
    if (!currentStreamDiv) return;

    var text = currentStreamDiv._simFull;
    var idx = currentStreamDiv._simIdx;
    // 每 tick 输出更多字符（长文本加速）
    var charsPerTick = text.length > 500 ? 8 : (text.length > 200 ? 4 : 2);
    var delay = text.length > 500 ? 5 : (text.length > 200 ? 8 : 20);

    if (idx < text.length) {
        var end = Math.min(idx + charsPerTick, text.length);
        var chunk = text.slice(idx, end);
        currentStreamDiv._rawText += chunk;
        currentStreamDiv._simIdx = end;

        // 检测是否包含 markdown 块级语法（标题、表格、代码块等），是则走渲染路径
        var hasMarkdown = /```|\*\*|^#{1,6}\s|^\|.*\|$|^>\s?|^[-*+]\s|^\d+\.\s|\[.*\]\(.*\)/m.test(text);
        if (hasMarkdown) {
            currentStreamDiv.innerHTML = renderMarkdown(currentStreamDiv._rawText) +
                '<span class="typing-cursor">&#x258A;</span>';
        } else {
            currentStreamDiv.textContent = currentStreamDiv._rawText;
            // 追加光标
            var cursor = document.createElement('span');
            cursor.className = 'typing-cursor';
            cursor.innerHTML = '&#x258A;';
            currentStreamDiv.appendChild(cursor);
        }
        currentStreamDiv._rendered = true;
        scrollToBottom();

        _simulateTimer = setTimeout(_simulateTick, delay);
    } else {
        _isSimStreaming = false;
        finishStream();
        // 不立即结束 processing：可能还有后续消息（如工具调用结果）。
        // 使用 1.5 秒防抖——如果这段时间内没有新消息到达，才认为对话结束。
        if (_finalMsgTimer) clearTimeout(_finalMsgTimer);
        _finalMsgTimer = setTimeout(function () {
            _finalMsgTimer = null;
            setProcessing(false);
        }, 1500);
    }
}

function appendStreamToken(token) {
    if (!currentStreamDiv) {
        var container = document.getElementById('messages');
        currentStreamDiv = document.createElement('div');
        currentStreamDiv.className = 'msg-bubble msg-streaming';
        currentStreamDiv._rawText = '';
        container.appendChild(currentStreamDiv);
    }

    currentStreamDiv._rawText += token;

    // 节流 markdown 渲染（50ms 间隔），避免频繁 DOM 操作
    if (_streamRenderTimer) clearTimeout(_streamRenderTimer);
    _streamRenderTimer = setTimeout(function () {
        if (currentStreamDiv && currentStreamDiv._rawText) {
            currentStreamDiv.innerHTML = renderMarkdown(currentStreamDiv._rawText) +
                '<span class="typing-cursor">&#x258A;</span>';
            currentStreamDiv._rendered = true;
            scrollToBottom();
        }
    }, 50);

    // 立即显示纯文本（保证用户第一时间看到内容）
    if (!currentStreamDiv._rendered) {
        currentStreamDiv.textContent = currentStreamDiv._rawText;
        scrollToBottom();
    }
}

function finishStream() {
    if (_streamRenderTimer) {
        clearTimeout(_streamRenderTimer);
        _streamRenderTimer = null;
    }
    if (_simulateTimer) {
        clearTimeout(_simulateTimer);
        _simulateTimer = null;
    }
    // 如果正在执行 simulateStreaming 打字机动画，不要打断它
    // 除非是打字机自己调用 finishStream（此时 _simulateTimer 已被 clearTimeout 清掉）
    if (_isSimStreaming && _simulateTimer === null) {
        // 这是来自 turn_complete 等事件的 finishStream 调用
        // 打字机已经通过 else 分支自行结束了，安全
    } else if (_isSimStreaming) {
        // 打字机还在运行，不要打断
        return;
    }
    if (currentStreamDiv) {
        currentStreamDiv.classList.remove('msg-streaming');
        currentStreamDiv.classList.add('msg-bot');
        // 最终渲染完整 markdown
        // 优先使用 _simFull（完整文本），其次 _rawText（打字机已输出的部分）
        var rawMarkdown = currentStreamDiv._simFull || currentStreamDiv._rawText || '';
        if (rawMarkdown) {
            currentStreamDiv.innerHTML = renderMarkdown(rawMarkdown);
            currentStreamDiv._rendered = true;
            // 自动检测并渲染文件下载链接（PPT、PDF、代码等）
            _renderFileLinks(currentStreamDiv, rawMarkdown);
            // 保存引用供 setProcessing(false) 时添加下载按钮
            _lastFinishedStreamDiv = currentStreamDiv;
        }
        currentStreamDiv = null;
    }
    // 渲染 Mermaid 图表
    setTimeout(_renderMermaidCharts, 100);
}

/**
 * 给消息气泡添加"下载 Markdown"按钮
 */
function addDownloadBtn(msgDiv, rawText) {
    var btn = document.createElement('button');
    btn.className = 'download-btn';
    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 下载 Markdown';
    btn.title = '下载为 .md 文件';
    btn.onclick = function (e) {
        e.stopPropagation();
        var blob = new Blob([rawText], { type: 'text/markdown;charset=utf-8' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'report_' + Date.now() + '.md';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };
    msgDiv.appendChild(btn);
}

/**
 * 自动检测消息内容中的可下载文件路径，渲染为可点击的下载链接。
 * 支持的后缀: .pptx, .pdf, .html, .xlsx, .docx, .png, .jpg, .svg, .py, .js, .ts 等
 */
function _renderFileLinks(msgDiv, rawMarkdown) {
    // 匹配 output/ 或 /output/ 路径中的文件
    var fileRe = /(?:^|[\s，。、：:!()\[\]{}\/])(\/?output\/[\w\u4e00-\u9fff/\-.]+\.\w+)/gi;
    var matches = rawMarkdown.match(fileRe);
    if (!matches || matches.length === 0) return;

    // 去重，统一去掉前导 /
    var uniqueFiles = [];
    var seen = {};
    // 只显示二进制文件类型的下载链接（PPT/Word/Excel/PDF/图片等）
    var _binaryExts = ['pptx', 'pdf', 'docx', 'xlsx', 'xls', 'png', 'jpg', 'jpeg', 'gif', 'svg', 'zip'];
    for (var i = 0; i < matches.length; i++) {
        var f = matches[i].trim();
        // 去掉前导 / 和可能被捕获的 markdown 链接语法字符
        f = f.replace(/^[\(\)\/]+/, '');
        if (seen[f]) continue;
        // 只处理二进制文件
        var ext = f.split('.').pop().toLowerCase();
        if (_binaryExts.indexOf(ext) < 0) continue;
        seen[f] = true;
        uniqueFiles.push(f);
    }

    if (uniqueFiles.length === 0) return;

    // 创建文件下载区域
    var fileLinksDiv = document.createElement('div');
    fileLinksDiv.className = 'file-links';
    var label = document.createElement('div');
    label.className = 'file-links-label';
    label.textContent = '\uD83D\uDCC1 生成的文件：';
    fileLinksDiv.appendChild(label);

    for (var j = 0; j < uniqueFiles.length; j++) {
        var filePath = uniqueFiles[j];
        var link = document.createElement('a');
        link.className = 'file-link';
        link.href = '/' + filePath;
        link.download = filePath.split('/').pop();
        link.target = '_blank';

        // 图标
        var icon = document.createElement('span');
        icon.className = 'file-link-icon';
        icon.innerHTML = _getFileIcon(filePath);
        link.appendChild(icon);

        // 文件名
        var nameSpan = document.createElement('span');
        nameSpan.textContent = filePath;
        link.appendChild(nameSpan);

        fileLinksDiv.appendChild(link);
    }

    msgDiv.appendChild(fileLinksDiv);
}

/**
 * 根据文件后缀返回对应的 SVG 图标
 */
function _getFileIcon(filePath) {
    var ext = filePath.split('.').pop().toLowerCase();
    var icons = {
        pptx: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M12 17v-5"/><path d="M9 14h6"/></svg>',
        pdf: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
        html: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
        xlsx: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/></svg>',
        docx: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
        png: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>',
        py: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
    };
    return icons[ext] || '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>';
}

// --- Utilities ---

/**
 * 判断消息是否为纯验证状态消息（不需要展示给用户）
 */
function _isStatusOnlyMsg(content) {
    var trimmed = content.trim();
    // 验证状态消息（"检测到报告文件已生成"等无意义信息）
    if (/^✅\s*检测到/.test(trimmed)) return true;
    // 极短的无意义内容
    if (trimmed.length <= 2 && /^[✅⚠️📋📊🔧→⚡✨⏳🤔]+$/.test(trimmed)) return true;
    return false;
}

/**
 * 判断消息是否为步骤进度信息（需要以紧凑样式展示）
 */
function _isStepProgressMsg(content) {
    var trimmed = content.trim();
    // 步骤进度：✅ step_2: ... 或 ❌ step_3: ...
    if (/^[✅❌]\s*step_\d+/.test(trimmed)) return true;
    // 计划执行完成
    if (/^✅\s*计划执行完成/.test(trimmed)) return true;
    return false;
}

function createTable(headers, rows) {
    var table = document.createElement('table');
    table.className = 'data-table';
    var thead = document.createElement('thead');
    var headerRow = document.createElement('tr');
    headers.forEach(function (h) {
        var th = document.createElement('th');
        th.textContent = h;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    (rows || []).forEach(function (row) {
        var tr = document.createElement('tr');
        row.forEach(function (cell) {
            var td = document.createElement('td');
            td.textContent = cell;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
}

/**
 * 块级 Markdown 渲染器：逐行解析，支持标题、表格、链接、代码块、引用、列表、分割线等。
 */
function renderMarkdown(text) {
    var lines = text.split('\n');
    var result = [];
    var inCodeBlock = false;
    var codeContent = '';
    var codeLang = '';  // 代码块语言标识
    var inTable = false;
    var tableRows = [];
    var inList = false;
    var listType = '';
    var listItems = [];
    var paraBuffer = '';

    function flushList() {
        if (!inList) return;
        var tag = listType === 'ol' ? 'ol' : 'ul';
        var html = '<' + tag + ' class="md-list">';
        for (var i = 0; i < listItems.length; i++) {
            html += '<li>' + renderInline(listItems[i]) + '</li>';
        }
        html += '</' + tag + '>';
        result.push(html);
        listItems = [];
        inList = false;
    }

    function flushTable() {
        if (!inTable || tableRows.length < 2) {
            if (tableRows.length > 0) {
                for (var i = 0; i < tableRows.length; i++) {
                    result.push('<p>' + renderInline(tableRows[i].join(' | ')) + '</p>');
                }
            }
            tableRows = [];
            inTable = false;
            return;
        }
        var html = '<table class="data-table"><thead><tr>';
        var headers = tableRows[0];
        for (var h = 0; h < headers.length; h++) {
            html += '<th>' + renderInline(headers[h].trim()) + '</th>';
        }
        html += '</tr></thead><tbody>';
        for (var r = 1; r < tableRows.length; r++) {
            html += '<tr>';
            for (var c = 0; c < tableRows[r].length; c++) {
                html += '<td>' + renderInline(tableRows[r][c].trim()) + '</td>';
            }
            html += '</tr>';
        }
        html += '</tbody></table>';
        result.push(html);
        tableRows = [];
        inTable = false;
    }

    function flushParagraph(buf) {
        if (buf.trim()) {
            result.push('<p>' + renderInline(buf.trim()) + '</p>');
        }
    }

    function isTableRow(line) {
        return /^\|.+\|$/.test(line.trim());
    }

    for (var i = 0; i < lines.length; i++) {
        var line = lines[i];

        // 代码块
        if (/^```/.test(line)) {
            flushList();
            flushTable();
            flushParagraph(paraBuffer);
            paraBuffer = '';
            if (!inCodeBlock) {
                inCodeBlock = true;
                codeContent = '';
                codeLang = line.replace(/^```\s*/, '').trim().toLowerCase();
            } else {
                // Mermaid 图表：渲染为 SVG
                if (codeLang === 'mermaid' || _isMermaidContent(codeContent)) {
                    result.push('<div class="mermaid-chart">' + _escapeHtml(codeContent) + '</div>');
                } else {
                    result.push('<pre class="code-block">' + _escapeHtml(codeContent) + '</pre>');
                }
                inCodeBlock = false;
                codeContent = '';
                codeLang = '';
            }
            continue;
        }

        if (inCodeBlock) {
            codeContent += (codeContent ? '\n' : '') + line;
            continue;
        }

        // 空行
        if (line.trim() === '') {
            flushList();
            flushTable();
            flushParagraph(paraBuffer);
            paraBuffer = '';
            continue;
        }

        // 分割线
        if (/^(-{3,}|\*{3,}|_{3,})$/.test(line.trim())) {
            flushList();
            flushTable();
            flushParagraph(paraBuffer);
            paraBuffer = '';
            result.push('<hr class="md-hr">');
            continue;
        }

        // 标题
        var headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
        if (headingMatch) {
            flushList();
            flushTable();
            flushParagraph(paraBuffer);
            paraBuffer = '';
            var lv = headingMatch[1].length;
            result.push('<h' + lv + ' class="md-heading">' + renderInline(headingMatch[2]) + '</h' + lv + '>');
            continue;
        }

        // 引用
        if (/^>\s?/.test(line)) {
            flushList();
            flushTable();
            flushParagraph(paraBuffer);
            paraBuffer = '';
            result.push('<blockquote class="md-quote"><p>' + renderInline(line.replace(/^>\s?/, '')) + '</p></blockquote>');
            continue;
        }

        // 无序列表
        var ulMatch = line.match(/^(\s*)[-*+]\s+(.+)$/);
        if (ulMatch) {
            flushTable();
            flushParagraph(paraBuffer);
            paraBuffer = '';
            if (!inList || listType !== 'ul') { flushList(); inList = true; listType = 'ul'; }
            listItems.push(ulMatch[2]);
            continue;
        }

        // 有序列表
        var olMatch = line.match(/^(\s*)\d+\.\s+(.+)$/);
        if (olMatch) {
            flushTable();
            flushParagraph(paraBuffer);
            paraBuffer = '';
            if (!inList || listType !== 'ol') { flushList(); inList = true; listType = 'ol'; }
            listItems.push(olMatch[2]);
            continue;
        }

        // 表格行
        if (isTableRow(line)) {
            flushList();
            flushParagraph(paraBuffer);
            paraBuffer = '';
            // 跳过分隔行 (|---|---|)
            if (/^\|[\s\-:|]+\|$/.test(line.trim())) continue;
            var cells = line.trim().replace(/^\||\|$/g, '').split('|');
            tableRows.push(cells);
            if (!inTable) inTable = true;
            continue;
        }

        // 普通段落
        if (inTable && !isTableRow(line)) { flushTable(); }
        paraBuffer += (paraBuffer ? '\n' : '') + line;
    }

    // 刷新剩余缓冲区
    flushList();
    flushTable();
    flushParagraph(paraBuffer);

    // 如果没产生任何块级输出，退化为简单内联渲染
    if (result.length === 0 && text.trim()) {
        return renderInline(text.trim());
    }
    return result.join('\n');
}

/** HTML 转义（用于代码块内容） */
function _escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** 行内元素渲染：粗体、斜体、行内代码、链接、图片 */
function renderInline(text) {
    // HTML 转义
    text = _escapeHtml(text);
    // 图片 ![alt](url) — 必须在链接之前处理
    text = text.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" class="md-img" loading="lazy">');
    // 粗体 **text**
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // 斜体 *text*
    text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    // 行内代码 `code`
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    // 链接 [text](url)
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    // 换行符保留<br>（段落内换行）
    text = text.replace(/\n/g, '<br>');
    return text;
}

function scrollToBottom() {
    var container = document.getElementById('messages');
    container.scrollTop = container.scrollHeight;
}

function updateSessionInfo() {
    var el = document.getElementById('session-info');
    if (el && sessionId) {
        el.innerHTML = '<div class="session-dot" style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#5c6070;margin-right:8px;flex-shrink:0;"></div><span>' + sessionId + '</span>';
    }
}

function updateSystemStatus(state, text) {
    var el = document.getElementById('system-status');
    if (!el) return;
    var color, glow;
    if (state === 'active') { color = '#e8a838'; glow = '0 0 8px #e8a838'; }
    else if (state === 'error') { color = '#e84057'; glow = '0 0 8px #e84057'; }
    else { color = '#5c6070'; glow = 'none'; }
    el.innerHTML = '<div class="status-indicator" style="background:' + color + ';box-shadow:' + glow + ';"></div><span>' + text + '</span>';
}

// --- Boot ---

window.addEventListener('load', init);