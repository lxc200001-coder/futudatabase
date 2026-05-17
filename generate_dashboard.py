#!/usr/bin/env python3
"""
从 all_summary.xlsx（信号扫描sheet）生成 ECharts 可视化 HTML 仪表盘
"""
import json
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime

# 中->英列名映射（all_summary.xlsx 信号扫描 sheet → 内部使用）
COL_MAP = {
    "股票代码": "symbol",
    "均线周期": "ma",
    "时间": "datetime",
    "收盘价": "close",
    "HA收盘价": "ha_close",
    "HA均线值": "ma_value",
    "趋势方向": "dir",
    "信号": "signal",
    "买入信号时间": "buy_signal_time",
    "买入信号收盘价": "buy_signal_close",
    "距离买入信号已过天数": "buy_signal_days",
    "卖出信号时间": "sell_signal_time",
    "卖出信号收盘价": "sell_signal_close",
    "距离卖出信号已过天数": "sell_signal_days",
    "综合评分": "score",
    "预计持仓进度": "hold_progress",
    "持仓日化收益率": "daily_return",
    "股票名称": "stock_name",
    "距离买入信号收盘价涨跌幅": "buy_signal_change",
    "距离卖出信号收盘价涨跌幅": "sell_signal_change",
}

# 表格中展示的列（英文key → 中文标签）
TABLE_COLS = [
    ("symbol", "代码"),
    ("ma", "均线"),
    ("datetime", "时间"),
    ("close", "收盘价"),
    ("signal", "信号"),
    ("score", "综合评分"),
    ("hold_progress", "持仓进度"),
    ("daily_return", "日化收益率"),
    ("dir", "方向"),
    ("buy_signal_days", "买入天数"),
]


def load_data():
    """优先加载 all_summary.xlsx，其次 fallback 到 scan_result"""
    # 尝试 all_summary.xlsx（可能在 trades/ 或 results/）
    summary_path = None
    for p in ["trades/all_summary.xlsx", "results/all_summary.xlsx"]:
        if os.path.exists(p):
            summary_path = p
            break
    if summary_path is not None:
        try:
            xl = pd.ExcelFile(summary_path)
            if "信号扫描" in xl.sheet_names:
                df = pd.read_excel(summary_path, sheet_name="信号扫描")
                # 映射列名
                rename = {k: v for k, v in COL_MAP.items() if k in df.columns}
                df = df.rename(columns=rename)
                # 确保关键列存在
                for eng_col in ["symbol", "ma", "datetime", "signal"]:
                    if eng_col not in df.columns:
                        print(f"错误: 信号扫描sheet缺少必要列 '{eng_col}'")
                        return None
                # 类型转换
                df["datetime"] = pd.to_datetime(df["datetime"])
                df["ma"] = df["ma"].astype(int)
                # 信号值标准化：中文 → 英文
                if "signal" in df.columns:
                    df["signal"] = df["signal"].replace({
                        "买入": "BUY",
                        "卖出": "SELL",
                        "持有": "HOLD",
                        "观察": "WATCH",
                    })
                # 方向值标准化：中文 → 数值
                if "dir" in df.columns:
                    df["dir"] = df["dir"].map({"多头": 1, "空头": -1}).astype(int)
                if "buy_signal_time" in df.columns:
                    df["buy_signal_time"] = pd.to_datetime(df["buy_signal_time"])
                if "sell_signal_time" in df.columns:
                    df["sell_signal_time"] = pd.to_datetime(df["sell_signal_time"])
                print(f"读取 all_summary.xlsx → 信号扫描: {len(df)} 行")
                return df, summary_path
        except Exception as e:
            print(f"读取 all_summary.xlsx 失败: {e}")

    # fallback: scan_result
    files = sorted(glob.glob("results/scan_result_*.xlsx"))
    if files:
        fpath = files[-1]
        print(f"读取（fallback）: {fpath}")
        df = pd.read_excel(fpath, sheet_name="all_raw")
        df["datetime"] = pd.to_datetime(df["datetime"])
        if "buy_signal_time" in df.columns:
            df["buy_signal_time"] = pd.to_datetime(df["buy_signal_time"])
        if "sell_signal_time" in df.columns:
            df["sell_signal_time"] = pd.to_datetime(df["sell_signal_time"])
        return df, fpath

    print("错误: 未找到 all_summary.xlsx 或 scan_result_*.xlsx")
    return None


def _js(val):
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, (pd.Series, pd.Index)):
        return val.tolist()
    if hasattr(val, "isoformat"):
        return val.isoformat()
    # numpy types → Python native
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, float) and pd.isna(val):
        return None
    # catch-all: numpy generic scalar → Python native
    if isinstance(val, np.generic):
        return val.item()
    return val


def build_json_data(signals):

    # 1. 概览
    buy_cnt = int((signals["signal"] == "BUY").sum())
    sell_cnt = int((signals["signal"] == "SELL").sum())
    overview = {"buy": buy_cnt, "sell": sell_cnt, "total": buy_cnt + sell_cnt}

    # 2. 是否有评分数据
    has_score = "score" in signals.columns and signals["score"].notna().any()
    has_score = bool(has_score)

    # 4. 信号表格（最近一条信号/股）
    latest = signals.loc[signals.groupby("symbol")["datetime"].idxmax()]
    table_rows = []
    for _, r in latest.iterrows():
        table_rows.append({c: _js(r[c]) for c in signals.columns})

    # 5. 买入信号时间线
    buy_timeline = signals[signals["signal"] == "BUY"].copy()
    if len(buy_timeline) > 0:
        buy_timeline["date"] = buy_timeline["datetime"].dt.date
        tl_agg = buy_timeline.groupby("date").size().reset_index(name="count")
        timeline_json = {"dates": [_js(d) for d in tl_agg["date"]], "counts": tl_agg["count"].tolist()}
    else:
        timeline_json = {"dates": [], "counts": []}

    # 6. 评分气泡图（全部有评分的个股，颜色区分信号类型）
    bubble_data = []
    per_stock = signals.loc[signals.groupby("symbol")["datetime"].idxmax()]
    scored = per_stock[per_stock["score"].notna()]
    for _, r in scored.iterrows():
        bubble_data.append({
            "symbol": str(r.get("symbol", "")),
            "name": str(r.get("stock_name", "")) if pd.notna(r.get("stock_name")) else "",
            "score": round(float(r["score"]), 1),
            "signal": str(r.get("signal", "")),
            "buy_days": int(r["buy_signal_days"]) if pd.notna(r.get("buy_signal_days")) else 0,
            "buy_change": round(float(r["buy_signal_change"]), 2) if pd.notna(r.get("buy_signal_change")) else 0,
            "sell_change": round(float(r["sell_signal_change"]), 2) if pd.notna(r.get("sell_signal_change")) else 0,
        })
    bubble_data.sort(key=lambda x: x["score"], reverse=True)

    # 7. 按信号分类的个股列表
    BUY_COLS = ["symbol", "stock_name", "score", "close", "buy_signal_time", "buy_signal_close", "buy_signal_days", "buy_signal_change"]
    BUY_LABELS = ["代码", "名称", "评分", "收盘价", "买入时间", "买入价", "买入天数", "距买入价涨跌幅"]
    SELL_COLS = ["symbol", "stock_name", "score", "close", "sell_signal_time", "sell_signal_close", "sell_signal_days", "sell_signal_change"]
    SELL_LABELS = ["代码", "名称", "评分", "收盘价", "卖出时间", "卖出价", "卖出天数", "距卖出价涨跌幅"]

    def _signal_rows(sig_type, cols):
        sub = signals[signals["signal"] == sig_type].copy()
        if sub.empty:
            return [], cols
        # 每只股票取最新一条
        latest_sig = sub.loc[sub.groupby("symbol")["datetime"].idxmax()]
        rows = []
        for _, r in latest_sig.iterrows():
            rows.append({c: _js(r[c]) for c in cols})
        rows.sort(key=lambda x: x.get("score") or 0, reverse=True)
        return rows, cols

    signal_sections = {}
    for sig in ["BUY", "SELL", "HOLD", "WATCH"]:
        if sig in ("BUY", "HOLD"):
            rows, cols = _signal_rows(sig, BUY_COLS)
            signal_sections[sig] = {"rows": rows, "cols": cols, "labels": BUY_LABELS}
        else:
            rows, cols = _signal_rows(sig, SELL_COLS)
            signal_sections[sig] = {"rows": rows, "cols": cols, "labels": SELL_LABELS}

    return {
        "overview": overview,
        "bubble": bubble_data,
        "table": table_rows,
        "timeline": timeline_json,
        "columns": list(signals.columns),
        "has_score": has_score,
        "table_cols": TABLE_COLS,
        "signal_sections": signal_sections,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def generate_html(data):
    encoded = json.dumps(data, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>信号扫描仪表盘</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:#f0f2f5; color:#333; }}
.header {{ background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460); color:#fff; padding:24px 32px; }}
.header h1 {{ font-size:22px; font-weight:600; }}
.header p {{ font-size:13px; opacity:.7; margin-top:4px; }}
.container {{ max-width:1400px; margin:0 auto; padding:16px; }}
.cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:16px; }}
.card {{ background:#fff; border-radius:10px; padding:20px; box-shadow:0 1px 4px rgba(0,0,0,.06); }}
.card .num {{ font-size:32px; font-weight:700; }}
.card .label {{ font-size:13px; color:#888; margin-top:4px; }}
.card .num.buy {{ color:#27ae60; }}
.card .num.sell {{ color:#e74c3c; }}
.card .num.total {{ color:#2c3e50; }}
.card .num.stocks {{ color:#95a5a6; }}
@media (max-width:768px) {{ .cards {{ grid-template-columns:repeat(2,1fr); }} }}
.chart-box {{ background:#fff; border-radius:10px; padding:16px; box-shadow:0 1px 4px rgba(0,0,0,.06); }}
.chart-box h3 {{ font-size:14px; color:#555; margin-bottom:8px; }}
.chart {{ width:100%; height:320px; }}
.table-wrap {{ background:#fff; border-radius:10px; padding:16px; box-shadow:0 1px 4px rgba(0,0,0,.06); overflow-x:auto; }}
.table-wrap h3 {{ font-size:14px; color:#555; margin-bottom:8px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; white-space:nowrap; }}
th {{ text-align:left; padding:8px 10px; background:#f8f9fa; border-bottom:2px solid #dee2e6; cursor:pointer; user-select:none; }}
th:hover {{ background:#e9ecef; }}
td {{ padding:7px 10px; border-bottom:1px solid #eee; }}
tr:hover {{ background:#f5f6fa; }}
.tag {{ display:inline-block; padding:1px 8px; border-radius:4px; font-size:12px; font-weight:600; }}
.tag-buy {{ background:#e8f8e8; color:#27ae60; }}
.tag-sell {{ background:#fde8e8; color:#e74c3c; }}
.dir-up {{ color:#e74c3c; }}
.dir-down {{ color:#27ae60; }}
.filter-bar {{ margin-bottom:12px; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
.filter-bar label {{ font-size:13px; color:#666; }}
.filter-bar select {{ padding:4px 8px; border:1px solid #ddd; border-radius:4px; font-size:13px; }}
.filter-bar input {{ padding:4px 8px; border:1px solid #ddd; border-radius:4px; font-size:13px; width:160px; }}
@media (max-width:768px) {{ .cards {{ grid-template-columns:repeat(2,1fr); }} .grid2 {{ grid-template-columns:1fr; }} }}
.tab-bar {{ display:flex; gap:0; margin-bottom:16px; background:#fff; border-radius:8px; overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
.tab {{ padding:10px 28px; border:none; background:#fff; font-size:14px; cursor:pointer; color:#666; transition:all .2s; }}
.tab:hover {{ background:#f5f6fa; }}
.tab.active {{ background:#1a1a2e; color:#fff; font-weight:600; }}
.tab-content {{ display:none; }}
.tab-content.active {{ display:block; }}
</style>
</head>
<body>
<div class="header">
  <h1>信号扫描仪表盘</h1>
  <p>生成时间: {data["generated_at"]}</p>
</div>
<div class="container" style="padding-bottom:32px;">
  <div class="tab-bar">
    <button class="tab active" onclick="switchTab('dashboard')">信号扫描</button>
    <button class="tab" onclick="switchTab('bubble')">评分气泡</button>
  </div>
  <div id="tab-dashboard" class="tab-content active"></div>
  <div id="tab-bubble" class="tab-content"></div>
</div>

<script>
var D = {encoded};

// 概览卡片
var cardHtml = '<div class="cards">' + [
  {{num:D.overview.buy, label:'买入信号 (BUY)', cls:'buy'}},
  {{num:D.overview.sell, label:'卖出信号 (SELL)', cls:'sell'}},
  {{num:D.overview.total, label:'有效信号合计', cls:'total'}},
  {{num:D.table.length, label:'触发信号的个股', cls:'stocks'}},
].map(function(c){{ return '<div class="card"><div class="num '+c.cls+'">'+c.num+'</div><div class="label">'+c.label+'</div></div>'}}).join('')+'</div>';

// 图表容器（主页不再显示气泡图，请在"评分气泡"标签页查看）
var chartsHtml = '';

// 表格
var sortKey=null, sortDir=1;
function renderTable() {{
  var sig = (document.getElementById('fSig')||{{}}).value||'';
  var ma = (document.getElementById('fMa')||{{}}).value||'';
  var kw = ((document.getElementById('fKw')||{{}}).value||'').toLowerCase();
  var rows = D.table.filter(function(r){{
    if(sig && r.signal!==sig) return false;
    if(ma && r.ma!==parseInt(ma)) return false;
    if(kw && !r.symbol.toLowerCase().includes(kw)) return false;
    return true;
  }});
  if(sortKey) rows.sort(function(a,b){{ var va=a[sortKey],vb=b[sortKey]; if(va==null)return 1; if(vb==null)return -1; return va<vb?-sortDir:va>vb?sortDir:0; }});

  var h='<div class="filter-bar">';
  h+='<label>信号:</label><select id="fSig" onchange="renderTable()"><option value="">全部</option><option value="BUY">BUY</option><option value="SELL">SELL</option></select>';
  h+='<label>均线:</label><select id="fMa" onchange="renderTable()"><option value="">全部</option>'+[5,10,20,30,60].map(function(m){{return '<option value="'+m+'">'+m+'</option>'}}).join('')+'</select>';
  h+='<label>搜索:</label><input id="fKw" placeholder="代码..." oninput="renderTable()">';
  h+='<span style="font-size:12px;color:#888;margin-left:auto;">'+rows.length+' 条</span></div>';

  h+='<table><thead><tr>';
  D.table_cols.forEach(function(c){{ h+='<th onclick="sortBy(\\''+c[0]+'\\')">'+c[1]+(sortKey===c[0]?(sortDir>0?' ▲':' ▼'):'')+'</th>'; }});
  h+='</tr></thead><tbody>';
  rows.forEach(function(r){{
    var tagCls = r.signal==='BUY'?'tag-buy':'tag-sell';
    var dirCls = r.dir===1?'dir-up':'dir-down';
    h+='<tr>';
    h+='<td><strong>'+r.symbol+'</strong></td>';
    h+='<td>'+r.ma+'</td>';
    h+='<td>'+(r.datetime||'-')+'</td>';
    h+='<td>'+(r.close!=null?r.close.toFixed(2):'-')+'</td>';
    h+='<td><span class="tag '+tagCls+'">'+(r.signal||'-')+'</span></td>';
    h+='<td>'+(r.score!=null?r.score.toFixed(1):'-')+'</td>';
    h+='<td>'+(r.hold_progress!=null?(r.hold_progress*100).toFixed(1)+'%':'-')+'</td>';
    h+='<td>'+(r.daily_return!=null?(r.daily_return*100).toFixed(2)+'%':'-')+'</td>';
    h+='<td class="'+dirCls+'">'+(r.dir===1?'↑ 多头':'↓ 空头')+'</td>';
    h+='<td>'+(r.buy_signal_days!=null?r.buy_signal_days:'-')+'</td>';
    h+='</tr>';
  }});
  h+='</tbody></table>';
  document.getElementById('signalTable').innerHTML = h;
}}
function sortBy(k) {{ if(sortKey===k) sortDir=-sortDir; else {{ sortKey=k; sortDir=1; }} renderTable(); }}

// 渲染 ECharts — 可复用的气泡图渲染器
var SIG_COLORS = {{'BUY':'#27ae60','SELL':'#e74c3c','HOLD':'#2980b9','WATCH':'#95a5a6'}};
function renderBubbleChart(domId) {{
  if(!D.bubble || !D.bubble.length) return;
  var chart = echarts.init(document.getElementById(domId));

  var sigOrder = ['BUY','SELL','HOLD','WATCH'];
  var seriesData = [];
  sigOrder.forEach(function(sig) {{
    var pts = D.bubble.filter(function(d){{return d.signal===sig;}});
    if(!pts.length) return;
    pts.sort(function(a,b){{return b.score-a.score;}});
    seriesData.push({{
      name:SIG_NAMES[sig], type:'scatter',
      data:pts.map(function(d,i){{
        var chg = (sig==='BUY'||sig==='HOLD') ? d.buy_change : d.sell_change;
        var absChg = Math.abs(chg);
        var sz = Math.max(8,Math.min(50,Math.sqrt(absChg)*3+8));
        var item = {{value:[d.score,chg,absChg,d.symbol,d.name]}};
        if(i<3 && sz>22) item.label = {{show:true,formatter:function(p){{return (p.value[3]||'').replace(/^(US\\.|CC\\.)/,'');}},fontSize:11,fontWeight:'bold',color:'#fff',position:'inside'}};
        return item;
      }}),
      symbolSize:function(d){{var a=(d.value||d)[2];return Math.max(8,Math.min(50,Math.sqrt(a)*3+8));}},
      labelLayout:{{hideOverlap:true}},
      itemStyle:{{color:SIG_COLORS[sig],opacity:0.7}}
    }});
  }});

  if(seriesData.length) {{
    chart.setOption({{
      tooltip:{{formatter:function(p){{var v=p.value||p.data;return (v[4]||v[3])+' ('+v[3]+')<br/>评分: '+v[0]+'<br/>涨跌幅: '+v[1].toFixed(2)+'%<br/>'+p.seriesName;}}}},
      legend:{{top:0}},
      grid:{{left:55,right:40,bottom:50,top:50}},
      xAxis:{{type:'value',show:true,name:'评分',nameLocation:'middle',nameGap:30,axisLabel:{{fontSize:12}}}},
      yAxis:{{type:'value',show:true,name:'涨跌幅 (%)',nameLocation:'middle',nameGap:45,axisLabel:{{fontSize:12,formatter:function(v){{return v+'%';}}}}}},
      dataZoom:[
        {{type:'inside',xAxisIndex:0}},
        {{type:'inside',yAxisIndex:0}},
        {{type:'slider',xAxisIndex:0,height:20,bottom:5,showDataShadow:false,borderColor:'#ddd',fillerColor:'rgba(60,140,220,0.15)'}}
      ],
      series:seriesData
    }});
  }}
  window.addEventListener('resize',function(){{chart.resize();}});
  return chart;
}}
function initCharts() {{
  // 气泡图仅在"评分气泡"标签页中由 switchTab 按需渲染
}}
// Tab 切换
function switchTab(name) {{
  document.querySelectorAll('.tab-content').forEach(function(el){{ el.classList.remove('active'); }});
  document.querySelectorAll('.tab').forEach(function(el){{ el.classList.remove('active'); }});
  document.getElementById('tab-'+name).classList.add('active');
  var tabs = document.querySelector('.tab-bar').children;
  for(var i=0;i<tabs.length;i++) {{ if(tabs[i].textContent.includes(name==='dashboard'?'信号':'评分')) {{ tabs[i].classList.add('active'); }} }}
  if(name==='bubble') renderBubbleChart('fullBubbleChart');
}}

// 按信号分类的4个板块（可排序）
var SIG_NAMES = {{'BUY':'买入', 'SELL':'卖出', 'HOLD':'持有', 'WATCH':'观察'}};
// 排序状态
var sigSort = {{}};
function sortSig(sig, key) {{
  var st = sigSort[sig];
  if (st.key === key) st.dir = -st.dir;
  else {{ st.key = key; st.dir = 1; }}
  _renderSigTableBody(sig);
}}
function _renderSigTableBody(sig) {{
  var st = sigSort[sig];
  var rows = st.rows.slice().sort(function(a,b){{
    var va = a[st.key], vb = b[st.key];
    if (va == null) return 1; if (vb == null) return -1;
    if (typeof va === 'string') return va < vb ? -st.dir : va > vb ? st.dir : 0;
    return (va - vb) * st.dir;
  }});
  var h = '';
  rows.forEach(function(r){{
    h += '<tr>';
    st.cols.forEach(function(c){{
      var v = r[c];
      if (c === 'close' || c === 'buy_signal_close' || c === 'sell_signal_close') v = v != null ? v.toFixed(2) : '-';
      else if (c === 'score') v = v != null ? v.toFixed(1) : '-';
      else if (c === 'buy_signal_change' || c === 'sell_signal_change') v = v != null ? Number(v).toFixed(2) + '%' : '-';
      else if (c === 'buy_signal_time' || c === 'sell_signal_time') v = v || '-';
      else if (c === 'stock_name') {{ v = v || '-'; h += '<td style="max-width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="'+v+'">'+v+'</td>'; return; }}
      else v = v != null ? v : '-';
      h += '<td>' + (c === 'symbol' ? '<strong>' + v + '</strong>' : v) + '</td>';
    }});
    h += '</tr>';
  }});
  document.getElementById('sigBody-' + sig).innerHTML = h;
}}
function renderSignalSections() {{
  var h = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;min-width:0;">';
  ['BUY','SELL','HOLD','WATCH'].forEach(function(sig){{
    var sec = D.signal_sections[sig];
    if (!sec || !sec.rows.length) {{ var c0=sig==='BUY'?'#27ae60':sig==='SELL'?'#e74c3c':sig==='HOLD'?'#2980b9':'#95a5a6'; h+='<div class="chart-box" style="min-width:0;border-top:4px solid '+c0+'"><h3>'+SIG_NAMES[sig]+'信号 <span style="color:'+c0+'">0</span></h3><p style="color:#aaa;font-size:13px;padding:12px 0;">暂无数据</p></div>'; return; }}
    var defKey = (sig === 'BUY' || sig === 'HOLD') ? 'buy_signal_days' : 'sell_signal_days';
    sigSort[sig] = {{key: defKey, dir: 1, rows: sec.rows.slice(), cols: sec.cols, labels: sec.labels}};
    var sigColor = sig==='BUY'?'#27ae60':sig==='SELL'?'#e74c3c':sig==='HOLD'?'#2980b9':'#95a5a6';
    h+='<div class="chart-box" style="min-width:0;border-top:4px solid '+sigColor+'"><h3>'+SIG_NAMES[sig]+'信号 <span style="color:'+sigColor+'">'+sec.rows.length+'</span></h3>';
    h+='<div style="overflow-x:auto;"><table style="font-size:12px;width:100%;">';
    h+='<thead id="sigHead-'+sig+'"><tr>';
    sec.labels.forEach(function(l, i){{ h+='<th onclick="sortSig(\\''+sig+'\\',\\''+sec.cols[i]+'\\')">'+l+'</th>'; }});
    h+='</tr></thead><tbody id="sigBody-'+sig+'"></tbody></table></div></div>';
  }});
  h += '</div>';
  return h;
}}
function initSigTables() {{
  ['BUY','SELL','HOLD','WATCH'].forEach(function(sig){{ if (sigSort[sig]) _renderSigTableBody(sig); }});
}}

document.getElementById('tab-dashboard').innerHTML = cardHtml + chartsHtml + renderSignalSections() + '<div class="table-wrap"><h3>信号明细（最新信号/股）</h3><div id="signalTable"></div></div>';
document.getElementById('tab-bubble').innerHTML = D.bubble && D.bubble.length
  ? '<div class="chart-box" style="min-height:calc(100vh - 220px);display:flex;flex-direction:column;"><h3>评分气泡图</h3><div id="fullBubbleChart" style="width:100%;flex:1;min-height:400px;"></div></div>'
  : '<div class="chart-box"><h3>评分气泡图</h3><p style="color:#aaa;font-size:13px;padding:20px 0;">暂无气泡图数据</p></div>';
renderTable();
initCharts();
initSigTables();
</script>
</body>
</html>"""


def main():
    result = load_data()
    if result is None:
        return
    df, fpath = result

    signals = df[df["signal"] != "NONE"].copy()
    data_json = build_json_data(signals)
    html = generate_html(data_json)

    out_dir = os.path.dirname(fpath) or "results"
    out_path = os.path.join(out_dir, "dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"仪表盘已生成: {out_path}")


if __name__ == "__main__":
    main()
