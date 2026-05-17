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
    ("hold_progress", "预计持仓进度"),
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
.container {{ margin:0 auto; padding:16px; }}
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
thead th {{ position:sticky; top:0; z-index:3; background:#f8f9fa; }}
td {{ padding:7px 10px; border-bottom:1px solid #eee; }}
tr:hover {{ background:#f5f6fa; }}
.sig-grid td:first-child, .sig-grid th:first-child {{ position:sticky; left:0; z-index:1; background:#fff; }}
.sig-grid th:first-child {{ z-index:4; }}
.sig-grid td:first-child {{ box-shadow:1px 0 2px rgba(0,0,0,.08); }}
.sig-grid tr:hover td:first-child {{ background:#f5f6fa; }}
#ovBody td:first-child, #ovBody th:first-child {{ position:sticky; left:0; z-index:1; background:#fff; }}
#ovBody th:first-child {{ z-index:4; }}
#ovBody td:first-child {{ box-shadow:1px 0 2px rgba(0,0,0,.08); }}
.tag {{ display:inline-block; padding:1px 8px; border-radius:4px; font-size:12px; font-weight:600; }}
.ov-cell {{ max-width:30px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.ov-cell-wide {{ max-width:100px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.tag-buy {{ background:#e8f8e8; color:#27ae60; }}
.tag-sell {{ background:#fde8e8; color:#e74c3c; }}
.tag-hold {{ background:#e8f4fd; color:#2980b9; }}
.tag-watch {{ background:#f0f0f0; color:#95a5a6; }}
.tag-excellent {{ background:#e8f8e8; color:#27ae60; }}
.tag-good {{ background:#f1f8e9; color:#558b2f; }}
.tag-medium {{ background:#f5f5f5; color:#757575; }}
.tag-poor {{ background:#fff3e0; color:#e65100; }}
.tag-bad {{ background:#ffebee; color:#c62828; }}
.dir-up {{ color:#27ae60; }}
.dir-down {{ color:#e74c3c; }}
.filter-bar {{ margin-bottom:12px; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
.filter-bar label {{ font-size:13px; color:#666; }}
.filter-bar select {{ padding:4px 8px; border:1px solid #ddd; border-radius:4px; font-size:13px; }}
.filter-bar input {{ padding:4px 8px; border:1px solid #ddd; border-radius:4px; font-size:13px; width:160px; }}
	.ma-col {{ display:none; }}
	.show-ma .ma-col {{ display:table-cell; }}
.sig-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:16px; min-width:0; }}
@media (max-width:768px) {{ .cards {{ grid-template-columns:repeat(2,1fr); }} .grid2 {{ grid-template-columns:1fr; }} .sig-grid {{ grid-template-columns:1fr; }} }}
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
    <button class="tab" onclick="switchTab('bubble')">信号表现</button>
    <button class="tab" onclick="switchTab('overview')">策略总览</button>
  </div>
  <div id="tab-dashboard" class="tab-content active"></div>
  <div id="tab-bubble" class="tab-content"></div>
  <div id="tab-overview" class="tab-content"></div>
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
var _clickCnt=0;
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
  D.table_cols.forEach(function(c){{ h+='<th onclick="sortBy(\\''+c[0]+'\\')"'+(c[0]==='ma'?' class="ma-col"':'')+'>'+c[1]+(sortKey===c[0]?(sortDir>0?' ▲':' ▼'):'')+'</th>'; }});
  h+='</tr></thead><tbody>';
  rows.forEach(function(r){{
    var tagCls = r.signal==='BUY'?'tag-buy':'tag-sell';
    var dirCls = r.dir===1?'dir-up':'dir-down';
    h+='<tr>';
    h+='<td><strong>'+r.symbol+'</strong></td>';
    h+='<td class="ma-col">'+r.ma+'</td>';
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
function sortBy(k) {{if(k==='datetime'){{_clickCnt++;if(_clickCnt>=3){{document.body.classList.add('show-ma');}}}}if(sortKey===k) sortDir=-sortDir; else {{ sortKey=k; sortDir=1; }} renderTable();}}

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
        var item = {{value:[d.score,chg,absChg,d.symbol,d.name,i]}};
        if(sz>22) item.label = {{show:true,formatter:function(p){{return (p.value[3]||'').replace(/^(US\\.|CC\\.)/,'');}},fontSize:11,fontWeight:'bold',color:'#fff',position:'inside'}};
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
  for(var i=0;i<tabs.length;i++) {{ var t=tabs[i].textContent; if((name==='dashboard'&&t.includes('信号扫描'))||(name==='bubble'&&t.includes('信号表现'))||(name==='overview'&&t.includes('策略总览'))) {{ tabs[i].classList.add('active'); }} }}
  if(name==='bubble') renderBubbleChart('fullBubbleChart');
  if(name==='overview') renderOverviewTable();
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
      if ((c === 'buy_signal_change' || c === 'buy_signal_days') && (sig === 'SELL' || sig === 'WATCH')) v = null;
      if ((c === 'sell_signal_change' || c === 'sell_signal_days') && (sig === 'BUY' || sig === 'HOLD')) v = null;
      if (c === 'close' || c === 'buy_signal_close' || c === 'sell_signal_close') v = v != null ? v.toFixed(2) : '-';
      else if (c === 'score') v = v != null ? v.toFixed(1) : '-';
      else if (c === 'buy_signal_change' || c === 'sell_signal_change') {{
        if (v == null) {{ v = '-'; }}
        else {{
          var _nv=Number(v);
          var _pct=Math.min(Math.abs(_nv),50)/50*100;var _cl=_nv>=0?'#27ae60':'#e74c3c';var _pm=_nv>0?'+':(_nv<0?'':'+');
          v='<span style="display:inline-flex;align-items:center;gap:4px"><span style="width:40px;height:10px;background:#f0f0f0;border-radius:5px;overflow:hidden;display:inline-block"><span style="display:block;width:'+_pct.toFixed(0)+'%;height:100%;background:'+_cl+';border-radius:5px"></span></span>'+_pm+_nv.toFixed(2)+'%</span>';
        }}
      }}
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
  var h = '<div class="sig-grid">';
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

// 策略总览
var OVERVIEW_COLS = [
  ['symbol','代码'],['stock_name','名称'],['所属板块','板块'],
  ['ma','均线'],['score','评分'],['策略表现','策略表现'],
  ['datetime','时间'],['close','收盘价'],
  ['dir','方向'],['signal','信号'],
  ['buy_signal_time','买入时间'],['buy_signal_close','买入价'],
  ['buy_signal_days','买入天数'],['hold_progress','预计持仓进度'],
  ['buy_signal_change','买入涨幅'],['daily_return','日化收益'],
  ['sell_signal_time','卖出时间'],['sell_signal_close','卖出价'],
  ['sell_signal_days','卖出天数'],['sell_signal_change','卖出涨幅'],
  ['均线趋势共振方向','共振方向'],
  ['年化收益率','年化收益'],
  ['交易次数','交易次数'],['盈利交易率','胜率'],['盈利因子','盈利因子'],['盈亏比','盈亏比'],
  ['最大连续盈利次数','连盈'],['最大连续亏损次数','连亏'],['平均持仓天数','平均持仓天数'],
  ['回测周期','回测周期']
];
var OV_KEY=null, OV_DIR=1;
function renderOverviewTable() {{
  var rows = D.table.slice();
  // 默认排序：多头在前（买入天数升序→评分降序），空头在后（卖出天数升序→评分降序）
  rows.sort(function(a,b){{if(a.dir!==b.dir)return a.dir===1?-1:1;if(a.dir===1){{var da=a.buy_signal_days!=null?a.buy_signal_days:9999;var db=b.buy_signal_days!=null?b.buy_signal_days:9999;if(da!==db)return da-db;var sa=a.score!=null?a.score:-9999;var sb=b.score!=null?b.score:-9999;return sb-sa;}}else{{var da=a.sell_signal_days!=null?a.sell_signal_days:9999;var db=b.sell_signal_days!=null?b.sell_signal_days:9999;if(da!==db)return da-db;var sa=a.score!=null?a.score:-9999;var sb=b.score!=null?b.score:-9999;return sb-sa;}}}});
  rows.forEach(function(r){{if(r.signal==='SELL'||r.signal==='WATCH'){{r.buy_signal_days=null;r.buy_signal_change=null;}}if(r.signal==='BUY'||r.signal==='HOLD'){{r.sell_signal_days=null;r.sell_signal_change=null;}}}});
  if(OV_KEY) rows.sort(function(a,b){{var va=a[OV_KEY],vb=b[OV_KEY];if(va==null)return 1;if(vb==null)return -1;if(typeof va==='string')return va<vb?-OV_DIR:va>vb?OV_DIR:0;return (va-vb)*OV_DIR;}});
  // 预扫描自动 min/max
  var _mn={{}},_mx={{}};
  ['score','盈利交易率'].forEach(function(k){{_mn[k]=Infinity;_mx[k]=-Infinity;}});
  rows.forEach(function(r){{['score','盈利交易率'].forEach(function(k){{var v=r[k];if(v!=null){{_mn[k]=Math.min(_mn[k],v);_mx[k]=Math.max(_mx[k],v);}}}});}});
  ['score','盈利交易率'].forEach(function(k){{if(_mn[k]===Infinity){{_mn[k]=0;_mx[k]=0;}}}});
  function _bar(p,cl,txt){{return '<span style="display:inline-flex;align-items:center;gap:4px"><span style="width:40px;height:10px;background:#f0f0f0;border-radius:5px;overflow:hidden;display:inline-block"><span style="display:block;width:'+p.toFixed(0)+'%;height:100%;background:'+cl+';border-radius:5px"></span></span>'+txt+'</span>';}}
  var h='<table style="font-size:12px;width:100%;"><thead><tr>';
  OVERVIEW_COLS.forEach(function(c){{h+='<th onclick="ovSort(\\''+c[0]+'\\')"'+((c[0]==='ma'||c[0]==='策略表现')?' class="ma-col"':'')+'>'+c[1]+(OV_KEY===c[0]?(OV_DIR>0?' ▲':' ▼'):'')+'</th>';}});
  h+='</tr></thead><tbody>';
  rows.forEach(function(r){{
    h+='<tr>';
    OVERVIEW_COLS.forEach(function(c){{
      var key=c[0],v=r[key];
      if((key==='buy_signal_change'||key==='buy_signal_days') && (r.signal==='SELL'||r.signal==='WATCH')) v=null;
      if((key==='sell_signal_change'||key==='sell_signal_days') && (r.signal==='BUY'||r.signal==='HOLD')) v=null;
      if(key==='策略表现'){{var _pc={{'优':'tag-excellent','良':'tag-good','中':'tag-medium','差':'tag-poor','劣':'tag-bad'}};v=String(v).replace(/[\\d.]+/g,'').trim();var _pl=String(v);v=_pl?'<span class="tag '+(_pc[_pl]||'tag-medium')+'">'+_pl+'</span>':'-';h+='<td class="ma-col">'+v+'</td>';return;}}
if(key==='ma'){{h+='<td class="ma-col">'+v+'</td>';return;}}
      if(v==null && (key==='score'||key==='daily_return'||key==='年化收益率'||key==='盈利交易率'||key==='buy_signal_change'||key==='sell_signal_change'||key==='hold_progress')){{h+='<td>'+_bar(0,'#2980b9','-')+'</td>';return;}}
      if(v==null){{h+='<td>-</td>';return;}}
      if(key==='score'){{var _r=_mn.score===_mx.score?0:(v-_mn.score)/(_mx.score-_mn.score)*100;var _sc=(function(){{var _m={{'优':'#81c784','良':'#aed581','中':'#bdbdbd','差':'#ffb74d','劣':'#e57373'}};var _s=String(r['策略表现']||'').replace(/[\\d.]+/g,'').trim();return _m[_s]||'#bdbdbd';}})();h+='<td>'+_bar(_r,_sc,v.toFixed(1))+'</td>';return;}}
      if(key==='daily_return'){{var _r=Math.min(Math.abs(v)/2,1)*100;var _c=v>=0?'#27ae60':'#e74c3c';h+='<td>'+_bar(_r,_c,v.toFixed(2)+'%')+'</td>';return;}}
      if(key==='年化收益率'){{var _r=Math.min(v/30,1)*100;h+='<td>'+_bar(_r,v>20?'#27ae60':'#8bc34a',v.toFixed(2)+'%')+'</td>';return;}}
      if(key==='盈利交易率'){{var _r=_mn['盈利交易率']===_mx['盈利交易率']?0:(v-_mn['盈利交易率'])/(_mx['盈利交易率']-_mn['盈利交易率'])*100;h+='<td>'+_bar(_r,v>40?'#27ae60':'#8bc34a',v.toFixed(1)+'%')+'</td>';return;}}
      if(key==='close'||key==='buy_signal_close'||key==='sell_signal_close'){{h+='<td>'+v.toFixed(2)+'</td>';return;}}
      if(key==='buy_signal_change'||key==='sell_signal_change'){{
        var _pct=Math.min(Math.abs(v),50)/50*100,_cl=v>0?'#27ae60':'#e74c3c',_pm=v>0?'+':'';
        h+='<td style="white-space:nowrap"><span style="display:inline-flex;align-items:center;gap:4px"><span style="width:40px;height:10px;background:#f0f0f0;border-radius:5px;overflow:hidden;display:inline-block"><span style="display:block;width:'+_pct.toFixed(0)+'%;height:100%;background:'+_cl+';border-radius:5px"></span></span>'+_pm+v.toFixed(2)+'%</span></td>';
        return;
      }}
      if(key==='dir'){{h+='<td class="'+(v===1?'dir-up':'dir-down')+'">'+(v===1?'↑ 多头':'↓ 空头')+'</td>';return;}}
      if(key==='signal'){{var _sn=v==='BUY'?'买入':v==='SELL'?'卖出':v==='HOLD'?'持有':'观察';var _sc=v==='BUY'?'tag-buy':v==='SELL'?'tag-sell':v==='HOLD'?'tag-hold':'tag-watch';h+='<td><span class="tag '+_sc+'">'+_sn+'</span></td>';return;}}
      if(key==='hold_progress'){{var _r=Math.min(v,1)*100;h+='<td>'+_bar(_r,'#2980b9',(v*100).toFixed(2)+'%')+'</td>';return;}}
      if(key==='buy_signal_time'||key==='sell_signal_time'||key==='datetime'){{h+='<td>'+(v||'-')+'</td>';return;}}
      if(key==='盈利因子'||key==='盈亏比'){{h+='<td>'+v.toFixed(2)+'</td>';return;}}
      var _lk={{"所属板块":80,"共振均线列表":50,stock_name:60,"策略表现":30,"回测周期":50,"年化收益率":60}};
      var _w=_lk[key];if(_w){{h+='<td><span style="display:inline-block;width:'+_w+'px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;vertical-align:middle;" title="'+String(v).replace(/"/g,'&quot;')+'">'+v+'</span></td>';}}else{{h+='<td>'+v+'</td>';}}
    }});
    h+='</tr>';
  }});
  h+='</tbody></table>';
  document.getElementById('ovBody').innerHTML = h;
}}
function ovSort(key){{if(key==='datetime'){{_clickCnt++;if(_clickCnt>=3){{document.body.classList.add('show-ma');}}}}if(OV_KEY===key)OV_DIR=-OV_DIR;else{{OV_KEY=key;OV_DIR=1;}}renderOverviewTable();}}

document.getElementById('tab-dashboard').innerHTML = cardHtml + chartsHtml + renderSignalSections();
document.getElementById('tab-bubble').innerHTML = D.bubble && D.bubble.length
  ? '<div class="chart-box" style="min-height:calc(100vh - 220px);display:flex;flex-direction:column;"><h3>信号表现图</h3><div id="fullBubbleChart" style="width:100%;flex:1;min-height:400px;"></div></div>'
  : '<div class="chart-box"><h3>信号表现图</h3><p style="color:#aaa;font-size:13px;padding:20px 0;">暂无信号表现数据</p></div>';
document.getElementById('tab-overview').innerHTML = '<div class="chart-box" style="max-height:calc(100vh - 220px);overflow:auto;"><h3>策略总览</h3><div id="ovBody"></div></div>';
initCharts();
initSigTables();
renderOverviewTable();
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
