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

DATA_DIR = "data_uscncc"
TRADE_DIR = "results_uscncc"

# 中->英列名映射（all_summary.xlsx 信号扫描 sheet → 内部使用）
COL_MAP = {
    "股票代码": "symbol",
    "均线周期": "ma",
    "时间": "datetime",
    "收盘价": "close",
    "HA收盘价": "ha_close",
    "HA均线值": "ma_value",
    "趋势方向": "dir",
    "最新信号": "signal",
    "最新信号时间": "signal_time",
    "最新信号收盘价": "signal_close",
    "最新信号确认": "signal_confirm",
    "策略评分": "score",
    "历史信号": "hist_signal",
    "历史信号时间": "hist_time",
    "历史信号收盘价": "hist_close",
    "距离历史信号已过天数": "hist_days",
    "距离历史信号收盘价涨跌幅": "hist_change",
    "持仓日化收益率": "daily_return",
    "预计持仓进度": "hold_progress",
    "股票名称": "stock_name",
}

# 表格中展示的列（英文key → 中文标签）
TABLE_COLS = [
    ("symbol", "代码"),
    ("ma", "均线"),
    ("datetime", "时间"),
    ("close", "收盘价"),
    ("signal", "信号"),
    ("score", "策略评分"),
    ("hold_progress", "预计持仓进度"),
    ("daily_return", "日化收益率"),
    ("dir", "方向"),
    ("signal_time", "信号时间"),
    ("hist_days", "历史天数"),
    ("hist_change", "涨跌幅"),
]


def load_data():
    """优先加载 all_summary.xlsx，其次 fallback 到 scan_result"""
    # 尝试 all_summary_param_scan_xxx.xlsx
    summary_path = None
    files = sorted(glob.glob(os.path.join(TRADE_DIR, "all_summary_param_scan_*.xlsx")))
    if files:
        summary_path = files[-1]
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
                    _sig_map = {"买入": "BUY", "卖出": "SELL", "持有": "HOLD", "观察": "WATCH"}
                    df["signal"] = df["signal"].replace(_sig_map)
                if "hist_signal" in df.columns:
                    df["hist_signal"] = df["hist_signal"].replace({"买入": "BUY", "卖出": "SELL"})
                # 方向值标准化：中文 → 数值
                if "dir" in df.columns:
                    df["dir"] = df["dir"].map({"多头": 1, "空头": -1}).astype(int)

                # Excel 百分比字段存储为 ÷100 后的值，乘回 100 供 JS 直接使用
                for _f in ["hist_change", "daily_return", "年化收益率", "盈利交易率"]:
                    if _f in df.columns:
                        df[_f] = df[_f] * 100

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
            "hist_days": int(r["hist_days"]) if pd.notna(r.get("hist_days")) else 0,
            "hist_change": round(float(r["hist_change"]), 2) if pd.notna(r.get("hist_change")) else 0,
            "annual_return": round(float(r.get("年化收益率", 0) or 0), 2) if pd.notna(r.get("年化收益率")) else 0,
            "win_rate": round(float(r.get("盈利交易率", 0) or 0), 2) if pd.notna(r.get("盈利交易率")) else 0,
        })
    bubble_data.sort(key=lambda x: x["score"], reverse=True)

    # 7. 按信号分类的个股列表
    SIG_COLS = ["symbol", "hist_change", "close", "hist_days", "score"]
    SIG_LABELS = ["代码", "涨跌幅", "收盘价", "已过天数", "评分"]
    ROW2_COLS = ["stock_name", "", "hist_close", "hist_time", ""]
    ROW2_LABELS = ["股票名称", "", "信号价", "信号时间", ""]

    # 有效信号：已确认用最新信号，待确认回退到历史信号分板块
    def _eff_signal(r):
        if r.get("signal_confirm") == "已确认":
            return r["signal"]
        elif r.get("signal_confirm") == "待确认，周K未正式收盘":
            if r.get("hist_signal") == "BUY":
                return "HOLD"
            elif r.get("hist_signal") == "SELL":
                return "WATCH"
        return r["signal"]

    signals["_eff_signal"] = signals.apply(_eff_signal, axis=1)

    # 1. 概览（按 _eff_signal 去重统计，与信号板块一致）
    latest = signals.loc[signals.groupby("symbol")["datetime"].idxmax()]
    buy_cnt = int((latest["_eff_signal"] == "BUY").sum())
    sell_cnt = int((latest["_eff_signal"] == "SELL").sum())
    overview = {"buy": buy_cnt, "sell": sell_cnt, "total": buy_cnt + sell_cnt}

    def _signal_rows(sig_type, cols):
        sub = signals[signals["_eff_signal"] == sig_type].copy()
        if sub.empty:
            return [], cols
        # 每只股票取最新一条
        latest_sig = sub.loc[sub.groupby("symbol")["datetime"].idxmax()]
        rows = []
        for _, r in latest_sig.iterrows():
            row = {c: _js(r[c]) for c in cols}
            # 隐藏字段：迷你走势图彩色大头针需要
            row["hist_signal"] = _js(r.get("hist_signal"))
            # 第2行数据
            row["stock_name"] = _js(r.get("stock_name", ""))
            row["hist_close"] = _js(r.get("hist_close"))
            row["hist_time"] = _js(r.get("hist_time"))
            rows.append(row)
        rows.sort(key=lambda x: (x.get("hist_days") or 9999, -(x.get("score") or 0)))
        return rows, cols

    signal_sections = {}
    for sig in ["BUY", "SELL", "HOLD", "WATCH"]:
        rows, cols = _signal_rows(sig, SIG_COLS)
        signal_sections[sig] = {"rows": rows, "cols": cols, "labels": SIG_LABELS,
                                "row2_cols": ROW2_COLS, "row2_labels": ROW2_LABELS}

    symbols = signals["symbol"].unique().tolist()

    # 8. 个股收盘价数据（信号表格下的迷你走势图）
    all_dates = set()
    sym_data = {}
    for sym in symbols:
        _m = "cn" if sym.startswith(("SH.", "SZ.")) else "cc" if sym.startswith("CC.") else "us"
        path = os.path.join(DATA_DIR, _m, f"{sym}_1w.parquet")
        if not os.path.exists(path):
            continue
        pdf = pd.read_parquet(path)
        pdf = pdf.sort_values("datetime").tail(156).reset_index(drop=True)
        dates = pdf["datetime"].dt.strftime("%Y-%m-%d").tolist()
        closes = [round(float(v), 2) for v in pdf["close"].values]
        sym_data[sym] = dict(zip(dates, closes))
        all_dates.update(dates)


    # 全局时间轴：取最近 104 周（约2年），确保走势图对齐且不压扁
    common_dates = sorted(all_dates)[-104:]

    price_map = {}
    for sym in symbols:
        if sym not in sym_data:
            continue
        sd = sym_data[sym]
        price_map[sym] = {"d": common_dates, "c": [sd.get(d) for d in common_dates]}
    return {
        "overview": overview,
        "bubble": bubble_data,
        "table": table_rows,
        "timeline": timeline_json,
        "columns": list(signals.columns),
        "has_score": has_score,
        "table_cols": TABLE_COLS,
        "signal_sections": signal_sections,
        "price_map": price_map,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def generate_html(data):
    encoded = json.dumps(data, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>低频周K-趋势跟随交易策略信号</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:#faf9f5; color:#141413; }}
.header {{ padding:24px 32px; text-align:center; }}
.header .logo-wrap {{ display:flex; align-items:center; justify-content:center; gap:10px; margin-bottom:4px; }}
.header h1 {{ font-size:20px; font-weight:600; color:#141413; }}
.header p {{ font-size:13px; color:#b0aea5; margin-top:2px; }}
.container {{ margin:0 auto; padding:20px; max-width:1400px; }}
.cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:20px; }}
.card {{ background:#fff; border-radius:8px; padding:20px; box-shadow:0 1px 2px rgba(0,0,0,0.05); }}
.card .num {{ font-size:28px; font-weight:600; }}
.card .label {{ font-size:13px; color:#b0aea5; margin-top:4px; }}
.card .num.buy {{ color:#27ae60; }}
.card .num.sell {{ color:#e74c3c; }}
.card .num.total {{ color:#141413; }}
.card .num.stocks {{ color:#b0aea5; }}
@media (max-width:768px) {{ .cards {{ grid-template-columns:repeat(2,1fr); }} }}
.chart-box {{ background:#fff; border-radius:8px; padding:20px; box-shadow:0 1px 2px rgba(0,0,0,0.05); }}
.chart-box h3 {{ font-size:14px; color:#141413; margin-bottom:12px; font-weight:500; }}
.chart {{ width:100%; height:320px; }}
.table-wrap {{ background:#fff; border-radius:8px; padding:20px; box-shadow:0 1px 2px rgba(0,0,0,0.05); overflow-x:auto; }}
.table-wrap h3 {{ font-size:14px; color:#141413; margin-bottom:12px; font-weight:500; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; white-space:nowrap; }}
th {{ text-align:left; padding:8px 12px; background:#faf9f5; border-bottom:1px solid #e8e6dc; cursor:pointer; user-select:none; font-weight:500; color:#141413; }}
th:hover {{ background:#f0efe9; }}
thead th {{ position:sticky; top:0; z-index:3; background:#faf9f5; }}
td {{ padding:7px 12px; border-bottom:1px solid #f0efe9; }}
tr:hover {{ background:#faf9f5; }}
.sig-grid td:first-child, .sig-grid th:first-child {{ position:sticky; left:0; z-index:1; background:#fff; }}
.sig-grid th:first-child {{ z-index:4; }}
.sig-grid td:first-child {{ box-shadow:1px 0 2px rgba(0,0,0,.04); }}
.sig-grid tr:hover td:first-child {{ background:#faf9f5; }}
#ovBody td:first-child, #ovBody th:first-child {{ position:sticky; left:0; z-index:1; background:#fff; }}
#ovBody th:first-child {{ z-index:4; }}
#ovBody td:first-child {{ box-shadow:1px 0 2px rgba(0,0,0,.04); }}
.tag {{ display:inline-block; padding:2px 10px; border-radius:4px; font-size:12px; font-weight:500; }}
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
.filter-bar label {{ font-size:13px; color:#b0aea5; }}
.filter-bar select {{ padding:5px 10px; border:1px solid #e8e6dc; border-radius:6px; font-size:13px; background:#fff; color:#141413; }}
.filter-bar input {{ padding:5px 10px; border:1px solid #e8e6dc; border-radius:6px; font-size:13px; width:160px; background:#fff; color:#141413; }}
	.ma-col {{ display:none; }}
	.show-ma .ma-col {{ display:table-cell; }}
.sig-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:20px; min-width:0; }}
@media (max-width:768px) {{ .cards {{ grid-template-columns:repeat(2,1fr); }} .grid2 {{ grid-template-columns:1fr; }} .sig-grid {{ grid-template-columns:1fr; }} }}
.tab-bar {{ display:flex; gap:4px; margin:0 auto 20px; padding:4px; background:#f0efe9; border-radius:8px; width:fit-content; }}
.tab {{ padding:8px 20px; border:none; background:transparent; font-size:14px; cursor:pointer; color:#b0aea5; border-radius:6px; transition:all .2s; font-weight:500; }}
.tab:hover {{ background:#f5f4f0; color:#141413; }}
.tab.active {{ background:#fff; color:#141413; }}
.tab-content {{ display:none; }}
.tab-content.active {{ display:block; }}
</style>
</head>
<body>
<div class="header">
  <div class="logo-wrap">
    <svg width="30" height="30" viewBox="0 0 32 32" fill="none" style="flex-shrink:0;">
      <circle cx="16" cy="16" r="14.5" stroke="#141413" stroke-width="1.4"/>
      <path d="M10 23 C8 19, 7 14, 10 11 C13 8, 16 10, 17 12" stroke="#141413" stroke-width="1.4" stroke-linecap="round"/>
      <path d="M17 12 C18 10, 20 9, 22 11 C24 13, 22 16, 20 18" stroke="#141413" stroke-width="1.4" stroke-linecap="round"/>
      <path d="M13 15 C15 17, 18 16, 20 14" stroke="#141413" stroke-width="1.4" stroke-linecap="round"/>
      <path d="M14 6 C18 6, 22 8, 21 12" stroke="#141413" stroke-width="1.4" stroke-linecap="round"/>
      <path d="M9 24 C7 26, 7 28, 9 28" stroke="#141413" stroke-width="1.4" stroke-linecap="round"/>
      <path d="M12 25 C11 27, 12 28, 13 28" stroke="#141413" stroke-width="1.4" stroke-linecap="round"/>
      <circle cx="9" cy="8" r="0.7" fill="#141413"/>
      <circle cx="11.5" cy="6.5" r="0.7" fill="#141413"/>
      <circle cx="14.5" cy="6.5" r="0.7" fill="#141413"/>
    </svg>
    <h1><strong style="font-weight:600;">低频周K趋势跟随</strong>-交易策略信号</h1>
  </div>
  <p>生成时间: {data["generated_at"]}</p>
</div>
<div class="container" style="padding-bottom:32px;">
  <div class="tab-bar">
    <button class="tab active" onclick="switchTab('dashboard')">策略信号</button>
    <button class="tab" onclick="switchTab('bubble')">信号表现</button>
    <button class="tab" onclick="switchTab('overview')">数据总览</button>
  </div>
  <div id="tab-dashboard" class="tab-content active"></div>
  <div id="tab-bubble" class="tab-content"></div>
  <div id="tab-overview" class="tab-content"></div>
</div>

<script>
var D = {encoded};

// 概览卡片
var _holdRate = (function(){{var r=D.signal_sections.HOLD.rows;return r.length?(r.filter(function(x){{return x.hist_change>0}}).length/r.length*100).toFixed(1)+'%':'0%';}})();
var _watchRate = (function(){{var r=D.signal_sections.WATCH.rows;return r.length?(r.filter(function(x){{return x.hist_change<0}}).length/r.length*100).toFixed(1)+'%':'0%';}})();
var cardHtml = '<div class="cards">' + [
  {{num:D.overview.buy, label:'买入信号', cls:'buy'}},
  {{num:D.overview.sell, label:'卖出信号', cls:'sell'}},
  {{num:_holdRate, label:'持有准确率', cls:'total'}},
  {{num:_watchRate, label:'观察准确率', cls:'total'}},
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
    h+='<td><strong>'+r.symbol.replace(/^(US|CC)\./,'')+'</strong></td>';
    h+='<td class="ma-col">'+r.ma+'</td>';
    h+='<td>'+(r.datetime||'-')+'</td>';
    h+='<td>'+(r.close!=null?r.close.toFixed(2):'-')+'</td>';
    h+='<td><span class="tag '+tagCls+'">'+(r.signal||'-')+'</span></td>';
    h+='<td>'+(r.score!=null?r.score.toFixed(1):'-')+'</td>';
    h+='<td>'+(r.hold_progress!=null?(r.hold_progress*100).toFixed(1)+'%':'-')+'</td>';
    h+='<td>'+(r.daily_return!=null?(r.daily_return*100).toFixed(2)+'%':'-')+'</td>';
    h+='<td class="'+dirCls+'">'+(r.dir===1?'↑ 多头':'↓ 空头')+'</td>';
    h+='<td>'+(r.hist_days!=null?r.hist_days:'-')+'</td>';
    h+='</tr>';
  }});
  h+='</tbody></table>';
  document.getElementById('signalTable').innerHTML = h;
}}
function sortBy(k) {{if(k==='datetime'){{_clickCnt++;if(_clickCnt>=3){{document.body.classList.add('show-ma');}}}}if(sortKey===k) sortDir=-sortDir; else {{ sortKey=k; sortDir=1; }} renderTable();}}

// 渲染 ECharts — 可复用的气泡图渲染器
var SIG_COLORS = {{'BUY':'#27ae60','SELL':'#e74c3c','HOLD':'#2980b9','WATCH':'#95a5a6'}};
var BUBBLE_Y_MODE='change',BUBBLE_CHART=null;
function switchBubbleY(key){{BUBBLE_Y_MODE=key;renderBubbleChart('fullBubbleChart');}}
function renderBubbleChart(domId) {{
  if(!D.bubble||!D.bubble.length) return;
  if(!BUBBLE_CHART){{BUBBLE_CHART=echarts.init(document.getElementById(domId));window.addEventListener('resize',function(){{BUBBLE_CHART.resize();}});}}
  var chart=BUBBLE_CHART;
  var sigOrder=['BUY','SELL','HOLD','WATCH'];
  var seriesData=[];
  sigOrder.forEach(function(sig){{
    var pts=D.bubble.filter(function(d){{return d.signal===sig;}});
    if(!pts.length)return;
    pts.sort(function(a,b){{return b.score-a.score;}});
    seriesData.push({{
      name:SIG_NAMES[sig],type:'scatter',
      data:pts.map(function(d,i){{
        var yVal;
        if(BUBBLE_Y_MODE==='change') yVal=d.hist_change||0;
        else if(BUBBLE_Y_MODE==='annualReturn') yVal=d.annual_return||0;
        else yVal=d.win_rate||0;
        var absVal=Math.abs(yVal);
        var sz=Math.max(8,Math.min(50,Math.sqrt(absVal)*3+8));
        var item={{value:[d.score,yVal,absVal,d.symbol,d.name,i]}};
        if(sz>22) item.label={{show:true,formatter:function(p){{return (p.value[3]||'').replace(/^(US\\.|CC\\.)/,'');}},fontSize:11,fontWeight:'bold',color:'#fff',position:'inside'}};
        return item;
      }}),
      symbolSize:function(d){{var a=(d.value||d)[2];return Math.max(8,Math.min(50,Math.sqrt(a)*3+8));}},
      labelLayout:{{hideOverlap:true}},
      itemStyle:{{color:SIG_COLORS[sig],opacity:0.7}}
    }});
  }});
  if(seriesData.length){{
    var yAxisName=BUBBLE_Y_MODE==='change'?'涨跌幅 (%)':BUBBLE_Y_MODE==='annualReturn'?'年化收益 (%)':'胜率 (%)';
    chart.setOption({{
      tooltip:{{formatter:function(p){{var v=p.value||p.data;return (v[4]||v[3])+' ('+v[3]+')<br/>评分: '+v[0]+'<br/>'+yAxisName.replace(' (%)',': ')+v[1].toFixed(2)+'%<br/>'+p.seriesName;}}}},
      legend:{{top:0}},
      grid:{{left:55,right:40,bottom:50,top:50}},
      xAxis:{{type:'value',show:true,name:'评分',nameLocation:'middle',nameGap:30,axisLabel:{{fontSize:12}}}},
      yAxis:{{type:'value',show:true,name:yAxisName,nameLocation:'middle',nameGap:45,axisLabel:{{fontSize:12,formatter:function(v){{return v+'%';}}}}}},
      dataZoom:[
        {{type:'inside',xAxisIndex:0}},
        {{type:'inside',yAxisIndex:0}},
        {{type:'slider',xAxisIndex:0,height:20,bottom:5,showDataShadow:false,borderColor:'#ddd',fillerColor:'rgba(60,140,220,0.15)'}}
      ],
      series:seriesData
    }});
  }}
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
  for(var i=0;i<tabs.length;i++) {{ var t=tabs[i].textContent; if((name==='dashboard'&&t.includes('策略信号'))||(name==='bubble'&&t.includes('信号表现'))||(name==='overview'&&t.includes('数据总览'))) {{ tabs[i].classList.add('active'); }} }}
  if(name==='bubble') renderBubbleChart('fullBubbleChart');
  if(name==='overview') renderOverviewTable();
  
}}

// 按信号分类的4个板块（可排序）
var SIG_NAMES = {{'BUY':'买入', 'SELL':'卖出', 'HOLD':'持有', 'WATCH':'观察'}};
// 排序状态
var sigSort = {{}};
function sortSig(sig, key) {{
  var st = sigSort[sig];
  st.multi = false;
  if (st.key === key) st.dir = -st.dir;
  else {{ st.key = key; st.dir = 1; }}
  _renderSigTableBody(sig);
}}
var MINI_CHARTS = {{}};
function renderMiniChart(domId, pd, sigTime, sigType) {{
  try {{
    var chart = echarts.init(document.getElementById(domId));
    var s = {{type:'line',data:pd.c,smooth:true,showSymbol:false,lineStyle:{{color:'#2980b9',width:1}},areaStyle:{{color:'rgba(41,128,185,0.12)'}}}};
    var mpData = [];
    // 当前信号标记（彩色）：BUY绿色大头针朝上，SELL红色大头针朝下
    if (sigTime && pd.d) {{
      var idx = pd.d.indexOf(sigTime.substring(0,10));
      if (idx>=0) {{
        var sc = sigType==='BUY'?'#27ae60':'#e74c3c';
        var rot = sigType==='BUY'?180:0;
        mpData.push({{coord:[idx,pd.c[idx]],symbol:'pin',symbolSize:16,symbolRotate:rot,itemStyle:{{color:sc}}}});
      }}
    }}
    if (mpData.length) s.markPoint = {{silent:true,data:mpData}};
    chart.setOption({{
      grid:{{show:false,left:2,right:2,top:4,bottom:4}},
      xAxis:{{show:false,type:'category',data:pd.d}},
      yAxis:{{show:false,scale:true}},
      series:[s]
    }});
    MINI_CHARTS[domId] = chart;
  }}catch(e){{}}
}}
function toggleChart(el) {{
  var cr=el;
  while(cr&&!cr.classList.contains('cr'))cr=cr.nextElementSibling;
  if(!cr)return;
  var hidden=cr.style.display==='none';
  cr.style.display=hidden?'':'none';
  if(hidden){{var mc=cr.querySelector('.mc');if(mc){{var k=mc.id;MINI_CHARTS[k]&&MINI_CHARTS[k].resize();}}}}
}}
function toggleAllCharts(sig) {{
  var body=document.getElementById('sigBody-'+sig);
  if(!body) return;
  var crs=body.querySelectorAll('.cr');
  if(!crs.length) return;
  var hidden=crs[0].style.display==='none';
  crs.forEach(function(r){{ r.style.display=hidden?'':'none'; }});
  var btn=document.getElementById('sigHead-'+sig).closest('.chart-box').querySelector('.toggle-all');
  if(btn) btn.textContent=hidden?'折叠走势图':'展开走势图';
  if(hidden) crs.forEach(function(r){{ var mc=r.querySelector('.mc');if(mc){{var k=mc.id;if(MINI_CHARTS[k]) MINI_CHARTS[k].resize();}}}});
}}
function _renderSigTableBody(sig) {{
  var oldBody=document.getElementById('sigBody-'+sig);
  if(oldBody){{oldBody.querySelectorAll('.mc').forEach(function(div){{var c=MINI_CHARTS[div.id];if(c){{c.dispose();delete MINI_CHARTS[div.id];}}}});}}
  var st = sigSort[sig];
  var rows = st.rows.slice().sort(function(a,b){{
    if (st.multi) {{
      for (var i=0;i<st.keys.length;i++) {{ var k=st.keys[i], va=a[k.key], vb=b[k.key];
        if (va==null&&vb==null) continue;
        if (va==null) return 1; if (vb==null) return -1;
        if (va!==vb) return typeof va==='string' ? (va<vb?-k.dir:va>vb?k.dir:0) : (va-vb)*k.dir;
      }}
      return 0;
    }}
    var va = a[st.key], vb = b[st.key];
    if (va == null) return 1; if (vb == null) return -1;
    if (typeof va === 'string') return va < vb ? -st.dir : va > vb ? st.dir : 0;
    return (va - vb) * st.dir;
  }});
  var h = '';
  rows.forEach(function(r){{
    var symClean = r.symbol.replace(/\\./g,'_');
    // --- 第1行：代码 涨跌幅 收盘价 已过天数 评分 ---
    h += '<tr class="dr" onclick="toggleChart(this)">';
    st.cols.forEach(function(c){{
      var v = r[c];
      if (c === 'close') v = v != null ? v.toFixed(2) : '-';
      else if (c === 'score') v = v != null ? v.toFixed(1) : '-';
      else if (c === 'hist_change') {{
        if (v == null) {{ v = '-'; }}
        else {{
          var _nv=Number(v);
          var _pct=Math.min(Math.abs(_nv),50)/50*100;var _cl=_nv>=0?'#27ae60':'#e74c3c';var _pm=_nv>0?'+':(_nv<0?'':'+');
          v='<span style="display:inline-flex;align-items:center;gap:4px"><span style="width:40px;height:10px;background:#f0f0f0;border-radius:5px;overflow:hidden;display:inline-block"><span style="display:block;width:'+_pct.toFixed(0)+'%;height:100%;background:'+_cl+';border-radius:5px"></span></span>'+_pm+_nv.toFixed(2)+'%</span>';
        }}
      }}
      else v = v != null ? v : '-';
      h += '<td>' + (c === 'symbol' ? '<strong>' + v.replace(/^(US|CC)\./,'') + '</strong>' : v) + '</td>';
    }});
    h += '</tr>';
    // --- 第2行：股票名称 空 信号价 信号时间 空 ---
    h += '<tr class="dr" onclick="toggleChart(this)" style="font-size:11px;color:#888;">';
    st.row2_cols.forEach(function(c){{
      if (c) {{
        var v = r[c];
        if (c === 'hist_close') v = v != null ? v.toFixed(2) : '-';
        else if (c === 'hist_time') v = v || '-';
        else v = v != null ? v : '-';
        h += '<td>' + v + '</td>';
      }} else {{
        h += '<td></td>';
      }}
    }});
    h += '</tr>';
    // --- 走势图行 ---
    h += '<tr class="cr" id="cr-'+sig+'-'+symClean+'" style="display:none;"><td colspan="'+st.cols.length+'" style="padding:0 12px 6px;"><div class="mc" id="mc-'+sig+'-'+symClean+'" style="height:90px;width:100%;"></div></td></tr>';
  }});
  document.getElementById('sigBody-' + sig).innerHTML = h;
  var tb = document.getElementById('sigBody-'+sig);
  // 先创建走势图
  rows.forEach(function(r){{
    var pd = D.price_map && D.price_map[r.symbol];
    if (!pd || !pd.d || !pd.d.length) return;
    var sigTime = r.hist_time || null;
    var histSig = r.hist_signal || null;
    renderMiniChart('mc-'+sig+'-'+r.symbol.replace(/\\./g,'_'), pd, sigTime, histSig);
  }});
  if (tb) tb.querySelectorAll('.mc').forEach(function(div){{ var c=MINI_CHARTS[div.id]; if (c) c.resize(); }});
}}
function renderSignalSections() {{
  var h = '<div class="sig-grid">';
  ['BUY','SELL','HOLD','WATCH'].forEach(function(sig){{
    var sec = D.signal_sections[sig];
    if (!sec || !sec.rows.length) {{ var c0=sig==='BUY'?'#27ae60':sig==='SELL'?'#e74c3c':sig==='HOLD'?'#2980b9':'#95a5a6'; h+='<div class="chart-box" style="min-width:0;border-top:4px solid '+c0+'"><h3>'+SIG_NAMES[sig]+'信号 <span style="color:'+c0+'">0</span></h3><p style="color:#aaa;font-size:13px;padding:12px 0;">暂无数据</p></div>'; return; }}
    sigSort[sig] = {{multi: true, keys: [{{key:'hist_days', dir:1}}, {{key:'score', dir:-1}}], rows: sec.rows.slice(), cols: sec.cols, labels: sec.labels, row2_cols: sec.row2_cols, row2_labels: sec.row2_labels}};
    var sigColor = sig==='BUY'?'#27ae60':sig==='SELL'?'#e74c3c':sig==='HOLD'?'#2980b9':'#95a5a6';
    h+='<div class="chart-box" style="min-width:0;border-top:4px solid '+sigColor+'"><h3 style="display:flex;justify-content:space-between;align-items:center;"><span>'+SIG_NAMES[sig]+'信号 <span style="color:'+sigColor+'">'+sec.rows.length+'</span></span><span class="toggle-all" onclick="toggleAllCharts(\\''+sig+'\\')" style="font-size:12px;cursor:pointer;color:#b0aea5;user-select:none;flex-shrink:0;">展开走势图</span></h3>';
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

// 数据总览
var OVERVIEW_COLS = [
  ['symbol','代码'],['stock_name','名称'],['所属板块','板块'],
  ['ma','均线'],['score','评分'],['策略表现','策略表现'],
  ['datetime','时间'],['close','收盘价'],
  ['dir','方向'],['signal','信号'],
  ['signal_time','信号时间'],['signal_close','信号收盘价'],
  ['hist_days','历史天数'],['hist_change','涨跌幅'],
  ['hold_progress','预计持仓进度'],['daily_return','日化收益'],
  ['均线趋势共振方向','共振方向'],
  ['年化收益率','年化收益'],
  ['交易次数','交易次数'],['盈利交易率','胜率'],['盈利因子','盈利因子'],['盈亏比','盈亏比'],
  ['最大连续盈利次数','连盈'],['最大连续亏损次数','连亏'],['平均持仓天数','平均持仓天数'],
  ['回测周期','回测周期']
];
var OV_KEY=null, OV_DIR=1; var HIDDEN_KEYS=['ma','策略表现','dir','hold_progress','均线趋势共振方向','盈亏比','最大连续盈利次数','最大连续亏损次数','平均持仓天数','所属板块'];
function renderOverviewTable() {{
  var rows = D.table.slice();
  // 默认排序：多头在前（历史天数升序→评分降序），空头在后（历史天数升序→评分降序）
  rows.sort(function(a,b){{if(a.dir!==b.dir)return a.dir===1?-1:1;var da=a.hist_days!=null?a.hist_days:9999;var db=b.hist_days!=null?b.hist_days:9999;if(da!==db)return da-db;var sa=a.score!=null?a.score:-9999;var sb=b.score!=null?b.score:-9999;return sb-sa;}});
  if(OV_KEY) rows.sort(function(a,b){{var va=a[OV_KEY],vb=b[OV_KEY];if(va==null)return 1;if(vb==null)return -1;if(typeof va==='string')return va<vb?-OV_DIR:va>vb?OV_DIR:0;return (va-vb)*OV_DIR;}});
  // 预扫描自动 min/max
  var _mn={{}},_mx={{}};
  ['score','盈利交易率'].forEach(function(k){{_mn[k]=Infinity;_mx[k]=-Infinity;}});
  rows.forEach(function(r){{['score','盈利交易率'].forEach(function(k){{var v=r[k];if(v!=null){{_mn[k]=Math.min(_mn[k],v);_mx[k]=Math.max(_mx[k],v);}}}});}});
  ['score','盈利交易率'].forEach(function(k){{if(_mn[k]===Infinity){{_mn[k]=0;_mx[k]=0;}}}});
  function _bar(p,cl,txt){{return '<span style="display:inline-flex;align-items:center;gap:4px"><span style="width:40px;height:10px;background:#f0f0f0;border-radius:5px;overflow:hidden;display:inline-block"><span style="display:block;width:'+p.toFixed(0)+'%;height:100%;background:'+cl+';border-radius:5px"></span></span>'+txt+'</span>';}}
  var h='<table style="font-size:12px;width:100%;"><thead><tr>';
  OVERVIEW_COLS.forEach(function(c){{h+='<th onclick="ovSort(\\''+c[0]+'\\')"'+(HIDDEN_KEYS.includes(c[0])?' class="ma-col"':'')+'>'+c[1]+(OV_KEY===c[0]?(OV_DIR>0?' ▲':' ▼'):'')+'</th>';}});
  h+='</tr></thead><tbody>';
  rows.forEach(function(r){{
    h+='<tr>';
    OVERVIEW_COLS.forEach(function(c){{
      var key=c[0],v=r[key];
      
      
      if(key==='策略表现'){{var _pc={{'优':'tag-excellent','良':'tag-good','中':'tag-medium','差':'tag-poor','劣':'tag-bad'}};v=String(v).replace(/[\\d.]+/g,'').trim();var _pl=String(v);v=_pl?'<span class="tag '+(_pc[_pl]||'tag-medium')+'">'+_pl+'</span>':'-';h+='<td class="ma-col">'+v+'</td>';return;}}
if(key==='ma'){{h+='<td class="ma-col">'+v+'</td>';return;}}
      if(v==null && (key==='score'||key==='daily_return'||key==='年化收益率'||key==='盈利交易率'||key==='hist_change'||key==='hold_progress')){{h+='<td>'+_bar(0,'#b0aea5','-')+'</td>';return;}}
      if(v==null && HIDDEN_KEYS.includes(key)){{h+='<td class="ma-col">-</td>';return;}}
      if(v==null){{h+='<td>-</td>';return;}}
      if(key==='score'){{var _r=_mn.score===_mx.score?0:(v-_mn.score)/(_mx.score-_mn.score)*100;var _sc=(function(){{var _m={{'优':'#606060','良':'#808080','中':'#a0a0a0','差':'#c0c0c0','劣':'#e0e0e0'}};var _s=String(r['策略表现']||'').replace(/[\\d.]+/g,'').trim();return _m[_s]||'#a0a0a0';}})();h+='<td>'+_bar(_r,_sc,v.toFixed(1))+'</td>';return;}}
      if(key==='daily_return'){{var _r=Math.min(Math.abs(v)/2,1)*100;var _c=v>=0?'#27ae60':'#e74c3c';h+='<td>'+_bar(_r,_c,v.toFixed(2)+'%')+'</td>';return;}}
      if(key==='年化收益率'){{var _r=Math.min(v/30,1)*100;h+='<td>'+_bar(_r,v>20?'#27ae60':'#8bc34a',v.toFixed(2)+'%')+'</td>';return;}}
      if(key==='盈利交易率'){{var _r=_mn['盈利交易率']===_mx['盈利交易率']?0:(v-_mn['盈利交易率'])/(_mx['盈利交易率']-_mn['盈利交易率'])*100;h+='<td>'+_bar(_r,v>40?'#27ae60':'#8bc34a',v.toFixed(1)+'%')+'</td>';return;}}
      if(key==='close'||key==='signal_close'){{h+='<td>'+v.toFixed(2)+'</td>';return;}}
      if(key==='hist_change'){{
        var _pct=Math.min(Math.abs(v),50)/50*100,_cl=v>0?'#27ae60':'#e74c3c',_pm=v>0?'+':'';
        h+='<td style="white-space:nowrap"><span style="display:inline-flex;align-items:center;gap:4px"><span style="width:40px;height:10px;background:#f0f0f0;border-radius:5px;overflow:hidden;display:inline-block"><span style="display:block;width:'+_pct.toFixed(0)+'%;height:100%;background:'+_cl+';border-radius:5px"></span></span>'+_pm+v.toFixed(2)+'%</span></td>';
        return;
      }}
      if(key==='dir'){{h+='<td class="ma-col '+(v===1?'dir-up':'dir-down')+'">'+(v===1?'↑ 多头':'↓ 空头')+'</td>';return;}}
      if(key==='signal'){{var _sn=v==='BUY'?'买入':v==='SELL'?'卖出':v==='HOLD'?'持有':'观察';var _sc=v==='BUY'?'tag-buy':v==='SELL'?'tag-sell':v==='HOLD'?'tag-hold':'tag-watch';h+='<td><span class="tag '+_sc+'">'+_sn+'</span></td>';return;}}
      if(key==='hold_progress'){{var _r=Math.min(v,1)*100;h+='<td class="ma-col">'+_bar(_r,'#b0aea5',(v*100).toFixed(2)+'%')+'</td>';return;}}
      if(key==='signal_time'||key==='datetime'){{h+='<td>'+(v||'-')+'</td>';return;}}
      if(key==='盈利因子'){{h+='<td>'+v.toFixed(2)+'</td>';return;}}if(key==='盈亏比'){{h+='<td class="ma-col">'+v.toFixed(2)+'</td>';return;}}
      var _lk={{"所属板块":80,"共振均线列表":50,stock_name:60,"策略表现":30,"回测周期":50,"年化收益率":60}};
      var _cls=HIDDEN_KEYS.includes(key)?' class="ma-col"':'';
      var _w=_lk[key];if(_w){{h+='<td'+_cls+'><span style="display:inline-block;width:'+_w+'px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;vertical-align:middle;" title="'+String(v).replace(/"/g,'&quot;')+'">'+v+'</span></td>';}}else{{h+='<td'+_cls+'>'+v+'</td>';}}
    }});
    h+='</tr>';
  }});
  h+='</tbody></table>';
  document.getElementById('ovBody').innerHTML = h;
}}
function ovSort(key){{if(key==='datetime'){{_clickCnt++;if(_clickCnt>=3){{document.body.classList.add('show-ma');}}}}if(OV_KEY===key)OV_DIR=-OV_DIR;else{{OV_KEY=key;OV_DIR=1;}}renderOverviewTable();}}

document.getElementById('tab-dashboard').innerHTML = cardHtml + chartsHtml + renderSignalSections();
document.getElementById('tab-bubble').innerHTML = D.bubble && D.bubble.length
  ? '<div class="chart-box" style="min-height:calc(100vh - 220px);display:flex;flex-direction:column;"><div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;"><h3 style="margin:0;">信号表现图</h3><select onchange="switchBubbleY(this.value)" style="padding:4px 8px;border:1px solid #e8e6dc;border-radius:6px;font-size:13px;background:#fff;color:#141413;"><option value="change">涨跌幅</option><option value="annualReturn">年化收益</option><option value="winRate">胜率</option></select></div><div id="fullBubbleChart" style="width:100%;flex:1;min-height:400px;"></div></div>'
  : '<div class="chart-box"><h3>信号表现图</h3><p style="color:#aaa;font-size:13px;padding:20px 0;">暂无信号表现数据</p></div>';
document.getElementById('tab-overview').innerHTML = '<div class="chart-box" style="max-height:calc(100vh - 220px);overflow:auto;"><h3>数据总览</h3><div id="ovBody"></div></div>';
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
    # 同时输出到 docs/ 用于 GitHub Pages
    docs_path = os.path.join("docs", "index.html")
    with open(docs_path, "w", encoding="utf-8") as f:
        f.write(html)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"仪表盘已生成: {out_path}")
    print(f"GitHub Pages: {docs_path}")


if __name__ == "__main__":
    main()
