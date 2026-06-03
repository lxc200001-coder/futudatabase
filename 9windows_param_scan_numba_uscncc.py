import os
import glob
import json
import warnings
from datetime import datetime
import concurrent.futures
import pandas as pd
import numpy as np
from plotly.io import to_json
from tqdm import tqdm
from openpyxl.styles import Alignment
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
import seaborn as sns
import plotly.graph_objects as go

# 屏蔽pandas concat空列FutureWarning（不影响功能）
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation", category=FutureWarning)


def _load_top_turnover_map():
    """读取最新 top_turnover 文件，返回 {代码: 排名} 映射表。"""
    files = sorted(glob.glob(os.path.join("symbols", "top_turnover_*.csv")))
    if not files:
        return {}
    try:
        df = pd.read_csv(files[-1])
        return dict(zip(df["代码"].dropna().astype(str), df["排名"].dropna().astype(int)))
    except Exception:
        return {}


def _apply_sheet_format(ws):
    """设置自动筛选、冻结首行+首列、表头加高4倍+文字自动换行"""
    if ws.max_row and ws.max_column:
        ws.auto_filter.ref = ws.dimensions
        ws.freeze_panes = "B2"
    # 首行表头加高到默认4倍，文字自动换行
    ws.row_dimensions[1].height = 60
    for cell in ws[1]:
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")

# =========================================================
# 配置
# =========================================================
DATA_DIR = "data_uscncc"
TRADE_DIR = "results_uscncc"
SYMBOL_FILE = "symbols/symbols.csv"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TRADE_DIR, exist_ok=True)

INITIAL_CASH = 10000
FEE_RATE = 0.001

DEFAULT_KTYPE = "week,day"     # 默认K线周期: week(周K) / day(日K) / 60m(60分钟K) / all(三者全部) / week,day(逗号拼接)
DEFAULT_MARKET = "US,CC"    # 默认市场: all / US / CN / CC / US,CC
MA_MODE = "continuous"    # 默认MA序列类型: continuous=连续回测 / jump=跳跃回测
_HEATMAP_CACHE = {}  # 热力图看板数据缓存: {BAR_INTERVAL: {market: [(code, rows, ws, best_ma, name), ...]}}

# 中文映射
DIR_MAP = {1: "多头", -1: "空头"}
SIG_MAP = {"BUY": "买入", "SELL": "卖出", "HOLD": "持有", "WATCH": "观察"}


def _market_subdir(code):
    """根据股票代码返回市场子目录名：us/cn/cc"""
    if code.startswith("CC."):
        return "cc"
    elif code.startswith(("SH.", "SZ.")):
        return "cn"
    elif code.startswith("US."):
        return "us"
    raise ValueError(f"未知代码前缀: {code}")


def _find_data_file(code):
    """在 data_uscncc/{ktype_dir}/{market}/ 中查找 parquet 文件，兼容旧目录。"""
    if code.startswith("CC."):
        market = "cc"
    elif code.startswith(("SH.", "SZ.")):
        market = "cn"
    elif code.startswith("US."):
        market = "us"
    else:
        raise ValueError(f"未知代码前缀: {code}")

    _dir_map = {"1W": "1w", "1D": "1d", "60m": "60m"}
    _ktype_dir = _dir_map.get(BAR_INTERVAL, "")
    path = os.path.join(DATA_DIR, _ktype_dir, market, f"{code}{FILE_SUFFIX}.parquet")
    return path if os.path.exists(path) else None

def apply_cn_mapping(df):
    """统一应用趋势方向和信号的中文映射"""
    if "趋势方向" in df.columns:
        df["趋势方向"] = df["趋势方向"].map(DIR_MAP)
    if "信号" in df.columns:
        df["信号"] = df["信号"].map(SIG_MAP)
    if "最新信号" in df.columns:
        df["最新信号"] = df["最新信号"].map(SIG_MAP)


KLINE_MAP = {
    "1D": {"display": "日K", "suffix": "_1d", "period": 252, "ma_range": list(range(2, 181))},
    "1W": {"display": "周K", "suffix": "_1w", "period": 52, "ma_range": list(range(2, 61))},
    "60m": {"display": "60分钟K", "suffix": "_60m", "period": 1638, "ma_range": list(range(2, 359))},
}

def generate_ma_list(bar_interval, ma_mode="continuous"):
    """根据K线周期和MA模式生成MA列表。

    周线: 强制 continuous (step=1)
    日线 continuous: 2..180 (step=1) / jump: 2,4,6..180 (step=2)
    60m  continuous: 2..358 (step=1) / jump: 2,6,10..358 (step=4)
    """
    if bar_interval == "1W" or ma_mode == "continuous":
        return KLINE_MAP[bar_interval]["ma_range"]
    _steps = {"1D": 2, "60m": 4}
    step = _steps[bar_interval]
    _max = {"1D": 181, "60m": 359}
    return list(range(2, _max[bar_interval], step))

BAR_INTERVAL = "1D" if DEFAULT_KTYPE == "day" else "1W"
MA_LIST = generate_ma_list(BAR_INTERVAL, MA_MODE)
FILE_SUFFIX = KLINE_MAP[BAR_INTERVAL]["suffix"]
TRADING_PERIOD = KLINE_MAP[BAR_INTERVAL]["period"]
KTYPE_DIR_MAP = {"1W": "1w", "1D": "1d", "60m": "60m"}
TRADE_SUBDIR = KTYPE_DIR_MAP.get(BAR_INTERVAL, "")
for _m in ("us", "cn", "cc"):
    os.makedirs(os.path.join(TRADE_DIR, TRADE_SUBDIR, _m), exist_ok=True)
    os.makedirs(os.path.join(TRADE_DIR, TRADE_SUBDIR, _m, "heatmaps"), exist_ok=True)
os.makedirs(os.path.join(TRADE_DIR, TRADE_SUBDIR, "heatmaps"), exist_ok=True)
STEP_MONTHS = {"1W": 12, "1D": 6, "60m": 3}.get(BAR_INTERVAL, 12)
WINDOW_START_DATE = "2000-01-03"

# ---- 命令行参数解析（前置，仅在作为主程序运行时生效）----
def _setup_ktype(ktype, ma_mode="continuous"):
    """设置回测周期的全局变量（供 worker 进程调用）。"""
    global BAR_INTERVAL, MA_LIST, FILE_SUFFIX, TRADING_PERIOD, TRADE_SUBDIR, STEP_MONTHS, MA_MODE
    MA_MODE = ma_mode
    if ktype == "60m":
        BAR_INTERVAL = "60m"
    elif ktype == "day":
        BAR_INTERVAL = "1D"
    else:
        BAR_INTERVAL = "1W"
    MA_LIST = generate_ma_list(BAR_INTERVAL, MA_MODE)
    FILE_SUFFIX = KLINE_MAP[BAR_INTERVAL]["suffix"]
    TRADING_PERIOD = KLINE_MAP[BAR_INTERVAL]["period"]
    TRADE_SUBDIR = KTYPE_DIR_MAP.get(BAR_INTERVAL, "")
    STEP_MONTHS = {"1W": 12, "1D": 6, "60m": 3}.get(BAR_INTERVAL, 12)
    for _m in ("us", "cn", "cc"):
        os.makedirs(os.path.join(TRADE_DIR, TRADE_SUBDIR, _m), exist_ok=True)
        os.makedirs(os.path.join(TRADE_DIR, TRADE_SUBDIR, _m, "heatmaps"), exist_ok=True)
    os.makedirs(os.path.join(TRADE_DIR, TRADE_SUBDIR, "heatmaps"), exist_ok=True)


# 百分比字段（原始值=百分比数值，如 5.23 表示 5.23%；
# 输出时 ÷100 再设 Excel 单元格格式为 0.00%，实现 Excel 原生百分比显示）
PCT_COLS = [
    "预计持仓进度", "预计涨幅进度",
    "距离历史信号收盘价涨跌幅", "持仓日化收益率",
    "平均每笔收益率", "平均盈利比", "平均亏损比",
    "收益率", "年化收益率", "买入持有收益率", "超额收益率",
    "最大回撤", "盈利交易率",
]

# =========================================================
# 股票列表
# =========================================================
def init_symbols_file(path):
    if not os.path.exists(path):
        pd.DataFrame({"code": ["US.TSLA", "US.AAPL", "US.NVDA", "US.MSFT"]}).to_csv(path, index=False)

def load_symbols(path):
    return pd.read_csv(path)["code"].dropna().tolist()

# =========================================================
# HA计算
# =========================================================
def calc_heikin_ashi(df):
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = np.zeros(len(df))
    ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2

    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2

    return ha_close, ha_open


@njit
def _numba_backtest(close, buy, sell, initial_cash, fee_rate):
    """Numba 加速核心回测：单次遍历计算交易记录和资金曲线。

    返回 (trades_arr, equity_arr, n_trades)。

    trades_arr 列:
    [entry_price, position(股数), entry_index, exit_price, exit_index,
     buy_fee, sell_fee, pnl,
     cash_before_buy, cash_after_buy,
     cash_before_sell, cash_after_sell,
     is_force_close]
    """
    n = len(close)
    max_trades = n
    trades = np.zeros((max_trades, 13))

    available_cash = initial_cash
    realized_cash = initial_cash  # 累计已平仓盈亏（用于资金曲线）
    position = 0.0
    entry_price_val = 0.0
    entry_idx = 0
    trade_count = 0
    equity = np.zeros(n)

    for i in range(n):
        p = close[i]

        if np.isnan(p) or p <= 0:
            equity[i] = equity[i - 1] if i > 0 else initial_cash
            continue

        # ---- 开仓 ----
        if buy[i] and position == 0:
            shares = int(available_cash / (p * (1 + fee_rate)))
            if shares > 0:
                cost = shares * p
                fee = cost * fee_rate
                cash_before_buy = available_cash
                available_cash -= (cost + fee)
                position = shares
                entry_price_val = p
                entry_idx = i
                trades[trade_count, 8] = cash_before_buy
                trades[trade_count, 9] = available_cash

        # ---- 平仓 ----
        elif sell[i] and position > 0:
            sell_value = position * p
            sell_fee = sell_value * fee_rate
            cash_before_sell = available_cash
            available_cash += (sell_value - sell_fee)

            buy_fee = entry_price_val * position * fee_rate
            pnl = (p - entry_price_val) * position - buy_fee - sell_fee

            trades[trade_count, 0] = entry_price_val
            trades[trade_count, 1] = position
            trades[trade_count, 2] = entry_idx
            trades[trade_count, 3] = p
            trades[trade_count, 4] = i
            trades[trade_count, 5] = buy_fee
            trades[trade_count, 6] = sell_fee
            trades[trade_count, 7] = pnl
            trades[trade_count, 10] = cash_before_sell
            trades[trade_count, 11] = available_cash
            trades[trade_count, 12] = 0  # 正常平仓
            trade_count += 1

            realized_cash += pnl
            position = 0.0

        # ---- 资金曲线 ----
        if position > 0:
            floating = (p - entry_price_val) * position
            equity[i] = max(realized_cash + floating, 1e-6)
        else:
            equity[i] = max(realized_cash, 1e-6)

    # 末尾强平
    if position > 0:
        p = close[-1]
        sell_value = position * p
        sell_fee = sell_value * fee_rate
        buy_fee = entry_price_val * position * fee_rate
        pnl = (p - entry_price_val) * position - buy_fee - sell_fee
        cash_before_sell = available_cash
        available_cash += (sell_value - sell_fee)

        trades[trade_count, 0] = entry_price_val
        trades[trade_count, 1] = position
        trades[trade_count, 2] = entry_idx
        trades[trade_count, 3] = p
        trades[trade_count, 4] = n - 1
        trades[trade_count, 5] = buy_fee
        trades[trade_count, 6] = sell_fee
        trades[trade_count, 7] = pnl
        # cash_before/after_buy 已在开仓时写入 idx 8/9，不覆盖
        trades[trade_count, 10] = cash_before_sell
        trades[trade_count, 11] = available_cash
        trades[trade_count, 12] = 1  # 强平
        trade_count += 1
        realized_cash += pnl
        equity[-1] = max(realized_cash, 1e-6)

    return trades[:trade_count], equity, trade_count


def _build_trades_from_arrays(code_val, market_val, datetime_arr, ma_len,
                               trades_arr, equity_arr, n_trades):
    """从 Numba 返回的裸数组构建交易 DataFrame（无 pandas 中间层）。"""
    if n_trades == 0:
        return pd.DataFrame(), equity_arr

    backtest_period = f"{pd.Timestamp(datetime_arr[0]).date()} ~ {pd.Timestamp(datetime_arr[-1]).date()}"

    records = []
    for j in range(n_trades):
        t = trades_arr[j]
        entry_idx = int(t[2])
        exit_idx = int(t[4])

        entry_time = pd.Timestamp(datetime_arr[entry_idx])
        exit_time = pd.Timestamp(datetime_arr[exit_idx])

        entry_price_val = t[0]
        position_val = int(t[1])
        buy_fee = t[5]
        sell_fee = t[6]
        pnl = t[7]
        total_fee = buy_fee + sell_fee
        cost_basis = entry_price_val * position_val + buy_fee
        return_pct = pnl / cost_basis * 100 if cost_basis > 0 else 0
        hold_kbars = exit_idx - entry_idx
        hold_days = (exit_time - entry_time).days
        is_force = t[12] == 1

        cash_before_buy = t[8] if t[8] != 0 else None
        cash_after_buy = t[9] if t[9] != 0 else None
        cash_before_sell = t[10]
        cash_after_sell = t[11]

        records.append({
            "股票代码": code_val,
            "市场": market_val,
            "K线周期": BAR_INTERVAL,
            "均线周期": ma_len,

            "开仓时间": entry_time,
            "开仓价格": float(entry_price_val),
            "买入股数": position_val,

            "平仓时间": exit_time,
            "平仓价格": float(t[3]),
            "卖出股数": position_val,

            "交易状态": "未平仓(强制结算)" if is_force else "已平仓",
            "订单盈亏类型": "盈利" if pnl > 0 else "亏损",

            "收益金额": pnl,
            "收益率(%)": float(return_pct),

            "买入手续费": buy_fee,
            "卖出手续费": sell_fee,
            "总手续费": total_fee,

            "开仓前可用现金": cash_before_buy,
            "开仓后可用现金": cash_after_buy,
            "平仓前可用现金": cash_before_sell,
            "平仓后可用现金": cash_after_sell,

            "持仓K线数": hold_kbars,
            "持仓天数": hold_days,

            "回测周期": backtest_period
        })

    trades_df = pd.DataFrame(records)
    return trades_df, equity_arr


# =========================================================
# 最近信号
# =========================================================
def get_last_signal_info(df):

    today = pd.Timestamp.today().normalize()

    if df.iloc[-1]["buy"]:
        signal = "BUY"
    elif df.iloc[-1]["sell"]:
        signal = "SELL"
    elif df.iloc[-1]["dir"] == 1:
        signal = "HOLD"
    else:
        signal = "WATCH"

    buy_rows = df[df["buy"]]

    if len(buy_rows) > 0:
        last_buy = buy_rows.iloc[-1]
        buy_time = pd.to_datetime(last_buy["datetime"])
        buy_days = (today - buy_time.normalize()).days
    else:
        buy_time = pd.NaT
        buy_days = None

    sell_rows = df[df["sell"]]

    if len(sell_rows) > 0:
        last_sell = sell_rows.iloc[-1]
        sell_time = pd.to_datetime(last_sell["datetime"])
        sell_days = (today - sell_time.normalize()).days
    else:
        sell_time = pd.NaT
        sell_days = None

    last = df.iloc[-1]
    last_close = float(last["close"])


    # 取最后一次信号（买入或卖出，取较晚的那次）
    hist_signal = None
    hist_time = pd.NaT
    hist_close = None
    hist_days = None
    if len(buy_rows) > 0 and len(sell_rows) > 0:
        lb = pd.to_datetime(buy_rows.iloc[-1]["datetime"])
        ls = pd.to_datetime(sell_rows.iloc[-1]["datetime"])
        if lb >= ls:
            hist_signal = "买入"
            hist_time = lb
            hist_close = float(buy_rows.iloc[-1]["close"])
        else:
            hist_signal = "卖出"
            hist_time = ls
            hist_close = float(sell_rows.iloc[-1]["close"])
    elif len(buy_rows) > 0:
        hist_signal = "买入"
        hist_time = pd.to_datetime(buy_rows.iloc[-1]["datetime"])
        hist_close = float(buy_rows.iloc[-1]["close"])
    elif len(sell_rows) > 0:
        hist_signal = "卖出"
        hist_time = pd.to_datetime(sell_rows.iloc[-1]["datetime"])
        hist_close = float(sell_rows.iloc[-1]["close"])

    # 若选出的历史信号未确认（<5天），回退到上一次已确认的信号
    if hist_signal is not None:
        _hd = (today - hist_time.normalize()).days
        if _hd < 5:
            _cutoff = today - pd.Timedelta(days=5)
            _cb = buy_rows[pd.to_datetime(buy_rows["datetime"]).dt.normalize() <= _cutoff]
            _cs = sell_rows[pd.to_datetime(sell_rows["datetime"]).dt.normalize() <= _cutoff]
            if len(_cs) > 0 and (len(_cb) == 0 or pd.to_datetime(_cs.iloc[-1]["datetime"]) >= pd.to_datetime(_cb.iloc[-1]["datetime"])):
                hist_signal = "卖出"
                hist_time = pd.to_datetime(_cs.iloc[-1]["datetime"])
                hist_close = float(_cs.iloc[-1]["close"])
            elif len(_cb) > 0:
                hist_signal = "买入"
                hist_time = pd.to_datetime(_cb.iloc[-1]["datetime"])
                hist_close = float(_cb.iloc[-1]["close"])
            else:
                hist_signal = None
                hist_time = pd.NaT
                hist_close = None

    if hist_signal is not None:
        hist_days = (today - hist_time.normalize()).days
        hist_change = (last_close - hist_close) / hist_close * 100 if hist_close else None
        # 持仓日化收益率：空头趋势反转符号（做空视角下跌为盈、上涨为亏）
        _dir = int(last["dir"])
        if hist_change is not None and _dir == -1:
            hist_daily = (-hist_change) / hist_days if hist_days and hist_days > 0 else None
        else:
            hist_daily = hist_change / hist_days if hist_days and hist_days > 0 else None
    else:
        hist_change = None
        hist_daily = None

    # 最新信号确认（仅周线需要判断，日线/60分钟默认已确认）
    if BAR_INTERVAL == "1W":
        if signal in ("BUY", "SELL") and (
            (signal == "BUY" and buy_days is not None and buy_days < 5) or
            (signal == "SELL" and sell_days is not None and sell_days < 5)
        ):
            confirm = "待确认，周K未正式收盘"
        else:
            confirm = "已确认"
    else:
        confirm = "已确认"

    _date = lambda v: pd.Timestamp(v).date()
    _opt_date = lambda v: _date(v) if pd.notna(v) else None

    return {
        "时间": _opt_date(last["datetime"]),
        "收盘价": last_close,
        "HA收盘价": float(last["ha_close"]),
        "HA均线值": float(last["ma"]) if not pd.isna(last["ma"]) else None,
        "趋势方向": int(last["dir"]),
        "最新信号": signal,
        "最新信号时间": _opt_date(last["datetime"]),
        "最新信号收盘价": last_close,
        "最新信号确认": confirm,

        "历史信号": hist_signal,
        "历史信号时间": _opt_date(hist_time),
        "历史信号收盘价": hist_close,
        "距离历史信号已过天数": hist_days,
        "距离历史信号收盘价涨跌幅": hist_change,
        "持仓日化收益率": hist_daily,
    }

# =========================================================
# 交易回测
# =========================================================
def _make_trade_record(code_val, ma_len, entry_price, position, entry_time, entry_index,
                       exit_price, exit_time, exit_index,
                       cash_before_open, cash_after_open,
                       cash_before_close, cash_after_close,
                       status, backtest_period,
                       market_val=""):
    buy_fee = entry_price * position * FEE_RATE
    sell_value = exit_price * position
    sell_fee = sell_value * FEE_RATE
    total_fee = buy_fee + sell_fee
    pnl = (exit_price - entry_price) * position - total_fee
    cost_basis = entry_price * position + buy_fee
    return_pct = pnl / cost_basis * 100 if cost_basis > 0 else 0
    hold_kbars = exit_index - entry_index
    hold_days = int((exit_time - entry_time) / np.timedelta64(1, 'D'))

    return {
        "股票代码": code_val,
        "市场": market_val,
        "K线周期": BAR_INTERVAL,
        "均线周期": ma_len,

        "开仓时间": entry_time,
        "开仓价格": float(entry_price),
        "买入股数": position,

        "平仓时间": exit_time,
        "平仓价格": float(exit_price),
        "卖出股数": position,

        "交易状态": status,
        "订单盈亏类型": "盈利" if pnl > 0 else "亏损",

        "收益金额": pnl,
        "收益率(%)": float(return_pct),

        "买入手续费": buy_fee,
        "卖出手续费": sell_fee,
        "总手续费": total_fee,

        "开仓前可用现金": cash_before_open if cash_before_open is not None else None,
        "开仓后可用现金": cash_after_open if cash_after_open is not None else None,
        "平仓前可用现金": cash_before_close if cash_before_close is not None else None,
        "平仓后可用现金": cash_after_close if cash_after_close is not None else None,

        "持仓K线数": hold_kbars,
        "持仓天数": hold_days,

        "回测周期": backtest_period
    }

def equity_curve(df, trades_df):

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    if trades_df is None or trades_df.empty:
        return np.full(len(df), INITIAL_CASH)

    trades_df = trades_df.copy()
    trades_df["开仓时间"] = pd.to_datetime(trades_df["开仓时间"])
    trades_df["平仓时间"] = pd.to_datetime(trades_df["平仓时间"])

    equity = np.zeros(len(df))

    cash = INITIAL_CASH
    trade_idx = 0

    trades_df = trades_df.sort_values("平仓时间").reset_index(drop=True)

    # numpy arrays for faster row access
    datetime_arr = df["datetime"].values
    close_arr = df["close"].values
    open_arr = trades_df["开仓时间"].values
    close_t_arr = trades_df["平仓时间"].values
    pnl_arr = trades_df["收益金额"].values
    entry_arr = trades_df["开仓价格"].values
    shares_arr = trades_df["买入股数"].values
    n_trades = len(trades_df)

    for i in range(len(df)):

        t = datetime_arr[i]
        price = close_arr[i]

        while trade_idx < n_trades and close_t_arr[trade_idx] <= t:
            cash += pnl_arr[trade_idx]
            trade_idx += 1

        floating = 0

        if trade_idx < n_trades:
            if open_arr[trade_idx] <= t < close_t_arr[trade_idx]:
                floating = (price - entry_arr[trade_idx]) * shares_arr[trade_idx]

        equity[i] = max(cash + floating, 1e-6)

    return equity

# =========================================================
# 风险指标
# =========================================================
def max_drawdown(curve):
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    return abs(dd.min())

def sharpe_from_equity(curve):

    curve = np.array(curve)
    curve = np.where(curve <= 0, INITIAL_CASH, curve)

    log_ret = np.diff(np.log(curve))

    if len(log_ret) < 3:
        return 0

    std = np.std(log_ret)

    if std == 0:
        return 0

    return np.mean(log_ret) / (std + 1e-9) * np.sqrt(TRADING_PERIOD)

# =========================================================
# 连续统计
# =========================================================
def streaks(trades_df):

    if trades_df.empty:
        return 0, 0

    pnl = trades_df["收益金额"].values

    win = loss = 0
    max_win = max_loss = 0

    for x in pnl:

        if x > 0:
            win += 1
            loss = 0
        else:
            loss += 1
            win = 0

        max_win = max(max_win, win)
        max_loss = max(max_loss, loss)

    return max_win, max_loss

# =========================================================
# 策略评分
# =========================================================
def normalize(x, min_v, max_v):

    if pd.isna(x):
        return 0

    if max_v == min_v:
        return 0

    x = max(min_v, min(x, max_v))

    return (x - min_v) / (max_v - min_v)

def calc_score_row(row):

    cagr_score = normalize(row.get("年化收益率", 0), 0, 30) * 100
    sharpe_score = normalize(row.get("夏普比率", 0), 0, 2) * 100
    dd_score = (1 - normalize(row.get("最大回撤", 0), 0, 50)) * 100
    pf_score = normalize(row.get("盈利因子", 0), 1, 3) * 100
    win_score = normalize(row.get("盈利交易率", 0), 30, 80) * 100
    trade_score = normalize(min(row.get("交易次数", 0), 100), 10, 100) * 100

    score = (
        cagr_score * 0.30 +
        sharpe_score * 0.25 +
        dd_score * 0.20 +
        pf_score * 0.15 +
        win_score * 0.05 +
        trade_score * 0.05
    )

    return score

# =========================================================
# 窗口生成
# =========================================================
def generate_windows(df=None, end_date=None):
    """生成累积扩展窗口：起点固定，终点按月步长递增。
       所有股票使用相同的 end_date 以保证窗口一致。
       传 df 时（单个股票）取该股票数据的最后一天作为终点。"""
    if end_date is None and df is not None:
        dates = pd.to_datetime(df["datetime"])
        end_date = dates.max()
    elif end_date is None:
        return []

    start = pd.Timestamp(WINDOW_START_DATE)
    end = pd.Timestamp(end_date)

    windows = []
    cur = start + pd.DateOffset(months=STEP_MONTHS)

    while cur < end:
        windows.append((start, cur))
        cur += pd.DateOffset(months=STEP_MONTHS)

    # 追加一个完整步长窗口，替代非整年兜底
    windows.append((start, cur))

    return windows

# =========================================================
# 汇总
# =========================================================
def build_summary(trades_df, ma_len, df, equity_arr=None):

    start = pd.to_datetime(df["datetime"].iloc[0])
    end = pd.to_datetime(df["datetime"].iloc[-1])

    market_val = str(df.get("market", pd.Series([""])).iloc[0]) if "market" in df.columns else ""

    if equity_arr is not None:
        curve = equity_arr
    else:
        curve = equity_curve(df, trades_df)

    final = curve[-1]

    if trades_df.empty:

        return {
            "股票代码": df["code"].iloc[0],
            "市场": market_val,
            "K线周期": BAR_INTERVAL,
            "均线周期": ma_len,

            "收益率": 0,
            "年化收益率": 0,
            "买入持有收益率": 0,
            "超额收益率": 0,
            "平均每笔收益率": 0,

            "最大回撤": 0,
            "夏普比率": 0,
            "卡尔玛比率": 0,

            "交易次数": 0,
            "盈利交易率": 0,
            "盈利因子": 0,
            "盈亏比": 0,

            "平均盈利": 0,
            "平均盈利比": 0,
            "平均亏损": 0,
            "平均亏损比": 0,
            "最大单笔盈利": 0,
            "最大单笔亏损": 0,

            "最大连续盈利次数": 0,
            "最大连续亏损次数": 0,

            "平均持仓天数": 0,

            "初始资金": INITIAL_CASH,
            "最终资金": INITIAL_CASH,

        }

    ret = (final / INITIAL_CASH - 1) * 100
    years = max((end - start).days / 365, 1 / 365)
    cagr = ((final / INITIAL_CASH) ** (1 / years) - 1) * 100

    buy_hold = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    alpha = ret - buy_hold

    mdd = max_drawdown(curve) * 100
    sh = sharpe_from_equity(curve)
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    win = trades_df[trades_df["收益金额"] > 0]
    loss = trades_df[trades_df["收益金额"] < 0]

    win_rate = len(win) / len(trades_df) * 100
    pf = win["收益金额"].sum() / abs(loss["收益金额"].sum()) if len(loss) else 0

    avg_win = win["收益金额"].mean() if len(win) else 0
    avg_loss = abs(loss["收益金额"].mean()) if len(loss) else 0

    avg_win_pct = win["收益率(%)"].mean() if len(win) else 0
    avg_loss_pct = loss["收益率(%)"].mean() if len(loss) else 0

    payoff = avg_win / avg_loss if avg_loss else 0

    max_win_trade = trades_df["收益金额"].max()
    max_loss_trade = trades_df["收益金额"].min()

    max_win_streak, max_loss_streak = streaks(trades_df)

    avg_hold = trades_df["持仓天数"].mean()

    return {
        "股票代码": df["code"].iloc[0],
        "市场": market_val,
        "K线周期": BAR_INTERVAL,
        "均线周期": ma_len,

        "收益率": float(ret),
        "年化收益率": float(cagr),
        "买入持有收益率": float(buy_hold),
        "超额收益率": float(alpha),
        "平均每笔收益率": float(trades_df["收益率(%)"].mean()) if not trades_df.empty else 0,

        "最大回撤": float(mdd),
        "夏普比率": float(sh),
        "卡尔玛比率": float(calmar),

        "交易次数": int(len(trades_df)),
        "盈利交易率": float(win_rate),
        "盈利因子": float(pf),
        "盈亏比": float(payoff),

        "平均盈利": float(avg_win),
        "平均盈利比": float(avg_win_pct),
        "平均亏损": float(avg_loss),
        "平均亏损比": float(avg_loss_pct),
        "最大单笔盈利": float(max_win_trade),
        "最大单笔亏损": float(max_loss_trade),

        "最大连续盈利次数": max_win_streak,
        "最大连续亏损次数": max_loss_streak,

        "平均持仓天数": float(avg_hold),

        "初始资金": INITIAL_CASH,
        "最终资金": float(final),

    }

# =========================================================
# 列排序
# =========================================================
COLUMN_ORDER = [
    "股票代码", "市场", "K线周期", "均线周期", "策略评分"
]

END_COLUMNS = ["窗口", "窗口内有效数据周期"]

def reorder_columns(df):
    cols = df.columns.tolist()
    ordered = [c for c in COLUMN_ORDER if c in cols]
    rest = [c for c in cols if c not in COLUMN_ORDER and c not in END_COLUMNS]
    end = [c for c in END_COLUMNS if c in cols]
    return df[ordered + rest + end]

# =========================================================
# 参数稳定性分析
# =========================================================
def build_window_stability(summary_rows):
    """对每个累积窗口阶段计算参数稳定性（同 calc_param_stability 逻辑，按阶段展开）。

    输入：某只股票所有窗口的汇总行。
    输出：每行 = (均线周期, 窗口) 的稳定性指标，
          排序 = 股票代码 ↑ | 窗口 ↑ | 均线周期 ↑
    """
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return pd.DataFrame()

    # 排除评分全为0的窗口
    valid = df.groupby("窗口")["策略评分"].transform("max") > 0
    df = df[valid]
    if df.empty:
        return pd.DataFrame()

    windows = sorted(df["窗口"].unique())
    if not windows:
        return pd.DataFrame()

    # 每窗口内按策略评分排名
    df["窗口内排名"] = df.groupby(["股票代码", "窗口"])["策略评分"].rank(ascending=False, method="min")

    code_val = df["股票代码"].iloc[0]
    bar_val = df["K线周期"].iloc[0]

    def _norm(series, higher_is_better=True):
        lo, hi = series.min(), series.max()
        if hi == lo:
            return pd.Series(0.5, index=series.index)
        return (series - lo) / (hi - lo) if higher_is_better else (hi - series) / (hi - lo)

    all_stages = []
    for i, w in enumerate(windows):
        stage_df = df[df["窗口"].isin(windows[:i + 1])]

        stats = stage_df.groupby("均线周期").agg(
            窗口数量=("窗口", "nunique"),
            盈利窗口数量=("年化收益率", lambda x: (x > 0).sum()),
            策略评分排名平均值=("窗口内排名", "mean"),
            策略评分排名第一次数=("窗口内排名", lambda x: (x == 1).sum()),
            策略评分排名Top3占比=("窗口内排名", lambda x: (x <= 3).sum() / max(len(x), 1) * 100),
            策略评分排名标准差=("窗口内排名", "std"),
            年化收益率平均值=("年化收益率", "mean"),
            年化收益率标准差=("年化收益率", "std"),
        ).reset_index()

        stats["盈利窗口占比"] = stats["盈利窗口数量"] / stats["窗口数量"] * 100
        stats["年化收益率标准差"] = stats["年化收益率标准差"].fillna(0)
        stats["策略评分排名标准差"] = stats["策略评分排名标准差"].fillna(0)

        # 归一化加权评分
        avg_rank_n = _norm(stats["策略评分排名平均值"], higher_is_better=False)
        top3_n = _norm(stats["策略评分排名Top3占比"], higher_is_better=True)
        std_n = _norm(stats["策略评分排名标准差"], higher_is_better=False)
        cagr_n = _norm(stats["年化收益率平均值"], higher_is_better=True)
        win_rate_n = _norm(stats["盈利窗口占比"], higher_is_better=True)
        cagr_std_n = _norm(stats["年化收益率标准差"], higher_is_better=False)

        stats["参数稳定性评分"] = (
            0.20 * avg_rank_n + 0.10 * top3_n + 0.15 * std_n
            + 0.25 * cagr_n + 0.20 * win_rate_n + 0.10 * cagr_std_n
        )

        stats.insert(0, "窗口", w)
        stats.insert(0, "K线周期", bar_val)
        stats.insert(0, "股票代码", code_val)
        all_stages.append(stats)

    result = pd.concat(all_stages, ignore_index=True)
    col_order = [
        "股票代码", "K线周期", "均线周期", "窗口", "窗口数量",
        "盈利窗口数量", "盈利窗口占比",
        "年化收益率平均值", "年化收益率标准差",
        "策略评分排名Top3占比", "策略评分排名平均值",
        "策略评分排名标准差", "策略评分排名第一次数",
        "参数稳定性评分", "是否最优",
    ]
    result = result.reindex(columns=col_order)
    result = result.sort_values(["股票代码", "窗口", "均线周期"]).reset_index(drop=True)

    # 标记每股票每窗口的最优参数（不改变已有排序）
    result["是否最优"] = ""
    idx = (
        result
        .sort_values(["参数稳定性评分", "策略评分排名标准差", "年化收益率平均值"],
                      ascending=[False, True, False])
        .groupby(["股票代码", "窗口"], sort=False)
        .head(1)
        .index
    )
    result.loc[idx, "是否最优"] = "最优"
    return result


# =========================================================
# 评分矩阵（明细+排名）
# =========================================================
def build_score_matrix(summary_rows):
    """从窗口回测汇总行构建评分明细和排名透视表"""
    df = pd.DataFrame(summary_rows)

    # 排除评分全为0的窗口
    valid = df.groupby("窗口")["策略评分"].transform("max") > 0
    df = df[valid]

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 评分明细透视
    score_pivot = df.pivot_table(
        index=["股票代码", "K线周期", "均线周期"],
        columns="窗口",
        values="策略评分",
        aggfunc="first"
    )
    score_pivot = score_pivot[sorted(score_pivot.columns)]
    score_pivot = score_pivot.reset_index()

    # 评分排名（每股票每窗口内独立排名）
    df["窗口内排名"] = df.groupby(["股票代码", "窗口"])["策略评分"].rank(ascending=False, method="min")

    rank_pivot = df.pivot_table(
        index=["股票代码", "K线周期", "均线周期"],
        columns="窗口",
        values="窗口内排名",
        aggfunc="first"
    )
    rank_pivot = rank_pivot[sorted(rank_pivot.columns)]
    rank_pivot = rank_pivot.reset_index()

    return score_pivot, rank_pivot


# =========================================================
# 参数扫描热力图
# =========================================================
def _build_metric_heatmap_figure(code, scan_rows, metric_name, metric_title,
                                  colorscale, center, best_ma=None, stock_name=""):
    """构建单指标参数扫描热力图，返回 go.Figure 或 None。"""
    if not scan_rows:
        return None
    df = pd.DataFrame(scan_rows)
    if df.empty or df["窗口"].nunique() < 2:
        return None
    if metric_name not in df.columns:
        return None

    pivot = df.pivot_table(index="均线周期", columns="窗口", values=metric_name, aggfunc="first")
    pivot = pivot[sorted(pivot.columns, key=lambda c: str(c))]
    if pivot.empty:
        return None

    annot_text = [[f"{v:.1f}" if pd.notna(v) else "" for v in row] for row in pivot.values]
    # 每列第1名★标记
    for col_idx, col_name in enumerate(pivot.columns):
        col_data = pivot[col_name].dropna()
        if col_data.empty:
            continue
        ranked = col_data.sort_values() if metric_name == "最大回撤" else col_data.sort_values(ascending=False)
        for label, _ in ranked.head(1).items():
            row_idx = list(pivot.index).index(label)
            _raw = annot_text[row_idx][col_idx]
            if _raw:
                annot_text[row_idx][col_idx] = f"{_raw}"

    y_labels = [str(ma) for ma in pivot.index]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=pivot.values, x=[str(c) for c in pivot.columns], y=y_labels,
        text=annot_text, texttemplate="%{text}", textfont=dict(size=10),
        colorscale=colorscale, zmid=0 if center else None,
        hovertemplate="窗口: %{x}<br>均线: %{y}<br>值: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{code} {stock_name} {metric_title} 参数扫描热力图", font=dict(size=15)),
        xaxis=dict(title="回测窗口", tickangle=45),
        yaxis=dict(title="均线周期", dtick=1),
        height=max(500, len(pivot.index) * 26), width=max(700, len(pivot.columns) * 110),
        margin=dict(l=80, r=40, t=80, b=80), paper_bgcolor="white",
    )
    return fig


def _build_sensitivity_figure(code, scan_rows, stock_name=""):
    """构建参数敏感性折线图，返回 go.Figure 或 None。"""
    if not scan_rows:
        return None
    df = pd.DataFrame(scan_rows)
    if df.empty:
        return None
    grouped = df.groupby("均线周期")["策略评分"].agg(["mean", "std"]).dropna()
    if len(grouped) < 3:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grouped.index, y=grouped["mean"],
        mode="lines+markers", name="均值",
        line=dict(color="#2980b9", width=2), marker=dict(size=5),
        error_y=dict(type="data", array=grouped["std"], visible=True, thickness=1.5, color="#2980b9"),
    ))
    fig.add_hline(y=0, line=dict(color="#cccccc", width=1, dash="dash"))
    fig.update_layout(
        title=dict(text=f"{code} {stock_name} 参数敏感性分析（均值±标准差）", font=dict(size=15)),
        xaxis=dict(title="均线周期"), yaxis=dict(title="策略评分"),
        height=420, width=850, margin=dict(l=60, r=40, t=60, b=60),
        paper_bgcolor="white", showlegend=False,
    )
    return fig


def _build_stability_figure(code, ws_df, best_ma=None, stock_name=""):
    """构建全窗口参数稳定性热力图，返回 go.Figure 或 None。"""
    if ws_df is None or ws_df.empty:
        return None
    pivot = ws_df.pivot_table(index="均线周期", columns="窗口", values="参数稳定性评分", aggfunc="first")
    pivot = pivot[sorted(pivot.columns)]
    if pivot.empty:
        return None

    annot_text = [[f"{v:.3f}" if pd.notna(v) else "" for v in row] for row in pivot.values]
    # 每列第1名★标记
    for col_idx, col_name in enumerate(pivot.columns):
        col_data = pivot[col_name].dropna()
        if col_data.empty:
            continue
        ranked = col_data.sort_values(ascending=False)
        for label, _ in ranked.head(1).items():
            row_idx = list(pivot.index).index(label)
            _raw = annot_text[row_idx][col_idx]
            if _raw:
                annot_text[row_idx][col_idx] = f"{_raw}"

    y_labels = [str(ma) for ma in pivot.index]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=pivot.values, x=[str(c) for c in pivot.columns], y=y_labels,
        text=annot_text, texttemplate="%{text}", textfont=dict(size=10),
        colorscale="RdYlGn", zmid=0.5,
        hovertemplate="窗口: %{x}<br>均线: %{y}<br>稳定性评分: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{code} {stock_name} 全窗口参数稳定性热力图", font=dict(size=15)),
        xaxis=dict(title="窗口", tickangle=45),
        yaxis=dict(title="均线周期", dtick=1),
        height=max(500, len(pivot.index) * 26), width=max(700, len(pivot.columns) * 110),
        margin=dict(l=80, r=40, t=80, b=80), paper_bgcolor="white",
    )
    return fig


def generate_all_stock_best_ma_heatmap(all_ws, save_dir="heatmaps", return_fig=False):
    """生成全股票各窗口最优参数热力图（Plotly HTML 版本）。
    行=股票代码, 列=窗口, 值=最优均线周期。
    末尾3列为股性指标（灰色背景不上色）。
    return_fig=True 时返回 go.Figure，不写文件。
    """
    if all_ws is None or all_ws.empty:
        return None if return_fig else None

    best = all_ws[all_ws["是否最优"] == "最优"].copy()
    if best.empty:
        return

    pivot = best.pivot_table(
        index="股票代码", columns="窗口", values="均线周期", aggfunc="first"
    )
    pivot = pivot[sorted(pivot.columns)]
    if pivot.empty:
        return

    # --- 计算股性指标 ---
    def _row_personality(r):
        vals = r.dropna()
        if len(vals) < 2:
            return pd.Series([0, 0])
        changes = int((np.diff(vals.values) != 0).sum())
        std_val = vals.std(ddof=0)
        return pd.Series([changes, std_val])

    metrics_df = pivot.apply(_row_personality, axis=1)
    changes_col = metrics_df.iloc[:, 0]
    std_col = metrics_df.iloc[:, 1]
    score_col = (changes_col.combine(std_col, lambda c, s: round(max(0, 100 - c * 5 - s * 2), 1)))

    extra_cols = ["最优参数变动次数", "最优参数标准差", "股性评分"]
    extra_data = pd.DataFrame({
        "最优参数变动次数": changes_col,
        "最优参数标准差": std_col.round(2),
        "股性评分": score_col,
    }, index=pivot.index)

    n_stocks, n_windows = pivot.shape
    n_extra = len(extra_cols)

    # 合并 z：窗口列用原始值，附加列填 0（通过色阶映射为灰色）
    all_data = pivot.copy()
    for col in extra_cols:
        all_data[col] = 0.0
    z_full = all_data.values.astype(float)

    # 标注矩阵
    annot_text = []
    for ri in range(n_stocks):
        row = []
        for ci in range(n_windows):
            v = pivot.iloc[ri, ci]
            row.append(f"{int(v)}" if pd.notna(v) else "")
        for ei, col_name in enumerate(extra_cols):
            v = extra_data.iloc[ri, ei]
            if col_name == "最优参数变动次数":
                row.append(f"{int(v)}" if pd.notna(v) else "")
            elif col_name == "最优参数标准差":
                row.append(f"{v:.1f}" if pd.notna(v) else "")
            else:
                row.append(f"{v:.1f}" if pd.notna(v) else "")
        annot_text.append(row)

    # 色阶：0→灰色，>0→YlOrRd
    z_min = 0
    z_max = max(np.nanmax(pivot.values), 2)
    custom_scale = [
        [0, "#f0f0f0"],
        [1.0 / z_max, "#ffffcc"],
        [1, "#bd0026"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z_full, zmin=z_min, zmax=z_max,
        colorscale=custom_scale,
        x=[str(c) for c in all_data.columns],
        y=list(all_data.index),
        text=annot_text, texttemplate="%{text}", textfont=dict(size=9),
        hovertemplate="股票: %{y}<br>窗口: %{x}<br>值: %{text}<extra></extra>",
    ))

    # 附加列与窗口列的竖分隔线
    fig.add_shape(type="line",
        x0=n_windows - 0.5, x1=n_windows - 0.5,
        y0=-0.5, y1=n_stocks - 0.5,
        line=dict(color="#999999", width=2),
    )

    fig.update_layout(
        title=dict(text="全股票各窗口最优参数变动情况", font=dict(size=16)),
        xaxis=dict(tickangle=45),
        yaxis=dict(title="股票代码"),
        height=max(400, n_stocks * 22),
        width=max(800, n_windows * 90 + n_extra * 100),
        margin=dict(l=120, r=60, t=80, b=120),
        paper_bgcolor="white",
    )

    if return_fig:
        return fig
    _p = os.path.join(save_dir, f"all_全股票各窗口最优参数变动情况热力图{FILE_SUFFIX}.html")
    fig.write_html(_p, include_plotlyjs="cdn", config={"displayModeBar": False})
    print(f"全股票最优参数热力图: {_p}")


# =========================================================

def _sanitize_json_for_html(data_json):
    """净化 JSON 字符串，确保安全嵌入 HTML script 标签。"""
    # 防止 </script> 提前关闭 script 标签
    return data_json.replace("</script>", "<\\/script>")


# =========================================================
# 统一热力图看板（全周期 × 全市场）
# =========================================================

def generate_heatmap_dashboard(cache_data):
    """在所有回测完成后生成统一热力图看板，覆盖所有已跑的 ktype × market 组合。

    cache_data: {BAR_INTERVAL: {market: [(code, rows, ws_df, best_ma, name), ...]}}
    输出: {TRADE_DIR}/统一热力图看板.html
    """
    if not cache_data:
        return

    METRIC_CONFIG = [
        ("策略评分", "策略评分", "RdYlGn", True),
        ("年化收益率", "年化收益率", "RdYlGn", True),
        ("夏普比率", "夏普比率", "RdYlGn", True),
        ("最大回撤", "最大回撤", "OrRd", False),
    ]
    SENSITIVITY_KEY = "参数敏感性分析"
    STABILITY_KEY = "参数稳定性评分"

    ALL_KTYPES = ["1W", "1D", "60m"]
    ALL_MARKETS = ["us", "cc", "cn"]

    payload_data = {}  # {ktype: {market: {stocks: [...], figures: {code: {type: fig_json}}}}}

    for _bi in ALL_KTYPES:
        payload_data[_bi] = {}
        for _mkt_id in ALL_MARKETS:
            entries = cache_data.get(_bi, {}).get(_mkt_id, [])
            if not entries:
                payload_data[_bi][_mkt_id] = None
                continue
            stock_list = []
            figures_data = {}
            all_ws_list = []
            _kt_lbl = {"1W":"周K","1D":"日K","60m":"60分"}.get(_bi,_bi)
            for code, scan_rows, ws_df, best_ma, stock_name in tqdm(entries, desc=f"  看板({_kt_lbl},{_mkt_id.upper()})", unit="stock"):
                stock_list.append({"code": code, "name": stock_name or ""})
                figs = {}
                for chart_label, metric_name, colorscale, center in METRIC_CONFIG:
                    fig = _build_metric_heatmap_figure(
                        code, scan_rows, metric_name, chart_label,
                        colorscale, center, best_ma=best_ma, stock_name=stock_name,
                    )
                    if fig is not None:
                        d = json.loads(to_json(fig))
                        if "layout" in d and "template" in d["layout"]:
                            del d["layout"]["template"]
                        figs[chart_label] = d
                fig = _build_sensitivity_figure(code, scan_rows, stock_name=stock_name)
                if fig is not None:
                    d = json.loads(to_json(fig))
                    if "layout" in d and "template" in d["layout"]:
                        del d["layout"]["template"]
                    figs[SENSITIVITY_KEY] = d
                fig = _build_stability_figure(code, ws_df, best_ma=best_ma, stock_name=stock_name)
                if fig is not None:
                    d = json.loads(to_json(fig))
                    if "layout" in d and "template" in d["layout"]:
                        del d["layout"]["template"]
                    figs[STABILITY_KEY] = d
                figures_data[code] = figs
                if ws_df is not None and not ws_df.empty:
                    all_ws_list.append(ws_df)
            stock_list.sort(key=lambda x: x["code"])
            if all_ws_list:
                all_ws_concat = pd.concat(all_ws_list, ignore_index=True)
                _fig = generate_all_stock_best_ma_heatmap(all_ws_concat, return_fig=True)
                if _fig is not None:
                    _d = json.loads(to_json(_fig))
                    _d.get("layout", {}).pop("template", None)
                    figures_data["__ALL__"] = {"全股票最优参数变动": _d}
                    stock_list.insert(0, {"code": "__ALL__", "name": "📊 全股票汇总"})
            payload_data[_bi][_mkt_id] = {"stocks": stock_list, "figures": figures_data}

    _json_str = _sanitize_json_for_html(json.dumps(payload_data, ensure_ascii=False))
    _ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = _build_heatmap_dashboard_html(_json_str, timestamp=_ts)
    _p = os.path.join(TRADE_DIR, "统一热力图看板.html")
    os.makedirs(TRADE_DIR, exist_ok=True)
    with open(_p, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"统一热力图看板: {_p}")


def _build_heatmap_dashboard_html(data_json, timestamp=""):
    """生成包含 ktype + market 双导航栏的统一热力图看板 HTML。"""
    _tmpl = os.path.join(os.path.dirname(__file__), "heatmap_dashboard_template.html")
    with open(_tmpl, "r", encoding="utf-8") as _f:
        return _f.read().replace("__DATA__", data_json).replace("__TIMESTAMP__", timestamp)


# =========================================================
# 主程序
# =========================================================
def _process_one_stock(code, windows=None, ktype=None, ma_mode="continuous"):
    """Process a single stock. Returns (all_rows, stability_dfs, signal_map)."""
    if ktype:
        _setup_ktype(ktype, ma_mode)
    path = _find_data_file(code)
    if not path:
        return [], [], {}, None, None
    df = pd.read_parquet(path)
    df = df.sort_values("datetime")
    df["code"] = code
    out_file = os.path.join(TRADE_DIR, TRADE_SUBDIR, _market_subdir(code), f"{code}_trades.xlsx")
    stock_name = str(df["stock_name"].iloc[0]) if "stock_name" in df.columns else ""
    stock_plates = str(df["plates"].iloc[0]) if "plates" in df.columns else ""
    stock_all_rows = []
    stock_stability_dfs = []

    # =============================================
    # 累积扩展窗口回测（覆盖 MA_LIST 全部 60 个参数）
    # =============================================
    if windows is None:
        windows = generate_windows(df)
    window_trades_by_ma = {ma: [] for ma in MA_LIST}
    window_summary_rows = []

    # 预计算 HA 和滚动均线（所有窗口起点相同，全量数据一次算完）
    ha_close_full, _ = calc_heikin_ashi(df)
    ma_cache = {}
    for ma in MA_LIST:
        ma_cache[ma] = ha_close_full.rolling(ma, min_periods=ma).mean()

    # 提前提取每只股票的固定值
    code_val = code
    market_val = str(df.get("market", pd.Series([""])).iloc[0]) if "market" in df.columns else ""

    for ws, we in windows:

        window_mask = (pd.to_datetime(df["datetime"]) >= ws) & (pd.to_datetime(df["datetime"]) < we)
        df_w = df[window_mask]  # 只读切片，不 copy

        if df_w.empty:
            continue

        window_label = f"{ws.date()}~{we.date()}"
        eff_start = pd.to_datetime(df_w["datetime"]).min()
        eff_end = pd.to_datetime(df_w["datetime"]).max()
        effective_range = f"{eff_start.date()}~{eff_end.date()}"

        # 每窗口预计算一次（各 MA 共用）
        close_w_arr = df_w["close"].values.astype(np.float64)
        datetime_w_arr = df_w["datetime"].values

        for ma in MA_LIST:
            ma_arr = ma_cache[ma][window_mask].values.astype(np.float64)

            # numpy 直接计算买卖信号（无 pandas 列赋值）
            dir_arr = np.zeros(len(ma_arr), dtype=np.int8)
            dir_arr[0] = -1
            dir_arr[1:] = np.where(ma_arr[1:] > ma_arr[:-1], 1, -1)

            buy_arr = np.zeros(len(ma_arr), dtype=np.bool_)
            buy_arr[1:] = (dir_arr[1:] == 1) & (dir_arr[:-1] == -1)

            sell_arr = np.zeros(len(ma_arr), dtype=np.bool_)
            sell_arr[1:] = (dir_arr[1:] == -1) & (dir_arr[:-1] == 1)

            trades_arr, equity_arr, n_trades = _numba_backtest(
                close_w_arr, buy_arr, sell_arr, INITIAL_CASH, FEE_RATE
            )

            trades, _ = _build_trades_from_arrays(
                code_val, market_val, datetime_w_arr, ma,
                trades_arr, equity_arr, n_trades
            )

            if not trades.empty:
                trades["窗口"] = window_label
                trades["窗口内有效数据周期"] = effective_range
                window_trades_by_ma[ma].append(trades)

            summary = build_summary(trades, ma, df_w, equity_arr=equity_arr)
            summary["窗口"] = window_label
            summary["窗口内有效数据周期"] = effective_range
            summary["策略评分"] = calc_score_row(summary)

            window_summary_rows.append(summary)
            stock_all_rows.append(summary)

    # =============================================
    # 交易日志明细 -> 写 parquet（替代 Excel，避免 worker 中慢速 I/O）
    # =============================================
    window_trades_merged = [t for ma in MA_LIST for t in window_trades_by_ma[ma] if not t.empty]
    if window_trades_merged:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            _tdf = pd.concat(window_trades_merged, ignore_index=True, sort=False)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            _tdf.to_parquet(out_file.replace(".xlsx", "_trades.parquet"))

    # =============================================
    # 参数扫描结果汇总（计算在内存中，Excel 由主进程统一写入）
    # =============================================
    window_stability_df = None
    best_stab_ma = None
    if window_summary_rows:
        ws_df = reorder_columns(pd.DataFrame(window_summary_rows))
        apply_cn_mapping(ws_df)

        score_pivot, rank_pivot = build_score_matrix(window_summary_rows)

        window_stability_df = build_window_stability(window_summary_rows) if window_summary_rows and len(window_summary_rows) > len(MA_LIST) else None

        if window_stability_df is not None and not window_stability_df.empty:
            _windows = sorted(window_stability_df["窗口"].unique())
            _target = _windows[-2] if len(_windows) >= 2 else _windows[-1]
            last_ws_df = window_stability_df[window_stability_df["窗口"] == _target]
            best_stab_ma = (
                last_ws_df
                .sort_values(["参数稳定性评分", "策略评分排名标准差"], ascending=[False, True])
                .iloc[0]["均线周期"]
            )

    # =============================================
    # 从全量数据计算当前信号（用于信号扫描，复用预计算的 HA 和均线）
    # 直接修改 df 列，无需 .copy()——get_last_signal_info 只读最后一行
    # =============================================
    _top_turnover_map = _load_top_turnover_map()
    signal_map = {}
    for ma in MA_LIST:
        df["ha_close"] = ha_close_full
        df["ma"] = ma_cache[ma]
        df["dir"] = np.where(df["ma"] > df["ma"].shift(1), 1, -1)
        df["buy"] = (df["dir"] == 1) & (df["dir"].shift(1) == -1)
        df["sell"] = (df["dir"] == -1) & (df["dir"].shift(1) == 1)
        signal_info = get_last_signal_info(df)
        signal_info["K线周期"] = BAR_INTERVAL
        signal_info["均线周期"] = ma
        signal_info["股票代码"] = code
        signal_info["股票名称"] = stock_name
        signal_info["所属板块"] = stock_plates
        _turnover_rank = _top_turnover_map.get(code)
        signal_info["是否全市场成交额前200"] = "是" if _turnover_rank else "否"
        signal_info["全市场成交额排名"] = _turnover_rank
        signal_info["市场"] = market_val
        signal_map[ma] = signal_info

    return stock_all_rows, stock_stability_dfs, signal_map, window_stability_df, out_file, stock_name, best_stab_ma


def _round_display(df, pct_cols=None):
    """输出前统一处理浮点列：
    - 百分比字段 ÷100 → Excel 原生百分比格式
    - 其余 float 列保留 2 位小数
    - 参数稳定性评分保留 3 位小数
    不影响原始计算精度。
    """
    df = df.copy()
    pct_set = set(pct_cols or [])
    for col in df.select_dtypes(include=["float", "float64"]).columns:
        if col in pct_set:
            df[col] = (df[col] / 100.0).round(4)
        elif col == "参数稳定性评分":
            df[col] = df[col].round(6)
        else:
            df[col] = df[col].round(2)
    return df


def _set_pct_format(ws, df, pct_cols):
    """将指定列设为 Excel 百分比格式 0.00%"""
    from openpyxl.utils import get_column_letter
    col_map = {name: idx for idx, name in enumerate(df.columns, 1)}
    for col_name in pct_cols:
        if col_name not in col_map:
            continue
        col_letter = get_column_letter(col_map[col_name])
        for cell in ws[col_letter]:
            if cell.row > 1:
                cell.number_format = '0.00%'


def _build_signal_scan(all_df, signal_maps, stab_best, _unused=None):
    """构建信号扫描表（匹配 2windows_param_scan_numba.py 输出结构）。"""
    if not signal_maps:
        return pd.DataFrame()

    # 取最后一个窗口做评分查询
    last_df = None
    if all_df is not None and not all_df.empty and "窗口" in all_df.columns:
        _wins = sorted(all_df["窗口"].unique())
        if _wins:
            last_df = all_df[all_df["窗口"] == _wins[-1]]

    # 回测指标字段（2.py 参考清单）
    BT_COLS = ["策略评分", "收益率", "年化收益率", "买入持有收益率", "超额收益率", "平均每笔收益率",
               "最大回撤", "夏普比率", "卡尔玛比率",
               "交易次数", "盈利交易率", "盈利因子", "盈亏比",
               "平均盈利", "平均盈利比", "平均亏损", "平均亏损比", "最大单笔盈利", "最大单笔亏损",
               "最大连续盈利次数", "最大连续亏损次数", "平均持仓天数",
               "初始资金", "最终资金", "窗口", "窗口内有效数据周期"]

    rows = []
    for code, sig_map in signal_maps.items():
        # 确定最优MA
        best_ma = None
        if stab_best is not None and not stab_best.empty:
            _m = stab_best[stab_best["股票代码"] == code]
            if not _m.empty:
                best_ma = int(_m.iloc[0]["均线周期"])

        sig = None
        if best_ma is not None and best_ma in sig_map:
            sig = sig_map[best_ma]
        elif sig_map:
            if last_df is not None:
                _sr = last_df[last_df["股票代码"] == code]
                if not _sr.empty and "策略评分" in _sr.columns:
                    _best_idx = _sr["策略评分"].idxmax()
                    best_ma = int(_sr.loc[_best_idx, "均线周期"])
                    sig = sig_map.get(best_ma)
            if sig is None:
                best_ma = list(sig_map.keys())[0]
                sig = sig_map[best_ma]

        if sig is None:
            continue

        # 从 signal_info 复制所有信号字段，再覆盖回测指标
        row = dict(sig)
        if last_df is not None and "均线周期" in last_df.columns:
            _mask = (last_df["股票代码"] == code) & (last_df["均线周期"] == best_ma)
            _match = last_df[_mask]
            if not _match.empty:
                _r = _match.iloc[0]
                for col in BT_COLS:
                    if col in _r:
                        row[col] = _r[col]

        # 均线趋势共振（2.py 原版逻辑）
        trend_lookup = {ma: info.get("趋势方向") for ma, info in sig_map.items()}
        _mas = [m for m in sorted(trend_lookup.keys()) if m <= best_ma]
        _dirs = [trend_lookup[m] for m in _mas if trend_lookup.get(m) is not None]
        if _dirs and all(d == 1 for d in _dirs):
            row["均线趋势共振方向"] = "多头共振"
            row["共振均线数量"] = len(_mas)
            row["共振均线列表"] = ",".join(str(m) for m in _mas)
        elif _dirs and all(d == -1 for d in _dirs):
            row["均线趋势共振方向"] = "空头共振"
            row["共振均线数量"] = len(_mas)
            row["共振均线列表"] = ",".join(str(m) for m in _mas)
        else:
            row["均线趋势共振方向"] = "无"
            row["共振均线数量"] = 0
            row["共振均线列表"] = ""

        # 预计持仓进度 = 距离历史信号已过天数 / 平均持仓天数（仅多头有效）
        _elapsed = row.get("距离历史信号已过天数")
        _avg_hold = row.get("平均持仓天数")
        if row.get("趋势方向") == 1 and _elapsed is not None and _avg_hold is not None and _avg_hold > 0:
            row["预计持仓进度"] = round(min(_elapsed / _avg_hold * 100, 100), 2)
        else:
            row["预计持仓进度"] = None

        # 预计涨幅进度 = 距离历史信号收盘价涨跌幅 / 平均每笔收益率（仅多头且收益率 > 0）
        _hist_change = row.get("距离历史信号收盘价涨跌幅")
        _avg_trade_ret = row.get("平均每笔收益率")
        if row.get("趋势方向") == 1 and _hist_change is not None and _avg_trade_ret is not None and _avg_trade_ret > 0:
            row["预计涨幅进度"] = round(min(_hist_change / _avg_trade_ret * 100, 100), 2)
        else:
            row["预计涨幅进度"] = None

        # 窗口内有效数据天数 = 解析 "窗口内有效数据周期" 日期范围
        _win_period = row.get("窗口内有效数据周期", "")
        if isinstance(_win_period, str) and "~" in _win_period:
            try:
                parts = _win_period.split("~")
                _d1 = pd.Timestamp(parts[0])
                _d2 = pd.Timestamp(parts[1])
                row["窗口内有效数据天数"] = (_d2 - _d1).days
            except Exception:
                row["窗口内有效数据天数"] = None
        else:
            row["窗口内有效数据天数"] = None

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    signal_df = pd.DataFrame(rows)

    # 排序：多头在前（已过天数升序、策略评分降序），空头在后
    bull = signal_df[signal_df.get("趋势方向", pd.Series(-1, index=signal_df.index)) == 1].sort_values(
        ["距离历史信号已过天数", "策略评分"], ascending=[True, False]
    )
    bear = signal_df[signal_df.get("趋势方向", pd.Series(-1, index=signal_df.index)) != 1].sort_values(
        ["距离历史信号已过天数", "策略评分"], ascending=[True, False]
    )
    signal_df = pd.concat([bull, bear], ignore_index=True)

    # 策略表现分档（2.py 原版）
    def _tier(v):
        if pd.isna(v):
            return None
        if v >= 80:
            return "1优"
        elif v >= 60:
            return "2良"
        elif v >= 40:
            return "3中"
        elif v >= 20:
            return "4差"
        else:
            return "5劣"
    signal_df["策略表现"] = signal_df.get("策略评分", pd.Series(float("nan"))).apply(_tier)

    apply_cn_mapping(signal_df)
    return signal_df


SIGNAL_COLS = [
    "股票代码", "股票名称", "所属板块", "是否全市场成交额前200", "全市场成交额排名", "市场", "K线周期", "均线周期",
    "策略评分", "策略表现",
    "时间", "收盘价", "HA收盘价", "HA均线值",
    "趋势方向", "最新信号", "最新信号时间", "最新信号收盘价", "最新信号确认",
    "历史信号", "历史信号时间", "历史信号收盘价", "距离历史信号已过天数",
    "预计持仓进度",
    "距离历史信号收盘价涨跌幅", "预计涨幅进度",
    "持仓日化收益率",
    "均线趋势共振方向", "共振均线数量", "共振均线列表",
    "收益率", "年化收益率", "买入持有收益率", "超额收益率", "平均每笔收益率",
    "最大回撤", "夏普比率", "卡尔玛比率",
    "交易次数", "盈利交易率", "盈利因子", "盈亏比",
    "平均盈利", "平均盈利比", "平均亏损", "平均亏损比", "最大单笔盈利", "最大单笔亏损",
    "最大连续盈利次数", "最大连续亏损次数", "平均持仓天数",
    "初始资金", "最终资金",
    "窗口", "窗口内有效数据周期", "窗口内有效数据天数",
]




def _write_summary_excel(out_path, signal_df, all_df, score_matrix, rank_matrix,
                          window_stability_dfs, market_label):
    """写入汇总 Excel（信号扫描 + WalkForward + 最优参数 + 统计逻辑）+ 4 个独立 parquet。"""
    _base = out_path.replace(".xlsx", "")

    # 独立 parquet：回测汇总
    if all_df is not None and not all_df.empty:
        _all_out = reorder_columns(all_df)
        apply_cn_mapping(_all_out)
        _all_out = _round_display(_all_out, PCT_COLS)
        _all_out.to_parquet(f"{_base}_回测汇总.parquet", index=False)

    # 独立 parquet：策略评分明细 / 排名
    if score_matrix is not None and not score_matrix.empty:
        _round_display(score_matrix).to_parquet(f"{_base}_策略评分明细.parquet", index=False)
    if rank_matrix is not None and not rank_matrix.empty:
        _round_display(rank_matrix).to_parquet(f"{_base}_策略评分排名.parquet", index=False)

    # 独立 parquet：全窗口参数稳定性分析
    if window_stability_dfs:
        _ws = pd.concat(window_stability_dfs, ignore_index=True) if isinstance(window_stability_dfs, list) else window_stability_dfs
        _wpct = ["盈利窗口占比", "年化收益率平均值"]
        _ws_out = _round_display(_ws, _wpct)
        _ws_out.to_parquet(f"{_base}_全窗口参数稳定性分析.parquet", index=False)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # --- 1. 信号扫描（列顺序匹配 2py）---
        sig_out = signal_df[[c for c in SIGNAL_COLS if c in signal_df.columns]]
        sig_out = _round_display(sig_out, PCT_COLS)
        sig_out.to_excel(writer, sheet_name="信号扫描", index=False)
        _set_pct_format(writer.sheets["信号扫描"], sig_out, PCT_COLS)

        # 预计持仓进度列：实心填充数据条
        if "预计持仓进度" in sig_out.columns:
            from openpyxl.formatting.rule import DataBarRule
            from openpyxl.utils import get_column_letter
            _col_letter = get_column_letter(list(sig_out.columns).index("预计持仓进度") + 1)
            _nrows = len(sig_out)
            if _nrows > 0:
                _rule = DataBarRule(start_type="min", end_type="max",
                                    color="5B9BD5",  # 蓝色实心填充
                                    showValue=True,
                                    minLength=None, maxLength=None)
                writer.sheets["信号扫描"].conditional_formatting.add(
                    f"{_col_letter}2:{_col_letter}{_nrows + 1}", _rule
                )

        # 预计涨幅进度列：绿色实心填充数据条
        if "预计涨幅进度" in sig_out.columns:
            from openpyxl.formatting.rule import DataBarRule
            from openpyxl.utils import get_column_letter
            _col_letter = get_column_letter(list(sig_out.columns).index("预计涨幅进度") + 1)
            _nrows = len(sig_out)
            if _nrows > 0:
                _rule = DataBarRule(start_type="min", end_type="max",
                                    color="27AE60",
                                    showValue=True,
                                    minLength=None, maxLength=None)
                writer.sheets["信号扫描"].conditional_formatting.add(
                    f"{_col_letter}2:{_col_letter}{_nrows + 1}", _rule
                )

        # --- 2. Walk-Forward回测汇总（如果有）---
        if "Walk-Forward回测汇总" in [s for s in signal_df.columns if False]:
            pass  # 已通过 seq_trades_dfs 写入

        # --- 3. 各窗口最优参数变动情况 ---
        if window_stability_dfs:
            _ws2 = pd.concat(window_stability_dfs, ignore_index=True) if isinstance(window_stability_dfs, list) else window_stability_dfs
            best_all_ws = _ws2[_ws2["是否最优"] == "最优"].copy()
            if not best_all_ws.empty:
                pivot_best_all = best_all_ws.pivot_table(
                    index="股票代码", columns="窗口", values="均线周期", aggfunc="first"
                )
                pivot_best_all = pivot_best_all[sorted(pivot_best_all.columns)]

                def _row_personality(r):
                    vals = r.dropna()
                    if len(vals) < 2:
                        return pd.Series([0, 0])
                    changes = int((np.diff(vals.values) != 0).sum())
                    std_val = vals.std(ddof=0)
                    return pd.Series([changes, std_val])

                metrics_df = pivot_best_all.apply(_row_personality, axis=1)
                pivot_best_all["最优参数变动次数"] = metrics_df.iloc[:, 0]
                pivot_best_all["最优参数标准差"] = metrics_df.iloc[:, 1].round(2)
                pivot_best_all["股性评分"] = metrics_df.apply(
                    lambda r: round(max(0, 100 - r.iloc[0] * 5 - r.iloc[1] * 2), 1), axis=1
                )
                pivot_best_all.to_excel(writer, sheet_name="各窗口最优参数变动情况")

        # --- 7. 统计逻辑 ---
        logic_rows = [
            # ── Sheet 说明 ──
            {"类型": "Sheet说明", "名称": "信号扫描",
             "统计逻辑": "每只股票用参数稳定性最优的均线周期，显示当前信号（买入/卖出/持有/观察）及该参数在最后一个窗口的回测指标（收益率、最大回撤、夏普比率等），多头在前空头在后按策略评分降序排列；均线趋势共振分析检测各周期方向一致性"},
            {"类型": "Sheet说明", "名称": "回测汇总",
             "统计逻辑": "所有股票所有累积窗口所有MA的完整回测结果汇总（独立parquet文件）"},
            {"类型": "Sheet说明", "名称": "策略评分明细",
             "统计逻辑": "透视表，行=股票代码+K线周期+均线周期，列=窗口时间区间，值=策略评分（独立parquet文件）"},
            {"类型": "Sheet说明", "名称": "策略评分排名",
             "统计逻辑": "透视表，同上结构，值改为窗口内排名（独立parquet文件）"},
            {"类型": "Sheet说明", "名称": "全窗口参数稳定性分析",
             "统计逻辑": "个股层面，对每个累积窗口阶段计算参数稳定性（独立parquet文件）"},
            {"类型": "Sheet说明", "名称": "各窗口最优参数变动情况",
             "统计逻辑": "透视表，行=股票代码，列=窗口，值=均线周期；选取每只股票每个窗口中参数稳定性评分最高的均线周期，展示最优参数随窗口变化的趋势；末尾3列为股性指标：最优参数变动次数（相邻窗口间最优参数切换次数）、最优参数标准差（最优参数的分散程度）、股性评分（固定扣分公式 = max(0, 100 − 变动次数×5 − 标准差×2)，变动越少越稳定得分越高）"},
            {"类型": "", "名称": "", "统计逻辑": ""},
            # ── 评分模型 ──
            {"类型": "评分模型", "名称": "策略评分",
             "统计逻辑": "Score = 0.30*CAGR + 0.25*Sharpe + 0.20*(1-最大回撤) + 0.15*盈利因子 + 0.05*盈利交易率 + 0.05*交易次数；子指标min-max归一化，CAGR:0-30, Sharpe:0-2, 回撤:0-50, 盈利因子:1-3, 盈利率:30-80, 交易次数:10-100，加权求和范围0~100"},
            {"类型": "评分模型", "名称": "参数稳定性评分",
             "统计逻辑": "Score = 0.20*avg_rank_n + 0.10*top3_n + 0.15*std_n + 0.25*cagr_n + 0.20*win_rate_n + 0.10*cagr_std_n；各指标min-max归一化，排名类占0.45，收益类(均值+占比+标准差)占0.55，越高越稳定；输出保留3位小数"},
            # ── 基本字段 ──
            {"类型": "基本字段", "名称": "股票代码",
             "统计逻辑": "富途格式股票代码，如 US.AAPL / CC.BTC"},
            {"类型": "基本字段", "名称": "股票名称",
             "统计逻辑": "股票中文名称，来源于 parquet 数据文件"},
            {"类型": "基本字段", "名称": "所属板块",
             "统计逻辑": "股票行业/板块分类，来源于 parquet 数据文件"},
            {"类型": "基本字段", "名称": "是否全市场成交额前200",
             "统计逻辑": "是否在当日成交额前200美股列表中，由9download_uscncc.py --top-turnover生成，用于筛选高流动性标的"},
            {"类型": "基本字段", "名称": "全市场成交额排名",
             "统计逻辑": "该股在当日美股成交额中的排名（1~200），仅当是否全市场成交额前200为是时有效"},
            {"类型": "基本字段", "名称": "市场",
             "统计逻辑": "US=美股, CN=A股, CC=加密货币"},
            {"类型": "基本字段", "名称": "K线周期",
             "统计逻辑": "1W=周K, 1D=日K, 60m=60分钟K"},
            {"类型": "基本字段", "名称": "均线周期",
             "统计逻辑": "参数稳定性分析中倒数第二个窗口（仅1个窗口时取唯一窗口）参数稳定性评分最高的均线周期，作为该股票的最优参数，跳过最后一个未完整窗口"},
            # ── 策略表现 ──
            {"类型": "策略表现", "名称": "策略表现",
             "统计逻辑": "策略评分分档标签：≥80为「1优」，≥60为「2良」，≥40为「3中」，≥20为「4差」，<20为「5劣」"},
            # ── 信号字段 ──
            {"类型": "信号字段", "名称": "时间",
             "统计逻辑": "K线时间戳，格式 yyyy-MM-dd"},
            {"类型": "信号字段", "名称": "收盘价",
             "统计逻辑": "原始K线收盘价（未复权）"},
            {"类型": "信号字段", "名称": "HA收盘价",
             "统计逻辑": "Heikin Ashi 收盘价 = (HA开盘价 + HA最高价 + HA最低价 + HA收盘价) / 4，平滑后的价格用于计算MA"},
            {"类型": "信号字段", "名称": "HA均线值",
             "统计逻辑": "HA收盘价的简单移动平均（SMA），周期=全市场全窗口参数稳定性分析选出的最优均线周期"},
            {"类型": "信号字段", "名称": "趋势方向",
             "统计逻辑": "MA值 > MA.shift(1) 为「多头」，否则为「空头」"},
            {"类型": "信号字段", "名称": "最新信号",
             "统计逻辑": "MA方向变化判断：dir 由 -1→1 为买入，1→-1 为卖出；非信号状态时多头为持有、空头为观察"},
            {"类型": "信号字段", "名称": "最新信号时间",
             "统计逻辑": "最近一次 BUY/SELL 信号出现的 K 线时间"},
            {"类型": "信号字段", "名称": "最新信号收盘价",
             "统计逻辑": "最新信号时间对应的原始收盘价"},
            {"类型": "信号字段", "名称": "最新信号确认",
             "统计逻辑": "仅周线生效：BUY/SELL信号且距离最近一次信号<5根K线为「待确认，周K未正式收盘」，否则为「已确认」；日线和60分钟始终为「已确认」"},
            {"类型": "信号字段", "名称": "历史信号",
             "统计逻辑": "倒数第二次出现的 BUY/SELL 信号方向"},
            {"类型": "信号字段", "名称": "历史信号时间",
             "统计逻辑": "倒数第二次信号出现的 K 线时间"},
            {"类型": "信号字段", "名称": "历史信号收盘价",
             "统计逻辑": "历史信号时间对应的原始收盘价"},
            {"类型": "信号字段", "名称": "距离历史信号已过天数",
             "统计逻辑": "当前最新K线日期 − 历史信号日期，单位自然日"},
            {"类型": "信号字段", "名称": "预计持仓进度",
             "统计逻辑": "仅多头计算 = min(已过天数 / 平均持仓天数 × 100%, 100%)，反映当前持仓占平均持仓周期的进度"},
            {"类型": "信号字段", "名称": "预计涨幅进度",
             "统计逻辑": "仅多头且平均每笔收益率>0时计算 = min(涨跌幅 / 平均每笔收益率 × 100%, 100%)，反映当前涨幅已实现的平均收益进度"},
            {"类型": "信号字段", "名称": "距离历史信号收盘价涨跌幅",
             "统计逻辑": "(当前收盘价 − 历史信号收盘价) / 历史信号收盘价 × 100%"},
            {"类型": "信号字段", "名称": "持仓日化收益率",
             "统计逻辑": "自最新信号以来的日均收益率 = 总涨跌幅% / 持有天数；多头趋势直接计算，空头趋势反转符号（做空视角下跌为盈、上涨为亏）"},
            # ── 共振分析 ──
            {"类型": "共振分析", "名称": "均线趋势共振方向",
             "统计逻辑": "统计最优参数以下所有均线周期方向，全部为多头时标记为「多头共振」，全部为空头时标记为「空头共振」，否则为「无」"},
            {"类型": "共振分析", "名称": "共振均线数量",
             "统计逻辑": "与共振方向一致的均线周期数量"},
            {"类型": "共振分析", "名称": "共振均线列表",
             "统计逻辑": "与共振方向一致的均线周期列表"},
            # ── 回测指标 ──
            {"类型": "回测指标", "名称": "收益率",
             "统计逻辑": "最后一个窗口的总收益率 = (最终资金 − 初始资金) / 初始资金 × 100%"},
            {"类型": "回测指标", "名称": "年化收益率",
             "统计逻辑": "CAGR = (最终资金/初始资金)^(1/年数) − 1，年数 = 窗口实际天数/365"},
            {"类型": "回测指标", "名称": "买入持有收益率",
             "统计逻辑": "同期简单买入持有策略的收益率 = (窗口最后收盘价 − 窗口最初收盘价) / 窗口最初收盘价 × 100%"},
            {"类型": "回测指标", "名称": "超额收益率",
             "统计逻辑": "策略年化收益率 − 买入持有年化收益率，衡量策略相对基准的超额收益"},
            {"类型": "回测指标", "名称": "平均每笔收益率",
             "统计逻辑": "所有交易收益率(%)的算术平均值 = sum(每笔收益率%) / 交易次数，衡量单笔交易的平均收益水平"},
            {"类型": "回测指标", "名称": "最大回撤",
             "统计逻辑": "资金曲线从峰值到谷底的最大跌幅 = max(1 − 当日资金/当日之前峰值资金) × 100%"},
            {"类型": "回测指标", "名称": "夏普比率",
             "统计逻辑": "Sharpe Ratio = (策略年化收益率 − 无风险利率) / 年化波动率，无风险利率取2%，衡量风险调整后收益；>1为良好，>2为优秀"},
            {"类型": "回测指标", "名称": "卡尔玛比率",
             "统计逻辑": "Calmar Ratio = 年化收益率 / 最大回撤（绝对值），衡量收益与最大回撤的比值；越高说明承担单位回撤获取的收益越多"},
            {"类型": "回测指标", "名称": "交易次数",
             "统计逻辑": "回测窗口内的总交易次数（每次买入+卖出算一回合），反映策略活跃度"},
            {"类型": "回测指标", "名称": "盈利交易率",
             "统计逻辑": "盈利交易次数 / 总交易次数 × 100%，衡量策略的胜率"},
            {"类型": "回测指标", "名称": "盈利因子",
             "统计逻辑": "总盈利 / 总亏损绝对值；>1表示整体盈利，>2表示盈利能力良好"},
            {"类型": "回测指标", "名称": "盈亏比",
             "统计逻辑": "平均盈利 / 平均亏损（绝对值），衡量单次盈利与亏损的比例；>2为良好"},
            {"类型": "回测指标", "名称": "平均盈利",
             "统计逻辑": "所有盈利交易的平均盈利金额"},
            {"类型": "回测指标", "名称": "平均盈利比",
             "统计逻辑": "所有盈利交易的平均盈亏百分比"},
            {"类型": "回测指标", "名称": "平均亏损",
             "统计逻辑": "所有亏损交易的平均亏损金额（正数表示）"},
            {"类型": "回测指标", "名称": "平均亏损比",
             "统计逻辑": "所有亏损交易的平均盈亏百分比（负数）"},
            {"类型": "回测指标", "名称": "最大单笔盈利",
             "统计逻辑": "所有盈利交易中最大的一笔盈利金额"},
            {"类型": "回测指标", "名称": "最大单笔亏损",
             "统计逻辑": "所有亏损交易中最大的一笔亏损金额（正数表示）"},
            {"类型": "回测指标", "名称": "最大连续盈利次数",
             "统计逻辑": "交易序列中连续盈利的最大次数，反映策略的一致性"},
            {"类型": "回测指标", "名称": "最大连续亏损次数",
             "统计逻辑": "交易序列中连续亏损的最大次数，反映策略的回撤深度"},
            {"类型": "回测指标", "名称": "平均持仓天数",
             "统计逻辑": "所有交易持仓天数的平均值 = 总持仓天数 / 交易次数"},
            {"类型": "回测指标", "名称": "初始资金",
             "统计逻辑": "回测起始资金，统一设定为 10,000"},
            {"类型": "回测指标", "名称": "最终资金",
             "统计逻辑": "回测结束后账户总资金 = 初始资金 + 累计盈亏"},
            # ── 窗口信息 ──
            {"类型": "窗口信息", "名称": "窗口",
             "统计逻辑": "回测窗口的时间区间标签，格式 起始日期~结束日期；起始日期为各周期数据的最早日期，步长：周线12个月/日线6个月/60分钟3个月，所有股票共享同一套窗口列表"},
            {"类型": "窗口信息", "名称": "窗口内有效数据周期",
             "统计逻辑": "该窗口实际数据的起止日期区间，格式 起始日期~结束日期；若股票上市晚于窗口起始，起始日期为数据首日"},
            {"类型": "窗口信息", "名称": "窗口内有效数据天数",
             "统计逻辑": "从窗口内有效数据周期解析出的实际天数 = 结束日期 − 起始日期"},
            # ── 参数选择 ──
            {"类型": "参数选择", "名称": "最优均线周期",
             "统计逻辑": "参数稳定性分析中倒数第二个窗口（仅1个窗口时取唯一窗口）参数稳定性评分最高的均线周期，选作信号扫描使用的参数"},
            # ── 参数稳定性 ──
            {"类型": "参数稳定性", "名称": "窗口数量",
             "统计逻辑": "该均线周期参与计算的窗口总数"},
            {"类型": "参数稳定性", "名称": "盈利窗口占比",
             "统计逻辑": "盈利窗口数量 / 窗口数量 × 100%"},
            {"类型": "参数稳定性", "名称": "盈利窗口数量",
             "统计逻辑": "年化收益率 > 0 的窗口数量"},
            {"类型": "参数稳定性", "名称": "年化收益率平均值",
             "统计逻辑": "该均线周期在所有窗口中年化收益率的算术平均值"},
            {"类型": "参数稳定性", "名称": "年化收益率标准差",
             "统计逻辑": "该均线周期在所有窗口中年化收益率的标准差，衡量收益波动性"},
            {"类型": "参数稳定性", "名称": "策略评分排名平均值",
             "统计逻辑": "该均线周期在各窗口中策略评分排名的算术平均值，越低越好"},
            {"类型": "参数稳定性", "名称": "策略评分排名标准差",
             "统计逻辑": "该均线周期在各窗口中策略评分排名的标准差，越低越稳定"},
            {"类型": "参数稳定性", "名称": "策略评分排名第一次数",
             "统计逻辑": "该均线周期在各窗口中排名第一的次数，衡量夺冠能力"},
            {"类型": "参数稳定性", "名称": "参数稳定性评分",
             "统计逻辑": "Score = 0.20*avg_rank_n + 0.10*top3_n + 0.15*std_n + 0.25*cagr_n + 0.20*win_rate_n + 0.10*cagr_std_n；各指标min-max归一化，排名类占0.45，收益类(均值+占比+标准差)占0.55，越高越稳定；输出保留3位小数"},
            {"类型": "参数稳定性", "名称": "策略评分排名Top3占比",
             "统计逻辑": "该均线周期在各窗口中排名前三的次数占比"},
            {"类型": "参数稳定性", "名称": "是否最优",
             "统计逻辑": "每只股票每个窗口内参数稳定性评分最高者标记为「最优」，并列时以策略评分排名标准差升序+年化收益率平均值降序决胜，确保唯一"},
        ]
        pd.DataFrame(logic_rows).to_excel(writer, sheet_name="统计逻辑", index=False)

        for ws in writer.sheets.values():
            _apply_sheet_format(ws)

    print(f"{market_label}汇总Excel: {out_path}")


def run_trade():

    symbols = load_symbols(SYMBOL_FILE)

    # 过滤出有数据文件的股票，用于进度条总计数
    available = [s for s in symbols if _find_data_file(s)]

    # 扫描全市场数据，取最早和最晚日期作为窗口范围
    global_start = pd.Timestamp("2099-12-31")
    global_end = pd.Timestamp("2000-01-01")
    for s in available:
        try:
            _path = _find_data_file(s)
            if not _path:
                continue
            _tmp = pd.read_parquet(_path, columns=["datetime"])
            _min = pd.to_datetime(_tmp["datetime"]).min()
            _max = pd.to_datetime(_tmp["datetime"]).max()
            if _min < global_start:
                global_start = _min
            if _max > global_end:
                global_end = _max
        except Exception:
            continue
    global WINDOW_START_DATE
    WINDOW_START_DATE = str(global_start.date())
    windows = generate_windows(end_date=global_end)
    print(f"窗口范围: {global_start.date()} ~ {global_end.date()}, 共 {len(windows)} 个窗口")

    # 按市场分组
    def _market_group(code):
        if code.startswith("CC."):
            return "cc"
        elif code.startswith(("SH.", "SZ.")):
            return "cn"
        elif code.startswith("US."):
            return "us"
        raise ValueError(f"未知代码前缀: {code}")

    market_order = ["us", "cc", "cn"]
    market_groups = {m: [s for s in available if _market_group(s) == m] for m in market_order}

    all_rows = []
    all_signal_maps = {}
    window_stability_dfs = []
    _heatmap_cache = []  # (code, summary_rows, stability_df, best_ma, stock_name)

    for mkt in market_order:
        group = market_groups[mkt]
        if not group:
            continue

        print(f"\n===== 开始回测 {mkt.upper()} 市场（{len(group)} 只股票）=====")

        market_rows = []
        market_signal_maps = {}
        market_window_stability = []

        _results = {}
        _kt = {"1W": "week", "1D": "day", "60m": "60m"}.get(BAR_INTERVAL, "week")
        _workers = os.cpu_count() - 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=_workers) as executor:
            future_to_code = {executor.submit(_process_one_stock, code, windows, _kt, MA_MODE): code for code in group}
            with tqdm(total=len(group), desc=f"{mkt.upper()}回测", unit="stock") as pbar:
                    for future in concurrent.futures.as_completed(future_to_code):
                        code = future_to_code[future]
                        try:
                            _results[code] = future.result()
                        except Exception as e:
                            _results[code] = e
                        finally:
                            pbar.update(1)

        _success = 0
        _failed = []
        for code in sorted(_results.keys()):
                result = _results[code]
                if isinstance(result, Exception):
                    _failed.append((code, str(result)))
                else:
                    _success += 1
                    s_all, s_stab, sig_map, s_ws, out_f, stock_name, best_ma = result
                    market_rows.extend(s_all)
                    market_signal_maps[code] = sig_map
                    if s_ws is not None and not s_ws.empty:
                        market_window_stability.append(s_ws)
                    # 收集热力图数据（主进程统一生成，避免 worker 中 matplotlib 开销）
                    _heatmap_cache.append((code, s_all, s_ws, best_ma, stock_name))
        tqdm.write(f"  {mkt.upper()} 完成: {_success} 成功")
        if _failed:
            tqdm.write(f"  {mkt.upper()} 失败: {len(_failed)} 只 — {'; '.join(f'{c}({e})' for c, e in _failed)}")

        # ---- 本市场汇总 ----
        if not market_rows:
            continue

        market_df = pd.DataFrame(market_rows)

        # 稳定性分析取最优参数
        stab_mkt = None
        if market_window_stability:
            all_ws_mkt = pd.concat(market_window_stability, ignore_index=True)
            _wins_mkt = sorted(all_ws_mkt["窗口"].unique())
            _target = _wins_mkt[-2] if len(_wins_mkt) >= 2 else _wins_mkt[-1]
            last_ws_mkt = all_ws_mkt[all_ws_mkt["窗口"] == _target]
            stab_mkt = (
                last_ws_mkt
                .sort_values(["参数稳定性评分", "策略评分排名标准差"], ascending=[False, True])
                .groupby("股票代码", sort=False)
                .head(1)
                .reset_index(drop=True)
            )

        # 本市场信号扫描
        signal_mkt = _build_signal_scan(market_df, market_signal_maps, stab_mkt, all_signal_maps)

        # 收集热力图数据到全局缓存（统一看板在回测完成后生成）
        global _HEATMAP_CACHE
        _mkt_entries = [(c, r, w, b, s) for c, r, w, b, s in _heatmap_cache
                        if _market_subdir(c) == mkt]
        if _mkt_entries:
            _HEATMAP_CACHE.setdefault(BAR_INTERVAL, {})[mkt] = _mkt_entries

        # 累计到全市场
        all_rows.extend(market_rows)
        all_signal_maps.update(market_signal_maps)
        window_stability_dfs.extend(market_window_stability)

    # =============================================
    # 全市场汇总
    # =============================================
    if not all_rows:
        return

    all_df = pd.DataFrame(all_rows)
    score_all, rank_all = build_score_matrix(all_rows)

    stab_best = None
    if window_stability_dfs:
        all_ws_stab = pd.concat(window_stability_dfs, ignore_index=True)
        _wins = sorted(all_ws_stab["窗口"].unique())
        _target_ws = _wins[-2] if len(_wins) >= 2 else _wins[-1]
        last_ws = all_ws_stab[all_ws_stab["窗口"] == _target_ws]
        stab_best = (
            last_ws
            .sort_values(["参数稳定性评分", "策略评分排名标准差"], ascending=[False, True])
            .groupby("股票代码", sort=False)
            .head(1)
            .reset_index(drop=True)
        )

    signal_all = _build_signal_scan(all_df, all_signal_maps, stab_best, all_signal_maps)

    if signal_all.empty:
        return

    # TradingView 配置
    tv_lines = []
    missing_codes = []
    for code in sorted(available):
        match = None
        if stab_best is not None:
            m = stab_best[stab_best["股票代码"] == code]
            if not m.empty:
                match = int(m.iloc[0]["均线周期"])
        if match is not None:
            ticker = code.split(".", 1)[1] if "." in code else code
            tv_lines.append(f'    autoMAPool.put("{ticker}", {match})')
        else:
            missing_codes.append(code)

    tv_path = os.path.join(TRADE_DIR, TRADE_SUBDIR, "tradingview_params.txt")
    with open(tv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tv_lines) + "\n")
    print(f"\nTradingView 配置文件: {tv_path} 共 {len(tv_lines)} 只股票")

    if missing_codes:
        print("以下股票未找到最优参数，已跳过:")
        for c in missing_codes:
            print(f"  {c}")

    # 全市场汇总 Excel
    date_str = pd.Timestamp.today().strftime("%Y%m%d")
    all_out = os.path.join(TRADE_DIR, TRADE_SUBDIR, f"{date_str}_{TRADE_SUBDIR}_信号汇总.xlsx")
    _write_summary_excel(all_out, signal_all, all_df, score_all, rank_all,
                        window_stability_dfs, "全市场")
    print(f"全市场完成: {all_out}")


# =========================================================
# 富途自选股分组同步
# =========================================================
# 同步和导出功能已移除

# =========================================================
def _generate_ktype_comparison():
    """读取各周期信号汇总Excel，生成各周期数据对比Excel。"""
    import glob as _glob
    _kt_labels = {"1w": "（周线）", "1d": "（日线）", "60m": "（60分钟）"}
    _dfs = {}
    for _kt_name in ["1w", "1d", "60m"]:
        _files = sorted(_glob.glob(os.path.join(TRADE_DIR, _kt_name, "*_信号汇总.xlsx")))
        if not _files:
            continue
        _df = pd.read_excel(_files[-1], sheet_name="信号扫描")
        if _df.empty:
            continue
        # 重命名对比字段，加周期后缀
        _rename = {}
        for _col in _df.columns:
            if _col in ("股票代码", "股票名称", "所属板块", "是否全市场成交额前200",
                        "全市场成交额排名", "市场"):
                continue
            _rename[_col] = f"{_col}{_kt_labels[_kt_name]}"
        _df2 = _df.rename(columns=_rename)
        _df2 = _df2.set_index("股票代码")
        # 非对比字段仅保留第一个周期（1w）的，其余丢弃避免 join 冲突
        if _kt_name != "1w":
            _non_compare = ["股票名称", "所属板块", "是否全市场成交额前200",
                           "全市场成交额排名", "市场"]
            _df2 = _df2.drop(columns=[c for c in _non_compare if c in _df2.columns])
        _dfs[_kt_name] = _df2

    if not _dfs:
        return

    # 合并：outer join 保留所有股票
    _merged = None
    for _kt_name, _df in _dfs.items():
        if _merged is None:
            _merged = _df
        else:
            _merged = _merged.join(_df, how="outer")

    _merged = _merged.reset_index()

    # 按字段分类（用于 Excel 组合）
    _field_groups = [
        ("策略评分", ["策略评分"]),
        ("收益类", ["收益率", "年化收益率", "买入持有收益率", "超额收益率", "平均每笔收益率"]),
        ("风险类", ["最大回撤", "夏普比率", "卡尔玛比率"]),
        ("交易统计", ["交易次数", "盈利交易率", "盈利因子", "盈亏比"]),
        ("每笔统计", ["平均盈利", "平均盈利比", "平均亏损", "平均亏损比",
                       "最大单笔盈利", "最大单笔亏损"]),
        ("连续性", ["最大连续盈利次数", "最大连续亏损次数", "平均持仓天数"]),
        ("持仓进度", ["预计持仓进度", "预计涨幅进度", "持仓日化收益率"]),
        ("资金", ["初始资金", "最终资金"]),
        ("信号/方向", ["趋势方向", "最新信号", "最新信号确认",
                       "历史信号", "距离历史信号已过天数",
                       "距离历史信号收盘价涨跌幅"]),
        ("窗口", ["窗口", "窗口内有效数据周期", "窗口内有效数据天数"]),
    ]

    # 构建输出列顺序：基本信息 + 各分类字段（3周期并排）
    _base_cols = ["股票代码", "股票名称", "所属板块",
                  "是否全市场成交额前200", "全市场成交额排名", "市场"]
    _order = [c for c in _base_cols if c in _merged.columns]
    for _gname, _fields in _field_groups:
        for _f in _fields:
            for _kt_label in ["（周线）", "（日线）", "（60分钟）"]:
                _col = f"{_f}{_kt_label}"
                if _col in _merged.columns:
                    _order.append(_col)

    _out = _merged[[c for c in _order if c in _merged.columns]]

    # 写入 Excel 并添加组合
    _out_path = os.path.join(TRADE_DIR, "各周期数据对比.xlsx")
    with pd.ExcelWriter(_out_path, engine="openpyxl") as _writer:
        _out.to_excel(_writer, sheet_name="周期对比", index=False)
        _ws = _writer.sheets["周期对比"]

        # 添加组合（分组）
        from openpyxl.utils import get_column_letter
        _col_idx = len(_base_cols) + 1  # 第一组起始列（1-indexed）
        for _gname, _fields in _field_groups:
            _gcols = 0
            for _f in _fields:
                for _kt_label in ["（周线）", "（日线）", "（60分钟）"]:
                    if f"{_f}{_kt_label}" in _out.columns:
                        _gcols += 1
            if _gcols > 1:
                _start_letter = get_column_letter(_col_idx)
                _end_letter = get_column_letter(_col_idx + _gcols - 1)
                _ws.column_dimensions.group(_start_letter, _end_letter, hidden=False)
            _col_idx += _gcols

        # 表头加粗
        for _cell in _ws[1]:
            _cell.font = _cell.font.copy(bold=True)

    print(f"各周期数据对比: {_out_path}")


if __name__ == "__main__":
    import argparse
    import sys

    # 兼容旧版 sync 参数
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        print("sync 模式已移除")
        sys.exit(0)

    parser = argparse.ArgumentParser(description="多窗口参数扫描回测")
    parser.add_argument("--ktype", default=DEFAULT_KTYPE,
                        help=f"K线周期: day/周K, week/周K, 60m/60分钟, 逗号拼接如week,day, all=三者全部 (默认: {DEFAULT_KTYPE})")
    parser.add_argument("--market", default=DEFAULT_MARKET,
                        help=f"市场: US/CN/CC/US,CN/all (默认: {DEFAULT_MARKET})")
    parser.add_argument("--ma-mode", choices=["continuous", "jump"], default=MA_MODE,
                        help="MA序列类型: continuous=连续回测, jump=跳跃回测(日线step=2,60m step=4,周线强制连续)")
    _CLI_ARGS = parser.parse_args()
    MA_MODE = _CLI_ARGS.ma_mode

    # 解析 ktype 列表（支持逗号拼接）
    _KT_MAP = {"day": "日K", "week": "周K", "60m": "60分钟K", "all": "全部"}
    _raw = _CLI_ARGS.ktype.lower().replace("，", ",").split(",")
    _ktypes = []
    for _k in _raw:
        _k = _k.strip()
        if _k == "all":
            _ktypes = ["week", "day", "60m"]
            break
        if _k in _KT_MAP:
            if _k not in _ktypes:
                _ktypes.append(_k)
    if not _ktypes:
        print(f"错误: 无效的 --ktype '{_CLI_ARGS.ktype}'，可选 week/day/60m/all 或逗号拼接")
        sys.exit(1)

    # 初始设置（用第一个 ktype 初始化全局变量）
    _setup_ktype(_ktypes[0], MA_MODE)

    def _filter_symbols(symbols):
        markets = _CLI_ARGS.market.upper().split(",")
        if "ALL" in markets:
            return symbols
        prefixes = []
        for m in markets:
            m = m.strip()
            if m == "US":
                prefixes.append("US.")
            elif m == "CN":
                prefixes.extend(("SH.", "SZ."))
            elif m == "CC":
                prefixes.append("CC.")
        def _match(s):
            return any(s.startswith(p) for p in prefixes)
        return [s for s in symbols if _match(s)]

    import __main__ as _mod
    _orig_load = _mod.load_symbols
    _mod.load_symbols = lambda path: _filter_symbols(_orig_load(path))

    init_symbols_file(SYMBOL_FILE)

    import time as _t
    for i, _kt in enumerate(_ktypes):
        _t0 = _t.time()
        print(f"\n{'='*60}")
        print(f"  开始 {_kt} 回测 ({i+1}/{len(_ktypes)})")
        print(f"{'='*60}")
        _setup_ktype(_kt, MA_MODE)
        run_trade()
        _elapsed = _t.time() - _t0
        print(f"  [{_kt}] 完成，耗时 {int(_elapsed//60)}分{int(_elapsed%60)}秒")

    # 三个周期都跑了才生成各周期数据对比
    if set(_ktypes) == {"week", "day", "60m"}:
        _generate_ktype_comparison()

    # 统一热力图看板（覆盖所有已跑的 ktype × market）
    if _HEATMAP_CACHE:
        generate_heatmap_dashboard(_HEATMAP_CACHE)