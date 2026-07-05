import os
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
    """从 DuckDB top_turnover_stock_rank 表读取最新排名，返回 {代码: 排名} 映射表。"""
    result = {}
    try:
        import duckdb
        _db_path = os.path.join(os.path.dirname(__file__), "database", "market.duckdb")
        if not os.path.exists(_db_path):
            return result
        _con = duckdb.connect(_db_path, read_only=True)
        for _r in _con.execute("""
            SELECT code, rank FROM top_turnover_stock_rank
            WHERE datetime = (SELECT MAX(datetime) FROM top_turnover_stock_rank)
        """).fetchall():
            result[str(_r[0])] = int(_r[1])
        _con.close()
    except Exception:
        pass
    return result


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
TRADE_DIR = "results_uscncc"

os.makedirs(TRADE_DIR, exist_ok=True)

INITIAL_CASH = 10000
FEE_RATE = 0.001
TRADE_MODE = "close"   # "close" 按信号K线收盘价成交 / "open" 按下根K线开盘价成交
SLIPPAGE = 0.0         # 滑点（百分比，如 0.001 = 0.1%）

DEFAULT_KTYPE = "week,day"     # 默认ktype: week(周K) / day(日K) / all(两者全部) / week,day(逗号拼接)
DEFAULT_MARKET = "US,CC"    # 默认market: all / US / CN / CC / US,CC
MA_MODE = "continuous"    # 默认MA序列类型: continuous=连续回测 / jump=跳跃回测
_HEATMAP_CACHE = {}  # 热力图看板数据缓存: {BAR_INTERVAL: {market: [(code, rows, ws, best_ma, name, sig_map), ...]}}

# 中文映射
DIR_MAP = {1: "多头", -1: "空头"}
SIG_MAP = {"BUY": "买入", "SELL": "卖出", "HOLD": "持有", "WATCH": "观察"}


def _market_subdir(code):
    """根据code返回market子目录名：us/cn/cc"""
    if code.startswith("CC."):
        return "cc"
    elif code.startswith(("SH.", "SZ.")):
        return "cn"
    elif code.startswith("US."):
        return "us"
    raise ValueError(f"未知代码前缀: {code}")


_DB_CONN = None  # 进程级全局 DuckDB 连接

def _get_db_conn():
    """获取进程级 DuckDB 连接（复用）。"""
    global _DB_CONN
    if _DB_CONN is None:
        import duckdb as _dk
        _db_path = os.path.join(os.path.dirname(__file__), "database", "market.duckdb")
        if os.path.exists(_db_path):
            try:
                _DB_CONN = _dk.connect(_db_path, read_only=True)
            except Exception:
                pass
    return _DB_CONN


def _load_data(code):
    """从 DuckDB 读取 K 线数据（复用连接）。"""
    _kt = {"1W": "1w", "1D": "1d"}.get(BAR_INTERVAL, "")
    _con = _get_db_conn()
    if _con is not None:
        for _tbl in [f"klines_{_kt}", f"v_klines_{_kt}"]:
            try:
                _df = _con.execute(
                    f'SELECT * FROM {_tbl} WHERE code = ? ORDER BY datetime', [code]
                ).fetchdf()
                if not _df.empty:
                    return _df
            except Exception:
                pass
    return None

def apply_cn_mapping(df):
    """统一应用direction和信号的中文映射"""
    if "direction" in df.columns:
        df["direction"] = df["direction"].map(DIR_MAP)
    if "signal" in df.columns:
        df["signal"] = df["signal"].map(SIG_MAP)
    if "signal" in df.columns:
        df["signal"] = df["signal"].map(SIG_MAP)


KLINE_MAP = {
    "1D": {"display": "日K", "suffix": "_1d", "period": 252, "ma_range": list(range(2, 181))},
    "1W": {"display": "周K", "suffix": "_1w", "period": 52, "ma_range": list(range(2, 61))},
}

def generate_ma_list(bar_interval, ma_mode="continuous"):
    """根据ktype和MA模式生成MA列表。

    周线: 强制 continuous (step=1)
    日线 continuous: 2..180 (step=1) / jump: 2,4,6..180 (step=2)
    """
    if bar_interval == "1W" or ma_mode == "continuous":
        return KLINE_MAP[bar_interval]["ma_range"]
    if bar_interval == "1D":
        return list(range(2, 181, 2))
    return KLINE_MAP[bar_interval]["ma_range"]

BAR_INTERVAL = "1D" if DEFAULT_KTYPE == "day" else "1W"
MA_LIST = generate_ma_list(BAR_INTERVAL, MA_MODE)
FILE_SUFFIX = KLINE_MAP[BAR_INTERVAL]["suffix"]
TRADING_PERIOD = KLINE_MAP[BAR_INTERVAL]["period"]
KTYPE_DIR_MAP = {"1W": "1w", "1D": "1d"}
TRADE_SUBDIR = KTYPE_DIR_MAP.get(BAR_INTERVAL, "")
for _m in ("us", "cn", "cc"):
    os.makedirs(os.path.join(TRADE_DIR, TRADE_SUBDIR, _m), exist_ok=True)
    os.makedirs(os.path.join(TRADE_DIR, TRADE_SUBDIR, _m, "heatmaps"), exist_ok=True)
os.makedirs(os.path.join(TRADE_DIR, TRADE_SUBDIR, "heatmaps"), exist_ok=True)
STEP_MONTHS = {"1W": 12, "1D": 6}.get(BAR_INTERVAL, 12)
WINDOW_START_DATE = "2000-01-03"

# ---- 命令行参数解析（前置，仅在作为主程序运行时生效）----
def _setup_ktype(ktype, ma_mode="continuous"):
    """设置backtest_period的全局变量（供 worker 进程调用）。"""
    global BAR_INTERVAL, MA_LIST, FILE_SUFFIX, TRADING_PERIOD, TRADE_SUBDIR, STEP_MONTHS, MA_MODE, TRADE_MODE, SLIPPAGE
    MA_MODE = ma_mode
    if ktype == "day":
        BAR_INTERVAL = "1D"
    else:
        BAR_INTERVAL = "1W"
    MA_LIST = generate_ma_list(BAR_INTERVAL, MA_MODE)
    FILE_SUFFIX = KLINE_MAP[BAR_INTERVAL]["suffix"]
    TRADING_PERIOD = KLINE_MAP[BAR_INTERVAL]["period"]
    TRADE_SUBDIR = KTYPE_DIR_MAP.get(BAR_INTERVAL, "")
    STEP_MONTHS = {"1W": 12, "1D": 6}.get(BAR_INTERVAL, 12)
    for _m in ("us", "cn", "cc"):
        os.makedirs(os.path.join(TRADE_DIR, TRADE_SUBDIR, _m), exist_ok=True)
        os.makedirs(os.path.join(TRADE_DIR, TRADE_SUBDIR, _m, "heatmaps"), exist_ok=True)
    os.makedirs(os.path.join(TRADE_DIR, TRADE_SUBDIR, "heatmaps"), exist_ok=True)


# 百分比字段（原始值=百分比数值，如 5.23 表示 5.23%；
# 输出时 ÷100 再设 Excel 单元格格式为 0.00%，实现 Excel 原生百分比显示）
PCT_COLS = [
    "est_hold_progress", "est_gain",
    "change_since_last_signal", "holding_daily_return",
    "avg_trade_return", "avg_win_pct", "avg_loss_pct",
    "total_return", "cagr", "buy_hold_return", "excess_return",
    "max_drawdown", "win_rate",
]

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
def _numba_backtest(close, buy, sell, initial_cash, fee_rate,
                    slippage=0.0, trade_mode=0, open_arr=None):
    """Numba 加速核心回测：单次遍历计算交易记录和资金曲线。

    trade_mode: 0=按信号K线收盘价成交, 1=按下根K线开盘价成交
    slippage: 滑点比例（如 0.001 = 0.1%），买入加滑点，卖出减滑点

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
    realized_cash = initial_cash
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

        # ---- 确定成交价 ----
        if trade_mode == 1 and open_arr is not None and i < n - 1:
            exec_price = open_arr[i + 1]  # 按下根K线开盘价
        else:
            exec_price = p  # 按本根K线收盘价
        exec_price = max(exec_price, 1e-10)  # 防止零值

        # ---- 开仓 ----
        if buy[i] and position == 0:
            buy_price = exec_price * (1 + slippage)
            shares = int(available_cash / (buy_price * (1 + fee_rate)))
            if shares > 0:
                cost = shares * buy_price
                fee = cost * fee_rate
                cash_before_buy = available_cash
                available_cash -= (cost + fee)
                position = shares
                entry_price_val = buy_price
                entry_idx = i
                trades[trade_count, 8] = cash_before_buy
                trades[trade_count, 9] = available_cash

        # ---- 平仓 ----
        elif sell[i] and position > 0:
            sell_price = exec_price * (1 - slippage)
            sell_value = position * sell_price
            sell_fee = sell_value * fee_rate
            cash_before_sell = available_cash
            available_cash += (sell_value - sell_fee)

            buy_fee = entry_price_val * position * fee_rate
            pnl = (sell_price - entry_price_val) * position - buy_fee - sell_fee

            trades[trade_count, 0] = entry_price_val
            trades[trade_count, 1] = position
            trades[trade_count, 2] = entry_idx
            trades[trade_count, 3] = sell_price
            trades[trade_count, 4] = i
            trades[trade_count, 5] = buy_fee
            trades[trade_count, 6] = sell_fee
            trades[trade_count, 7] = pnl
            trades[trade_count, 10] = cash_before_sell
            trades[trade_count, 11] = available_cash
            trades[trade_count, 12] = 0
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
        sell_price = p * (1 - slippage)
        sell_value = position * sell_price
        sell_fee = sell_value * fee_rate
        buy_fee = entry_price_val * position * fee_rate
        pnl = (sell_price - entry_price_val) * position - buy_fee - sell_fee
        cash_before_sell = available_cash
        available_cash += (sell_value - sell_fee)

        trades[trade_count, 0] = entry_price_val
        trades[trade_count, 1] = position
        trades[trade_count, 2] = entry_idx
        trades[trade_count, 3] = sell_price
        trades[trade_count, 4] = n - 1
        trades[trade_count, 5] = buy_fee
        trades[trade_count, 6] = sell_fee
        trades[trade_count, 7] = pnl
        trades[trade_count, 10] = cash_before_sell
        trades[trade_count, 11] = available_cash
        trades[trade_count, 12] = 1
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
            "code": code_val,
            "market": market_val,
            "ktype": BAR_INTERVAL,
            "ma": ma_len,

            "entry_time": entry_time,
            "entry_price": float(entry_price_val),
            "buy_shares": position_val,

            "exit_time": exit_time,
            "exit_price": float(t[3]),
            "sell_shares": position_val,

            "trade_status": "未平仓(强制结算)" if is_force else "已平仓",
            "pnl_type": "盈利" if pnl > 0 else "亏损",

            "pnl": pnl,
            "return_pct": float(return_pct),

            "buy_fee": buy_fee,
            "sell_fee": sell_fee,
            "total_fee": total_fee,

            "cash_before_entry": cash_before_buy,
            "cash_after_entry": cash_after_buy,
            "cash_before_exit": cash_before_sell,
            "cash_after_exit": cash_after_sell,

            "hold_bars": hold_kbars,
            "hold_days": hold_days,

            "backtest_period": backtest_period
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
        _buy_idx = df["datetime"].searchsorted(buy_time, side="left")
        buy_kbars = len(df) - 1 - min(_buy_idx, len(df) - 1)
    else:
        buy_time = pd.NaT
        buy_kbars = None

    sell_rows = df[df["sell"]]

    if len(sell_rows) > 0:
        last_sell = sell_rows.iloc[-1]
        sell_time = pd.to_datetime(last_sell["datetime"])
        _sell_idx = df["datetime"].searchsorted(sell_time, side="left")
        sell_kbars = len(df) - 1 - min(_sell_idx, len(df) - 1)
    else:
        sell_time = pd.NaT
        sell_kbars = None

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

    # 若选出的last_signal未确认（<5天），回退到上一次已确认的信号
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
        # K 线数 = 信号所在位置到末尾的 bar 数量
        _hist_idx = df["datetime"].searchsorted(hist_time, side="left")
        _hist_idx = min(_hist_idx, len(df) - 1)
        hist_kbars = len(df) - 1 - _hist_idx
        hist_change = (last_close - hist_close) / hist_close * 100 if hist_close else None
        # holding_daily_return：空头趋势反转符号（做空视角下跌为盈、上涨为亏）
        _dir = int(last["dir"])
        if hist_change is not None and _dir == -1:
            hist_daily = (-hist_change) / hist_days if hist_days and hist_days > 0 else None
        else:
            hist_daily = hist_change / hist_days if hist_days and hist_days > 0 else None
    else:
        hist_change = None
        hist_daily = None
        hist_kbars = None

    # signal_confirmed：信号出现在最新 K 线则待确认，下一根 K 线才确认
    if signal in ("BUY", "SELL"):
        _kbars = buy_kbars if signal == "BUY" else sell_kbars
        if _kbars is not None and _kbars < 1:
            confirm = "待确认（信号与最新K线同根）"
        else:
            confirm = "已确认"
    else:
        confirm = "已确认"

    _date = lambda v: pd.Timestamp(v).date()
    _opt_date = lambda v: _date(v) if pd.notna(v) else None

    return {
        "signal_date": _opt_date(last["datetime"]),
        "close": last_close,
        "ha_close": float(last["ha_close"]),
        "ha_ma": float(last["ma"]) if not pd.isna(last["ma"]) else None,
        "direction": int(last["dir"]),
        "signal": signal,
        "signal_time": _opt_date(last["datetime"]),
        "signal_close": last_close,
        "signal_confirmed": confirm,

        "last_signal": hist_signal,
        "last_signal_time": _opt_date(hist_time),
        "last_signal_close": hist_close,
        "bars_since_last_signal": hist_kbars,
        "change_since_last_signal": hist_change,
        "holding_daily_return": hist_daily,
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
        "code": code_val,
        "market": market_val,
        "ktype": BAR_INTERVAL,
        "ma": ma_len,

        "entry_time": entry_time,
        "entry_price": float(entry_price),
        "buy_shares": position,

        "exit_time": exit_time,
        "exit_price": float(exit_price),
        "sell_shares": position,

        "trade_status": status,
        "pnl_type": "盈利" if pnl > 0 else "亏损",

        "pnl": pnl,
        "return_pct": float(return_pct),

        "buy_fee": buy_fee,
        "sell_fee": sell_fee,
        "total_fee": total_fee,

        "cash_before_entry": cash_before_open if cash_before_open is not None else None,
        "cash_after_entry": cash_after_open if cash_after_open is not None else None,
        "cash_before_exit": cash_before_close if cash_before_close is not None else None,
        "cash_after_exit": cash_after_close if cash_after_close is not None else None,

        "hold_bars": hold_kbars,
        "hold_days": hold_days,

        "backtest_period": backtest_period
    }

def equity_curve(df, trades_df):

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    if trades_df is None or trades_df.empty:
        return np.full(len(df), INITIAL_CASH)

    trades_df = trades_df.copy()
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

    equity = np.zeros(len(df))

    cash = INITIAL_CASH
    trade_idx = 0

    trades_df = trades_df.sort_values("exit_time").reset_index(drop=True)

    # numpy arrays for faster row access
    datetime_arr = df["datetime"].values
    close_arr = df["close"].values
    open_arr = trades_df["entry_time"].values
    close_t_arr = trades_df["exit_time"].values
    pnl_arr = trades_df["pnl"].values
    entry_arr = trades_df["entry_price"].values
    shares_arr = trades_df["buy_shares"].values
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

    pnl = trades_df["pnl"].values

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
# strategy_score
# =========================================================
def normalize(x, min_v, max_v):

    if pd.isna(x):
        return 0

    if max_v == min_v:
        return 0

    x = max(min_v, min(x, max_v))

    return (x - min_v) / (max_v - min_v)

def calc_score_row(row):

    cagr_score = normalize(row.get("cagr", 0), 0, 30) * 100
    sharpe_score = normalize(row.get("sharpe_ratio", 0), 0, 2) * 100
    dd_score = (1 - normalize(row.get("max_drawdown", 0), 0, 50)) * 100
    pf_score = normalize(row.get("profit_factor", 0), 1, 3) * 100
    win_score = normalize(row.get("win_rate", 0), 30, 80) * 100
    trade_score = normalize(min(row.get("trade_count", 0), 100), 10, 100) * 100

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
# window_label生成
# =========================================================
def generate_windows(df=None, end_date=None):
    """生成累积扩展window_label：起点固定，终点按月步长递增。
       所有股票使用相同的 end_date 以保证window_label一致。
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

    # 追加一个完整步长window_label，替代非整年兜底
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
            "code": df["code"].iloc[0],
            "market": market_val,
            "ktype": BAR_INTERVAL,
            "ma": ma_len,

            "total_return": 0,
            "cagr": 0,
            "buy_hold_return": 0,
            "excess_return": 0,
            "avg_trade_return": 0,

            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "calmar_ratio": 0,

            "trade_count": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "payoff_ratio": 0,

            "avg_win": 0,
            "avg_win_pct": 0,
            "avg_loss": 0,
            "avg_loss_pct": 0,
            "max_win": 0,
            "max_loss": 0,

            "max_win_streak": 0,
            "max_loss_streak": 0,

            "avg_hold_days": 0,
            "avg_hold_bars": 0,

            "initial_cash": INITIAL_CASH,
            "final_cash": INITIAL_CASH,

        }

    ret = (final / INITIAL_CASH - 1) * 100
    years = max((end - start).days / 365, 1 / 365)
    cagr = ((final / INITIAL_CASH) ** (1 / years) - 1) * 100

    buy_hold = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    alpha = ret - buy_hold

    mdd = max_drawdown(curve) * 100
    sh = sharpe_from_equity(curve)
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    win = trades_df[trades_df["pnl"] > 0]
    loss = trades_df[trades_df["pnl"] < 0]

    win_rate = len(win) / len(trades_df) * 100
    pf = win["pnl"].sum() / abs(loss["pnl"].sum()) if len(loss) else 0

    avg_win = win["pnl"].mean() if len(win) else 0
    avg_loss = abs(loss["pnl"].mean()) if len(loss) else 0

    avg_win_pct = win["return_pct"].mean() if len(win) else 0
    avg_loss_pct = loss["return_pct"].mean() if len(loss) else 0

    payoff = avg_win / avg_loss if avg_loss else 0

    max_win_trade = trades_df["pnl"].max()
    max_loss_trade = trades_df["pnl"].min()

    max_win_streak, max_loss_streak = streaks(trades_df)

    avg_hold = trades_df["hold_days"].mean()
    avg_hold_bars = trades_df["hold_bars"].mean()

    return {
        "code": df["code"].iloc[0],
        "market": market_val,
        "ktype": BAR_INTERVAL,
        "ma": ma_len,

        "total_return": float(ret),
        "cagr": float(cagr),
        "buy_hold_return": float(buy_hold),
        "excess_return": float(alpha),
        "avg_trade_return": float(trades_df["return_pct"].mean()) if not trades_df.empty else 0,

        "max_drawdown": float(mdd),
        "sharpe_ratio": float(sh),
        "calmar_ratio": float(calmar),

        "trade_count": int(len(trades_df)),
        "win_rate": float(win_rate),
        "profit_factor": float(pf),
        "payoff_ratio": float(payoff),

        "avg_win": float(avg_win),
        "avg_win_pct": float(avg_win_pct),
        "avg_loss": float(avg_loss),
        "avg_loss_pct": float(avg_loss_pct),
        "max_win": float(max_win_trade),
        "max_loss": float(max_loss_trade),

        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,

        "avg_hold_days": float(avg_hold),
        "avg_hold_bars": float(avg_hold_bars),

        "initial_cash": INITIAL_CASH,
        "final_cash": float(final),

    }

# =========================================================
# 列排序
# =========================================================
COLUMN_ORDER = [
    "code", "market", "ktype", "ma", "strategy_score"
]

END_COLUMNS = ["window_label", "effective_range"]

def reorder_columns(df):
    cols = df.columns.tolist()
    ordered = [c for c in COLUMN_ORDER if c in cols]
    rest = [c for c in cols if c not in COLUMN_ORDER and c not in END_COLUMNS]
    end = [c for c in END_COLUMNS if c in cols]
    return df[ordered + rest + end]

# =========================================================
# stability分析
# =========================================================
def build_window_stability(summary_rows):
    """对每个累积window_label阶段计算stability（同 calc_param_stability 逻辑，按阶段展开）。

    输入：某只股票所有window_label的汇总行。
    输出：每行 = (ma, window_label) 的稳定性指标，
          排序 = code ↑ | window_label ↑ | ma ↑
    """
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return pd.DataFrame()

    # 排除评分全为0的window_label
    valid = df.groupby("window_label")["strategy_score"].transform("max") > 0
    df = df[valid]
    if df.empty:
        return pd.DataFrame()

    windows = sorted(df["window_label"].unique())
    if not windows:
        return pd.DataFrame()

    # 每window_label内按strategy_score排名
    df["window_rank"] = df.groupby(["code", "window_label"])["strategy_score"].rank(ascending=False, method="min")

    code_val = df["code"].iloc[0]
    bar_val = df["ktype"].iloc[0]

    def _norm(series, higher_is_better=True):
        lo, hi = series.min(), series.max()
        if hi == lo:
            return pd.Series(0.5, index=series.index)
        return (series - lo) / (hi - lo) if higher_is_better else (hi - series) / (hi - lo)

    all_stages = []
    for i, w in enumerate(windows):
        stage_df = df[df["window_label"].isin(windows[:i + 1])]

        stats = stage_df.groupby("ma").agg(
            window_count=("window_label", "nunique"),
            win_window_count=("cagr", lambda x: (x > 0).sum()),
            avg_score_rank=("window_rank", "mean"),
            rank_first_count=("window_rank", lambda x: (x == 1).sum()),
            rank_top3_pct=("window_rank", lambda x: (x <= 3).sum() / max(len(x), 1) * 100),
            score_rank_std=("window_rank", "std"),
            avg_cagr=("cagr", "mean"),
            cagr_std=("cagr", "std"),
        ).reset_index()

        stats["win_window_pct"] = stats["win_window_count"] / stats["window_count"] * 100
        stats["cagr_std"] = stats["cagr_std"].fillna(0)
        stats["score_rank_std"] = stats["score_rank_std"].fillna(0)

        # 归一化加权评分
        avg_rank_n = _norm(stats["avg_score_rank"], higher_is_better=False)
        top3_n = _norm(stats["rank_top3_pct"], higher_is_better=True)
        std_n = _norm(stats["score_rank_std"], higher_is_better=False)
        cagr_n = _norm(stats["avg_cagr"], higher_is_better=True)
        win_rate_n = _norm(stats["win_window_pct"], higher_is_better=True)
        cagr_std_n = _norm(stats["cagr_std"], higher_is_better=False)

        stats["stability_score"] = (
            0.20 * avg_rank_n + 0.10 * top3_n + 0.15 * std_n
            + 0.25 * cagr_n + 0.20 * win_rate_n + 0.10 * cagr_std_n
        )

        stats.insert(0, "window_label", w)
        stats.insert(0, "ktype", bar_val)
        stats.insert(0, "code", code_val)
        all_stages.append(stats)

    result = pd.concat(all_stages, ignore_index=True)
    col_order = [
        "code", "ktype", "ma", "window_label", "window_count",
        "win_window_count", "win_window_pct",
        "avg_cagr", "cagr_std",
        "rank_top3_pct", "avg_score_rank",
        "score_rank_std", "rank_first_count",
        "stability_score", "is_best",
    ]
    result = result.reindex(columns=col_order)
    result = result.sort_values(["code", "window_label", "ma"]).reset_index(drop=True)

    # 标记每股票每window_label的最优参数（不改变已有排序）
    result["is_best"] = ""
    idx = (
        result
        .sort_values(["stability_score", "score_rank_std", "avg_cagr"],
                      ascending=[False, True, False])
        .groupby(["code", "window_label"], sort=False)
        .head(1)
        .index
    )
    result.loc[idx, "is_best"] = "最优"
    return result


# =========================================================
# 评分矩阵（明细+排名）
# =========================================================
def build_score_matrix(summary_rows):
    """从window_label回测汇总行构建评分明细和排名透视表"""
    df = pd.DataFrame(summary_rows)

    # 排除评分全为0的window_label
    valid = df.groupby("window_label")["strategy_score"].transform("max") > 0
    df = df[valid]

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 评分明细透视
    score_pivot = df.pivot_table(
        index=["code", "ktype", "ma"],
        columns="window_label",
        values="strategy_score",
        aggfunc="first"
    )
    score_pivot = score_pivot[sorted(score_pivot.columns)]
    score_pivot = score_pivot.reset_index()

    # 评分排名（每股票每window_label内独立排名）
    df["window_rank"] = df.groupby(["code", "window_label"])["strategy_score"].rank(ascending=False, method="min")

    rank_pivot = df.pivot_table(
        index=["code", "ktype", "ma"],
        columns="window_label",
        values="window_rank",
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
    if df.empty or df["window_label"].nunique() < 2:
        return None
    if metric_name not in df.columns:
        return None

    pivot = df.pivot_table(index="ma", columns="window_label", values=metric_name, aggfunc="first")
    pivot = pivot[sorted(pivot.columns, key=lambda c: str(c))]
    if pivot.empty:
        return None

    annot_text = [[f"{v:.1f}" if pd.notna(v) else "" for v in row] for row in pivot.values]
    # 每列第1名★标记
    for col_idx, col_name in enumerate(pivot.columns):
        col_data = pivot[col_name].dropna()
        if col_data.empty:
            continue
        ranked = col_data.sort_values() if metric_name == "max_drawdown" else col_data.sort_values(ascending=False)
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
        hovertemplate="window_label: %{x}<br>均线: %{y}<br>值: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{code} {stock_name} {metric_title} 参数扫描热力图", font=dict(size=15)),
        xaxis=dict(title="回测window_label", tickangle=45),
        yaxis=dict(title="ma", dtick=1),
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
    grouped = df.groupby("ma")["strategy_score"].agg(["mean", "std"]).dropna()
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
        xaxis=dict(title="ma"), yaxis=dict(title="strategy_score"),
        height=420, width=850, margin=dict(l=60, r=40, t=60, b=60),
        paper_bgcolor="white", showlegend=False,
    )
    return fig


def _build_stability_figure(code, ws_df, best_ma=None, stock_name=""):
    """构建全window_labelstability热力图，返回 go.Figure 或 None。"""
    if ws_df is None or ws_df.empty:
        return None
    pivot = ws_df.pivot_table(index="ma", columns="window_label", values="stability_score", aggfunc="first")
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
        hovertemplate="window_label: %{x}<br>均线: %{y}<br>稳定性评分: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{code} {stock_name} 全window_labelstability热力图", font=dict(size=15)),
        xaxis=dict(title="window_label", tickangle=45),
        yaxis=dict(title="ma", dtick=1),
        height=max(500, len(pivot.index) * 26), width=max(700, len(pivot.columns) * 110),
        margin=dict(l=80, r=40, t=80, b=80), paper_bgcolor="white",
    )
    return fig


def generate_all_stock_best_ma_heatmap(all_ws, save_dir="heatmaps", return_fig=False):
    """生成全股票各window_label最优参数热力图（Plotly HTML 版本）。
    行=code, 列=window_label, 值=最优ma。
    末尾3列为股性指标（灰色背景不上色）。
    return_fig=True 时返回 go.Figure，不写文件。
    """
    if all_ws is None or all_ws.empty:
        return None if return_fig else None

    best = all_ws[all_ws["is_best"] == "最优"].copy()
    if best.empty:
        return

    pivot = best.pivot_table(
        index="code", columns="window_label", values="ma", aggfunc="first"
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

    extra_cols = ["最优param_changes", "param_change_std", "personality_score"]
    extra_data = pd.DataFrame({
        "最优param_changes": changes_col,
        "param_change_std": std_col.round(2),
        "personality_score": score_col,
    }, index=pivot.index)

    n_stocks, n_windows = pivot.shape
    n_extra = len(extra_cols)

    # 合并 z：window_label列用原始值，附加列填 0（通过色阶映射为灰色）
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
            if col_name == "最优param_changes":
                row.append(f"{int(v)}" if pd.notna(v) else "")
            elif col_name == "param_change_std":
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
        hovertemplate="股票: %{y}<br>window_label: %{x}<br>值: %{text}<extra></extra>",
    ))

    # 附加列与window_label列的竖分隔线
    fig.add_shape(type="line",
        x0=n_windows - 0.5, x1=n_windows - 0.5,
        y0=-0.5, y1=n_stocks - 0.5,
        line=dict(color="#999999", width=2),
    )

    fig.update_layout(
        title=dict(text="全股票各window_labelparam_change", font=dict(size=16)),
        xaxis=dict(tickangle=45),
        yaxis=dict(title="code"),
        height=max(400, n_stocks * 22),
        width=max(800, n_windows * 90 + n_extra * 100),
        margin=dict(l=120, r=60, t=80, b=120),
        paper_bgcolor="white",
    )

    if return_fig:
        return fig
    _p = os.path.join(save_dir, f"all_全股票各window_labelparam_change热力图{FILE_SUFFIX}.html")
    fig.write_html(_p, include_plotlyjs="cdn", config={"displayModeBar": False})
    print(f"全股票最优参数热力图: {_p}")


# =========================================================

def _sanitize_json_for_html(data_json):
    """净化 JSON 字符串，确保安全嵌入 HTML script 标签。"""
    # 防止 </script> 提前关闭 script 标签
    return data_json.replace("</script>", "<\\/script>")


# =========================================================
# 统一热力图看板（全周期 × 全market）
# =========================================================

def generate_heatmap_dashboard(cache_data):
    """在所有回测完成后生成统一热力图看板，覆盖所有已跑的 ktype × market 组合。

    cache_data: {BAR_INTERVAL: {market: [(code, rows, ws_df, best_ma, name), ...]}}
    输出: {TRADE_DIR}/统一热力图看板.html
    """
    if not cache_data:
        return

    METRIC_CONFIG = [
        ("策略评分", "strategy_score", "RdYlGn", True),
        ("年化收益率", "cagr", "RdYlGn", True),
        ("夏普比率", "sharpe_ratio", "RdYlGn", True),
        ("最大回撤", "max_drawdown", "OrRd", False),
    ]
    SENSITIVITY_KEY = "参数敏感性分析"
    STABILITY_KEY = "参数稳定性评分"

    ALL_KTYPES = ["1W", "1D"]
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
            _kt_lbl = {"1W":"周K","1D":"日K"}.get(_bi,_bi)
            for code, scan_rows, ws_df, best_ma, stock_name, sig_map in tqdm(entries, desc=f"  看板({_kt_lbl},{_mkt_id.upper()})", unit="stock"):
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

                # ── K线图（LightweightCharts）──
                _bt_close = _bt_buy = _bt_sell = None
                if best_ma is not None:
                    _db_tbl = "klines_" + {"1W": "1w", "1D": "1d"}.get(_bi, "")
                    _df_k = pd.DataFrame()
                    _con = _get_db_conn()
                    if _con is not None:
                        try:
                            _df_k = _con.execute(
                                f'SELECT * FROM {_db_tbl} WHERE code = ? ORDER BY datetime', [code]
                            ).fetchdf()
                        except Exception:
                            pass
                    try:
                        _ha_close_k = (_df_k["open"] + _df_k["high"] + _df_k["low"] + _df_k["close"]) / 4
                        _ma_k = _ha_close_k.rolling(best_ma, min_periods=best_ma).mean()
                        _ma_diff = _ma_k.diff().fillna(0)
                        _buy_k = (_ma_diff > 0) & (_ma_diff.shift(1) <= 0)
                        _sell_k = (_ma_diff < 0) & (_ma_diff.shift(1) >= 0)
                        _bt_close = _df_k["close"].values.astype(np.float64)
                        _bt_open = _df_k["open"].values.astype(np.float64)
                        _bt_buy = _buy_k.values.astype(np.bool_)
                        _bt_sell = _sell_k.values.astype(np.bool_)
                        _bt_datetime = _df_k["datetime"].values

                        _candles = []
                        _ma_list = []
                        _signals_lst = []
                        _use_date = True
                        for _i in range(len(_df_k)):
                            _dt = pd.to_datetime(_df_k["datetime"].iloc[_i])
                            _t = _dt.strftime("%Y-%m-%d") if _use_date else str(int(_dt.timestamp()))
                            _candles.append({"time": _t, "open": float(_df_k["open"].iloc[_i]),
                                             "high": float(_df_k["high"].iloc[_i]),
                                             "low": float(_df_k["low"].iloc[_i]),
                                             "close": float(_df_k["close"].iloc[_i])})
                            _mv = _ma_k.iloc[_i]
                            if pd.notna(_mv):
                                _ma_list.append({"time": _t, "value": round(float(_mv), 2)})
                            if _buy_k.iloc[_i]:
                                _signals_lst.append({"time": _t, "position": "atPriceMiddle",
                                                     "price": float(_df_k["close"].iloc[_i]),
                                                     "color": "#26a69a", "shape": "circle", "text": "买入"})
                            if _sell_k.iloc[_i]:
                                _signals_lst.append({"time": _t, "position": "atPriceMiddle",
                                                     "price": float(_df_k["close"].iloc[_i]),
                                                     "color": "#ef5350", "shape": "circle", "text": "卖出"})
                        _lwc_figs = {
                            "_lwc": True,
                            "candles": _candles,
                            "mas": _ma_list,
                            "signals": _signals_lst,
                            "best_ma": int(best_ma) if best_ma is not None else 0,
                        }
                    except Exception:
                        _lwc_figs = None

                    # 资金曲线 + 交易明细
                    if _lwc_figs and _bt_close is not None:
                        try:
                            _tm = 1 if TRADE_MODE == "open" else 0
                            _trades_arr, _equity_arr, _n_tr = _numba_backtest(
                                _bt_close, _bt_buy, _bt_sell, INITIAL_CASH, FEE_RATE,
                                slippage=SLIPPAGE, trade_mode=_tm, open_arr=_bt_open,
                            )
                            _eq_times = [_c["time"] for _c in _candles]
                            _lwc_figs["equity"] = [
                                {"time": _eq_times[_i], "value": round(float(_equity_arr[_i]), 2)}
                                for _i in range(_n_tr)
                                if _equity_arr[_i] > 0
                            ]
                            # 交易明细
                            _tdf, _ = _build_trades_from_arrays(
                                "", "", _bt_datetime, int(best_ma),
                                _trades_arr, _equity_arr, _n_tr,
                            )
                            if not _tdf.empty:
                                _trade_cols = ["entry_time", "entry_price", "buy_shares",
                                               "exit_time", "exit_price", "sell_shares",
                                               "trade_status", "pnl_type",
                                               "pnl", "return_pct",
                                               "buy_fee", "sell_fee",
                                               "cash_before_entry", "cash_after_entry",
                                               "cash_before_exit", "cash_after_exit",
                                               "hold_bars"]
                                _trade_df = _tdf[[c for c in _trade_cols if c in _tdf.columns]].copy()
                                for _tc in ["entry_time", "exit_time"]:
                                    if _tc in _trade_df.columns:
                                        _trade_df[_tc] = _trade_df[_tc].apply(
                                            lambda x: str(pd.Timestamp(x).date()) if pd.notna(x) else "")
                                _lwc_figs["trade_table"] = _trade_df.to_dict(orient="records")
                                for _i, _r in enumerate(_lwc_figs["trade_table"]):
                                    _r["订单ID"] = _i + 1
                                _lwc_figs["trade_table"].reverse()
                        except Exception:
                            pass

                    # 从 scan_rows 提取 best_ma 的回测指标（最后一个window_label，最优 MA 来自倒数第二个window_label的稳定性分析）
                    if _lwc_figs:
                        _metrics = {}
                        if scan_rows and best_ma is not None:
                            try:
                                _bm = int(best_ma)
                                _windows = sorted(set(r.get("window_label", "") for r in scan_rows if r.get("window_label")))
                                _target_w = _windows[-1] if _windows else None
                                _matched = [r for r in scan_rows
                                            if r.get("window_label") == _target_w and
                                            r.get("ma") is not None and
                                            int(r["ma"]) == _bm]
                                if _matched:
                                    _row = _matched[0]
                                    for _k in ["total_return", "cagr", "buy_hold_return", "excess_return",
                                               "max_drawdown", "trade_count", "win_rate", "profit_factor", "payoff_ratio",
                                               "sharpe_ratio", "calmar_ratio", "avg_trade_return"]:
                                        if _k in _row and _row[_k] is not None:
                                            _metrics[_k] = round(float(_row[_k]), 4)
                            except Exception:
                                pass
                        _lwc_figs["metrics"] = _metrics
                        figs["K线图"] = _lwc_figs
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
    # 复制 lwc.js 到看板同目录
    import shutil
    _lwc_src = os.path.join(os.path.dirname(__file__), "lwc.js")
    if os.path.exists(_lwc_src):
        shutil.copy2(_lwc_src, os.path.join(TRADE_DIR, "lwc.js"))
    print(f"统一热力图看板: {_p}")


def _build_heatmap_dashboard_html(data_json, timestamp=""):
    """生成包含 ktype + market 双导航栏的统一热力图看板 HTML。"""
    _tmpl = os.path.join(os.path.dirname(__file__), "heatmap_dashboard_template.html")
    with open(_tmpl, "r", encoding="utf-8") as _f:
        return _f.read().replace("__DATA__", data_json).replace("__TIMESTAMP__", timestamp)


# =========================================================
# 主程序
# =========================================================
def _process_one_stock(code, windows=None, ktype=None, ma_mode="continuous",
                       trade_mode="close", slippage=0.0):
    """Process a single stock. Returns (all_rows, stability_dfs, signal_map)."""
    global TRADE_MODE, SLIPPAGE
    TRADE_MODE = trade_mode
    SLIPPAGE = slippage
    if ktype:
        _setup_ktype(ktype, ma_mode)
    df = _load_data(code)
    if df is None:
        return [], [], {}, None, None
    df["code"] = code
    out_file = os.path.join(TRADE_DIR, TRADE_SUBDIR, _market_subdir(code), f"{code}_trades.xlsx")
    stock_name = str(df["stock_name"].iloc[0]) if "stock_name" in df.columns else ""
    stock_plates = str(df["plates"].iloc[0]) if "plates" in df.columns else ""
    stock_all_rows = []
    stock_stability_dfs = []

    # =============================================
    # 累积扩展window_label回测（覆盖 MA_LIST 全部 60 个参数）
    # =============================================
    if windows is None:
        windows = generate_windows(df)
    window_trades_by_ma = {ma: [] for ma in MA_LIST}
    window_summary_rows = []

    # 预计算 HA 和滚动均线（所有window_label起点相同，全量数据一次算完）
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

        # 每window_label预计算一次（各 MA 共用）
        close_w_arr = df_w["close"].values.astype(np.float64)
        open_w_arr = df_w["open"].values.astype(np.float64)
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

            _tm = 1 if TRADE_MODE == "open" else 0
            trades_arr, equity_arr, n_trades = _numba_backtest(
                close_w_arr, buy_arr, sell_arr, INITIAL_CASH, FEE_RATE,
                slippage=SLIPPAGE, trade_mode=_tm, open_arr=open_w_arr
            )

            trades, _ = _build_trades_from_arrays(
                code_val, market_val, datetime_w_arr, ma,
                trades_arr, equity_arr, n_trades
            )

            if not trades.empty:
                trades["window_label"] = window_label
                trades["effective_range"] = effective_range
                window_trades_by_ma[ma].append(trades)

            summary = build_summary(trades, ma, df_w, equity_arr=equity_arr)
            summary["window_label"] = window_label
            summary["effective_range"] = effective_range
            summary["total_bars"] = len(df_w)
            summary["strategy_score"] = calc_score_row(summary)

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
            _windows = sorted(window_stability_df["window_label"].unique())
            _target = _windows[-2] if len(_windows) >= 2 else _windows[-1]
            last_ws_df = window_stability_df[window_stability_df["window_label"] == _target]
            best_stab_ma = (
                last_ws_df
                .sort_values(["stability_score", "score_rank_std"], ascending=[False, True])
                .iloc[0]["ma"]
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
        signal_info["ktype"] = BAR_INTERVAL
        signal_info["ma"] = ma
        signal_info["code"] = code
        signal_info["stock_name"] = stock_name
        signal_info["plates"] = stock_plates
        _turnover_rank = _top_turnover_map.get(code)
        signal_info["is_top200"] = "是" if _turnover_rank else "否"
        signal_info["market_rank"] = _turnover_rank
        signal_info["market"] = market_val
        signal_map[ma] = signal_info

    return stock_all_rows, stock_stability_dfs, signal_map, window_stability_df, out_file, stock_name, best_stab_ma


def _round_display(df, pct_cols=None):
    """输出前统一处理浮点列：
    - 百分比字段 ÷100 → Excel 原生百分比格式
    - 其余 float 列保留 2 位小数
    - stability_score保留 3 位小数
    不影响原始计算精度。
    """
    df = df.copy()
    pct_set = set(pct_cols or [])
    for col in df.select_dtypes(include=["float", "float64"]).columns:
        if col in pct_set:
            df[col] = (df[col] / 100.0).round(4)
        elif col == "stability_score":
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

    # 取最后一个window_label做评分查询（最优 MA 来自倒数第二个window_label的稳定性分析）
    last_df = None
    if all_df is not None and not all_df.empty and "window_label" in all_df.columns:
        _wins = sorted(all_df["window_label"].unique())
        if _wins:
            last_df = all_df[all_df["window_label"] == _wins[-1]]

    # 回测指标字段（2.py 参考清单）
    BT_COLS = ["strategy_score", "total_return", "cagr", "buy_hold_return", "excess_return", "avg_trade_return",
               "max_drawdown", "sharpe_ratio", "calmar_ratio",
               "trade_count", "win_rate", "profit_factor", "payoff_ratio",
               "avg_win", "avg_win_pct", "avg_loss", "avg_loss_pct", "max_win", "max_loss",
               "max_win_streak", "max_loss_streak", "avg_hold_bars", "avg_hold_days",
               "initial_cash", "final_cash", "window_label", "effective_range"]

    rows = []
    for code, sig_map in signal_maps.items():
        # 确定最优MA
        best_ma = None
        if stab_best is not None and not stab_best.empty:
            _m = stab_best[stab_best["code"] == code]
            if not _m.empty:
                best_ma = int(_m.iloc[0]["ma"])

        sig = None
        if best_ma is not None and best_ma in sig_map:
            sig = sig_map[best_ma]
        elif sig_map:
            if last_df is not None:
                _sr = last_df[last_df["code"] == code]
                if not _sr.empty and "strategy_score" in _sr.columns:
                    _best_idx = _sr["strategy_score"].idxmax()
                    best_ma = int(_sr.loc[_best_idx, "ma"])
                    sig = sig_map.get(best_ma)
            if sig is None:
                best_ma = list(sig_map.keys())[0]
                sig = sig_map[best_ma]

        if sig is None:
            continue

        # 从 signal_info 复制所有信号字段，再覆盖回测指标
        row = dict(sig)
        if last_df is not None and "ma" in last_df.columns:
            _mask = (last_df["code"] == code) & (last_df["ma"] == best_ma)
            _match = last_df[_mask]
            if not _match.empty:
                _r = _match.iloc[0]
                for col in BT_COLS:
                    if col in _r:
                        row[col] = _r[col]
                if "total_bars" in _r:
                    row["total_bars"] = int(_r["total_bars"])

        # 均线趋势共振（2.py 原版逻辑）
        trend_lookup = {ma: info.get("direction") for ma, info in sig_map.items()}
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

        # est_hold_progress = bars_since_last_signal / avg_hold_bars（仅多头有效）
        _elapsed = row.get("bars_since_last_signal")
        _avg_hold = row.get("avg_hold_bars")
        if row.get("direction") == 1 and _elapsed is not None and _avg_hold is not None and _avg_hold > 0:
            row["est_hold_progress"] = round(min(_elapsed / _avg_hold * 100, 100), 2)
        else:
            row["est_hold_progress"] = None

        # est_gain = change_since_last_signal / avg_trade_return（仅多头且total_return > 0）
        _hist_change = row.get("change_since_last_signal")
        _avg_trade_ret = row.get("avg_trade_return")
        if row.get("direction") == 1 and _hist_change is not None and _avg_trade_ret is not None and _avg_trade_ret > 0:
            row["est_gain"] = round(min(_hist_change / _avg_trade_ret * 100, 100), 2)
        else:
            row["est_gain"] = None

        # total_bars（由 _process_one_stock 预计算，上方已从 last_df 复制）

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    signal_df = pd.DataFrame(rows)

    # 排序：多头在前（已过K线数升序、strategy_score降序），空头在后
    bull = signal_df[signal_df.get("direction", pd.Series(-1, index=signal_df.index)) == 1].sort_values(
        ["bars_since_last_signal", "strategy_score"], ascending=[True, False]
    )
    bear = signal_df[signal_df.get("direction", pd.Series(-1, index=signal_df.index)) != 1].sort_values(
        ["bars_since_last_signal", "strategy_score"], ascending=[True, False]
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
    signal_df["策略表现"] = signal_df.get("strategy_score", pd.Series(float("nan"))).apply(_tier)

    apply_cn_mapping(signal_df)
    return signal_df


SIGNAL_COLS = [
    "code", "stock_name", "plates", "market", "is_top200", "market_rank", "ktype", "ma",
    "strategy_score", "策略表现",
    "signal_date", "close", "ha_close", "ha_ma",
    "direction", "signal", "signal_time", "signal_close", "signal_confirmed",
    "last_signal", "last_signal_time", "last_signal_close", "bars_since_last_signal",
    "est_hold_progress",
    "change_since_last_signal", "est_gain",
    "holding_daily_return",
    "均线趋势共振方向", "共振均线数量", "共振均线列表",
    "total_return", "cagr", "buy_hold_return", "excess_return", "avg_trade_return",
    "max_drawdown", "sharpe_ratio", "calmar_ratio",
    "trade_count", "win_rate", "profit_factor", "payoff_ratio",
    "avg_win", "avg_win_pct", "avg_loss", "avg_loss_pct", "max_win", "max_loss",
    "max_win_streak", "max_loss_streak", "avg_hold_bars", "avg_hold_days",
    "initial_cash", "final_cash",
    "window_label", "effective_range", "total_bars",
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

    # 独立 parquet：strategy_score明细 / 排名
    if score_matrix is not None and not score_matrix.empty:
        _round_display(score_matrix).to_parquet(f"{_base}_strategy_score明细.parquet", index=False)
    if rank_matrix is not None and not rank_matrix.empty:
        _round_display(rank_matrix).to_parquet(f"{_base}_strategy_score排名.parquet", index=False)

    # 独立 parquet：全window_labelstability分析
    if window_stability_dfs:
        _ws = pd.concat(window_stability_dfs, ignore_index=True) if isinstance(window_stability_dfs, list) else window_stability_dfs
        _wpct = ["win_window_pct", "avg_cagr"]
        _ws_out = _round_display(_ws, _wpct)
        _ws_out.to_parquet(f"{_base}_全window_labelstability分析.parquet", index=False)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # --- 1. 信号扫描（列顺序匹配 2py）---
        sig_out = signal_df[[c for c in SIGNAL_COLS if c in signal_df.columns]]
        sig_out = _round_display(sig_out, PCT_COLS)
        sig_out.to_excel(writer, sheet_name="信号扫描", index=False)
        _set_pct_format(writer.sheets["信号扫描"], sig_out, PCT_COLS)

        # est_hold_progress列：实心填充数据条
        if "est_hold_progress" in sig_out.columns:
            from openpyxl.formatting.rule import DataBarRule
            from openpyxl.utils import get_column_letter
            _col_letter = get_column_letter(list(sig_out.columns).index("est_hold_progress") + 1)
            _nrows = len(sig_out)
            if _nrows > 0:
                _rule = DataBarRule(start_type="min", end_type="max",
                                    color="5B9BD5",  # 蓝色实心填充
                                    showValue=True,
                                    minLength=None, maxLength=None)
                writer.sheets["信号扫描"].conditional_formatting.add(
                    f"{_col_letter}2:{_col_letter}{_nrows + 1}", _rule
                )

        # est_gain列：绿色实心填充数据条
        if "est_gain" in sig_out.columns:
            from openpyxl.formatting.rule import DataBarRule
            from openpyxl.utils import get_column_letter
            _col_letter = get_column_letter(list(sig_out.columns).index("est_gain") + 1)
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

        # --- 3. 各window_labelparam_change ---
        if window_stability_dfs:
            _ws2 = pd.concat(window_stability_dfs, ignore_index=True) if isinstance(window_stability_dfs, list) else window_stability_dfs
            best_all_ws = _ws2[_ws2["is_best"] == "最优"].copy()
            if not best_all_ws.empty:
                pivot_best_all = best_all_ws.pivot_table(
                    index="code", columns="window_label", values="ma", aggfunc="first"
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
                pivot_best_all["最优param_changes"] = metrics_df.iloc[:, 0]
                pivot_best_all["param_change_std"] = metrics_df.iloc[:, 1].round(2)
                pivot_best_all["personality_score"] = metrics_df.apply(
                    lambda r: round(max(0, 100 - r.iloc[0] * 5 - r.iloc[1] * 2), 1), axis=1
                )
                pivot_best_all.to_excel(writer, sheet_name="各window_labelparam_change")

        # --- 7. 统计逻辑 ---
        logic_rows = [
            # ── Sheet 说明 ──
            {"类型": "Sheet说明", "名称": "信号扫描",
             "统计逻辑": "每只股票用stability最优的ma，显示当前信号（买入/卖出/持有/观察）及该参数在最后一个window_label的回测指标（total_return、max_drawdown、sharpe_ratio等），多头在前空头在后按strategy_score降序排列；均线趋势共振分析检测各周期方向一致性"},
            {"类型": "Sheet说明", "名称": "回测汇总",
             "统计逻辑": "所有股票所有累积window_label所有MA的完整回测结果汇总（独立parquet文件）"},
            {"类型": "Sheet说明", "名称": "strategy_score明细",
             "统计逻辑": "透视表，行=code+ktype+ma，列=window_labelsignal_date区间，值=strategy_score（独立parquet文件）"},
            {"类型": "Sheet说明", "名称": "strategy_score排名",
             "统计逻辑": "透视表，同上结构，值改为window_rank（独立parquet文件）"},
            {"类型": "Sheet说明", "名称": "全window_labelstability分析",
             "统计逻辑": "个股层面，对每个累积window_label阶段计算stability（独立parquet文件）"},
            {"类型": "Sheet说明", "名称": "各window_labelparam_change",
             "统计逻辑": "透视表，行=code，列=window_label，值=ma；选取每只股票每个window_label中stability_score最高的ma，展示最优参数随window_label变化的趋势；末尾3列为股性指标：最优param_changes（相邻window_label间最优参数切换次数）、param_change_std（最优参数的分散程度）、personality_score（固定扣分公式 = max(0, 100 − 变动次数×5 − 标准差×2)，变动越少越稳定得分越高）"},
            {"类型": "", "名称": "", "统计逻辑": ""},
            # ── 评分模型 ──
            {"类型": "评分模型", "名称": "strategy_score",
             "统计逻辑": "Score = 0.30*CAGR + 0.25*Sharpe + 0.20*(1-max_drawdown) + 0.15*profit_factor + 0.05*win_rate + 0.05*trade_count；子指标min-max归一化，CAGR:0-30, Sharpe:0-2, 回撤:0-50, profit_factor:1-3, 盈利率:30-80, trade_count:10-100，加权求和范围0~100"},
            {"类型": "评分模型", "名称": "stability_score",
             "统计逻辑": "Score = 0.20*avg_rank_n + 0.10*top3_n + 0.15*std_n + 0.25*cagr_n + 0.20*win_rate_n + 0.10*cagr_std_n；各指标min-max归一化，排名类占0.45，收益类(均值+占比+标准差)占0.55，越高越稳定；输出保留3位小数"},
            # ── 基本字段 ──
            {"类型": "基本字段", "名称": "code",
             "统计逻辑": "富途格式code，如 US.AAPL / CC.BTC"},
            {"类型": "基本字段", "名称": "stock_name",
             "统计逻辑": "股票中文名称，来源于 parquet 数据文件"},
            {"类型": "基本字段", "名称": "plates",
             "统计逻辑": "股票行业/板块分类，来源于 parquet 数据文件"},
            {"类型": "基本字段", "名称": "is_top200",
             "统计逻辑": "是否在当日成交额前200美股列表中，由9download_uscncc.py --top-turnover生成，用于筛选高流动性标的"},
            {"类型": "基本字段", "名称": "market_rank",
             "统计逻辑": "该股在当日美股成交额中的排名（1~200），仅当is_top200为是时有效"},
            {"类型": "基本字段", "名称": "market",
             "统计逻辑": "US=美股, CN=A股, CC=加密货币"},
            {"类型": "基本字段", "名称": "ktype",
             "统计逻辑": "1W=周K, 1D=日K"},
            {"类型": "基本字段", "名称": "ma",
             "统计逻辑": "stability分析中倒数第二个window_label（仅1个window_label时取唯一window_label）stability_score最高的ma，作为该股票的最优参数，跳过最后一个未完整window_label"},
            # ── 策略表现 ──
            {"类型": "策略表现", "名称": "策略表现",
             "统计逻辑": "strategy_score分档标签：≥80为「1优」，≥60为「2良」，≥40为「3中」，≥20为「4差」，<20为「5劣」"},
            # ── 信号字段 ──
            {"类型": "信号字段", "名称": "signal_date",
             "统计逻辑": "K线signal_date戳，格式 yyyy-MM-dd"},
            {"类型": "信号字段", "名称": "close",
             "统计逻辑": "原始K线close（未复权）"},
            {"类型": "信号字段", "名称": "ha_close",
             "统计逻辑": "Heikin Ashi close = (HA开盘价 + HA最高价 + HA最低价 + ha_close) / 4，平滑后的价格用于计算MA"},
            {"类型": "信号字段", "名称": "ha_ma",
             "统计逻辑": "ha_close的简单移动平均（SMA），周期=全market全window_labelstability分析选出的最优ma"},
            {"类型": "信号字段", "名称": "direction",
             "统计逻辑": "MA值 > MA.shift(1) 为「多头」，否则为「空头」"},
            {"类型": "信号字段", "名称": "signal",
             "统计逻辑": "MA方向变化判断：dir 由 -1→1 为买入，1→-1 为卖出；非信号状态时多头为持有、空头为观察"},
            {"类型": "信号字段", "名称": "signal_time",
             "统计逻辑": "最近一次 BUY/SELL 信号出现的 K 线signal_date"},
            {"类型": "信号字段", "名称": "signal_close",
             "统计逻辑": "signal_time对应的原始close"},
            {"类型": "信号字段", "名称": "signal_confirmed",
             "统计逻辑": "BUY/SELL 信号与最新 K 线同根为「待确认（信号与最新K线同根）」，否则为「已确认」"},
            {"类型": "信号字段", "名称": "last_signal",
             "统计逻辑": "倒数第二次出现的 BUY/SELL 信号方向"},
            {"类型": "信号字段", "名称": "last_signal_time",
             "统计逻辑": "倒数第二次信号出现的 K 线signal_date"},
            {"类型": "信号字段", "名称": "last_signal_close",
             "统计逻辑": "last_signal_time对应的原始close"},
            {"类型": "信号字段", "名称": "bars_since_last_signal",
             "统计逻辑": "last_signal位置到最新K线的 bar 数量"},
            {"类型": "信号字段", "名称": "est_hold_progress",
             "统计逻辑": "仅多头计算 = min(bars_since_last_signal / avg_hold_bars × 100%, 100%)，反映当前持仓占平均持仓周期的进度"},
            {"类型": "信号字段", "名称": "est_gain",
             "统计逻辑": "仅多头且avg_trade_return>0时计算 = min(涨跌幅 / avg_trade_return × 100%, 100%)，反映当前涨幅已实现的平均收益进度"},
            {"类型": "信号字段", "名称": "change_since_last_signal",
             "统计逻辑": "(当前close − last_signal_close) / last_signal_close × 100%"},
            {"类型": "信号字段", "名称": "holding_daily_return",
             "统计逻辑": "自signal以来的日均total_return = 总涨跌幅% / 持有天数；多头趋势直接计算，空头趋势反转符号（做空视角下跌为盈、上涨为亏）"},
            # ── 共振分析 ──
            {"类型": "共振分析", "名称": "均线趋势共振方向",
             "统计逻辑": "统计最优参数以下所有ma方向，全部为多头时标记为「多头共振」，全部为空头时标记为「空头共振」，否则为「无」"},
            {"类型": "共振分析", "名称": "共振均线数量",
             "统计逻辑": "与共振方向一致的ma数量"},
            {"类型": "共振分析", "名称": "共振均线列表",
             "统计逻辑": "与共振方向一致的ma列表"},
            # ── 回测指标 ──
            {"类型": "回测指标", "名称": "total_return",
             "统计逻辑": "最后一个window_label的总total_return = (final_cash − initial_cash) / initial_cash × 100%"},
            {"类型": "回测指标", "名称": "cagr",
             "统计逻辑": "CAGR = (final_cash/initial_cash)^(1/年数) − 1，年数 = window_label实际天数/365"},
            {"类型": "回测指标", "名称": "buy_hold_return",
             "统计逻辑": "同期简单买入持有策略的total_return = (window_label最后close − window_label最初close) / window_label最初close × 100%"},
            {"类型": "回测指标", "名称": "excess_return",
             "统计逻辑": "策略cagr − 买入持有cagr，衡量策略相对基准的超额收益"},
            {"类型": "回测指标", "名称": "avg_trade_return",
             "统计逻辑": "所有交易return_pct的算术平均值 = sum(每笔total_return%) / trade_count，衡量单笔交易的平均收益水平"},
            {"类型": "回测指标", "名称": "max_drawdown",
             "统计逻辑": "资金曲线从峰值到谷底的最大跌幅 = max(1 − 当日资金/当日之前峰值资金) × 100%"},
            {"类型": "回测指标", "名称": "sharpe_ratio",
             "统计逻辑": "Sharpe Ratio = (策略cagr − 无风险利率) / 年化波动率，无风险利率取2%，衡量风险调整后收益；>1为良好，>2为优秀"},
            {"类型": "回测指标", "名称": "calmar_ratio",
             "统计逻辑": "Calmar Ratio = cagr / max_drawdown（绝对值），衡量收益与max_drawdown的比值；越高说明承担单位回撤获取的收益越多"},
            {"类型": "回测指标", "名称": "trade_count",
             "统计逻辑": "回测window_label内的总trade_count（每次买入+卖出算一回合），反映策略活跃度"},
            {"类型": "回测指标", "名称": "win_rate",
             "统计逻辑": "盈利trade_count / 总trade_count × 100%，衡量策略的胜率"},
            {"类型": "回测指标", "名称": "profit_factor",
             "统计逻辑": "总盈利 / 总亏损绝对值；>1表示整体盈利，>2表示盈利能力良好"},
            {"类型": "回测指标", "名称": "payoff_ratio",
             "统计逻辑": "avg_win / avg_loss（绝对值），衡量单次盈利与亏损的比例；>2为良好"},
            {"类型": "回测指标", "名称": "avg_win",
             "统计逻辑": "所有盈利交易的avg_win金额"},
            {"类型": "回测指标", "名称": "avg_win_pct",
             "统计逻辑": "所有盈利交易的平均盈亏百分比"},
            {"类型": "回测指标", "名称": "avg_loss",
             "统计逻辑": "所有亏损交易的avg_loss金额（正数表示）"},
            {"类型": "回测指标", "名称": "avg_loss_pct",
             "统计逻辑": "所有亏损交易的平均盈亏百分比（负数）"},
            {"类型": "回测指标", "名称": "max_win",
             "统计逻辑": "所有盈利交易中最大的一笔盈利金额"},
            {"类型": "回测指标", "名称": "max_loss",
             "统计逻辑": "所有亏损交易中最大的一笔亏损金额（正数表示）"},
            {"类型": "回测指标", "名称": "max_win_streak",
             "统计逻辑": "交易序列中连续盈利的最大次数，反映策略的一致性"},
            {"类型": "回测指标", "名称": "max_loss_streak",
             "统计逻辑": "交易序列中连续亏损的最大次数，反映策略的回撤深度"},
            {"类型": "回测指标", "名称": "avg_hold_days",
             "统计逻辑": "所有交易hold_days的平均值 = 总hold_days / trade_count"},
            {"类型": "回测指标", "名称": "avg_hold_bars",
             "统计逻辑": "所有交易hold_bars的平均值 = 总hold_bars / trade_count"},
            {"类型": "回测指标", "名称": "initial_cash",
             "统计逻辑": "回测起始资金，统一设定为 10,000"},
            {"类型": "回测指标", "名称": "final_cash",
             "统计逻辑": "回测结束后账户总资金 = initial_cash + 累计盈亏"},
            # ── window_label信息 ──
            {"类型": "window_label信息", "名称": "window_label",
             "统计逻辑": "回测window_label的signal_date区间标签，格式 起始日期~结束日期；起始日期为各周期数据的最早日期，步长：周线12个月/日线6个月/60分钟3个月，所有股票共享同一套window_label列表"},
            {"类型": "window_label信息", "名称": "effective_range",
             "统计逻辑": "该window_label实际数据的起止日期区间，格式 起始日期~结束日期；若股票上市晚于window_label起始，起始日期为数据首日"},
            {"类型": "window_label信息", "名称": "total_bars",
             "统计逻辑": "从effective_range解析出的实际天数 = 结束日期 − 起始日期"},
            # ── 参数选择 ──
            {"类型": "参数选择", "名称": "最优ma",
             "统计逻辑": "stability分析中倒数第二个window_label（仅1个window_label时取唯一window_label）stability_score最高的ma，选作信号扫描使用的参数"},
            # ── stability ──
            {"类型": "stability", "名称": "window_count",
             "统计逻辑": "该ma参与计算的window_label总数"},
            {"类型": "stability", "名称": "win_window_pct",
             "统计逻辑": "win_window_count / window_count × 100%"},
            {"类型": "stability", "名称": "win_window_count",
             "统计逻辑": "cagr > 0 的window_count"},
            {"类型": "stability", "名称": "avg_cagr",
             "统计逻辑": "该ma在所有window_label中cagr的算术平均值"},
            {"类型": "stability", "名称": "cagr_std",
             "统计逻辑": "该ma在所有window_label中cagr的标准差，衡量收益波动性"},
            {"类型": "stability", "名称": "avg_score_rank",
             "统计逻辑": "该ma在各window_label中strategy_score排名的算术平均值，越低越好"},
            {"类型": "stability", "名称": "score_rank_std",
             "统计逻辑": "该ma在各window_label中strategy_score排名的标准差，越低越稳定"},
            {"类型": "stability", "名称": "rank_first_count",
             "统计逻辑": "该ma在各window_label中排名第一的次数，衡量夺冠能力"},
            {"类型": "stability", "名称": "stability_score",
             "统计逻辑": "Score = 0.20*avg_rank_n + 0.10*top3_n + 0.15*std_n + 0.25*cagr_n + 0.20*win_rate_n + 0.10*cagr_std_n；各指标min-max归一化，排名类占0.45，收益类(均值+占比+标准差)占0.55，越高越稳定；输出保留3位小数"},
            {"类型": "stability", "名称": "rank_top3_pct",
             "统计逻辑": "该ma在各window_label中排名前三的次数占比"},
            {"类型": "stability", "名称": "is_best",
             "统计逻辑": "每只股票每个window_label内stability_score最高者标记为「最优」，并列时以score_rank_std升序+avg_cagr降序决胜，确保唯一"},
        ]
        pd.DataFrame(logic_rows).to_excel(writer, sheet_name="统计逻辑", index=False)

        for ws in writer.sheets.values():
            _apply_sheet_format(ws)

    print(f"{market_label}汇总Excel: {out_path}")


def run_trade():
    """全流程回测入口。从 watchlist 表加载code（按 --market 过滤）。"""
    import duckdb
    _dbp = os.path.join(os.path.dirname(__file__), "database", "market.duckdb")
    if not os.path.exists(_dbp):
        print("  market.duckdb 不存在，请先运行 download_data_uscncc.py")
        return
    _con = duckdb.connect(_dbp, read_only=True)
    _wl = [str(r[0]) for r in _con.execute("SELECT DISTINCT code FROM watchlist").fetchall()]
    _con.close()
    if not _wl:
        print("  watchlist 为空，请先运行 download_data_uscncc.py")
        return

    # 按 --market 过滤
    _mkt_prefixes = []
    for _m in (getattr(_CLI_ARGS, 'market', 'ALL') or 'ALL').upper().split(","):
        _m = _m.strip()
        if _m in ("ALL", "US"):
            _mkt_prefixes.append("US.")
        if _m in ("ALL", "CN"):
            _mkt_prefixes.extend(("SH.", "SZ."))
        if _m in ("ALL", "CC"):
            _mkt_prefixes.append("CC.")
    symbols = sorted(set(c for c in _wl if any(c.startswith(p) for p in _mkt_prefixes)))

    # 扫描全market数据，取最早和最晚日期作为window_label范围
    available = []
    global_start = pd.Timestamp("2099-12-31")
    global_end = pd.Timestamp("2000-01-01")
    for s in symbols:
        try:
            _df = _load_data(s)
            if _df is None or _df.empty:
                continue
            available.append(s)
            _min = pd.to_datetime(_df["datetime"]).min()
            _max = pd.to_datetime(_df["datetime"]).max()
            if _min < global_start:
                global_start = _min
            if _max > global_end:
                global_end = _max
        except Exception:
            continue
    global WINDOW_START_DATE
    # 按周期步长对齐window_label起点
    _s = {12: 12, 6: 6, 3: 3}.get(STEP_MONTHS, 12)
    _am = ((global_start.month - 1) // _s) * _s + 1
    WINDOW_START_DATE = f"{global_start.year}-{_am:02d}-03"
    windows = generate_windows(end_date=global_end)
    print(f"window_label范围: {WINDOW_START_DATE} ~ {global_end.date()}, 共 {len(windows)} 个window_label")

    # 按market分组
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

        print(f"\n===== 开始回测 {mkt.upper()} market（{len(group)} 只股票）=====")

        market_rows = []
        market_signal_maps = {}
        market_window_stability = []

        _results = {}
        _kt = {"1W": "week", "1D": "day"}.get(BAR_INTERVAL, "week")
        _workers = os.cpu_count() - 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=_workers) as executor:
            future_to_code = {executor.submit(_process_one_stock, code, windows, _kt, MA_MODE, TRADE_MODE, SLIPPAGE): code for code in group}
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
                    _heatmap_cache.append((code, s_all, s_ws, best_ma, stock_name, sig_map))
        tqdm.write(f"  {mkt.upper()} 完成: {_success} 成功")
        if _failed:
            tqdm.write(f"  {mkt.upper()} 失败: {len(_failed)} 只 — {'; '.join(f'{c}({e})' for c, e in _failed)}")

        # ---- 本market汇总 ----
        if not market_rows:
            continue

        market_df = pd.DataFrame(market_rows)

        # 稳定性分析取最优参数
        stab_mkt = None
        if market_window_stability:
            all_ws_mkt = pd.concat(market_window_stability, ignore_index=True)
            _wins_mkt = sorted(all_ws_mkt["window_label"].unique())
            _target = _wins_mkt[-2] if len(_wins_mkt) >= 2 else _wins_mkt[-1]
            last_ws_mkt = all_ws_mkt[all_ws_mkt["window_label"] == _target]
            stab_mkt = (
                last_ws_mkt
                .sort_values(["stability_score", "score_rank_std"], ascending=[False, True])
                .groupby("code", sort=False)
                .head(1)
                .reset_index(drop=True)
            )

        # 本market信号扫描
        signal_mkt = _build_signal_scan(market_df, market_signal_maps, stab_mkt, all_signal_maps)

        # 收集热力图数据到全局缓存（统一看板在回测完成后生成）
        global _HEATMAP_CACHE
        _mkt_entries = [(c, r, w, b, s, m) for c, r, w, b, s, m in _heatmap_cache
                        if _market_subdir(c) == mkt]
        if _mkt_entries:
            _HEATMAP_CACHE.setdefault(BAR_INTERVAL, {})[mkt] = _mkt_entries

        # 累计到全market
        all_rows.extend(market_rows)
        all_signal_maps.update(market_signal_maps)
        window_stability_dfs.extend(market_window_stability)

    # =============================================
    # 全market汇总
    # =============================================
    if not all_rows:
        return

    all_df = pd.DataFrame(all_rows)
    score_all, rank_all = build_score_matrix(all_rows)

    stab_best = None
    if window_stability_dfs:
        all_ws_stab = pd.concat(window_stability_dfs, ignore_index=True)
        _wins = sorted(all_ws_stab["window_label"].unique())
        _target_ws = _wins[-2] if len(_wins) >= 2 else _wins[-1]
        last_ws = all_ws_stab[all_ws_stab["window_label"] == _target_ws]
        stab_best = (
            last_ws
            .sort_values(["stability_score", "score_rank_std"], ascending=[False, True])
            .groupby("code", sort=False)
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
            m = stab_best[stab_best["code"] == code]
            if not m.empty:
                match = int(m.iloc[0]["ma"])
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

    # 全market汇总 Excel
    date_str = pd.Timestamp.today().strftime("%Y%m%d")
    all_out = os.path.join(TRADE_DIR, TRADE_SUBDIR, f"{date_str}_{TRADE_SUBDIR}_信号汇总.xlsx")
    _write_summary_excel(all_out, signal_all, all_df, score_all, rank_all,
                        window_stability_dfs, "全market")
    print(f"全market完成: {all_out}")


# =========================================================
# 富途自选股分组同步
# =========================================================
# 同步和导出功能已移除

# =========================================================
def generate_unified_signal_excel(ktypes_run):
    """读取各周期信号汇总Excel，生成统一信号汇总Excel。

    Sheet1: 信号扫描（合并所有周期，ktype排首位）
    Sheet2: 各周期信号对比（横向对比 + 方向共振统计）
    Sheet3~: {周期}_最优参数变动（每个周期一个sheet）
    SheetN: 统计逻辑
    """
    import glob as _glob
    _kt_dir = {"week": "1w", "day": "1d"}
    _kt_label = {"1w": "1W", "1d": "1D"}
    _kt_order = {"1W": 0, "1D": 1}
    _cmp_label = {"1w": "（周线）", "1d": "（日线）"}
    _kt_dirs_used = [_kt_dir[k] for k in ktypes_run if k in _kt_dir]

    if not _kt_dirs_used:
        return

    # ── 读取各周期数据 ──
    _sig_dfs = {}  # {1w: DataFrame}
    _param_dfs = {}  # {1w: DataFrame}
    _stats_df = None
    for _d in _kt_dirs_used:
        _files = sorted(_glob.glob(os.path.join(TRADE_DIR, _d, "*_信号汇总.xlsx")))
        if not _files:
            continue
        _fp = _files[-1]
        try:
            _df = pd.read_excel(_fp, sheet_name="信号扫描")
            if not _df.empty:
                _df["ktype"] = _kt_label[_d]
                _sig_dfs[_d] = _df
        except Exception:
            pass
        try:
            _pdf = pd.read_excel(_fp, sheet_name="各window_labelparam_change")
            if not _pdf.empty:
                _param_dfs[_d] = _pdf
        except Exception:
            pass
        if _stats_df is None:
            try:
                _stats_df = pd.read_excel(_fp, sheet_name="统计逻辑")
            except Exception:
                pass

    if not _sig_dfs:
        return

    _date_str = pd.Timestamp.today().strftime("%Y%m%d")
    _out_path = os.path.join(TRADE_DIR, f"{_date_str}_各周期信号汇总.xlsx")

    with pd.ExcelWriter(_out_path, engine="openpyxl") as _writer:

        # ── Sheet 1: 信号扫描（合并所有周期）──
        _all_sig = pd.concat(list(_sig_dfs.values()), ignore_index=True, sort=False)
        # 排序：按 1W 的direction→天数→评分排股票，再按 ktype 排个股
        _w1 = _all_sig[_all_sig["ktype"] == "1W"][["code", "direction", "bars_since_last_signal", "strategy_score"]].copy()
        _w1 = _w1.rename(columns={"direction": "_w1_dir", "bars_since_last_signal": "_w1_kbars", "strategy_score": "_w1_score"})
        _w1["_w1_dir"] = _w1["_w1_dir"].map({"多头": 0, "空头": 1}).fillna(1)
        _w1["_w1_kbars"] = _w1["_w1_kbars"].fillna(9999)
        _w1["_w1_score"] = -_w1["_w1_score"].fillna(0)
        _all_sig = _all_sig.merge(_w1, on="code", how="left")
        _all_sig["_w1_dir"] = _all_sig["_w1_dir"].fillna(1)
        _all_sig["_w1_kbars"] = _all_sig["_w1_kbars"].fillna(9999)
        _all_sig["_w1_score"] = _all_sig["_w1_score"].fillna(0)
        _all_sig["_k"] = _all_sig["ktype"].map(_kt_order).fillna(0)
        _all_sig = _all_sig.sort_values(["_w1_dir", "_w1_kbars", "_w1_score", "_k"]).drop(
            columns=["_w1_dir", "_w1_kbars", "_w1_score", "_k"], errors="ignore"
        ).reset_index(drop=True)
        _all_sig.to_excel(_writer, sheet_name="信号扫描", index=False)
        _set_pct_format(_writer.sheets["信号扫描"], _all_sig, PCT_COLS)
        # 日期列格式对齐 YYYY-MM-DD
        from openpyxl.utils import get_column_letter as _cl
        for _date_col in ["signal_date", "signal_time", "last_signal_time"]:
            if _date_col in _all_sig.columns:
                _ws = _writer.sheets["信号扫描"]
                _col_idx = list(_all_sig.columns).index(_date_col) + 1
                for _cell in _ws[_cl(_col_idx)]:
                    if _cell.row > 1 and isinstance(_cell.value, pd.Timestamp):
                        _cell.number_format = "YYYY-MM-DD"

        # 数据条（按各自周期范围）：est_hold_progress（蓝色）、est_gain（绿色）
        _kt_kt_map = {"1W": "1w", "1D": "1d"}
        for _col_name in ["est_hold_progress", "est_gain"]:
            if _col_name in _all_sig.columns:
                from openpyxl.formatting.rule import DataBarRule
                from openpyxl.utils import get_column_letter
                _col_letter = get_column_letter(list(_all_sig.columns).index(_col_name) + 1)
                _color = "5B9BD5" if _col_name == "est_hold_progress" else "70AD47"
                # 按 ktype 分组各自应用数据条范围
                _row = 2  # Excel 首行是表头
                for _kt in [_kt_label[d] for d in _kt_dirs_used]:
                    _count = (_all_sig["ktype"] == _kt).sum()
                    if _count > 0:
                        _end_row = _row + _count - 1
                        _bar_rule = DataBarRule(start_type="num", start_value=0, end_type="num", end_value=1,
                                                color=_color, showValue=True)
                        _writer.sheets["信号扫描"].conditional_formatting.add(
                            f"{_col_letter}{_row}:{_col_letter}{_end_row}", _bar_rule
                        )
                        _row += _count

        # ── Sheet 2: 各周期信号对比 ──
        _base_cols = ["code", "stock_name", "plates", "market",
                      "is_top200", "market_rank"]
        _cmp_fields = ["ma", "strategy_score", "策略表现", "direction", "signal"]

        # 按code合并各周期
        _merged = None
        for _d in _kt_dirs_used:
            _df = _sig_dfs[_d].copy()
            _rename = {c: f"{c}{_cmp_label[_d]}" for c in _df.columns
                       if c not in _base_cols and c != "ktype"}
            _renamed = _df.rename(columns=_rename)
            _keep = _base_cols + [f"{f}{_cmp_label[_d]}" for f in _cmp_fields]
            _keep = [c for c in _keep if c in _renamed.columns]
            _merged_part = _renamed[_keep].set_index("code")
            if _merged is None:
                _merged = _merged_part
            else:
                _merged_part = _merged_part.drop(columns=[c for c in _base_cols if c in _merged_part.columns], errors="ignore")
                _merged = _merged.join(_merged_part, how="outer")

        if _merged is not None:
            _merged = _merged.reset_index()
            # 计算方向共振
            _dir_cols = [f"direction{_cmp_label[d]}" for d in _kt_dirs_used]
            _exist_dir = [c for c in _dir_cols if c in _merged.columns]
            if _exist_dir:
                _merged["多头direction多周期共振数量"] = _merged[_exist_dir].apply(
                    lambda r: (r == "多头").sum(), axis=1)
                _merged["空头direction多周期共振数量"] = _merged[_exist_dir].apply(
                    lambda r: (r == "空头").sum(), axis=1)
                _merged = _merged.sort_values(
                    ["多头direction多周期共振数量", "code"],
                    ascending=[False, True]
                ).reset_index(drop=True)
            # 按字段分组重排列序
            _col_order = [c for c in _base_cols if c in _merged.columns]
            for _f in _cmp_fields:
                for _d in _kt_dirs_used:
                    _col = f"{_f}{_cmp_label[_d]}"
                    if _col in _merged.columns:
                        _col_order.append(_col)
            for _c in ["多头direction多周期共振数量", "空头direction多周期共振数量"]:
                if _c in _merged.columns:
                    _col_order.append(_c)
            _merged = _merged[[c for c in _col_order if c in _merged.columns]]
            _merged.to_excel(_writer, sheet_name="各周期信号对比", index=False)

        # ── Sheet 3~: 各周期最优参数变动 ──
        for _d, _pdf in _param_dfs.items():
            _sheet_name = f"{_d}_最优参数变动"
            _pdf.to_excel(_writer, sheet_name=_sheet_name, index=False)

        # ── 统计逻辑（追加新功能说明）──
        if _stats_df is not None:
            _new_rows = [
                {"类型": "Sheet说明", "名称": "各周期信号对比",
                 "统计逻辑": "横向对比各周期信号：行=code，列=ma/strategy_score/策略表现/direction/signal（加周期后缀）；末尾2列为多头/空头direction多周期共振数量，统计该股票在所有已跑周期中方向一致的个数"},
                {"类型": "基本字段", "名称": "ktype",
                 "统计逻辑": "1W=周K, 1D=日K；在信号扫描sheet中作为第一排序字段，顺序为周K→日K"},
                {"类型": "信号字段", "名称": "多头direction多周期共振数量",
                 "统计逻辑": "该股票在所有已跑周期中direction为「多头」的个数，反映多周期共振强度；仅在各周期信号对比sheet中出现"},
                {"类型": "信号字段", "名称": "空头direction多周期共振数量",
                 "统计逻辑": "该股票在所有已跑周期中direction为「空头」的个数，反映多周期共振强度；仅在各周期信号对比sheet中出现"},
            ]
            _stats_df = pd.concat([_stats_df, pd.DataFrame(_new_rows)], ignore_index=True, sort=False)
            _stats_df.to_excel(_writer, sheet_name="统计逻辑", index=False)

        for _ws in _writer.sheets.values():
            _apply_sheet_format(_ws)

    print(f"各周期信号汇总: {_out_path}")




if __name__ == "__main__":
    import argparse
    import sys

    # 兼容旧版 sync 参数
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        print("sync 模式已移除")
        sys.exit(0)

    parser = argparse.ArgumentParser(description="多window_label参数扫描回测")
    parser.add_argument("--ktype", default=DEFAULT_KTYPE,
                        help=f"ktype: day/周K, week/周K, 逗号拼接如week,day, all=两者全部 (默认: {DEFAULT_KTYPE})")
    parser.add_argument("--market", default=DEFAULT_MARKET,
                        help=f"market: US/CN/CC/US,CN/all (默认: {DEFAULT_MARKET})")
    parser.add_argument("--ma-mode", choices=["continuous", "jump"], default=MA_MODE,
                        help="MA序列类型: continuous=连续回测, jump=跳跃回测(日线step=2,周线强制连续)")
    parser.add_argument("--trade-mode", choices=["close", "open"], default=TRADE_MODE,
                        help="成交方式: close=信号K线收盘价成交, open=下根K线开盘价成交 (默认: close)")
    parser.add_argument("--slippage", type=float, default=SLIPPAGE,
                        help=f"滑点比例 (默认 {SLIPPAGE}, 如 0.001 = 0.1%%)")
    parser.add_argument("--save-cache", action="store_true",
                        help="回测完成后保存缓存到文件，下次可用 --from-cache 跳过回测直接生成看板")
    parser.add_argument("--from-cache", type=str, nargs="?", const="latest", default=None,
                        help="从缓存文件加载数据，跳过回测直接生成看板。指定路径或 latest（自动取最新）")
    _CLI_ARGS = parser.parse_args()
    MA_MODE = _CLI_ARGS.ma_mode
    TRADE_MODE = _CLI_ARGS.trade_mode
    SLIPPAGE = _CLI_ARGS.slippage

    # 解析 ktype 列表（支持逗号拼接）
    _KT_MAP = {"day": "日K", "week": "周K", "all": "全部"}
    _raw = _CLI_ARGS.ktype.lower().replace("，", ",").split(",")
    _ktypes = []
    for _k in _raw:
        _k = _k.strip()
        if _k == "all":
            _ktypes = ["week", "day"]
            break
        if _k in _KT_MAP:
            if _k not in _ktypes:
                _ktypes.append(_k)
    if not _ktypes:
        print(f"错误: 无效的 --ktype '{_CLI_ARGS.ktype}'，可选 week/day/all 或逗号拼接")
        sys.exit(1)

    # 初始设置（用第一个 ktype 初始化全局变量）
    _setup_ktype(_ktypes[0], MA_MODE)

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

    # --from-cache：跳过回测，直接从缓存文件生成看板
    if _CLI_ARGS.from_cache:
        import glob as _g, pickle as _pkl
        if _CLI_ARGS.from_cache == "latest":
            _cache_files = sorted(_g.glob(os.path.join(TRADE_DIR, "heatmap_cache_*.pkl")))
            if not _cache_files:
                print("错误: 未找到缓存文件")
                sys.exit(1)
            _cache_path = _cache_files[-1]
        else:
            _cache_path = _CLI_ARGS.from_cache
        print(f"加载缓存文件: {_cache_path}")
        with open(_cache_path, "rb") as _f:
            _HEATMAP_CACHE = _pkl.load(_f)
        if _HEATMAP_CACHE:
            generate_heatmap_dashboard(_HEATMAP_CACHE)
        sys.exit(0)

    # 统一信号汇总 Excel（合并所有已跑周期的信号 + 对比 + 最优参数）
    generate_unified_signal_excel(_ktypes)

    # 统一热力图看板（覆盖所有已跑的 ktype × market）
    if _HEATMAP_CACHE:
        generate_heatmap_dashboard(_HEATMAP_CACHE)
        # --save-cache：持久化缓存到文件
        if _CLI_ARGS.save_cache:
            import pickle as _pkl
            _cache_path = os.path.join(TRADE_DIR, f"heatmap_cache_{_t.strftime('%Y%m%d_%H%M%S')}.pkl")
            with open(_cache_path, "wb") as _f:
                _pkl.dump(_HEATMAP_CACHE, _f)
            print(f"缓存已保存: {_cache_path}")