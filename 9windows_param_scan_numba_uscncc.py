import os
import warnings
import concurrent.futures
import pandas as pd
import numpy as np
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

# 屏蔽pandas concat空列FutureWarning（不影响功能）
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation", category=FutureWarning)


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

# 按市场创建输出子目录
for _m in ("us", "cn", "cc"):
    os.makedirs(os.path.join(TRADE_DIR, _m), exist_ok=True)
    os.makedirs(os.path.join(TRADE_DIR, _m, "heatmaps"), exist_ok=True)
os.makedirs(os.path.join(TRADE_DIR, "heatmaps"), exist_ok=True)

INITIAL_CASH = 10000
FEE_RATE = 0.001

DEFAULT_KTYPE = "week"     # 默认K线周期: day / week
DEFAULT_MARKET = "US,CC"    # 默认市场: all / US / CN / CC / US,CC

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
    """在 data_uscncc/{us,cn,cc}/ 子目录中查找 parquet 文件"""
    if code.startswith("CC."):
        market = "cc"
    elif code.startswith(("SH.", "SZ.")):
        market = "cn"
    elif code.startswith("US."):
        market = "us"
    else:
        raise ValueError(f"未知代码前缀: {code}")
    path = os.path.join(DATA_DIR, market, f"{code}{FILE_SUFFIX}.parquet")
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
}
BAR_INTERVAL = "1D" if DEFAULT_KTYPE == "day" else "1W"
MA_LIST = KLINE_MAP[BAR_INTERVAL]["ma_range"]
FILE_SUFFIX = KLINE_MAP[BAR_INTERVAL]["suffix"]
TRADING_PERIOD = KLINE_MAP[BAR_INTERVAL]["period"]
STEP_YEARS = 1
WINDOW_START_DATE = "2000-01-03"

# ---- 命令行参数解析（前置，仅在作为主程序运行时生效）----
if __name__ == "__main__":
    import argparse
    import sys

    # 兼容旧版 sync 参数
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        print("sync 模式已移除")
        sys.exit(0)

    parser = argparse.ArgumentParser(description="多窗口参数扫描回测")
    parser.add_argument("--ktype", choices=["day", "week"], default=DEFAULT_KTYPE,
                        help=f"K线周期: day=日K, week=周K (默认: {DEFAULT_KTYPE})")
    parser.add_argument("--market", default=DEFAULT_MARKET,
                        help=f"市场: US/CN/CC/US,CN/all (默认: {DEFAULT_MARKET})")
    _CLI_ARGS = parser.parse_args()

    # 只用改 BAR_INTERVAL，其余从 KLINE_MAP 自动推导
    if _CLI_ARGS.ktype == "week":
        BAR_INTERVAL = "1W"
    else:
        BAR_INTERVAL = "1D"
    MA_LIST = KLINE_MAP[BAR_INTERVAL]["ma_range"]
    FILE_SUFFIX = KLINE_MAP[BAR_INTERVAL]["suffix"]
    TRADING_PERIOD = KLINE_MAP[BAR_INTERVAL]["period"]

# 百分比字段（原始值=百分比数值，如 5.23 表示 5.23%；
# 输出时 ÷100 再设 Excel 单元格格式为 0.00%，实现 Excel 原生百分比显示）
PCT_COLS = [
    "预计持仓进度",
    "距离历史信号收盘价涨跌幅", "持仓日化收益率",
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


def _build_trades_numba(df, ma_len):
    """替换 build_trades + equity_curve 的 Numba 加速版本。

    返回 (trades_df, equity_arr)。
    """
    code_val = df["code"].iloc[0]
    market_val = str(df.get("market", pd.Series([""])).iloc[0]) if "market" in df.columns else ""
    datetime_arr = df["datetime"].values

    close = df["close"].values.astype(np.float64)
    buy = df["buy"].values.astype(np.bool_)
    sell = df["sell"].values.astype(np.bool_)

    trades_arr, equity_arr, n_trades = _numba_backtest(close, buy, sell, INITIAL_CASH, FEE_RATE)

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

    # 最新信号确认
    if signal in ("BUY", "SELL") and (
        (signal == "BUY" and buy_days is not None and buy_days < 5) or
        (signal == "SELL" and sell_days is not None and sell_days < 5)
    ):
        confirm = "待确认，周K未正式收盘"
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


def build_trades(df, ma_len):

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    cash = INITIAL_CASH
    available_cash = cash

    position = 0
    entry_price = 0
    entry_time = None
    entry_index = 0

    trades = []

    backtest_period = f"{df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}"

    # numpy arrays for faster row access
    close_arr = df["close"].values
    buy_arr = df["buy"].values
    sell_arr = df["sell"].values
    datetime_arr = df["datetime"].values
    code_val = df["code"].iloc[0]
    market_val = str(df.get("market", pd.Series([""])).iloc[0]) if "market" in df.columns else ""

    for i in range(len(df)):

        price = float(close_arr[i])

        if pd.isna(price) or price <= 0:
            continue

        time = datetime_arr[i]

        if buy_arr[i] and position == 0:

            cash_before_buy = available_cash

            shares = int(available_cash / (price * (1 + FEE_RATE)))

            if shares <= 0:
                continue

            cost = shares * price
            buy_fee = cost * FEE_RATE

            available_cash -= (cost + buy_fee)
            cash_after_buy = available_cash

            position = shares
            entry_price = price
            entry_time = time
            entry_index = i

        elif sell_arr[i] and position > 0:

            cash_before_sell = available_cash
            sell_value = position * price
            sell_fee = sell_value * FEE_RATE
            available_cash += (sell_value - sell_fee)
            cash_after_sell = available_cash

            trades.append(_make_trade_record(
                code_val, ma_len, entry_price, position, entry_time, entry_index,
                price, time, i,
                cash_before_buy, cash_after_buy, cash_before_sell, cash_after_sell,
                "已平仓", backtest_period, market_val
            ))

            position = 0

    if position > 0:

        price = float(close_arr[-1])
        time = datetime_arr[-1]

        cash_before_sell = available_cash
        sell_value = position * price
        sell_fee = sell_value * FEE_RATE
        available_cash += (sell_value - sell_fee)
        cash_after_sell = available_cash

        trades.append(_make_trade_record(
            code_val, ma_len, entry_price, position, entry_time, entry_index,
            price, time, len(df) - 1,
            None, None, cash_before_sell, cash_after_sell,
            "未平仓(强制结算)", backtest_period, market_val
        ))

    return pd.DataFrame(trades)

# =========================================================
# 资金曲线
# =========================================================
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
# 5年滚动窗口生成
# =========================================================
def generate_windows(df=None, end_date=None):
    """生成累积扩展窗口：起点固定，终点每年步长递增。
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
    cur = start + pd.DateOffset(years=STEP_YEARS)

    while cur < end:
        windows.append((start, cur))
        cur += pd.DateOffset(years=STEP_YEARS)

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

            "最大回撤": 0,
            "夏普比率": 0,
            "卡尔玛比率": 0,

            "交易次数": 0,
            "盈利交易率": 0,
            "盈利因子": 0,
            "盈亏比": 0,

            "平均盈利": 0,
            "平均亏损": 0,
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

        "最大回撤": float(mdd),
        "夏普比率": float(sh),
        "卡尔玛比率": float(calmar),

        "交易次数": int(len(trades_df)),
        "盈利交易率": float(win_rate),
        "盈利因子": float(pf),
        "盈亏比": float(payoff),

        "平均盈利": float(avg_win),
        "平均亏损": float(avg_loss),
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
def calc_param_stability(summary_rows):
    df = pd.DataFrame(summary_rows)

    # 排除所有评分均为0的窗口（无交易窗口）
    valid = df.groupby("窗口")["策略评分"].transform("max") > 0
    df = df[valid]

    if df.empty or df["窗口"].nunique() < 2:
        return pd.DataFrame()

    # 每窗口内按策略评分排名（同分取最小排名）
    df["窗口内排名"] = df.groupby(["股票代码", "窗口"])["策略评分"].rank(ascending=False, method="min")

    # 提取固定信息
    code_val = df["股票代码"].iloc[0]
    bar_val = df["K线周期"].iloc[0]

    # 按均线周期汇总
    stats = df.groupby("均线周期").agg(
        窗口数量=("窗口", "nunique"),
        盈利窗口数量=("年化收益率", lambda x: (x > 0).sum()),
        策略评分排名平均值=("窗口内排名", "mean"),
        策略评分排名第一次数=("窗口内排名", lambda x: (x == 1).sum()),
        策略评分排名Top3占比=("窗口内排名", lambda x: (x <= 3).sum() / max(len(x), 1) * 100),
        策略评分排名标准差=("窗口内排名", "std"),
        年化收益率平均值=("年化收益率", "mean"),
        年化收益率标准差=("年化收益率", "std")
    ).reset_index()

    stats["盈利窗口占比"] = stats["盈利窗口数量"] / stats["窗口数量"] * 100
    stats["年化收益率平均值"] = stats["年化收益率平均值"]
    stats["年化收益率标准差"] = stats["年化收益率标准差"].fillna(0)

    stats["策略评分排名平均值"] = stats["策略评分排名平均值"]
    stats["策略评分排名标准差"] = stats["策略评分排名标准差"].fillna(0)

    # =========================================================
    # 参数稳定性综合评分（加权归一化，越高越好）
    # =========================================================
    # 指标方向：平均值(低→好)、标准差(低→好)、Top3占比(高→好)、第一次数(高→好)
    # 归一化方式：(max - v) / (max - min) 或 (v - min) / (max - min)
    def _norm(series, higher_is_better=True):
        lo, hi = series.min(), series.max()
        if hi == lo:
            return pd.Series(0.5, index=series.index)
        return (series - lo) / (hi - lo) if higher_is_better else (hi - series) / (hi - lo)

    avg_rank_n = _norm(stats["策略评分排名平均值"], higher_is_better=False)
    top3_n = _norm(stats["策略评分排名Top3占比"], higher_is_better=True)
    std_n = _norm(stats["策略评分排名标准差"], higher_is_better=False)
    cagr_n = _norm(stats["年化收益率平均值"], higher_is_better=True)
    win_rate_n = _norm(stats["盈利窗口占比"], higher_is_better=True)
    cagr_std_n = _norm(stats["年化收益率标准差"], higher_is_better=False)

    stats["参数稳定性综合评分"] = (
        0.20 * avg_rank_n + 0.10 * top3_n + 0.15 * std_n +
        0.25 * cagr_n + 0.20 * win_rate_n + 0.10 * cagr_std_n
    )

    # 排序后重排列顺序
    col_order = ["股票代码", "K线周期", "均线周期", "窗口数量",
                 "盈利窗口数量", "盈利窗口占比",
                 "年化收益率平均值", "年化收益率标准差",
                 "策略评分排名Top3占比", "策略评分排名平均值",
                 "策略评分排名标准差", "策略评分排名第一次数",
                 "参数稳定性综合评分", "是否最优"]
    stats.insert(0, "K线周期", bar_val)
    stats.insert(0, "股票代码", code_val)

    stats = stats.reindex(columns=col_order)

    # 标记最优参数（每股票一个最优）
    stats["是否最优"] = ""
    idx = (
        stats
        .sort_values(["参数稳定性综合评分", "策略评分排名标准差", "年化收益率平均值"],
                      ascending=[False, True, False])
        .head(1)
        .index
    )
    stats.loc[idx, "是否最优"] = "最优"
    return stats



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

        stats["参数稳定性综合评分"] = (
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
        "参数稳定性综合评分", "是否最优",
    ]
    result = result.reindex(columns=col_order)
    result = result.sort_values(["股票代码", "窗口", "均线周期"]).reset_index(drop=True)

    # 标记每股票每窗口的最优参数（不改变已有排序）
    result["是否最优"] = ""
    idx = (
        result
        .sort_values(["参数稳定性综合评分", "策略评分排名标准差", "年化收益率平均值"],
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
def generate_param_heatmap(code, scan_rows, save_dir="heatmaps", best_ma=None, stock_name=""):
    """从窗口回测数据生成参数扫描热力图"""
    if not scan_rows:
        return
    df = pd.DataFrame(scan_rows)
    if df.empty or df["窗口"].nunique() < 2:
        return

    code_safe = code.replace(".", "_")
    stock_name_safe = stock_name.replace(".", "_").replace(" ", "_").replace("(", "_").replace(")", "_").replace("/", "_") if stock_name else "未知"
    metrics = [
        ("策略评分", "策略评分", "RdYlGn", True),
        ("年化收益率", "年化收益率(%)", "RdYlGn", True),
        ("夏普比率", "夏普比率", "RdYlGn", True),
        ("最大回撤", "最大回撤(%)", "OrRd", False),
    ]

    for metric, title, cmap, center in metrics:
        if metric not in df.columns:
            continue
        pivot = df.pivot_table(index="均线周期", columns="窗口", values=metric, aggfunc="first")
        pivot = pivot[sorted(pivot.columns, key=lambda c: str(c))]

        if pivot.empty:
            continue

        # 构建标注矩阵
        annot_df = pivot.copy().astype(str)
        for col in pivot.columns:
            annot_df[col] = pivot[col].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "")

        fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.7), max(7, len(pivot) * 0.45)))
        sns.heatmap(
            pivot, annot=annot_df, fmt="", cmap=cmap,
            center=0 if center else None,
            linewidths=0.5, linecolor="#e0e0e0",
            ax=ax, cbar_kws={"shrink": 0.8}
        )

        # 每个窗口前3最优格标记：第1黑色、第2深灰、第3浅灰，数字白色
        rank_colors = ['#000000', '#555555', '#999999']
        top3_lookup = {}
        for col_idx, col_name in enumerate(pivot.columns):
            col_data = pivot[col_name].dropna()
            if col_data.empty:
                continue
            ranked = col_data.sort_values() if metric == "最大回撤" else col_data.sort_values(ascending=False)
            for rank, (label, _) in enumerate(ranked.head(3).items()):
                row_idx = pivot.index.get_loc(label)
                top3_lookup[(col_idx, row_idx)] = rank
                ax.add_patch(patches.Rectangle(
                    (col_idx + 0.02, row_idx + 0.02), 0.96, 0.96,
                    fill=True, color=rank_colors[rank], linewidth=0, zorder=2
                ))
        n_rows, n_cols = len(pivot.index), len(pivot.columns)
        for t in ax.texts:
            x, y = t.get_position()
            col, row = round(x - 0.5), round(y - 0.5)
            if 0 <= col < n_cols and 0 <= row < n_rows and (col, row) in top3_lookup:
                t.set_color("white")

        # Y轴标签：最优参数行加★
        if best_ma is not None and best_ma in pivot.index:
            labels = [str(ma) if ma != best_ma else f"★{ma}" for ma in pivot.index]
            ax.set_yticklabels(labels, rotation=0)
        ax.set_title(f"{code} {stock_name} {title} 参数扫描热力图", fontsize=14, fontweight="bold", pad=16)
        ax.set_xlabel("回测窗口", fontsize=11)
        ax.set_ylabel("均线周期", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        ax.text(0.5, 1.02,
                "▎黑/深灰/浅灰底 = 窗口内第1/2/3名   ▎★Y轴 = 最优参数   ▎行=均线周期 列=回测窗口 值越大颜色越暖（最大回撤除外）",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color="#666666")
        plt.tight_layout()
        path = os.path.join(save_dir, f"{code_safe}_{stock_name_safe}_{metric}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    # 参数敏感性折线图（各窗口均值 ± 标准差）
    grouped = df.groupby("均线周期")["策略评分"].agg(["mean", "std"]).dropna()
    if len(grouped) >= 3:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(grouped.index, grouped["mean"], "o-", color="#2980b9", linewidth=2, markersize=5)
        ax.fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.2, color="#2980b9"
        )
        ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")
        ax.set_xlabel("均线周期", fontsize=11)
        ax.set_ylabel("策略评分", fontsize=11)
        ax.set_title(f"{code} {stock_name} 参数敏感性分析（均值±标准差）", fontsize=14, fontweight="bold", pad=16)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, f"{code_safe}_{stock_name_safe}_参数敏感性分析.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()


def generate_stability_heatmap(code, ws_df, save_dir="heatmaps", best_ma=None, stock_name=""):
    """生成全窗口参数稳定性热力图。

    行=均线周期, 列=窗口, 值=参数稳定性综合评分。
    每列第1名黑色背景+白色文字。
    """
    if ws_df is None or ws_df.empty:
        return
    code_safe = code.replace(".", "_")
    stock_name_safe = stock_name.replace(".", "_").replace(" ", "_").replace("(", "_").replace(")", "_").replace("/", "_") if stock_name else "未知"

    pivot = ws_df.pivot_table(
        index="均线周期", columns="窗口", values="参数稳定性综合评分", aggfunc="first"
    )

    # 确保列按窗口顺序排列
    pivot = pivot[sorted(pivot.columns)]

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.6), max(7, len(pivot) * 0.45)))

    # 标注矩阵
    annot = pivot.map(lambda v: f"{v:.3f}" if pd.notna(v) else "")

    sns.heatmap(
        pivot, annot=annot, fmt="", cmap="RdYlGn",
        center=0.5, linewidths=0.5, linecolor="#e0e0e0",
        ax=ax, cbar_kws={"shrink": 0.8, "label": "参数稳定性综合评分"},
    )

    # 每列第1名标记黑色背景 + 白色文字
    for col_idx, col_name in enumerate(pivot.columns):
        col_data = pivot[col_name].dropna()
        if col_data.empty:
            continue
        best_label = col_data.idxmax()
        row_idx = pivot.index.get_loc(best_label)
        ax.add_patch(patches.Rectangle(
            (col_idx + 0.02, row_idx + 0.02), 0.96, 0.96,
            fill=True, color="#000000", linewidth=0, zorder=2,
        ))

    n_rows, n_cols = len(pivot.index), len(pivot.columns)
    for t in ax.texts:
        x, y = t.get_position()
        col, row = round(x - 0.5), round(y - 0.5)
        if 0 <= col < n_cols and 0 <= row < n_rows:
            col_name = pivot.columns[col]
            best = pivot[col_name].dropna().idxmax()
            if pivot.index[row] == best:
                t.set_color("white")

    ax.set_title(f"{code} {stock_name} 全窗口参数稳定性热力图", fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("窗口", fontsize=11)
    ax.set_ylabel("均线周期", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    # Y轴标签：最优参数行加★
    if best_ma is not None and best_ma in pivot.index:
        labels = [str(ma) if ma != best_ma else f"★{ma}" for ma in pivot.index]
        ax.set_yticklabels(labels, rotation=0)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{code_safe}_{stock_name_safe}_全窗口参数稳定性热力图.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_stock_best_ma_heatmap(all_ws, save_dir="heatmaps"):
    """生成全股票各窗口最优参数热力图。
    行=股票代码, 列=窗口, 值=最优均线周期。
    末尾3列为股性指标（不参与热力图上色）。
    """
    if all_ws is None or all_ws.empty:
        return

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

    # 窗口列（热力图） + 3 个附加列
    extra_cols = ["最优参数变动次数", "最优参数标准差", "股性评分"]
    extra_data = pd.DataFrame({
        "最优参数变动次数": changes_col,
        "最优参数标准差": std_col.round(2),
        "股性评分": score_col,
    }, index=pivot.index)

    n_stocks, n_windows = pivot.shape
    n_extra = len(extra_cols)
    fig_height = max(10, n_stocks * 0.22)
    fig_width = max(12, n_windows * 0.55 + n_extra * 0.9)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    annot = pivot.map(lambda v: f"{int(v)}" if pd.notna(v) else "")

    sns.heatmap(
        pivot, annot=annot, fmt="", cmap="YlOrRd",
        linewidths=0.3, linecolor="#e0e0e0",
        ax=ax, cbar_kws={"shrink": 0.6, "label": "最优均线周期"},
    )

    # --- 在热力图右侧绘制 3 个股性指标列（纯文本，不上色）---
    # 获取热力图的坐标范围
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cell_w = (x1 - x0) / n_windows
    cell_h = (y1 - y0) / n_stocks

    # 画浅灰背景区
    bg_x = x1
    bg_w = n_extra * cell_w
    rect = patches.Rectangle(
        (bg_x, y0), bg_w, y1 - y0,
        facecolor="#f0f0f0", edgecolor="none", zorder=-1
    )
    ax.add_patch(rect)

    # 竖分隔线
    sep = patches.Rectangle(
        (x1 - cell_w * 0.03, y0), cell_w * 0.06, y1 - y0,
        facecolor="#d0d0d0", edgecolor="none", zorder=-1
    )
    ax.add_patch(sep)

    # 添加文本
    for ri in range(n_stocks):
        for ei in range(n_extra):
            val = extra_data.iloc[ri, ei]
            label_str = f"{val}" if pd.notna(val) else ""
            tx = x1 + (ei + 0.5) * cell_w
            # heatmap y 轴从上到下，所以 ri=0 在最上面，对应 y1-cell_h/2
            ty = y1 - (ri + 0.5) * cell_h
            ax.text(tx, ty, label_str, ha="center", va="center",
                    fontsize=6, fontfamily="monospace")

    # 设置 x 轴标签包括窗口列 + 附加列
    all_labels = list(pivot.columns) + extra_cols
    ax.set_xticks(np.arange(len(all_labels)) + 0.5)
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)

    ax.set_title("全股票各窗口最优参数变动情况", fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("")
    ax.set_ylabel("股票代码", fontsize=12)
    ax.tick_params(axis="y", rotation=0, labelsize=6)

    plt.tight_layout()
    path = os.path.join(save_dir, "all_全股票各窗口最优参数变动情况热力图.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"全股票最优参数热力图: {path}")


# =========================================================
# 主程序
# =========================================================
def _process_one_stock(code, windows=None):
    """Process a single stock. Returns (all_rows, stability_dfs, signal_map)."""
    path = _find_data_file(code)
    if not path:
        return [], [], {}, None
    df = pd.read_parquet(path)
    df = df.sort_values("datetime")
    df["code"] = code
    out_file = os.path.join(TRADE_DIR, _market_subdir(code), f"{code}_trades.xlsx")
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

    for ws, we in windows:

        window_mask = (pd.to_datetime(df["datetime"]) >= ws) & (pd.to_datetime(df["datetime"]) < we)
        df_w = df[window_mask].copy()

        if df_w.empty:
            continue

        window_label = f"{ws.date()}~{we.date()}"
        eff_start = pd.to_datetime(df_w["datetime"]).min()
        eff_end = pd.to_datetime(df_w["datetime"]).max()
        effective_range = f"{eff_start.date()}~{eff_end.date()}"

        for ma in MA_LIST:
            df_w["ha_close"] = ha_close_full[window_mask]
            df_w["ma"] = ma_cache[ma][window_mask]
            df_w["dir"] = np.where(df_w["ma"] > df_w["ma"].shift(1), 1, -1)
            df_w["buy"] = (df_w["dir"] == 1) & (df_w["dir"].shift(1) == -1)
            df_w["sell"] = (df_w["dir"] == -1) & (df_w["dir"].shift(1) == 1)

            trades, equity_arr = _build_trades_numba(df_w, ma)

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
                .sort_values(["参数稳定性综合评分", "策略评分排名标准差"], ascending=[False, True])
                .iloc[0]["均线周期"]
            )

    # =============================================
    # 参数扫描热力图（PNG 写入开销小，保留在 worker 中）
    # =============================================
    if window_summary_rows:
        generate_param_heatmap(code, window_summary_rows, save_dir=os.path.join(TRADE_DIR, _market_subdir(code), "heatmaps"), best_ma=best_stab_ma, stock_name=stock_name)

    if window_stability_df is not None and not window_stability_df.empty:
        generate_stability_heatmap(
            code, window_stability_df,
            save_dir=os.path.join(TRADE_DIR, _market_subdir(code), "heatmaps"),
            best_ma=best_stab_ma,
            stock_name=stock_name,
        )

    # =============================================
    # 从全量数据计算当前信号（用于信号扫描，复用预计算的 HA 和均线）
    # =============================================
    signal_map = {}
    for ma in MA_LIST:
        d = df.copy()
        d["ha_close"] = ha_close_full
        d["ma"] = ma_cache[ma]
        d["dir"] = np.where(d["ma"] > d["ma"].shift(1), 1, -1)
        d["buy"] = (d["dir"] == 1) & (d["dir"].shift(1) == -1)
        d["sell"] = (d["dir"] == -1) & (d["dir"].shift(1) == 1)
        signal_info = get_last_signal_info(d)
        signal_info["K线周期"] = BAR_INTERVAL
        signal_info["均线周期"] = ma
        signal_info["股票代码"] = code
        signal_info["股票名称"] = stock_name
        signal_info["所属板块"] = stock_plates
        # 从 df 读取 market 字段
        _market_val = str(df.get("market", pd.Series([""])).iloc[0]) if "market" in df.columns else ""
        signal_info["市场"] = _market_val
        signal_map[ma] = signal_info

    return stock_all_rows, stock_stability_dfs, signal_map, window_stability_df, out_file


def _round_display(df, pct_cols=None):
    """输出前统一处理浮点列：
    - 百分比字段 ÷100 → Excel 原生百分比格式
    - 其余 float 列保留 2 位小数
    - 参数稳定性综合评分保留 3 位小数
    不影响原始计算精度。
    """
    df = df.copy()
    pct_set = set(pct_cols or [])
    for col in df.select_dtypes(include=["float", "float64"]).columns:
        if col in pct_set:
            df[col] = (df[col] / 100.0).round(4)
        elif col == "参数稳定性综合评分":
            df[col] = df[col].round(3)
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
    BT_COLS = ["策略评分", "收益率", "年化收益率", "买入持有收益率", "超额收益率",
               "最大回撤", "夏普比率", "卡尔玛比率",
               "交易次数", "盈利交易率", "盈利因子", "盈亏比",
               "平均盈利", "平均亏损", "最大单笔盈利", "最大单笔亏损",
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
    "股票代码", "股票名称", "所属板块", "市场", "K线周期", "均线周期",
    "策略评分", "策略表现",
    "时间", "收盘价", "HA收盘价", "HA均线值",
    "趋势方向", "最新信号", "最新信号时间", "最新信号收盘价", "最新信号确认",
    "历史信号", "历史信号时间", "历史信号收盘价", "距离历史信号已过天数",
    "预计持仓进度",
    "距离历史信号收盘价涨跌幅", "持仓日化收益率",
    "均线趋势共振方向", "共振均线数量", "共振均线列表",
    "收益率", "年化收益率", "买入持有收益率", "超额收益率",
    "最大回撤", "夏普比率", "卡尔玛比率",
    "交易次数", "盈利交易率", "盈利因子", "盈亏比",
    "平均盈利", "平均亏损", "最大单笔盈利", "最大单笔亏损",
    "最大连续盈利次数", "最大连续亏损次数", "平均持仓天数",
    "初始资金", "最终资金",
    "窗口", "窗口内有效数据周期", "窗口内有效数据天数",
]




def _write_summary_excel(out_path, signal_df, all_df, score_matrix, rank_matrix,
                          window_stability_dfs, market_label):
    """写入多 sheet 综合 Excel（信号扫描 + 汇总 + 评分 + 稳定性 + 统计逻辑）。"""
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

        # --- 2. 回测汇总 ---
        all_out = reorder_columns(all_df)
        apply_cn_mapping(all_out)
        all_out = _round_display(all_out, PCT_COLS)
        all_out.to_excel(writer, sheet_name="回测汇总", index=False)
        _set_pct_format(writer.sheets["回测汇总"], all_out, PCT_COLS)

        # --- 3. 策略评分明细 ---
        if score_matrix is not None and not score_matrix.empty:
            _round_display(score_matrix).to_excel(writer, sheet_name="策略评分明细", index=False)

        # --- 4. 策略评分排名 ---
        if rank_matrix is not None and not rank_matrix.empty:
            _round_display(rank_matrix).to_excel(writer, sheet_name="策略评分排名", index=False)

        # --- 5. 全窗口参数稳定性分析 ---
        if window_stability_dfs:
            _ws = pd.concat(window_stability_dfs, ignore_index=True) if isinstance(window_stability_dfs, list) else window_stability_dfs
            _wpct = ["盈利窗口占比", "年化收益率平均值"]
            _round_display(_ws, _wpct).to_excel(writer, sheet_name="全窗口参数稳定性分析", index=False)
            _set_pct_format(writer.sheets["全窗口参数稳定性分析"], _ws, _wpct)

        # --- 6. 各窗口最优参数变动情况 ---
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
             "统计逻辑": "所有股票所有累积窗口所有MA的完整回测结果汇总（含策略评分、窗口标签、收益/风险指标等）"},
            {"类型": "Sheet说明", "名称": "策略评分明细",
             "统计逻辑": "透视表，行=股票代码+K线周期+均线周期，列=窗口时间区间，值=策略评分"},
            {"类型": "Sheet说明", "名称": "策略评分排名",
             "统计逻辑": "透视表，同上结构，值改为窗口内排名（每窗口每股票内的参数间排名，同分取最小排名）"},
            {"类型": "Sheet说明", "名称": "全窗口参数稳定性分析",
             "统计逻辑": "个股层面，对每个累积窗口阶段计算参数稳定性（同参数稳定性分析逻辑，按窗口展开），排序=股票代码↑|窗口↑|均线周期↑"},
            {"类型": "Sheet说明", "名称": "各窗口最优参数变动情况",
             "统计逻辑": "透视表，行=股票代码，列=窗口，值=均线周期；选取每只股票每个窗口中参数稳定性综合评分最高的均线周期，展示最优参数随窗口变化的趋势；末尾3列为股性指标：最优参数变动次数（相邻窗口间最优参数切换次数）、最优参数标准差（最优参数的分散程度）、股性评分（固定扣分公式 = max(0, 100 − 变动次数×5 − 标准差×2)，变动越少越稳定得分越高）"},
            {"类型": "", "名称": "", "统计逻辑": ""},
            # ── 评分模型 ──
            {"类型": "评分模型", "名称": "策略评分",
             "统计逻辑": "Score = 0.30*CAGR + 0.25*Sharpe + 0.20*(1-最大回撤) + 0.15*盈利因子 + 0.05*盈利交易率 + 0.05*交易次数；子指标min-max归一化，CAGR:0-30, Sharpe:0-2, 回撤:0-50, 盈利因子:1-3, 盈利率:30-80, 交易次数:10-100，加权求和范围0~100"},
            {"类型": "评分模型", "名称": "参数稳定性综合评分",
             "统计逻辑": "Score = 0.20*avg_rank_n + 0.10*top3_n + 0.15*std_n + 0.25*cagr_n + 0.20*win_rate_n + 0.10*cagr_std_n；各指标min-max归一化，排名类占0.45，收益类(均值+占比+标准差)占0.55，越高越稳定；输出保留3位小数"},
            # ── 基本字段 ──
            {"类型": "基本字段", "名称": "股票代码",
             "统计逻辑": "富途格式股票代码，如 US.AAPL / CC.BTC"},
            {"类型": "基本字段", "名称": "股票名称",
             "统计逻辑": "股票中文名称，来源于 parquet 数据文件"},
            {"类型": "基本字段", "名称": "所属板块",
             "统计逻辑": "股票行业/板块分类，来源于 parquet 数据文件"},
            {"类型": "基本字段", "名称": "市场",
             "统计逻辑": "US=美股, CN=A股, CC=加密货币"},
            {"类型": "基本字段", "名称": "K线周期",
             "统计逻辑": "1D=日K, 1W=周K"},
            {"类型": "基本字段", "名称": "均线周期",
             "统计逻辑": "参数稳定性分析中倒数第二个窗口（仅1个窗口时取唯一窗口）策略评分最高的均线周期，作为该股票的最优参数，跳过最后一个未完整窗口"},
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
             "统计逻辑": "BUY/SELL信号且距离最近一次信号<5根K线为「待确认，K线未正式收盘」，否则为「已确认」"},
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
            {"类型": "回测指标", "名称": "平均亏损",
             "统计逻辑": "所有亏损交易的平均亏损金额（正数表示）"},
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
             "统计逻辑": "回测窗口的时间区间标签，格式 起始日期~结束日期；从2000-01-03起按1年步长递增，所有股票共享同一套窗口列表"},
            {"类型": "窗口信息", "名称": "窗口内有效数据周期",
             "统计逻辑": "该窗口实际数据的起止日期区间，格式 起始日期~结束日期；若股票上市晚于窗口起始，起始日期为数据首日"},
            {"类型": "窗口信息", "名称": "窗口内有效数据天数",
             "统计逻辑": "从窗口内有效数据周期解析出的实际天数 = 结束日期 − 起始日期"},
            # ── 参数选择 ──
            {"类型": "参数选择", "名称": "最优均线周期",
             "统计逻辑": "参数稳定性分析中倒数第二个窗口（仅1个窗口时取唯一窗口）策略评分最高的均线周期，选作信号扫描使用的参数"},
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
            {"类型": "参数稳定性", "名称": "参数稳定性综合评分",
             "统计逻辑": "Score = 0.20*avg_rank_n + 0.10*top3_n + 0.15*std_n + 0.25*cagr_n + 0.20*win_rate_n + 0.10*cagr_std_n；各指标min-max归一化，排名类占0.45，收益类(均值+占比+标准差)占0.55，越高越稳定；输出保留3位小数"},
            {"类型": "参数稳定性", "名称": "策略评分排名Top3占比",
             "统计逻辑": "该均线周期在各窗口中排名前三的次数占比"},
            {"类型": "参数稳定性", "名称": "是否最优",
             "统计逻辑": "每只股票每个窗口内参数稳定性综合评分最高者标记为「最优」，并列时以策略评分排名标准差升序+年化收益率平均值降序决胜，确保唯一"},
        ]
        pd.DataFrame(logic_rows).to_excel(writer, sheet_name="统计逻辑", index=False)

        for ws in writer.sheets.values():
            _apply_sheet_format(ws)

    print(f"{market_label}汇总Excel: {out_path}")


def run_trade():

    symbols = load_symbols(SYMBOL_FILE)

    # 过滤出有数据文件的股票，用于进度条总计数
    available = [s for s in symbols if _find_data_file(s)]

    # 扫描全市场数据，取最晚日期作为全局窗口终点
    global_end = pd.Timestamp("2000-01-01")
    for s in available:
        try:
            _path = _find_data_file(s)
            if not _path:
                continue
            _tmp = pd.read_parquet(_path, columns=["datetime"])
            _max = pd.to_datetime(_tmp["datetime"]).max()
            if _max > global_end:
                global_end = _max
        except Exception:
            continue
    windows = generate_windows(end_date=global_end)
    print(f"全局窗口终点: {global_end.date()}, 共 {len(windows)} 个窗口")

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

    for mkt in market_order:
        group = market_groups[mkt]
        if not group:
            continue

        print(f"\n===== 开始回测 {mkt.upper()} 市场（{len(group)} 只股票）=====")

        market_rows = []
        market_signal_maps = {}
        market_window_stability = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            future_to_code = {executor.submit(_process_one_stock, code, windows): code for code in group}
            _results = {}
            with tqdm(total=len(group), desc=f"{mkt.upper()}回测", unit="stock") as pbar:
                for future in concurrent.futures.as_completed(future_to_code):
                    code = future_to_code[future]
                    try:
                        _results[code] = future.result()
                    except Exception as e:
                        _results[code] = e
                    finally:
                        pbar.update(1)

            for code in sorted(_results.keys()):
                result = _results[code]
                if isinstance(result, Exception):
                    tqdm.write(f"  失败: {code} {result}")
                else:
                    s_all, s_stab, sig_map, s_ws, out_f = result
                    market_rows.extend(s_all)
                    market_signal_maps[code] = sig_map
                    if s_ws is not None and not s_ws.empty:
                        market_window_stability.append(s_ws)
                    tqdm.write(f"  完成: {code}")

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
                .sort_values(["参数稳定性综合评分", "策略评分排名标准差"], ascending=[False, True])
                .groupby("股票代码", sort=False)
                .head(1)
                .reset_index(drop=True)
            )

        # 本市场信号扫描
        signal_mkt = _build_signal_scan(market_df, market_signal_maps, stab_mkt, all_signal_maps)

        if not signal_mkt.empty:
            # 本市场全股票热力图
            mkt_dir = os.path.join(TRADE_DIR, mkt)
            if market_window_stability:
                all_ws_mkt = pd.concat(market_window_stability, ignore_index=True)
                generate_all_stock_best_ma_heatmap(
                    all_ws_mkt, save_dir=os.path.join(mkt_dir, "heatmaps")
                )

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
            .sort_values(["参数稳定性综合评分", "策略评分排名标准差"], ascending=[False, True])
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

    tv_path = os.path.join(TRADE_DIR, "tradingview_params.txt")
    with open(tv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tv_lines) + "\n")
    print(f"\nTradingView 配置文件: {tv_path} 共 {len(tv_lines)} 只股票")

    if missing_codes:
        print("以下股票未找到最优参数，已跳过:")
        for c in missing_codes:
            print(f"  {c}")

    # 全市场汇总 Excel
    date_str = pd.Timestamp.today().strftime("%Y%m%d")
    all_out = os.path.join(TRADE_DIR, f"all_summary_param_scan_{date_str}.xlsx")
    _write_summary_excel(all_out, signal_all, all_df, score_all, rank_all,
                        window_stability_dfs, "全市场")
    print(f"全市场完成: {all_out}")

    # 全股票各窗口最优参数热力图
    if window_stability_dfs:
        all_ws_hm = pd.concat(window_stability_dfs, ignore_index=True)
        generate_all_stock_best_ma_heatmap(
            all_ws_hm, save_dir=os.path.join(TRADE_DIR, "heatmaps")
        )

# =========================================================
# 富途自选股分组同步
# =========================================================
# 同步和导出功能已移除

# =========================================================
if __name__ == "__main__":
    # _CLI_ARGS 和 ktype 覆盖已在顶部 __main__ 块中完成
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
    run_trade()