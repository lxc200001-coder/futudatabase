import os
import warnings
import concurrent.futures
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from futu import OpenQuoteContext, KLType, AuType, RET_OK
from openpyxl.formatting.rule import DataBarRule
from openpyxl.utils import get_column_letter

# 补丁：openpyxl 3.1.5 DataBar 缺少 gradient 属性，通过 XML 注入实现实体填充
import openpyxl.formatting.rule as _rule_mod
_orig_data_bar_tree = _rule_mod.DataBar.to_tree
def _patched_data_bar_tree(self, tagname=None, namespace=None):
    el = _orig_data_bar_tree(self, tagname, namespace)
    if getattr(self, '_gradient', None) is not None:
        el.set('gradient', '0' if not self._gradient else '1')
    return el
_rule_mod.DataBar.to_tree = _patched_data_bar_tree

# 屏蔽pandas concat空列FutureWarning（不影响功能）
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation", category=FutureWarning)


def _apply_sheet_format(ws):
    """Set auto-filter and freeze first row + first column."""
    if ws.max_row and ws.max_column:
        ws.auto_filter.ref = ws.dimensions
        ws.freeze_panes = "B2"

# =========================================================
# 配置
# =========================================================
DATA_DIR = "data"
RESULT_DIR = "results"
TRADE_DIR = "trades"
SYMBOL_FILE = "symbols.csv"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(TRADE_DIR, exist_ok=True)

MA_LIST = [5, 10, 20, 30, 60]
INITIAL_CASH = 10000
FEE_RATE = 0.001

BAR_INTERVAL = "1W"
TRADING_PERIOD = 52
WINDOW_YEARS = 5
STEP_YEARS = 1
WINDOW_START_DATE = "2000-01-03"

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

# =========================================================
# 信号
# =========================================================
def calc_signal(df, ma_len):
    df = df.copy()
    df["ma"] = df["ha_close"].rolling(ma_len, min_periods=ma_len).mean()

    df["dir"] = np.where(df["ma"] > df["ma"].shift(1), 1, -1)
    df["buy"] = (df["dir"] == 1) & (df["dir"].shift(1) == -1)
    df["sell"] = (df["dir"] == -1) & (df["dir"].shift(1) == 1)

    return df

# =========================================================
# 最近信号
# =========================================================
def get_last_signal_info(df):

    today = pd.Timestamp.today().normalize()

    signal = "NONE"

    if df.iloc[-1]["buy"]:
        signal = "BUY"
    elif df.iloc[-1]["sell"]:
        signal = "SELL"

    buy_rows = df[df["buy"]]

    if len(buy_rows) > 0:
        last_buy = buy_rows.iloc[-1]
        buy_time = pd.to_datetime(last_buy["datetime"])
        buy_close = round(float(last_buy["close"]), 2)
        buy_days = (today - buy_time.normalize()).days
    else:
        buy_time = pd.NaT
        buy_close = None
        buy_days = None

    sell_rows = df[df["sell"]]

    if len(sell_rows) > 0:
        last_sell = sell_rows.iloc[-1]
        sell_time = pd.to_datetime(last_sell["datetime"])
        sell_close = round(float(last_sell["close"]), 2)
        sell_days = (today - sell_time.normalize()).days
    else:
        sell_time = pd.NaT
        sell_close = None
        sell_days = None

    last = df.iloc[-1]

    return {
        "时间": last["datetime"],
        "收盘价": round(float(last["close"]), 2),
        "HA收盘价": round(float(last["ha_close"]), 2),
        "HA均线值": round(float(last["ma"]), 2) if not pd.isna(last["ma"]) else None,
        "趋势方向": int(last["dir"]),
        "信号": signal,

        "买入信号时间": buy_time,
        "买入信号收盘价": buy_close,
        "距离买入信号已过天数": buy_days,

        "卖出信号时间": sell_time,
        "卖出信号收盘价": sell_close,
        "距离卖出信号已过天数": sell_days
    }

# =========================================================
# 交易回测
# =========================================================
def _make_trade_record(code_val, ma_len, entry_price, position, entry_time, entry_index,
                       exit_price, exit_time, exit_index,
                       cash_before_open, cash_after_open,
                       cash_before_close, cash_after_close,
                       status, backtest_period):
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
        "K线周期": BAR_INTERVAL,
        "均线周期": ma_len,

        "开仓时间": entry_time,
        "开仓价格": round(float(entry_price), 2),
        "买入股数": position,

        "平仓时间": exit_time,
        "平仓价格": round(float(exit_price), 2),
        "卖出股数": position,

        "交易状态": status,
        "订单盈亏类型": "盈利" if pnl > 0 else "亏损",

        "收益金额": round(pnl, 4),
        "收益率(%)": round(float(return_pct), 4),

        "买入手续费": round(buy_fee, 4),
        "卖出手续费": round(sell_fee, 4),
        "总手续费": round(total_fee, 4),

        "开仓前可用现金": round(cash_before_open, 2) if cash_before_open is not None else None,
        "开仓后可用现金": round(cash_after_open, 2) if cash_after_open is not None else None,
        "平仓前可用现金": round(cash_before_close, 2) if cash_before_close is not None else None,
        "平仓后可用现金": round(cash_after_close, 2) if cash_after_close is not None else None,

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

    for i in range(len(df)):

        price = round(float(close_arr[i]), 2)

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
                "已平仓", backtest_period
            ))

            position = 0

    if position > 0:

        price = round(float(close_arr[-1]), 2)
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
            "未平仓(强制结算)", backtest_period
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
# 综合评分
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
    sharpe_score = normalize(row.get("Sharpe", 0), 0, 2) * 100
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

    return round(score, 2)

# =========================================================
# 5年滚动窗口生成
# =========================================================
def generate_windows(df):
    dates = pd.to_datetime(df["datetime"])
    start = pd.Timestamp(WINDOW_START_DATE)
    end = dates.max()

    windows = []
    cur = start

    while cur <= end:
        windows.append((cur, cur + pd.DateOffset(years=WINDOW_YEARS)))
        cur += pd.DateOffset(years=STEP_YEARS)

    return windows

# =========================================================
# 汇总
# =========================================================
def build_summary(trades_df, ma_len, df):

    start = pd.to_datetime(df["datetime"].iloc[0])
    end = pd.to_datetime(df["datetime"].iloc[-1])

    period = f"{start} ~ {end}"

    curve = equity_curve(df, trades_df)

    final = curve[-1]

    if trades_df.empty:

        return {
            "股票代码": df["code"].iloc[0],
            "K线周期": BAR_INTERVAL,
            "均线周期": ma_len,

            "收益率": 0,
            "年化收益率": 0,
            "买入持有收益率": 0,
            "超额收益率": 0,

            "最大回撤": 0,
            "Sharpe": 0,
            "Calmar Ratio": 0,

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

            "回测周期": period
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
        "K线周期": BAR_INTERVAL,
        "均线周期": ma_len,

        "收益率": round(float(ret), 2),
        "年化收益率": round(float(cagr), 2),
        "买入持有收益率": round(float(buy_hold), 2),
        "超额收益率": round(float(alpha), 2),

        "最大回撤": round(float(mdd), 2),
        "Sharpe": round(float(sh), 4),
        "Calmar Ratio": round(float(calmar), 4),

        "交易次数": int(len(trades_df)),
        "盈利交易率": round(float(win_rate), 2),
        "盈利因子": round(float(pf), 4),
        "盈亏比": round(float(payoff), 4),

        "平均盈利": round(float(avg_win), 4),
        "平均亏损": round(float(avg_loss), 4),
        "最大单笔盈利": round(float(max_win_trade), 4),
        "最大单笔亏损": round(float(max_loss_trade), 4),

        "最大连续盈利次数": max_win_streak,
        "最大连续亏损次数": max_loss_streak,

        "平均持仓天数": round(float(avg_hold), 2),

        "初始资金": INITIAL_CASH,
        "最终资金": round(float(final), 2),

        "回测周期": period
    }

# =========================================================
# 列排序
# =========================================================
COLUMN_ORDER = [
    "股票代码", "K线周期", "均线周期", "综合评分"
]

END_COLUMNS = ["回测周期", "回测类型", "窗口", "窗口内有效数据日期"]

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
    valid = df.groupby("窗口")["综合评分"].transform("max") > 0
    df = df[valid]

    if df.empty or df["窗口"].nunique() < 2:
        return pd.DataFrame()

    # 每窗口内按综合评分排名（同分取最小排名）
    df["窗口内排名"] = df.groupby(["股票代码", "窗口"])["综合评分"].rank(ascending=False, method="min")

    # 提取固定信息
    code_val = df["股票代码"].iloc[0]
    bar_val = df["K线周期"].iloc[0]

    # 按均线周期汇总
    stats = df.groupby("均线周期").agg(
        窗口数量=("窗口", "nunique"),
        盈利窗口数量=("年化收益率", lambda x: (x > 0).sum()),
        综合评分排名平均值=("窗口内排名", "mean"),
        综合评分排名第一次数=("窗口内排名", lambda x: (x == 1).sum()),
        综合评分排名Top3占比=("窗口内排名", lambda x: round((x <= 3).sum() / max(len(x), 1) * 100, 1)),
        综合评分排名标准差=("窗口内排名", "std"),
        年化收益率平均值=("年化收益率", "mean"),
        年化收益率标准差=("年化收益率", "std")
    ).reset_index()

    stats["盈利窗口占比"] = (stats["盈利窗口数量"] / stats["窗口数量"] * 100).round(1)
    stats["年化收益率平均值"] = stats["年化收益率平均值"].round(2)
    stats["年化收益率标准差"] = stats["年化收益率标准差"].fillna(0).round(2)

    stats["综合评分排名平均值"] = stats["综合评分排名平均值"].round(2)
    stats["综合评分排名标准差"] = stats["综合评分排名标准差"].fillna(0).round(4)

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

    avg_rank_n = _norm(stats["综合评分排名平均值"], higher_is_better=False)
    top3_n = _norm(stats["综合评分排名Top3占比"], higher_is_better=True)
    std_n = _norm(stats["综合评分排名标准差"], higher_is_better=False)
    cagr_n = _norm(stats["年化收益率平均值"], higher_is_better=True)
    win_rate_n = _norm(stats["盈利窗口占比"], higher_is_better=True)
    cagr_std_n = _norm(stats["年化收益率标准差"], higher_is_better=False)

    stats["参数稳定性综合评分"] = (
        0.20 * avg_rank_n + 0.10 * top3_n + 0.15 * std_n +
        0.25 * cagr_n + 0.20 * win_rate_n + 0.10 * cagr_std_n
    ).round(4)

    # 排序后重排列顺序
    col_order = ["股票代码", "K线周期", "均线周期", "窗口数量",
                 "盈利窗口数量", "盈利窗口占比",
                 "年化收益率平均值", "年化收益率标准差",
                 "综合评分排名Top3占比", "综合评分排名平均值",
                 "综合评分排名标准差", "综合评分排名第一次数",
                 "参数稳定性综合评分"]
    stats.insert(0, "K线周期", bar_val)
    stats.insert(0, "股票代码", code_val)

    return stats.reindex(columns=col_order)


# =========================================================
# 评分矩阵（明细+排名）
# =========================================================
def build_score_matrix(summary_rows):
    """从窗口回测汇总行构建评分明细和排名透视表"""
    df = pd.DataFrame(summary_rows)

    # 排除评分全为0的窗口
    valid = df.groupby("窗口")["综合评分"].transform("max") > 0
    df = df[valid]

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 评分明细透视
    score_pivot = df.pivot_table(
        index=["股票代码", "K线周期", "均线周期"],
        columns="窗口",
        values="综合评分",
        aggfunc="first"
    )
    score_pivot = score_pivot[sorted(score_pivot.columns)]
    score_pivot = score_pivot.reset_index()

    # 评分排名（每股票每窗口内独立排名）
    df["窗口内排名"] = df.groupby(["股票代码", "窗口"])["综合评分"].rank(ascending=False, method="min")

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
# 主程序
# =========================================================
def _process_one_stock(code):
    """Process a single stock. Returns (all_rows, window_rows, stability_dfs)."""
    path = os.path.join(DATA_DIR, f"{code}_1w.parquet")
    if not os.path.exists(path):
        return [], [], []
    df = pd.read_parquet(path)
    df = df.sort_values("datetime")
    df["code"] = code
    out_file = os.path.join(TRADE_DIR, f"{code}_trades.xlsx")
    stock_all_rows = []
    stock_window_rows = []
    stock_stability_dfs = []

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:

        # =============================================
        # 全量回测
        # =============================================
        full_trades_list = []
        full_summary_rows = []
        window_trades_by_ma = {ma: [] for ma in MA_LIST}
        window_summary_rows = []

        # 全量数据的有效日期范围（用于 窗口内有效数据日期）
        full_eff_start = pd.to_datetime(df["datetime"]).min()
        full_eff_end = pd.to_datetime(df["datetime"]).max()
        full_effective_range = f"{full_eff_start.date()}~{full_eff_end.date()}"

        # Pre-compute HA once for all MA periods
        ha_close_full, _ = calc_heikin_ashi(df)
        df_with_ha = df.copy()
        df_with_ha["ha_close"] = ha_close_full

        for ma in MA_LIST:

            d = calc_signal(df_with_ha, ma)
            trades = build_trades(d, ma)
            if not trades.empty:
                trades["窗口"] = "FULL"
                trades["窗口内有效数据日期"] = full_effective_range
            full_trades_list.append(trades)

            summary = build_summary(trades, ma, df)

            signal_info = get_last_signal_info(d)

            summary.update(signal_info)

            summary["综合评分"] = calc_score_row(summary)
            summary["回测类型"] = "全量"
            summary["窗口"] = "FULL"

            full_summary_rows.append(summary)
            stock_all_rows.append(summary)

        # =============================================
        # 全量回测交易日志明细（合并所有MA）
        # =============================================
        full_trades_merged = [t for t in full_trades_list if not t.empty]
        if full_trades_merged:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                pd.concat(full_trades_merged, ignore_index=True, sort=False).to_excel(
                    writer, sheet_name="全量回测交易日志明细", index=False)

        # =============================================
        # 全量回测各周期结果汇总
        # =============================================
        full_summary_df = reorder_columns(pd.DataFrame(full_summary_rows))
        if "趋势方向" in full_summary_df.columns:
            full_summary_df["趋势方向"] = full_summary_df["趋势方向"].map({1: "多头", -1: "空头"})
        full_summary_df.to_excel(writer, sheet_name="全量回测各周期结果汇总", index=False)

        # =============================================
        # 全量回测最优参数结果（综合评分最高）
        # =============================================
        best_full_result = full_summary_df.loc[[full_summary_df["综合评分"].idxmax()]].copy()
        best_full_result.to_excel(writer, sheet_name="全量回测最优参数结果", index=False)

        # =============================================
        # 5年窗口回测
        # =============================================
        windows = generate_windows(df)

        for ws, we in windows:

            df_w = df[
                (pd.to_datetime(df["datetime"]) >= ws) &
                (pd.to_datetime(df["datetime"]) < we)
            ].copy()

            if df_w.empty:
                continue

            data_years = (df_w["datetime"].max() - df_w["datetime"].min()).days / 365.0
            if data_years < 2:
                continue

            window_label = f"{ws.date()}~{we.date()}"
            eff_start = pd.to_datetime(df_w["datetime"]).min()
            eff_end = pd.to_datetime(df_w["datetime"]).max()
            effective_range = f"{eff_start.date()}~{eff_end.date()}"

            ha_close_w, _ = calc_heikin_ashi(df_w)
            df_w_ha = df_w.copy()
            df_w_ha["ha_close"] = ha_close_w

            for ma in MA_LIST:
                d = calc_signal(df_w_ha, ma)
                trades = build_trades(d, ma)

                if not trades.empty:
                    trades_w = trades.copy()
                    trades_w["窗口"] = window_label
                    trades_w["窗口内有效数据日期"] = effective_range
                    window_trades_by_ma[ma].append(trades_w)

                summary = build_summary(trades, ma, df_w)
                summary["回测类型"] = "窗口"
                summary["窗口"] = window_label
                summary["窗口内有效数据日期"] = effective_range
                summary["综合评分"] = calc_score_row(summary)

                window_summary_rows.append(summary)
                stock_all_rows.append(summary)
                stock_window_rows.append(summary)

        # =============================================
        # 窗口回测交易日志明细（合并所有MA窗口）
        # =============================================
        window_trades_merged = []
        for ma in MA_LIST:
            window_trades_merged.extend(window_trades_by_ma[ma])
        window_trades_merged = [t for t in window_trades_merged if not t.empty]
        if window_trades_merged:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                pd.concat(window_trades_merged, ignore_index=True, sort=False).to_excel(
                    writer, sheet_name="窗口回测交易日志明细", index=False)

        # =============================================
        # 窗口回测汇总分析
        # =============================================
        if window_summary_rows:
            ws_df = reorder_columns(pd.DataFrame(window_summary_rows))
            if "趋势方向" in ws_df.columns:
                ws_df["趋势方向"] = ws_df["趋势方向"].map({1: "多头", -1: "空头"})
            ws_df.to_excel(writer, sheet_name="窗口回测各周期各窗口结果汇总", index=False)

            score_pivot, rank_pivot = build_score_matrix(window_summary_rows)
            if not score_pivot.empty:
                score_pivot.to_excel(writer, sheet_name="窗口回测综合评分明细", index=False)
            if not rank_pivot.empty:
                rank_pivot.to_excel(writer, sheet_name="窗口回测综合评分排名", index=False)

            stability_df = calc_param_stability(window_summary_rows)
            if not stability_df.empty:
                stability_df.to_excel(writer, sheet_name="窗口回测参数稳定性分析", index=False)
                stock_stability_dfs.append(stability_df)

                best_stab_ma = (
                    stability_df
                    .sort_values(["参数稳定性综合评分", "综合评分排名标准差"], ascending=[False, True])
                    .iloc[0]["均线周期"]
                )
                window_best_result = full_summary_df[full_summary_df["均线周期"] == best_stab_ma].copy()
                if not window_best_result.empty:
                    window_best_result.to_excel(writer, sheet_name="窗口回测最优参数结果", index=False)

                    best_full_tagged = best_full_result.copy()
                    best_full_tagged["最优参数来源"] = "全量评分最优"
                    window_best_tagged = window_best_result.copy()
                    window_best_tagged["最优参数来源"] = "窗口稳定性最优"

                    pd.concat(
                        [best_full_tagged, window_best_tagged],
                        ignore_index=True, sort=False
                    ).to_excel(writer, sheet_name="全量和窗口回测最优参数结果对比", index=False)

        for ws in writer.sheets.values():
            _apply_sheet_format(ws)

    return stock_all_rows, stock_window_rows, stock_stability_dfs


def run_trade():

    symbols = load_symbols(SYMBOL_FILE)

    # 过滤出有数据文件的股票，用于进度条总计数
    available = [s for s in symbols if os.path.exists(os.path.join(DATA_DIR, f"{s}_1w.parquet"))]


    all_rows = []
    window_rows = []
    stability_dfs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
        future_to_code = {executor.submit(_process_one_stock, code): code for code in available}
        with tqdm(total=len(available), desc="回测进度", unit="stock") as pbar:
            for future in concurrent.futures.as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    s_all, s_win, s_stab = future.result()
                    all_rows.extend(s_all)
                    window_rows.extend(s_win)
                    stability_dfs.extend(s_stab)
                    tqdm.write(f"完成: {code}")
                except Exception as e:
                    tqdm.write(f"失败: {code} {e}")
                finally:
                    pbar.update(1)

    if all_rows:

        all_df = pd.DataFrame(all_rows)

        full_df = all_df[all_df["回测类型"] == "全量"].copy()

        best_score_strategy = full_df.loc[
            full_df.groupby("股票代码")["综合评分"].idxmax()
        ].sort_values(by="综合评分", ascending=False)

        out = os.path.join(TRADE_DIR, "all_summary.xlsx")

        with pd.ExcelWriter(out, engine="openpyxl") as writer:

            # =========================================================
            # 计算窗口相关数据（提前算好，不依赖写入顺序）
            # =========================================================
            win_df = pd.DataFrame(window_rows) if window_rows else pd.DataFrame()
            score_all, rank_all = build_score_matrix(window_rows) if window_rows else (pd.DataFrame(), pd.DataFrame())
            all_stability = stab_best = compare = None
            win_best_full_result = pd.DataFrame()

            if window_rows and stability_dfs:
                all_stability = pd.concat(stability_dfs, ignore_index=True)
                # 每只股票选参数稳定性综合评分最高的参数，并列时取标准差最小的
                stab_best = (
                    all_stability
                    .sort_values(["参数稳定性综合评分", "综合评分排名标准差"], ascending=[False, True])
                    .groupby("股票代码", sort=False)
                    .head(1)
                    .reset_index(drop=True)
                )

                # 窗口回测个股评分最优参数 → 在全量回测中匹配回测结果
                win_best_full_result = full_df.merge(
                    stab_best[["股票代码", "均线周期"]],
                    on=["股票代码", "均线周期"],
                    how="inner"
                )

                if not best_score_strategy.empty:
                    # 窗口评分映射（股票+均线 → 综合评分）
                    win_score_map = full_df[["股票代码", "均线周期", "综合评分"]].copy()
                    win_score_map = win_score_map.rename(
                        columns={"均线周期": "窗口回测最优均线周期",
                                 "综合评分": "窗口回测最优均线周期回测结果综合评分"}
                    )

                    # 稳定性评分映射（股票+均线 → 参数稳定性综合评分）
                    stab_score_map = all_stability[["股票代码", "均线周期", "参数稳定性综合评分"]].copy()

                    compare = (
                        best_score_strategy[["股票代码", "K线周期", "均线周期", "综合评分"]]
                        .copy()
                        .rename(columns={
                            "均线周期": "全量回测最优均线周期",
                            "综合评分": "全量回测最优均线周期回测结果综合评分"
                        })
                        .merge(stab_best[["股票代码", "均线周期"]], on="股票代码", how="left")
                        .rename(columns={"均线周期": "窗口回测最优均线周期"})
                        .merge(win_score_map, on=["股票代码", "窗口回测最优均线周期"], how="left")
                        # 全量最优均线周期的稳定性评分
                        .merge(
                            stab_score_map.rename(columns={
                                "均线周期": "全量回测最优均线周期",
                                "参数稳定性综合评分": "全量回测最优均线周期参数稳定性综合评分"
                            }),
                            on=["股票代码", "全量回测最优均线周期"],
                            how="left"
                        )
                        # 窗口最优均线周期的稳定性评分
                        .merge(
                            stab_score_map.rename(columns={
                                "均线周期": "窗口回测最优均线周期",
                                "参数稳定性综合评分": "窗口回测最优均线周期参数稳定性综合评分"
                            }),
                            on=["股票代码", "窗口回测最优均线周期"],
                            how="left"
                        )
                    )

                    compare["参数一致性检验"] = compare.apply(
                        lambda r: "一致"
                        if pd.isna(r["窗口回测最优均线周期"])
                        or r["全量回测最优均线周期"] == r["窗口回测最优均线周期"]
                        else "不一致",
                        axis=1
                    )

                    def _diff_ratio(full_val, win_val):
                        if pd.isna(full_val) or full_val == 0 or pd.isna(win_val):
                            return None
                        return round((full_val - win_val) / full_val * 100, 2)

                    compare["综合评分差值比"] = compare.apply(
                        lambda r: _diff_ratio(
                            r.get("全量回测最优均线周期回测结果综合评分"),
                            r.get("窗口回测最优均线周期回测结果综合评分")
                        ), axis=1
                    )

                    compare["参数稳定性综合评分差值比"] = compare.apply(
                        lambda r: _diff_ratio(
                            r.get("全量回测最优均线周期参数稳定性综合评分"),
                            r.get("窗口回测最优均线周期参数稳定性综合评分")
                        ), axis=1
                    )

                    def _weighted_score(score, stability):
                        if pd.isna(score) or pd.isna(stability):
                            return None
                        return round(0.6 * (score / 100) + 0.4 * stability, 4)

                    compare["全量参数两项评分加权得分"] = compare.apply(
                        lambda r: _weighted_score(
                            r.get("全量回测最优均线周期回测结果综合评分"),
                            r.get("全量回测最优均线周期参数稳定性综合评分")
                        ), axis=1
                    )

                    compare["窗口参数两项评分加权得分"] = compare.apply(
                        lambda r: _weighted_score(
                            r.get("窗口回测最优均线周期回测结果综合评分"),
                            r.get("窗口回测最优均线周期参数稳定性综合评分")
                        ), axis=1
                    )

                    compare["最终选择均线周期来源"] = compare.apply(
                        lambda r: "全量"
                        if pd.isna(r.get("窗口回测最优均线周期"))
                        or (r.get("全量参数两项评分加权得分") or 0)
                           >= (r.get("窗口参数两项评分加权得分") or 0)
                        else "窗口",
                        axis=1
                    )

                    compare["最终选择均线周期"] = compare.apply(
                        lambda r: r["全量回测最优均线周期"]
                        if r["最终选择均线周期来源"] == "全量"
                        else r["窗口回测最优均线周期"],
                        axis=1
                    )

                    compare["最终选择均线周期回测结果综合评分"] = compare.apply(
                        lambda r: r["全量回测最优均线周期回测结果综合评分"]
                        if r["最终选择均线周期来源"] == "全量"
                        else r["窗口回测最优均线周期回测结果综合评分"],
                        axis=1
                    )

                    def _stock_nature(score):
                        if pd.isna(score):
                            return None
                        if score >= 80:
                            return "1优"
                        elif score >= 60:
                            return "2良"
                        elif score >= 40:
                            return "3中"
                        elif score >= 20:
                            return "4差"
                        else:
                            return "5劣"

                    compare["股性评价"] = compare["最终选择均线周期回测结果综合评分"].apply(_stock_nature)

                    # 排序：参数一致性检验升序 → 股性评价升序 → 综合评分降序
                    compare["_sort_key"] = compare["参数一致性检验"].map({"一致": 0, "不一致": 1})
                    compare = compare.sort_values(
                        ["_sort_key", "股性评价", "最终选择均线周期回测结果综合评分"],
                        ascending=[True, True, False]
                    ).drop(columns=["_sort_key"])

                    # 重排列顺序
                    compare_col_order = [
                        "股票代码", "K线周期",
                        "全量回测最优均线周期", "全量回测最优均线周期回测结果综合评分",
                        "全量回测最优均线周期参数稳定性综合评分",
                        "窗口回测最优均线周期", "窗口回测最优均线周期回测结果综合评分",
                        "窗口回测最优均线周期参数稳定性综合评分",
                        "参数一致性检验", "综合评分差值比", "参数稳定性综合评分差值比",
                        "全量参数两项评分加权得分", "窗口参数两项评分加权得分",
                        "最终选择均线周期来源", "最终选择均线周期",
                        "最终选择均线周期回测结果综合评分", "股性评价"
                    ]
                    compare = compare.reindex(columns=compare_col_order)

            # =========================================================
            # 信号扫描（用最终选择均线周期筛选）
            # =========================================================
            if compare is not None and not compare.empty:
                selected_map = compare[["股票代码", "最终选择均线周期", "股性评价"]].rename(
                    columns={"最终选择均线周期": "均线周期"}
                )
                watch_base = full_df.merge(selected_map, on=["股票代码", "均线周期"], how="inner")
                watch_df = watch_base[
                    (watch_base["距离买入信号已过天数"].notna()) &
                    (watch_base["距离买入信号已过天数"] < 30) &
                    (watch_base["距离买入信号已过天数"] > 4) &
                    (watch_base["趋势方向"] == 1) &
                    (watch_base["交易次数"] > 10) &
                    (watch_base["盈利交易率"] > 40)
                ].sort_values(
                    by=["综合评分", "距离买入信号已过天数"],
                    ascending=[False, True]
                )

                signal_df = full_df.merge(selected_map, on=["股票代码", "均线周期"], how="inner").copy()
                # 多头按距离买入信号天数排序，空头按距离卖出信号天数排序
                _bull = signal_df[signal_df["趋势方向"] == 1].sort_values(
                    by=["距离买入信号已过天数", "综合评分"], ascending=[True, False])
                _bear = signal_df[signal_df["趋势方向"] != 1].sort_values(
                    by=["距离卖出信号已过天数", "综合评分"], ascending=[True, False])
                signal_df = pd.concat([_bull, _bear], ignore_index=True)
            else:
                watch_df = best_score_strategy.copy()
                watch_df = watch_df[
                    (watch_df["距离买入信号已过天数"].notna()) &
                    (watch_df["距离买入信号已过天数"] < 30) &
                    (watch_df["距离买入信号已过天数"] > 4) &
                    (watch_df["趋势方向"] == 1) &
                    (watch_df["交易次数"] > 10) &
                    (watch_df["盈利交易率"] > 40)
                ].sort_values(
                    by=["综合评分", "距离买入信号已过天数"],
                    ascending=[False, True]
                )

                signal_df = full_df[full_df["信号"] != "NONE"].copy()
                _bull = signal_df[signal_df["趋势方向"] == 1].sort_values(
                    by=["距离买入信号已过天数", "综合评分"], ascending=[True, False])
                _bear = signal_df[signal_df["趋势方向"] != 1].sort_values(
                    by=["距离卖出信号已过天数", "综合评分"], ascending=[True, False])
                signal_df = pd.concat([_bull, _bear], ignore_index=True)

            # 信号扫描新增字段：距离买入/卖出信号收盘价涨跌幅
            signal_df["距离买入信号收盘价涨跌幅"] = signal_df.apply(
                lambda r: round((r["收盘价"] - r["买入信号收盘价"]) / r["买入信号收盘价"] * 100, 2)
                if pd.notna(r.get("买入信号收盘价")) and r["买入信号收盘价"] != 0 else None,
                axis=1
            )
            signal_df["距离卖出信号收盘价涨跌幅"] = signal_df.apply(
                lambda r: round((r["收盘价"] - r["卖出信号收盘价"]) / r["卖出信号收盘价"] * 100, 2)
                if pd.notna(r.get("卖出信号收盘价")) and r["卖出信号收盘价"] != 0 else None,
                axis=1
            )
            # 插入到对应天数字段后面
            buy_idx = signal_df.columns.get_loc("距离买入信号已过天数") + 1
            signal_df.insert(buy_idx, "距离买入信号收盘价涨跌幅", signal_df.pop("距离买入信号收盘价涨跌幅"))
            sell_idx = signal_df.columns.get_loc("距离卖出信号已过天数") + 1
            signal_df.insert(sell_idx, "距离卖出信号收盘价涨跌幅", signal_df.pop("距离卖出信号收盘价涨跌幅"))

            # =========================================================
            # 均线趋势共振分析
            # =========================================================
            trend_lookup = full_df.set_index(["股票代码", "均线周期"])["趋势方向"]

            def _calc_confluence(r):
                code = r["股票代码"]
                final_ma = r["均线周期"]
                mas = [m for m in MA_LIST if m <= final_ma]
                dirs = [trend_lookup.get((code, m)) for m in mas]
                dirs = [d for d in dirs if d is not None]
                if all(d == 1 for d in dirs):
                    return "多头共振", len(mas), ",".join(str(m) for m in mas)
                if all(d == -1 for d in dirs):
                    return "空头共振", len(mas), ",".join(str(m) for m in mas)
                return "无", 0, ""

            _confluence = signal_df.apply(_calc_confluence, axis=1, result_type="expand")
            signal_df["均线趋势共振方向"] = _confluence.iloc[:, 0]
            signal_df["共振均线数量"] = _confluence.iloc[:, 1]
            signal_df["共振均线列表"] = _confluence.iloc[:, 2]

            # 预计持仓进度 = 距离买入信号已过天数 / 平均持仓天数（仅多头时计算）
            signal_df["预计持仓进度"] = signal_df.apply(
                lambda r: round(r["距离买入信号已过天数"] / r["平均持仓天数"], 4)
                if pd.notna(r.get("距离买入信号已过天数"))
                   and r.get("平均持仓天数", 0) > 0
                   and r["趋势方向"] == 1
                else None,
                axis=1
            )
            # 插入到距离卖出信号收盘价涨跌幅后面
            _idx = signal_df.columns.get_loc("距离卖出信号收盘价涨跌幅") + 1
            signal_df.insert(_idx, "预计持仓进度", signal_df.pop("预计持仓进度"))

            # 信号扫描列重排
            _signal_cols = [
                "股票代码", "K线周期", "均线周期", "综合评分", "股性评价",
                "时间", "收盘价", "HA收盘价", "HA均线值",
                "趋势方向", "信号",
                "买入信号时间", "买入信号收盘价", "距离买入信号已过天数", "距离买入信号收盘价涨跌幅",
                "卖出信号时间", "卖出信号收盘价", "距离卖出信号已过天数", "距离卖出信号收盘价涨跌幅",
                "预计持仓进度",
                "均线趋势共振方向", "共振均线数量", "共振均线列表",
                "收益率", "年化收益率", "买入持有收益率", "超额收益率",
                "最大回撤", "Sharpe", "Calmar Ratio",
                "交易次数", "盈利交易率", "盈利因子", "盈亏比",
                "平均盈利", "平均亏损", "最大单笔盈利", "最大单笔亏损",
                "最大连续盈利次数", "最大连续亏损次数", "平均持仓天数",
                "初始资金", "最终资金",
                "回测周期", "回测类型", "窗口", "窗口内有效数据日期"
            ]
            signal_df = signal_df[[c for c in _signal_cols if c in signal_df.columns]]

            # =========================================================
            # 回测数据均线方向转译为中文
            # =========================================================
            dir_map = {1: "多头", -1: "空头"}
            for _df in [full_df, watch_df, signal_df, all_df, win_df, win_best_full_result]:
                if "趋势方向" in _df.columns:
                    _df["趋势方向"] = _df["趋势方向"].map(dir_map)

            # =========================================================
            # 按目标顺序写入
            # =========================================================

            # 1
            signal_df.to_excel(writer, sheet_name="信号扫描", index=False)
            # 信号扫描条件格式
            _ws = writer.sheets["信号扫描"]
            _nr = len(signal_df) + 1  # data row count (excl header)
            # 综合评分 - 绿色数据条 (自动最小/最大值, 实体填充)
            _sc = get_column_letter(signal_df.columns.get_loc("综合评分") + 1)
            _rule1 = DataBarRule(start_type="min", end_type="max", color="70AD47", showValue=True)
            _rule1.dataBar._gradient = False
            _ws.conditional_formatting.add(f"{_sc}2:{_sc}{_nr}", _rule1)
            # 预计持仓进度 - 蓝色数据条 (0~1, 实体填充)
            _pc = get_column_letter(signal_df.columns.get_loc("预计持仓进度") + 1)
            _rule2 = DataBarRule(start_type="num", start_value=0, end_type="num", end_value=1, color="5B9BD5", showValue=True)
            _rule2.dataBar._gradient = False
            _ws.conditional_formatting.add(f"{_pc}2:{_pc}{_nr}", _rule2)

            # 3
            if compare is not None and not compare.empty:
                compare.to_excel(writer, sheet_name="个股最终选择均线周期", index=False)

            # 4
            all_df[all_df["回测类型"] == "全量"].to_excel(writer, sheet_name="全量回测明细", index=False)

            # 5
            best_score_strategy.to_excel(writer, sheet_name="全量回测个股评分最优参数回测结果汇总", index=False)

            # 6
            if not win_df.empty:
                win_df.to_excel(writer, sheet_name="窗口回测明细", index=False)

            # 7
            if not score_all.empty:
                score_all.to_excel(writer, sheet_name="窗口回测综合评分明细", index=False)

            # 8
            if not rank_all.empty:
                rank_all.to_excel(writer, sheet_name="窗口回测综合评分排名", index=False)

            # 9
            if all_stability is not None and not all_stability.empty:
                all_stability.to_excel(writer, sheet_name="窗口回测参数稳定性分析", index=False)

            # 10
            if stab_best is not None and not stab_best.empty:
                stab_best.to_excel(writer, sheet_name="窗口回测个股评分最优参数", index=False)

                # 11
                if not win_best_full_result.empty:
                    win_best_full_result.to_excel(writer, sheet_name="窗口回测个股评分最优参数回测结果汇总", index=False)

            # =========================================================
            # 统计逻辑（Sheet说明 + 字段说明合并）
            # =========================================================
            logic_rows = [
                # ==================== Sheet级说明 ====================
                {"类型": "Sheet说明", "名称": "信号扫描",
                 "统计逻辑": "用最终选择均线周期筛选全量回测明细，保留信号非NONE的行；多头按距离买入信号天数升序+综合评分降序排，空头按距离卖出信号天数升序+综合评分降序排；并计算距离买入/卖出信号收盘价涨跌幅及均线趋势共振分析"},
                {"类型": "Sheet说明", "名称": "个股最终选择均线周期",
                 "统计逻辑": "合并全量最优均线周期和窗口稳定性最优均线周期，对比两参数的综合评分、稳定性评分，计算加权总分后选择最终均线周期，并输出股性评价"},
                {"类型": "Sheet说明", "名称": "全量回测明细",
                 "统计逻辑": "所有股票所有MA的全量回测汇总结果（含综合评分、信号状态、收益/风险指标等），回测类型=全量"},
                {"类型": "Sheet说明", "名称": "全量回测个股评分最优参数回测结果汇总",
                 "统计逻辑": "每只股票从全量回测明细中取综合评分最高的一条，跨股票按综合评分降序排列"},
                {"类型": "Sheet说明", "名称": "窗口回测明细",
                 "统计逻辑": "所有股票所有窗口所有MA的窗口回测汇总结果（含综合评分、窗口标签、有效数据日期等）"},
                {"类型": "Sheet说明", "名称": "窗口回测综合评分明细",
                 "统计逻辑": "透视表，行=股票代码+K线周期+均线周期，列=窗口时间区间，值=综合评分"},
                {"类型": "Sheet说明", "名称": "窗口回测综合评分排名",
                 "统计逻辑": "透视表，同上结构，值改为窗口内排名（每窗口每股票内的参数间排名，同分取最小排名）"},
                {"类型": "Sheet说明", "名称": "窗口回测参数稳定性分析",
                 "统计逻辑": "按均线周期聚合：窗口数量、综合评分排名平均值/第一次数/Top3占比/标准差、参数稳定性综合评分（加权归一化）"},
                {"类型": "Sheet说明", "名称": "窗口回测个股评分最优参数",
                 "统计逻辑": "从各股票参数稳定性分析中取参数稳定性综合评分最高的均线周期，并列时取标准差最小的"},
                {"类型": "Sheet说明", "名称": "窗口回测个股评分最优参数回测结果汇总",
                 "统计逻辑": "拿窗口稳定性最优参数选出的均线周期，去全量回测明细中匹配同股票+同均线的完整回测结果"},

                # ==================== 字段级说明 ====================
                {"类型": "", "名称": "", "统计逻辑": ""},

                {"类型": "基础信息", "名称": "股票代码",
                 "统计逻辑": "来自 symbols.csv，每只股票唯一标识"},
                {"类型": "基础信息", "名称": "K线周期",
                 "统计逻辑": "固定参数 BAR_INTERVAL（当前为1W周线）"},
                {"类型": "基础信息", "名称": "均线周期",
                 "统计逻辑": "来自 MA_LIST 循环遍历（5/10/20/30/60）"},
                {"类型": "基础信息", "名称": "回测周期",
                 "统计逻辑": "首条K线时间 ~ 最后一条K线时间（df datetime范围）"},

                {"类型": "信号逻辑", "名称": "信号",
                 "统计逻辑": "MA方向变化：dir由-1→1为BUY，1→-1为SELL"},
                {"类型": "信号逻辑", "名称": "趋势方向",
                 "统计逻辑": "ma > ma.shift(1) 为多头，否则空头"},
                {"类型": "信号逻辑", "名称": "买入信号时间",
                 "统计逻辑": "df中 buy=True 的最后一条记录时间"},
                {"类型": "信号逻辑", "名称": "卖出信号时间",
                 "统计逻辑": "df中 sell=True 的最后一条记录时间"},
                {"类型": "信号逻辑", "名称": "买入信号收盘价",
                 "统计逻辑": "最近buy信号时的收盘价"},
                {"类型": "信号逻辑", "名称": "卖出信号收盘价",
                 "统计逻辑": "最近sell信号时的收盘价"},
                {"类型": "信号逻辑", "名称": "距离买入信号已过天数",
                 "统计逻辑": "当前日期 - 最近buy信号日期（自然日差）"},
                {"类型": "信号逻辑", "名称": "距离卖出信号已过天数",
                 "统计逻辑": "当前日期 - 最近sell信号日期（自然日差）"},
                {"类型": "信号逻辑", "名称": "距离买入信号收盘价涨跌幅",
                 "统计逻辑": "(当前收盘价 - 买入信号收盘价) / 买入信号收盘价 × 100，正数表示现价高于买入信号价格"},
                {"类型": "信号逻辑", "名称": "距离卖出信号收盘价涨跌幅",
                 "统计逻辑": "(当前收盘价 - 卖出信号收盘价) / 卖出信号收盘价 × 100，正数表示现价高于卖出信号价格"},
                {"类型": "信号逻辑", "名称": "均线趋势共振方向",
                 "统计逻辑": "最终选择均线周期及以下各周期趋势方向全部一致时为多头共振/空头共振，否则为无"},
                {"类型": "信号逻辑", "名称": "共振均线数量",
                 "统计逻辑": "参与方向一致的均线周期个数"},
                {"类型": "信号逻辑", "名称": "共振均线列表",
                 "统计逻辑": "参与方向一致的均线周期列表（逗号分隔）"},
                {"类型": "信号逻辑", "名称": "预计持仓进度",
                 "统计逻辑": "距离买入信号已过天数 / 平均持仓天数，仅多头时计算，反映当前持仓在平均持仓中的进度比例"},

                {"类型": "收益类", "名称": "收益率",
                 "统计逻辑": "(最终资金 / 初始资金 - 1) × 100"},
                {"类型": "收益类", "名称": "年化收益率",
                 "统计逻辑": "(最终资金/初始资金)^(1/年数) - 1，按复利年化"},
                {"类型": "收益类", "名称": "买入持有收益率",
                 "统计逻辑": "(最后收盘价 / 第一个收盘价 - 1) × 100"},
                {"类型": "收益类", "名称": "超额收益率",
                 "统计逻辑": "策略收益率 - 买入持有收益率"},

                {"类型": "风险类", "名称": "最大回撤",
                 "统计逻辑": "(当前权益 - 历史最高权益) / 历史最高权益 的最小值绝对值"},
                {"类型": "风险类", "名称": "Sharpe",
                 "统计逻辑": "log(资金曲线收益)均值 / 标准差 × sqrt(TRADING_PERIOD)"},
                {"类型": "风险类", "名称": "Calmar Ratio",
                 "统计逻辑": "年化收益率 / 最大回撤绝对值"},

                {"类型": "交易统计", "名称": "交易次数",
                 "统计逻辑": "build_trades生成的平仓交易数量"},
                {"类型": "交易统计", "名称": "盈利交易率",
                 "统计逻辑": "盈利交易数 / 总交易数 × 100%"},
                {"类型": "交易统计", "名称": "盈利因子",
                 "统计逻辑": "所有盈利总和 / |所有亏损总和|"},
                {"类型": "交易统计", "名称": "盈亏比",
                 "统计逻辑": "平均单笔盈利 / 平均单笔亏损绝对值"},

                {"类型": "单笔交易", "名称": "平均盈利",
                 "统计逻辑": "所有盈利交易收益均值"},
                {"类型": "单笔交易", "名称": "平均亏损",
                 "统计逻辑": "所有亏损交易亏损均值绝对值"},
                {"类型": "单笔交易", "名称": "最大单笔盈利",
                 "统计逻辑": "单笔交易最大收益"},
                {"类型": "单笔交易", "名称": "最大单笔亏损",
                 "统计逻辑": "单笔交易最大亏损"},

                {"类型": "连续性", "名称": "最大连续盈利次数",
                 "统计逻辑": "连续盈利交易的最长连续计数"},
                {"类型": "连续性", "名称": "最大连续亏损次数",
                 "统计逻辑": "连续亏损交易的最长连续计数"},

                {"类型": "持仓", "名称": "平均持仓天数",
                 "统计逻辑": "每笔交易持仓天数均值（exit_time - entry_time）"},

                {"类型": "资金", "名称": "初始资金",
                 "统计逻辑": "固定参数 INITIAL_CASH"},
                {"类型": "资金", "名称": "最终资金",
                 "统计逻辑": "权益曲线最后一个值（含已实现+浮动盈亏）"},

                {"类型": "评分模型", "名称": "综合评分",
                 "统计逻辑": "Score = 0.30*CAGR + 0.25*Sharpe + 0.20*(1-最大回撤) + 0.15*盈利因子 + 0.05*盈利交易率 + 0.05*交易次数；子指标min-max归一化，CAGR:0-30, Sharpe:0-2, 回撤:0-50, 盈利因子:1-3, 盈利率:30-80, 交易次数:10-100，加权求和范围0~100"},

                {"类型": "参数稳定性", "名称": "盈利窗口数量",
                 "统计逻辑": "该均线周期在各窗口中的年化收益率>0的计数"},
                {"类型": "参数稳定性", "名称": "盈利窗口占比",
                 "统计逻辑": "盈利窗口数量 / 窗口数量 × 100%"},
                {"类型": "参数稳定性", "名称": "年化收益率平均值",
                 "统计逻辑": "各窗口年化收益率的算术平均值"},
                {"类型": "参数稳定性", "名称": "年化收益率标准差",
                 "统计逻辑": "各窗口年化收益率的标准差，衡量收益波动性"},
                {"类型": "参数稳定性", "名称": "参数稳定性综合评分",
                 "统计逻辑": "Score = 0.20*avg_rank_n + 0.10*top3_n + 0.15*std_n + 0.25*cagr_n + 0.20*win_rate_n + 0.10*cagr_std_n；各指标min-max归一化，排名类占0.45，收益类(均值+占比+标准差)占0.55，越高越稳定"},
            ]

            pd.DataFrame(logic_rows).to_excel(writer, sheet_name="统计逻辑", index=False)

            for ws in writer.sheets.values():
                _apply_sheet_format(ws)

        print("全市场完成:", out)

# =========================================================
if __name__ == "__main__":
    init_symbols_file(SYMBOL_FILE)
    run_trade()