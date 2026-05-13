import os
import pandas as pd
import numpy as np
from datetime import datetime
from futu import OpenQuoteContext, KLType, AuType, RET_OK

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

    ha_close, _ = calc_heikin_ashi(df)

    df["ha_close"] = ha_close
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

    def norm_price(x):
        return round(float(x), 2)

    for i in range(len(df)):

        price = norm_price(df["close"].iloc[i])

        if pd.isna(price) or price <= 0:
            continue

        time = df["datetime"].iloc[i]

        if df["buy"].iloc[i] and position == 0:

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

        elif df["sell"].iloc[i] and position > 0:

            cash_before_sell = available_cash

            sell_value = position * price
            sell_fee = sell_value * FEE_RATE

            buy_fee = entry_price * position * FEE_RATE
            total_fee = buy_fee + sell_fee

            pnl = (price - entry_price) * position - total_fee

            available_cash += (sell_value - sell_fee)
            cash_after_sell = available_cash

            cost_basis = entry_price * position + buy_fee
            return_pct = pnl / cost_basis * 100 if cost_basis > 0 else 0

            trades.append({
                "股票代码": df["code"].iloc[0],
                "K线周期": BAR_INTERVAL,
                "均线周期": ma_len,

                "开仓时间": entry_time,
                "开仓价格": entry_price,
                "买入股数": position,

                "平仓时间": time,
                "平仓价格": price,
                "卖出股数": position,

                "交易状态": "已平仓",
                "订单盈亏类型": "盈利" if pnl > 0 else "亏损",

                "收益金额": round(pnl, 4),
                "收益率(%)": round(return_pct, 4),

                "买入手续费": round(buy_fee, 4),
                "卖出手续费": round(sell_fee, 4),
                "总手续费": round(total_fee, 4),

                "开仓前可用现金": round(cash_before_buy, 2),
                "开仓后可用现金": round(cash_after_buy, 2),
                "平仓前可用现金": round(cash_before_sell, 2),
                "平仓后可用现金": round(cash_after_sell, 2),

                "持仓K线数": i - entry_index,
                "持仓天数": (time - entry_time).days,

                "回测周期": backtest_period
            })

            position = 0

    if position > 0:

        price = norm_price(df["close"].iloc[-1])
        time = df["datetime"].iloc[-1]

        cash_before_sell = available_cash

        sell_value = position * price
        sell_fee = sell_value * FEE_RATE

        buy_fee = entry_price * position * FEE_RATE
        total_fee = buy_fee + sell_fee

        pnl = (price - entry_price) * position - total_fee

        available_cash += (sell_value - sell_fee)
        cash_after_sell = available_cash

        cost_basis = entry_price * position + buy_fee
        return_pct = pnl / cost_basis * 100 if cost_basis > 0 else 0

        trades.append({
            "股票代码": df["code"].iloc[0],
            "K线周期": BAR_INTERVAL,
            "均线周期": ma_len,

            "开仓时间": entry_time,
            "开仓价格": entry_price,
            "买入股数": position,

            "平仓时间": time,
            "平仓价格": price,
            "卖出股数": position,

            "交易状态": "未平仓(强制结算)",
            "订单盈亏类型": "盈利" if pnl > 0 else "亏损",

            "收益金额": round(pnl, 4),
            "收益率(%)": round(return_pct, 4),

            "买入手续费": round(buy_fee, 4),
            "卖出手续费": round(sell_fee, 4),
            "总手续费": round(total_fee, 4),

            "开仓前可用现金": None,
            "开仓后可用现金": None,
            "平仓前可用现金": round(cash_before_sell, 2),
            "平仓后可用现金": round(cash_after_sell, 2),

            "持仓K线数": len(df) - entry_index,
            "持仓天数": (time - entry_time).days,

            "回测周期": backtest_period
        })

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

    for i in range(len(df)):

        t = df["datetime"].iloc[i]
        price = df["close"].iloc[i]

        while trade_idx < len(trades_df) and trades_df.loc[trade_idx, "平仓时间"] <= t:
            cash += trades_df.loc[trade_idx, "收益金额"]
            trade_idx += 1

        floating = 0

        if trade_idx < len(trades_df):
            row = trades_df.loc[trade_idx]

            if row["开仓时间"] <= t < row["平仓时间"]:
                floating = (price - row["开仓价格"]) * row["买入股数"]

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
    start = dates.min()
    end = dates.max()

    windows = []
    cur = start

    while cur + pd.DateOffset(years=WINDOW_YEARS) <= end:
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
# 主程序
# =========================================================
def run_trade():

    symbols = load_symbols(SYMBOL_FILE)

    all_rows = []
    scan_rows = []
    window_rows = []

    for code in symbols:

        try:

            path = os.path.join(DATA_DIR, f"{code}_1w.parquet")

            if not os.path.exists(path):
                continue

            df = pd.read_parquet(path)

            df = df.sort_values("datetime")
            df["code"] = code

            out_file = os.path.join(TRADE_DIR, f"{code}_trades.xlsx")

            summary_rows = []

            with pd.ExcelWriter(out_file, engine="openpyxl") as writer:

                for ma in MA_LIST:

                    d = calc_signal(df, ma)

                    trades = build_trades(d, ma)

                    trades.to_excel(writer, sheet_name=f"MA_{ma}", index=False)

                    summary = build_summary(trades, ma, df)

                    signal_info = get_last_signal_info(d)

                    summary.update(signal_info)

                    summary["综合评分"] = calc_score_row(summary)
                    summary["回测类型"] = "全量"
                    summary["窗口"] = "FULL"

                    summary_rows.append(summary)
                    all_rows.append(summary)
                    scan_rows.append(summary)

                # =============================================
                # 5年窗口回测
                # =============================================
                windows = generate_windows(df)

                for ws, we in windows:

                    df_w = df[
                        (pd.to_datetime(df["datetime"]) >= ws) &
                        (pd.to_datetime(df["datetime"]) < we)
                    ].copy()

                    if df_w.empty or len(df_w) < 10:
                        continue

                    window_label = f"{ws.date()}~{we.date()}"

                    for ma in MA_LIST:
                        d = calc_signal(df_w, ma)
                        trades = build_trades(d, ma)

                        summary = build_summary(trades, ma, df_w)
                        summary["回测类型"] = "窗口"
                        summary["窗口"] = window_label

                        summary_rows.append(summary)
                        all_rows.append(summary)
                        window_rows.append(summary)

                pd.DataFrame(summary_rows).to_excel(writer, sheet_name="汇总", index=False)

            print("完成:", code)

        except Exception as e:
            print("失败:", code, e)

    if all_rows:

        all_df = pd.DataFrame(all_rows)

        full_df = all_df[all_df["回测类型"] == "全量"].copy()

        best_cagr_strategy = full_df.loc[
            full_df.groupby("股票代码")["年化收益率"].idxmax()
        ].sort_values(by="年化收益率", ascending=False)

        best_score_strategy = full_df.loc[
            full_df.groupby("股票代码")["综合评分"].idxmax()
        ].sort_values(by="综合评分", ascending=False)

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

        scan_df = pd.DataFrame(scan_rows)

        signal_df = scan_df[scan_df["信号"] != "NONE"].copy()

        out = os.path.join(TRADE_DIR, "all_summary.xlsx")

        with pd.ExcelWriter(out, engine="openpyxl") as writer:

            all_df.to_excel(writer, sheet_name="全部回测", index=False)

            signal_df.to_excel(writer, sheet_name="信号扫描", index=False)

            best_cagr_strategy.to_excel(writer, sheet_name="个股年化最优参数", index=False)

            best_score_strategy.to_excel(writer, sheet_name="个股评分最优参数", index=False)

            watch_df.to_excel(writer, sheet_name="最近可关注股票", index=False)

            # =========================================================
            # 窗口回测汇总
            # =========================================================
            if window_rows:
                win_df = pd.DataFrame(window_rows)
                win_df.to_excel(writer, sheet_name="窗口回测明细", index=False)

                win_best = win_df.loc[
                    win_df.groupby(["股票代码", "窗口"])["年化收益率"].idxmax()
                ].sort_values(by="年化收益率", ascending=False)

                win_best.to_excel(writer, sheet_name="窗口个股最优", index=False)

            # =========================================================
            # 字段统计逻辑（详细版）
            # =========================================================
            schema_df = pd.DataFrame([

                # ========================= 基础信息 =========================
                {"字段": "股票代码", "统计逻辑": "来自 symbols.csv，每只股票唯一标识"},
                {"字段": "K线周期", "统计逻辑": "固定参数 BAR_INTERVAL（当前为1W周线）"},
                {"字段": "均线周期", "统计逻辑": "来自 MA_LIST 循环遍历（5/10/20/30/60）"},
                {"字段": "回测周期", "统计逻辑": "首条K线时间 ~ 最后一条K线时间（df datetime范围）"},

                # ========================= 信号逻辑 =========================
                {"字段": "信号", "统计逻辑": "MA方向变化：dir由-1→1为BUY，1→-1为SELL"},
                {"字段": "趋势方向", "统计逻辑": "ma > ma.shift(1) 为1，否则-1"},
                {"字段": "买入信号时间", "统计逻辑": "df中 buy=True 的最后一条记录时间"},
                {"字段": "卖出信号时间", "统计逻辑": "df中 sell=True 的最后一条记录时间"},
                {"字段": "距离买入信号已过天数", "统计逻辑": "当前日期 - 最近buy信号日期（自然日差）"},

                # ========================= 收益类 =========================
                {"字段": "收益率", "统计逻辑": "(最终资金 / 初始资金 - 1) × 100"},
                {"字段": "年化收益率", "统计逻辑": "(最终资金/初始资金)^(1/年数) - 1，按复利年化"},
                {"字段": "买入持有收益率", "统计逻辑": "(最后收盘价 / 第一个收盘价 - 1) × 100"},
                {"字段": "超额收益率", "统计逻辑": "策略收益率 - 买入持有收益率"},

                # ========================= 风险类 =========================
                {"字段": "最大回撤", "统计逻辑": "(当前权益 - 历史最高权益) / 历史最高权益 的最小值绝对值"},
                {"字段": "Sharpe", "统计逻辑": "log(资金曲线收益)均值 / 标准差 × sqrt(TRADING_PERIOD)"},
                {"字段": "Calmar Ratio", "统计逻辑": "年化收益率 / 最大回撤绝对值"},

                # ========================= 交易统计 =========================
                {"字段": "交易次数", "统计逻辑": "build_trades生成的平仓交易数量"},
                {"字段": "盈利交易率", "统计逻辑": "盈利交易数 / 总交易数 × 100%"},
                {"字段": "盈利因子", "统计逻辑": "所有盈利总和 / |所有亏损总和|"},
                {"字段": "盈亏比", "统计逻辑": "平均单笔盈利 / 平均单笔亏损绝对值"},

                # ========================= 单笔交易 =========================
                {"字段": "平均盈利", "统计逻辑": "所有盈利交易收益均值"},
                {"字段": "平均亏损", "统计逻辑": "所有亏损交易亏损均值绝对值"},
                {"字段": "最大单笔盈利", "统计逻辑": "单笔交易最大收益"},
                {"字段": "最大单笔亏损", "统计逻辑": "单笔交易最大亏损"},

                # ========================= 连续性 =========================
                {"字段": "最大连续盈利次数", "统计逻辑": "连续盈利交易的最长连续计数"},
                {"字段": "最大连续亏损次数", "统计逻辑": "连续亏损交易的最长连续计数"},

                # ========================= 持仓 =========================
                {"字段": "平均持仓天数", "统计逻辑": "每笔交易持仓天数均值（exit_time - entry_time）"},

                # ========================= 资金 =========================
                {"字段": "初始资金", "统计逻辑": "固定参数 INITIAL_CASH"},
                {"字段": "最终资金", "统计逻辑": "权益曲线最后一个值（含已实现+浮动盈亏）"},

                # ========================= 评分模型 =========================
                {"字段": "综合评分", "统计逻辑": "Score = 0.30*CAGR + 0.25*Sharpe + 0.20*(1-最大回撤) + 0.15*盈利因子 + 0.05*盈利交易率 + 0.05*交易次数稳定性；所有子指标先进行[min-max归一化]并限制在0~100区间（CAGR:0-30, Sharpe:0-2, 回撤:0-50, 盈利因子:1-3, 盈利率:30-80, 交易次数:10-100），再按权重加权求和，最终得分范围0~100"},

            ])

            schema_df.to_excel(writer, sheet_name="字段统计逻辑", index=False)

        print("全市场完成:", out)

# =========================================================
if __name__ == "__main__":
    init_symbols_file(SYMBOL_FILE)
    run_trade()