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

        # =========================
        # 买入
        # =========================
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

        # =========================
        # 卖出
        # =========================
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

    # =========================
    # 强制平仓
    # =========================
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
# 资金曲线（真实）
# =========================================================
def equity_curve(df, trades_df):

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    trades_df = trades_df.copy()
    trades_df["开仓时间"] = pd.to_datetime(trades_df["开仓时间"])
    trades_df["平仓时间"] = pd.to_datetime(trades_df["平仓时间"])

    equity = []

    for i in range(len(df)):

        t = df["datetime"].iloc[i]

        # 找到当前持仓状态
        active = trades_df[
            (trades_df["开仓时间"] <= t) &
            (trades_df["平仓时间"] > t)
        ]

        closed = trades_df[trades_df["平仓时间"] <= t]

        equity_value = INITIAL_CASH + closed["收益金额"].sum()

        # ⭐关键：如果有持仓，用浮动盈亏修正
        if not active.empty:
            row = active.iloc[0]
            entry = row["开仓价格"]
            size = row["买入股数"]
            price = df["close"].iloc[i]

            floating = (price - entry) * size
            equity_value += floating

        equity.append(equity_value)

    return np.array(equity)


# =========================================================
# 最大回撤
# =========================================================
def max_drawdown(curve):
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    return abs(dd.min())


# =========================================================
# Sharpe
# =========================================================
def sharpe_from_equity(curve):

    ret = np.diff(curve) / curve[:-1]

    if len(ret) < 3:
        return 0

    return np.mean(ret) / (np.std(ret) + 1e-9) * np.sqrt(TRADING_PERIOD)


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
# 汇总（完整指标体系）
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

    # ===================== Performance =====================
    ret = (final / INITIAL_CASH - 1) * 100
    years = max((end - start).days / 365, 1/365)
    cagr = ((final / INITIAL_CASH) ** (1 / years) - 1) * 100

    buy_hold = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    alpha = ret - buy_hold

    # ===================== Risk =====================
    mdd = max_drawdown(curve) * 100
    sh = sharpe_from_equity(curve)
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    # ===================== Trade =====================
    win = trades_df[trades_df["收益金额"] > 0]
    loss = trades_df[trades_df["收益金额"] < 0]

    win_rate = len(win) / len(trades_df) * 100
    pf = win["收益金额"].sum() / abs(loss["收益金额"].sum()) if len(loss) else 0

    avg_win = win["收益金额"].mean() if len(win) else 0
    avg_loss = abs(loss["收益金额"].mean()) if len(loss) else 0

    payoff = avg_win / avg_loss if avg_loss else 0

    # ===================== Distribution =====================
    max_win_trade = trades_df["收益金额"].max()
    max_loss_trade = trades_df["收益金额"].min()

    # ===================== Streak =====================
    max_win_streak, max_loss_streak = streaks(trades_df)

    # ===================== Holding =====================
    avg_hold = trades_df["持仓天数"].mean()

    return {
        "股票代码": df["code"].iloc[0],
        "均线周期": ma_len,

        # ================= Performance =================
        "收益率": round(float(ret), 2),
        "年化收益率": round(float(cagr), 2),
        "买入持有收益率": round(float(buy_hold), 2),
        "超额收益率": round(float(alpha), 2),

        # ================= Risk =================
        "最大回撤": round(float(mdd), 2),
        "Sharpe": round(float(sh), 4),
        "Calmar Ratio": round(float(calmar), 4),

        # ================= Trade =================
        "交易次数": int(len(trades_df)),
        "盈利交易率": round(float(win_rate), 2),
        "盈利因子": round(float(pf), 4),
        "盈亏比": round(float(payoff), 4),

        # ================= Distribution =================
        "平均盈利": round(float(avg_win), 4),
        "平均亏损": round(float(avg_loss), 4),
        "最大单笔盈利": round(float(max_win_trade), 4),
        "最大单笔亏损": round(float(max_loss_trade), 4),

        # ================= Streak =================
        "最大连续盈利次数": max_win_streak,
        "最大连续亏损次数": max_loss_streak,

        # ================= Holding =================
        "平均持仓天数": round(float(avg_hold), 2),

        # ================= Equity =================
        "初始资金": INITIAL_CASH,
        "最终资金": round(float(final), 2),

        # ================= Time =================
        "回测周期": period
    }


# =========================================================
# 主程序（含Excel输出）
# =========================================================
def run_trade():

    symbols = load_symbols(SYMBOL_FILE)
    all_rows = []

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

                    s = build_summary(trades, ma, df)
                    summary_rows.append(s)
                    all_rows.append(s)

                pd.DataFrame(summary_rows).to_excel(writer, sheet_name="汇总", index=False)

            print("完成:", code)

        except Exception as e:
            print("失败:", code, e)

    # ================= 全市场 =================
    if all_rows:

        all_df = pd.DataFrame(all_rows)

        best = all_df.loc[
            all_df.groupby("股票代码")["年化收益率"].idxmax()
        ]

        out = os.path.join(TRADE_DIR, "all_summary.xlsx")

        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            all_df.to_excel(writer, sheet_name="全部回测", index=False)
            best.to_excel(writer, sheet_name="最佳参数", index=False)

            # ⭐新增：说明sheet：指标说明
            doc_df = pd.DataFrame([
                {"模块": "MA回测逻辑", "计算逻辑": "HA均线方向变化生成买卖信号"},
                {"模块": "资金曲线", "计算逻辑": "初始资金 + 每笔交易收益累加形成资金曲线"},
                {"模块": "收益率", "计算逻辑": "最终资金 / 初始资金 - 1"},
                {"模块": "年化收益率(CAGR)", "计算逻辑": "基于复利计算 (Final/Initial)^(1/年数)-1"},
                {"模块": "买入持有收益率", "计算逻辑": "买入并持有至回测结束收益"},
                {"模块": "超额收益率(Alpha)", "计算逻辑": "策略收益 - 买入持有收益"},
                {"模块": "最大回撤", "计算逻辑": "资金曲线回撤峰值计算"},
                {"模块": "Sharpe Ratio", "计算逻辑": "收益均值/标准差 × sqrt(52)"},
                {"模块": "Calmar Ratio", "计算逻辑": "CAGR / |最大回撤|"},
                {"模块": "交易次数", "计算逻辑": "交易记录条数统计"},
                {"模块": "盈利交易率", "计算逻辑": "盈利交易 / 总交易"},
                {"模块": "盈利因子", "计算逻辑": "盈利总额 / 亏损总额绝对值"},
                {"模块": "盈亏比", "计算逻辑": "平均盈利 / 平均亏损"},
                {"模块": "平均盈利", "计算逻辑": "所有盈利交易平均值"},
                {"模块": "平均亏损", "计算逻辑": "所有亏损交易平均值"},
                {"模块": "最大单笔盈利", "计算逻辑": "单笔最大盈利交易"},
                {"模块": "最大单笔亏损", "计算逻辑": "单笔最大亏损交易"},
                {"模块": "最大连续盈利次数", "计算逻辑": "连续盈利次数最大值"},
                {"模块": "最大连续亏损次数", "计算逻辑": "连续亏损次数最大值"},
                {"模块": "平均持仓天数", "计算逻辑": "每笔交易持仓天数平均"}
            ])

            doc_df.to_excel(writer, sheet_name="指标说明", index=False)

        print("全市场完成:", out)


# =========================================================
if __name__ == "__main__":
    init_symbols_file(SYMBOL_FILE)
    run_trade()