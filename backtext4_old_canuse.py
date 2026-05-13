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

    equity = np.zeros(len(df))

    cash = INITIAL_CASH
    trade_idx = 0

    if trades_df is None or trades_df.empty:
        return np.full(len(df), INITIAL_CASH)

    required_cols = {"开仓时间", "平仓时间", "开仓价格", "买入股数"}

    if not required_cols.issubset(trades_df.columns):
        return np.full(len(df), INITIAL_CASH)

    trades_df = trades_df.sort_values("平仓时间").reset_index(drop=True)

    active_trade = None

    for i in range(len(df)):

        t = df["datetime"].iloc[i]
        price = df["close"].iloc[i]

        # =========================
        # 处理已平仓交易（O(1)推进）
        # =========================
        while trade_idx < len(trades_df) and trades_df.loc[trade_idx, "平仓时间"] <= t:
            cash += trades_df.loc[trade_idx, "收益金额"]
            trade_idx += 1

        # =========================
        # 当前持仓浮盈
        # =========================
        floating = 0

        if trade_idx < len(trades_df):
            row = trades_df.loc[trade_idx]

            if row["开仓时间"] <= t < row["平仓时间"]:
                floating = (price - row["开仓价格"]) * row["买入股数"]

        equity[i] = cash + floating

        equity[i] = max(equity[i], 1e-6)

    return equity


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

    curve = np.array(curve)

    # ⭐关键：过滤非法值
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
# 回测评分
# =========================================================
def normalize(x, min_v, max_v):
    if pd.isna(x):
        return 0
    if max_v == min_v:
        return 0

    # ⭐关键：防止超出区间导致爆分
    x = max(min_v, min(x, max_v))

    return (x - min_v) / (max_v - min_v)


def calc_score_row(row):

    # =========================
    # 边界（可后续调参）
    # =========================
    cagr_min, cagr_max = 0, 30
    sharpe_min, sharpe_max = 0, 2
    dd_min, dd_max = 0, 50
    pf_min, pf_max = 1, 3
    win_min, win_max = 30, 80
    trade_min, trade_max = 10, 100

    # =========================
    # 各子分数
    # =========================
    cagr_score = normalize(row.get("年化收益率", 0), cagr_min, cagr_max) * 100
    sharpe_score = normalize(row.get("Sharpe", 0), sharpe_min, sharpe_max) * 100

    dd_score = (1 - normalize(row.get("最大回撤", 0), dd_min, dd_max)) * 100

    pf_score = normalize(row.get("盈利因子", 0), pf_min, pf_max) * 100

    win_score = normalize(row.get("盈利交易率", 0), win_min, win_max) * 100

    trade_score = normalize(
        min(row.get("交易次数", 0), trade_max),
        trade_min,
        trade_max
    ) * 100

    # =========================
    # 总分
    # =========================
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
        "K线周期": BAR_INTERVAL,
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
                    s["综合评分"] = calc_score_row(s)
                    summary_rows.append(s)
                    all_rows.append(s)

                pd.DataFrame(summary_rows).to_excel(writer, sheet_name="汇总", index=False)

            print("完成:", code)

        except Exception as e:
            print("失败:", code, e)

    # ================= 全市场 =================
    if all_rows:

        all_df = pd.DataFrame(all_rows)

        all_df["综合评分"] = all_df.apply(calc_score_row, axis=1)

        # 每只股票：年化收益率最优策略
        best_cagr_strategy = all_df.loc[
            all_df.groupby("股票代码")["年化收益率"].idxmax()
        ].sort_values(
            by="年化收益率",
            ascending=False
        )

        # 每只股票：综合评分最优策略
        best_score_strategy = all_df.loc[
            all_df.groupby("股票代码")["综合评分"].idxmax()
        ].copy().sort_values(
            by="综合评分",
            ascending=False
        )

        out = os.path.join(TRADE_DIR, "all_summary.xlsx")

        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            all_df.to_excel(writer, sheet_name="全部回测", index=False)
            best_cagr_strategy.to_excel(writer, sheet_name="个股年化最优参数", index=False)
            best_score_strategy.to_excel(writer, sheet_name="个股评分最优参数", index=False)

            # ⭐新增：说明sheet：指标说明
            doc_df = pd.DataFrame([
                {"模块": "MA回测逻辑", "计算逻辑": "基于Heikin Ashi收盘价计算移动平均线，均线方向变化生成买卖信号"},
                {"模块": "信号生成", "计算逻辑": "MA上升(>前一周期)=多头趋势，下降=空头趋势；趋势切换产生交易信号"},
                {"模块": "资金曲线", "计算逻辑": "初始资金 + 已实现盈亏 + 持仓浮动盈亏（逐K线更新）"},
                {"模块": "收益率", "计算逻辑": "(最终资金 / 初始资金 - 1) × 100%"},
                {"模块": "年化收益率(CAGR)", "计算逻辑": "(最终资金/初始资金)^(1/年数) - 1，按复利计算"},
                {"模块": "买入持有收益率", "计算逻辑": "(回测结束价格 / 起始价格 - 1) × 100%"},
                {"模块": "超额收益率(Alpha)", "计算逻辑": "策略收益率 - 买入持有收益率"},
                {"模块": "最大回撤", "计算逻辑": "(当前资金曲线 - 历史最高点) / 历史最高点 的最小值绝对值"},
                {"模块": "Sharpe Ratio", "计算逻辑": "平均收益 / 收益标准差 × √TRADING_PERIOD（基于资金曲线收益率）"},
                {"模块": "Calmar Ratio", "计算逻辑": "年化收益率 / 最大回撤绝对值"},
                {"模块": "交易次数", "计算逻辑": "完整平仓交易记录数量"},
                {"模块": "盈利交易率", "计算逻辑": "盈利交易次数 / 总交易次数 × 100%"},
                {"模块": "盈利因子", "计算逻辑": "盈利总额 / 亏损总额绝对值"},
                {"模块": "盈亏比", "计算逻辑": "平均单笔盈利 / 平均单笔亏损绝对值"},
                {"模块": "平均盈利", "计算逻辑": "所有盈利交易收益均值"},
                {"模块": "平均亏损", "计算逻辑": "所有亏损交易亏损均值绝对值"},
                {"模块": "最大单笔盈利", "计算逻辑": "单笔交易最大盈利金额"},
                {"模块": "最大单笔亏损", "计算逻辑": "单笔交易最大亏损金额"},
                {"模块": "最大连续盈利次数", "计算逻辑": "连续盈利交易的最长连续长度"},
                {"模块": "最大连续亏损次数", "计算逻辑": "连续亏损交易的最长连续长度"},
                {"模块": "平均持仓天数", "计算逻辑": "每笔交易持仓时间平均值（天）"},
                {"模块": "初始资金", "计算逻辑": "回测初始本金（固定值）"},
                {"模块": "最终资金", "计算逻辑": "回测结束后的账户总资金（含浮盈）"},
                {"模块": "回测周期", "计算逻辑": "首条K线时间 ~ 最后一条K线时间"},
                {"模块": "综合评分", "计算逻辑": "Score = 0.30*CAGR + 0.25*Sharpe + 0.20*(1-最大回撤) + 0.15*盈利因子 + 0.05*盈利交易率 + 0.05*交易次数稳定性；所有子指标先进行[min-max归一化]并限制在0~100区间（CAGR:0-30, Sharpe:0-2, 回撤:0-50, 盈利因子:1-3, 盈利率:30-80, 交易次数:10-100），再按权重加权求和，最终得分范围0~100"}
            ])

            doc_df.to_excel(writer, sheet_name="指标说明", index=False)

        print("全市场完成:", out)


# =========================================================
if __name__ == "__main__":
    init_symbols_file(SYMBOL_FILE)
    run_trade()