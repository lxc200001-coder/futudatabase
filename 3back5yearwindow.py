import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
# 回测（核心逻辑不变）
# =========================================================
def build_trades(df, ma_len):
    cash = INITIAL_CASH
    available_cash = cash
    position = 0
    entry_price = 0
    entry_time = None
    entry_index = 0
    trades = []

    def norm_price(x): return round(float(x), 2)

    for i in range(len(df)):
        price = norm_price(df["close"].iloc[i])
        time = df["datetime"].iloc[i]

        if df["buy"].iloc[i] and position == 0:
            pre_cash = available_cash
            shares = int(available_cash / (price * (1 + FEE_RATE)))
            if shares <= 0: continue

            cost = shares * price
            buy_fee = cost * FEE_RATE
            available_cash -= (cost + buy_fee)

            position = shares
            entry_price = price
            entry_time = time
            entry_index = i

        elif df["sell"].iloc[i] and position > 0:
            pre_cash = available_cash
            sell_value = position * price
            sell_fee = sell_value * FEE_RATE
            buy_fee = entry_price * position * FEE_RATE
            total_fee = buy_fee + sell_fee

            pnl = (price - entry_price) * position - total_fee
            available_cash += (sell_value - sell_fee)

            trades.append({
                "股票代码": df["code"].iloc[0],
                "均线周期": ma_len,
                "开仓时间": entry_time,
                "开仓价格": entry_price,
                "买入股数": position,
                "平仓时间": time,
                "平仓价格": price,
                "卖出股数": position,
                "交易状态": "已平仓",
                "订单盈亏类型": "盈利" if pnl > 0 else "亏损",
                "收益率(%)": round(pnl / (entry_price * position) * 100, 4),
                "收益金额": round(pnl, 4),
                "买入手续费": round(buy_fee, 4),
                "卖出手续费": round(sell_fee, 4),
                "总手续费": round(total_fee, 4),
                "开仓后可用现金": round(pre_cash, 2),
                "平仓后可用现金": round(available_cash, 2),
                "持仓K线数": i - entry_index,
                "持仓天数": (i - entry_index) * 7
            })
            position = 0

    return pd.DataFrame(trades)

# =========================================================
# 汇总统计
# =========================================================
def build_summary(trades_df, ma_len, df):
    if trades_df.empty:
        return {"收益率": 0, "年化收益率": 0, "交易次数": 0, "盈利交易率": 0, "盈利因子": 0}

    total_return = trades_df["收益金额"].sum()
    final_cash = INITIAL_CASH + total_return

    total_return_pct = (final_cash / INITIAL_CASH - 1) * 100

    start_date = pd.to_datetime(df["datetime"].iloc[0])
    end_date = pd.to_datetime(df["datetime"].iloc[-1])
    years = max((end_date - start_date).days, 1) / 365

    annual_return = ((final_cash / INITIAL_CASH) ** (1 / years) - 1) * 100

    total_trades = len(trades_df)
    win_trades = len(trades_df[trades_df["收益金额"] > 0])
    win_rate = win_trades / total_trades * 100 if total_trades else 0

    gross_profit = trades_df[trades_df["收益金额"] > 0]["收益金额"].sum()
    gross_loss = abs(trades_df[trades_df["收益金额"] < 0]["收益金额"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss else 0

    return {
        "收益率": round(total_return_pct, 2),
        "年化收益率": round(annual_return, 2),
        "交易次数": total_trades,
        "盈利交易率": round(win_rate, 2),
        "盈利因子": round(profit_factor, 2)
    }

# =========================================================
# 5年滚动窗口
# =========================================================
def generate_windows(df):
    dates = pd.to_datetime(df["datetime"])
    start = pd.Timestamp("2000-01-03")
    end = dates.max()

    windows = []
    cur = start

    while cur + pd.DateOffset(years=WINDOW_YEARS) <= end:
        windows.append((cur, cur + pd.DateOffset(years=WINDOW_YEARS)))
        cur += pd.DateOffset(years=STEP_YEARS)

    return windows

# =========================================================
# 执行回测（核心升级）
# =========================================================
def run_trade():
    symbols = load_symbols(SYMBOL_FILE)
    all_summary_rows = []

    for code in symbols:
        path = os.path.join(DATA_DIR, f"{code}_1w.parquet")
        if not os.path.exists(path): continue

        df = pd.read_parquet(path).sort_values("datetime").reset_index(drop=True)
        df["code"] = code

        file_path = os.path.join(TRADE_DIR, f"{code.replace('.', '_')}_trades.xlsx")
        summary_rows = []

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:

            # =================================================
            # 1. FULL 回测
            # =================================================
            for ma in MA_LIST:
                df_tmp = calc_signal(df, ma)
                trades = build_trades(df_tmp, ma)
                summary = build_summary(trades, ma, df)

                summary_rows.append({
                    "股票代码": code,
                    "均线周期": ma,
                    **summary,
                    "窗口": "FULL",
                    "回测类型": "全量"
                })

                all_summary_rows.append({
                    "股票代码": code,
                    "均线周期": ma,
                    **summary,
                    "窗口": "FULL",
                    "回测类型": "全量"
                })

            # =================================================
            # 2. WINDOW 回测
            # =================================================
            windows = generate_windows(df)

            for w_idx, (start, end) in enumerate(windows):

                df_w = df[(pd.to_datetime(df["datetime"]) >= start) &
                          (pd.to_datetime(df["datetime"]) < end)].copy()

                if df_w.empty:
                    continue

                window_label = f"{start.date()}~{end.date()}"

                for ma in MA_LIST:
                    df_tmp = calc_signal(df_w, ma)
                    trades = build_trades(df_tmp, ma)
                    summary = build_summary(trades, ma, df_w)

                    summary_rows.append({
                        "股票代码": code,
                        "均线周期": ma,
                        **summary,
                        "窗口": window_label,
                        "回测类型": "窗口"
                    })

                    all_summary_rows.append({
                        "股票代码": code,
                        "均线周期": ma,
                        **summary,
                        "窗口": window_label,
                        "回测类型": "窗口"
                    })

            # =================================================
            # 写入汇总sheet（唯一汇总）
            # =================================================
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="汇总", index=False)

        print("完成:", code)

    # =========================================================
    # 全市场汇总
    # =========================================================
    if all_summary_rows:
        all_df = pd.DataFrame(all_summary_rows)

        best_df = all_df.loc[
            all_df.groupby("股票代码")["年化收益率"].idxmax()
        ].reset_index(drop=True)

        with pd.ExcelWriter(os.path.join(TRADE_DIR, "all_summary.xlsx"), engine="openpyxl") as writer:
            all_df.to_excel(writer, sheet_name="all", index=False)
            best_df.to_excel(writer, sheet_name="best", index=False)

        print("全市场汇总完成")

# =========================================================
# 主入口
# =========================================================
if __name__ == "__main__":
    init_symbols_file(SYMBOL_FILE)
    run_trade()