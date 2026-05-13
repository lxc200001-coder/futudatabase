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


# =========================================================
# 股票列表
# =========================================================
def init_symbols_file(path):

    if not os.path.exists(path):

        pd.DataFrame({
            "code": [
                "US.TSLA",
                "US.AAPL",
                "US.NVDA",
                "US.MSFT"
            ]
        }).to_csv(path, index=False)


def load_symbols(path):

    return (
        pd.read_csv(path)["code"]
        .dropna()
        .tolist()
    )


# =========================================================
# HA计算
# =========================================================
def calc_heikin_ashi(df):

    ha_close = (
        df["open"] +
        df["high"] +
        df["low"] +
        df["close"]
    ) / 4

    ha_open = np.zeros(len(df))

    ha_open[0] = (
        df["open"].iloc[0] +
        df["close"].iloc[0]
    ) / 2

    for i in range(1, len(df)):

        ha_open[i] = (
            ha_open[i - 1] +
            ha_close.iloc[i - 1]
        ) / 2

    return ha_close, ha_open


# =========================================================
# 信号
# =========================================================
def calc_signal(df, ma_len):

    df = df.copy()

    ha_close, _ = calc_heikin_ashi(df)

    df["ha_close"] = ha_close

    df["ma"] = (
        df["ha_close"]
        .rolling(
            ma_len,
            min_periods=ma_len
        )
        .mean()
    )

    df["dir"] = np.where(
        df["ma"] > df["ma"].shift(1),
        1,
        -1
    )

    df["buy"] = (
        (df["dir"] == 1) &
        (df["dir"].shift(1) == -1)
    )

    df["sell"] = (
        (df["dir"] == -1) &
        (df["dir"].shift(1) == 1)
    )

    return df


# =========================================================
# 交易回测
# =========================================================
def build_trades(df, ma_len):

    cash = INITIAL_CASH

    available_cash = cash

    position = 0

    entry_price = 0
    entry_time = None
    entry_index = 0

    trades = []

    def norm_price(x):

        return round(float(x), 2)

    for i in range(len(df)):

        price = norm_price(
            df["close"].iloc[i]
        )

        time = df["datetime"].iloc[i]

        # ================= BUY =================
        if (
            df["buy"].iloc[i]
            and position == 0
        ):

            pre_cash = available_cash

            shares = int(
                available_cash /
                (price * (1 + FEE_RATE))
            )

            if shares <= 0:
                continue

            cost = shares * price

            buy_fee = cost * FEE_RATE

            available_cash -= (
                cost + buy_fee
            )

            cash = available_cash

            position = shares

            entry_price = price
            entry_time = time
            entry_index = i

        # ================= SELL =================
        elif (
            df["sell"].iloc[i]
            and position > 0
        ):

            pre_cash = available_cash

            sell_value = position * price

            sell_fee = (
                sell_value * FEE_RATE
            )

            buy_fee = (
                entry_price *
                position *
                FEE_RATE
            )

            total_fee = (
                buy_fee + sell_fee
            )

            pnl = (
                (price - entry_price)
                * position
                - total_fee
            )

            available_cash += (
                sell_value - sell_fee
            )

            cash = available_cash

            trades.append({

                "股票代码":
                    df["code"].iloc[0],

                "均线周期":
                    ma_len,

                "开仓时间":
                    entry_time,

                "开仓价格":
                    entry_price,

                "买入股数":
                    position,

                "平仓时间":
                    time,

                "平仓价格":
                    price,

                "卖出股数":
                    position,

                "交易状态":
                    "已平仓",

                "订单盈亏类型":
                    (
                        "盈利"
                        if pnl > 0
                        else "亏损"
                    ),

                "收益率(%)":
                    round(
                        pnl /
                        (
                            entry_price *
                            position
                        ) * 100,
                        4
                    ),

                "收益金额":
                    round(pnl, 4),

                "买入手续费":
                    round(buy_fee, 4),

                "卖出手续费":
                    round(sell_fee, 4),

                "总手续费":
                    round(total_fee, 4),

                "开仓后可用现金":
                    round(pre_cash, 2),

                "平仓后可用现金":
                    round(
                        available_cash,
                        2
                    ),

                "持仓K线数":
                    i - entry_index,

                "持仓天数":
                    (
                        i - entry_index
                    ) * 7
            })

            position = 0

    # ================= 强制平仓 =================
    if position > 0:

        price = round(
            df["close"].iloc[-1],
            2
        )

        time = (
            df["datetime"].iloc[-1]
        )

        pre_cash = available_cash

        sell_value = (
            position * price
        )

        sell_fee = (
            sell_value * FEE_RATE
        )

        buy_fee = (
            entry_price *
            position *
            FEE_RATE
        )

        total_fee = (
            buy_fee + sell_fee
        )

        pnl = (
            (price - entry_price)
            * position
            - total_fee
        )

        available_cash += (
            sell_value - sell_fee
        )

        cash = available_cash

        trades.append({

            "股票代码":
                df["code"].iloc[0],

            "均线周期":
                ma_len,

            "开仓时间":
                entry_time,

            "开仓价格":
                entry_price,

            "买入股数":
                position,

            "平仓时间":
                time,

            "平仓价格":
                price,

            "卖出股数":
                position,

            "交易状态":
                "未平仓(强制结算)",

            "订单盈亏类型":
                (
                    "盈利"
                    if pnl > 0
                    else "亏损"
                ),

            "收益率(%)":
                round(
                    pnl /
                    (
                        entry_price *
                        position
                    ) * 100,
                    4
                ),

            "收益金额":
                round(pnl, 4),

            "买入手续费":
                round(buy_fee, 4),

            "卖出手续费":
                round(sell_fee, 4),

            "总手续费":
                round(total_fee, 4),

            "开仓后可用现金":
                round(pre_cash, 2),

            "平仓后可用现金":
                round(
                    available_cash,
                    2
                ),

            "持仓K线数":
                (
                    len(df)
                    - entry_index
                ),

            "持仓天数":
                (
                    len(df)
                    - entry_index
                ) * 7
        })

    return pd.DataFrame(trades)


# =========================================================
# 汇总统计
# =========================================================
def build_summary(
    trades_df,
    ma_len,
    df
):

    if trades_df.empty:

        return {

            "股票代码":
                df["code"].iloc[0],

            "均线周期":
                ma_len,

            "收益率":
                0,

            "年化收益率":
                0,

            "交易次数":
                0,

            "盈利交易率":
                0,

            "盈利因子":
                0
        }

    # ================= 总收益 =================
    total_return = (
        trades_df["收益金额"]
        .sum()
    )

    final_cash = (
        INITIAL_CASH
        + total_return
    )

    total_return_pct = (
        (
            final_cash
            / INITIAL_CASH
            - 1
        ) * 100
    )

    # ================= 修正后的年化收益率 =================
    start_date = pd.to_datetime(
        df["datetime"].iloc[0]
    )

    end_date = pd.to_datetime(
        df["datetime"].iloc[-1]
    )

    total_days = max(
        (end_date - start_date).days,
        1
    )

    years = total_days / 365

    annual_return = (
        (
            final_cash /
            INITIAL_CASH
        ) ** (1 / years) - 1
    ) * 100

    # ================= 交易统计 =================
    total_trades = len(
        trades_df
    )

    win_trades = len(

        trades_df[
            trades_df["收益金额"] > 0
        ]
    )

    win_rate = (

        win_trades /
        total_trades * 100

        if total_trades > 0
        else 0
    )

    # ================= 盈利因子 =================
    gross_profit = (

        trades_df[
            trades_df["收益金额"] > 0
        ]["收益金额"]
        .sum()
    )

    gross_loss = abs(

        trades_df[
            trades_df["收益金额"] < 0
        ]["收益金额"]
        .sum()
    )

    profit_factor = (

        gross_profit /
        gross_loss

        if gross_loss != 0
        else 0
    )

    return {

        "股票代码":
            trades_df["股票代码"].iloc[0],

        "均线周期":
            ma_len,

        "收益率":
            round(
                total_return_pct,
                2
            ),

        "年化收益率":
            round(
                annual_return,
                2
            ),

        "交易次数":
            total_trades,

        "盈利交易率":
            round(
                win_rate,
                2
            ),

        "盈利因子":
            round(
                profit_factor,
                2
            )
    }


# =========================================================
# 执行回测
# =========================================================
def run_trade():

    symbols = load_symbols(
        SYMBOL_FILE
    )

    # =====================================================
    # 所有股票汇总
    # =====================================================
    all_summary_rows = []

    for code in symbols:

        path = os.path.join(
            DATA_DIR,
            f"{code}_1w.parquet"
        )

        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)

        df = (
            df
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        df["code"] = code

        file_path = os.path.join(

            TRADE_DIR,

            f"{code.replace('.', '_')}"
            f"_trades.xlsx"
        )

        summary_rows = []

        with pd.ExcelWriter(
            file_path,
            engine="openpyxl"
        ) as writer:

            for ma in MA_LIST:

                df_tmp = calc_signal(
                    df,
                    ma
                )

                trades = build_trades(
                    df_tmp,
                    ma
                )

                # ================= 保存交易明细 =================
                trades.to_excel(

                    writer,

                    sheet_name=f"MA_{ma}",

                    index=False
                )

                # ================= 汇总 =================
                summary = build_summary(
                    trades,
                    ma,
                    df
                )

                summary_rows.append(
                    summary
                )

                # ================= 全市场汇总 =================
                all_summary_rows.append(
                    summary
                )

            # ================= 单股票汇总 =================
            summary_df = pd.DataFrame(
                summary_rows
            )

            summary_df.to_excel(

                writer,

                sheet_name="汇总",

                index=False
            )

        print("完成:", code)

    # =====================================================
    # 输出全市场汇总xlsx
    # =====================================================
    if len(all_summary_rows) > 0:

        all_df = pd.DataFrame(
            all_summary_rows
        )

        # =================================================
        # 每个股票最佳年化收益率
        # =================================================
        best_df = (

            all_df

            .sort_values(
                "年化收益率",
                ascending=False
            )

            .groupby("股票代码")

            .head(1)

            .reset_index(drop=True)
        )

        # =================================================
        # 输出xlsx
        # =================================================
        all_result_path = os.path.join(
            TRADE_DIR,
            "all_summary.xlsx"
        )

        with pd.ExcelWriter(
            all_result_path,
            engine="openpyxl"
        ) as writer:

            # ================= 全部数据 =================
            all_df.to_excel(

                writer,

                sheet_name="all",

                index=False
            )

            # ================= 最佳参数 =================
            best_df.to_excel(

                writer,

                sheet_name="best",

                index=False
            )

        print("全市场汇总完成")


# =========================================================
# 主入口
# =========================================================
if __name__ == "__main__":

    init_symbols_file(
        SYMBOL_FILE
    )

    run_trade()