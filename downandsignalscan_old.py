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
SYMBOL_FILE = "symbols.csv"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# HAKMA参数
MA_LIST = [5, 10, 20, 30, 60]


# =========================================================
# ① 股票列表
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
# ② 下载周线数据
# =========================================================
def fetch_weekly(
    code,
    start_str,
    end_str,
    quote_ctx
):

    all_data = []

    page_req_key = None

    last_max_time = None

    while True:

        ret, data, page_req_key = quote_ctx.request_history_kline(
            code=code,
            start=start_str,
            end=end_str,
            ktype=KLType.K_WEEK,
            autype=AuType.QFQ,
            page_req_key=page_req_key
        )

        if ret != RET_OK:

            print(code, "下载失败")

            break

        if data is None or len(data) == 0:
            break

        current_max = data["time_key"].max()

        if (
            last_max_time is not None
            and current_max <= last_max_time
        ):
            break

        last_max_time = current_max

        all_data.append(data)

        if not page_req_key:
            break

    if not all_data:
        return pd.DataFrame()

    return pd.concat(
        all_data,
        ignore_index=True
    )


def save_data(df, code):

    if df.empty:
        return

    file_path = os.path.join(
        DATA_DIR,
        f"{code}_1w.parquet"
    )

    df = df[[
        "code",
        "time_key",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover"
    ]].rename(columns={
        "time_key": "datetime"
    })

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    if os.path.exists(file_path):

        old = pd.read_parquet(file_path)

        df = pd.concat(
            [old, df],
            ignore_index=True
        )

    df = (
        df
        .drop_duplicates(["code", "datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    df.to_parquet(
        file_path,
        index=False
    )

    print(code, "下载完成:", len(df))


def run_download():

    init_symbols_file(SYMBOL_FILE)

    symbols = load_symbols(SYMBOL_FILE)

    end = datetime.now()

    start = datetime(2000, 1, 3)

    start_str = start.strftime("%Y-%m-%d")

    end_str = end.strftime("%Y-%m-%d")

    quote_ctx = OpenQuoteContext(
        host="127.0.0.1",
        port=11111
    )

    for code in symbols:

        print("\n下载:", code)

        df = fetch_weekly(
            code,
            start_str,
            end_str,
            quote_ctx
        )

        save_data(df, code)

    quote_ctx.close()


# =========================================================
# ③ HA计算
# =========================================================
def calc_heikin_ashi(df):

    ha_close = (
        df["open"]
        + df["high"]
        + df["low"]
        + df["close"]
    ) / 4

    ha_open = np.zeros(len(df))

    ha_open[0] = (
        df["open"].iloc[0]
        + df["close"].iloc[0]
    ) / 2

    for i in range(1, len(df)):

        ha_open[i] = (
            ha_open[i - 1]
            + ha_close.iloc[i - 1]
        ) / 2

    ha_dir = np.where(
        ha_close > ha_open,
        1,
        -1
    )

    return ha_close, ha_open, ha_dir


# =========================================================
# ④ 指标计算
# =========================================================
def calc_signal(df, ma_len):

    df = df.copy()

    ha_close, _, _ = calc_heikin_ashi(df)

    df["ha_close"] = ha_close

    df["ma"] = (
        df["ha_close"]
        .rolling(ma_len, min_periods=ma_len)
        .mean()
    )

    # 保持原逻辑：
    # MA相等时 = -1
    df["dir"] = np.where(
        df["ma"] > df["ma"].shift(1),
        1,
        -1
    )

    df["buy"] = (
        (df["dir"] == 1)
        & (df["dir"].shift(1) == -1)
    )

    df["sell"] = (
        (df["dir"] == -1)
        & (df["dir"].shift(1) == 1)
    )

    return df


# =========================================================
# ⑤ 获取最近信号信息
# =========================================================
def get_last_signal_info(df):

    today = pd.Timestamp.today().normalize()

    # =========================
    # BUY
    # =========================
    buy_rows = df[df["buy"]]

    if len(buy_rows) > 0:

        last_buy = buy_rows.iloc[-1]

        buy_time = pd.to_datetime(
            last_buy["datetime"]
        )

        buy_close = round(
            float(last_buy["close"]),
            2
        )

        buy_days = (
            today
            - buy_time.normalize()
        ).days

    else:

        buy_time = pd.NaT
        buy_close = None
        buy_days = None

    # =========================
    # SELL
    # =========================
    sell_rows = df[df["sell"]]

    if len(sell_rows) > 0:

        last_sell = sell_rows.iloc[-1]

        sell_time = pd.to_datetime(
            last_sell["datetime"]
        )

        sell_close = round(
            float(last_sell["close"]),
            2
        )

        sell_days = (
            today
            - sell_time.normalize()
        ).days

    else:

        sell_time = pd.NaT
        sell_close = None
        sell_days = None

    return (
        buy_time,
        buy_close,
        buy_days,
        sell_time,
        sell_close,
        sell_days
    )


# =========================================================
# ⑥ 指标筛选
# =========================================================
def run_scan():

    symbols = load_symbols(SYMBOL_FILE)

    results = []

    for code in symbols:

        path = os.path.join(
            DATA_DIR,
            f"{code}_1w.parquet"
        )

        if not os.path.exists(path):

            print(code, "无数据")

            continue

        df = pd.read_parquet(path)

        df = (
            df
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        if len(df) < max(MA_LIST) + 5:

            print(code, "数据不足")

            continue

        for ma in MA_LIST:

            df_tmp = calc_signal(df, ma)

            last = df_tmp.iloc[-1]

            # =========================
            # 当前信号
            # =========================
            signal = "NONE"

            if last["buy"]:
                signal = "BUY"

            elif last["sell"]:
                signal = "SELL"

            # =========================
            # 最近BUY/SELL信息
            # =========================
            (
                buy_time,
                buy_close,
                buy_days,
                sell_time,
                sell_close,
                sell_days
            ) = get_last_signal_info(df_tmp)

            results.append({

                "symbol":
                    code,

                "ma":
                    ma,

                "datetime":
                    last["datetime"],

                "close":
                    round(last["close"], 2),

                "ha_close":
                    round(last["ha_close"], 2),

                "ma_value":
                    round(last["ma"], 2),

                "dir":
                    int(last["dir"]),

                "signal":
                    signal,

                "buy_signal_time":
                    buy_time,

                "buy_signal_close":
                    buy_close,

                "buy_signal_days":
                    buy_days,

                "sell_signal_time":
                    sell_time,

                "sell_signal_close":
                    sell_close,

                "sell_signal_days":
                    sell_days
            })

            print(
                code,
                "MA",
                ma,
                signal
            )

    # =====================================================
    # DataFrame
    # =====================================================
    df_result = pd.DataFrame(results)

    # 仅信号
    signal_df = df_result[
        df_result["signal"] != "NONE"
    ].copy()

    # =====================================================
    # 控制台输出
    # =====================================================
    print("\n==============================")
    print("最新指标提醒")
    print("==============================")

    if len(signal_df) > 0:

        print(signal_df)

    else:

        print("无最新信号")

    # =====================================================
    # Excel输出
    # =====================================================
    excel_path = os.path.join(
        RESULT_DIR,
        f"scan_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:

        # 全部结果
        df_result.to_excel(
            writer,
            sheet_name="all",
            index=False
        )

        # 仅信号结果
        signal_df.to_excel(
            writer,
            sheet_name="signal",
            index=False
        )

    print("\n结果保存:", excel_path)


# =========================================================
# ⑦ 主入口
# =========================================================
if __name__ == "__main__":

    # 下载数据
    run_download()

    # 指标扫描
    run_scan()