import os
import time
import requests
import pandas as pd
import numpy as np

from collections import deque
from datetime import datetime
from futu import OpenQuoteContext, KLType, AuType, RET_OK

# =========================================================
# 配置
# =========================================================
DATA_DIR = "data"
RESULT_DIR = "results"
SYMBOL_FILE = "symbols.csv"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

MA_LIST = [5, 10, 20, 30, 60]

# =========================================================
# API限速
# =========================================================
REQUEST_LIMIT = 60
WINDOW_SECONDS = 30

request_times = deque()


def wait_rate_limit():

    now = time.time()

    while request_times and now - request_times[0] > WINDOW_SECONDS:
        request_times.popleft()

    if len(request_times) >= REQUEST_LIMIT:

        sleep_time = WINDOW_SECONDS - (now - request_times[0]) + 0.5

        sleep_time = max(0, sleep_time)

        print(f"\n触发限频，开始倒计时 {sleep_time:.1f} 秒")

        start = time.time()

        # 倒计时显示
        while True:

            elapsed = time.time() - start
            remaining = sleep_time - elapsed

            if remaining <= 0:
                break

            print(f"\r剩余等待: {remaining:.1f} 秒", end="", flush=True)

            time.sleep(0.2)

        print("\r剩余等待: 0.0 秒，继续执行        ")

    request_times.append(time.time())


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

    return pd.read_csv(path)["code"].dropna().tolist()


# =========================================================
# 全量开始日期（固定从 2000-01-03 开始）
# =========================================================
def get_start_date(code):
    return datetime(2000, 1, 3)


# =========================================================
# 下载周线数据
# =========================================================
def fetch_weekly(code, start_str, end_str, quote_ctx):

    all_data = []
    page_req_key = None
    last_max_time = None
    retry = 0

    while True:

        # 仅首页请求限频
        if page_req_key is None:
            wait_rate_limit()

        ret, data, page_req_key = quote_ctx.request_history_kline(
            code=code,
            start=start_str,
            end=end_str,
            ktype=KLType.K_WEEK,
            autype=AuType.QFQ,
            page_req_key=page_req_key
        )

        # 失败重试
        if ret != RET_OK:

            retry += 1

            print(code, f"请求失败，重试 {retry}/3")

            if retry >= 3:

                print(code, "下载失败")

                break

            time.sleep(2)

            continue

        retry = 0

        # 无数据
        if data is None or len(data) == 0:
            break

        # 防止分页重复
        current_max = data["time_key"].max()

        if last_max_time is not None and current_max <= last_max_time:
            break

        last_max_time = current_max

        all_data.append(data)

        # 无下一页
        if not page_req_key:
            break

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


# =========================================================
# Binance 周线数据（替代富途获取 BTC 等加密货币）
# =========================================================
def fetch_binance_weekly(code, start_str, end_str):
    """
    从 Binance 获取周 K 线数据，返回格式与 fetch_weekly 一致。
    仅支持 CC.BTCUSD → BTCUSDT。
    """
    symbol_map = {
        "CC.BTCUSD": "BTCUSDT",
        "CC.BTC": "BTCUSDT",
        "CC.ETHUSD": "ETHUSDT",
        "CC.ETH": "ETHUSDT",
    }
    binance_symbol = symbol_map.get(code)
    if binance_symbol is None:
        print(f"{code} 不支持 Binance 数据源")
        return pd.DataFrame()

    print(f"  Binance: {binance_symbol} 周线 {start_str} ~ {end_str}")

    base_url = "https://api.binance.com/api/v3/klines"
    start_ms = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_str).timestamp() * 1000)

    all_rows = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": binance_symbol,
            "interval": "1w",
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        try:
            resp = requests.get(base_url, params=params, timeout=15)
            resp.raise_for_status()
            klines = resp.json()
        except Exception as e:
            print(f"Binance 请求失败: {e}")
            break

        if not klines:
            break

        for k in klines:
            # [0]open_time_ms, [1]open, [2]high, [3]low, [4]close,
            # [5]volume, [6]close_time, [7]quote_vol, [8]trades, ...
            all_rows.append({
                "code": code,
                "time_key": pd.Timestamp(k[0], unit="ms"),  # ms -> datetime
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "turnover": float(k[7]),     # quote asset volume
            })

        # 下一页从最后一条的 close_time 开始
        current_start = klines[-1][6] + 1

        if len(klines) < 1000:
            break

        print(f"  Binance 分页: 已获取 {len(all_rows)} 条")

    if not all_rows:
        return pd.DataFrame()

    print(f"  Binance: 共获取 {len(all_rows)} 条周线数据")

    return pd.DataFrame(all_rows)


# =========================================================
# 保存数据
# =========================================================
def save_data(df, code):

    if df.empty:
        return

    file_path = os.path.join(DATA_DIR, f"{code}_1w.parquet")

    df = df[[
        "code",
        "time_key",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover"
    ]].rename(columns={"time_key": "datetime"})

    df["datetime"] = pd.to_datetime(df["datetime"])

    # 去重排序后直接保存（全量覆盖）
    df = (
        df
        .drop_duplicates(["code", "datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # 保存
    df.to_parquet(file_path, index=False)

    print(code, "保存完成:", len(df), "条")


# =========================================================
# 股票名称映射（批量按市场获取）
# =========================================================
def fetch_stock_basicinfo_map(symbols, quote_ctx):
    """获取股票名称，返回 ({code: name} dict, 正股代码列表)"""
    name_map = {}
    stock_codes = []
    group = {}
    for s in symbols:
        market = s.split(".")[0]
        group.setdefault(market, []).append(s)

    for market_prefix, codes in group.items():
        ret, data = quote_ctx.get_stock_basicinfo(
            market=market_prefix, code_list=codes
        )
        if ret == RET_OK and data is not None:
            for _, row in data.iterrows():
                code = row["code"]
                name_map[code] = row.get("name", "")
                if row.get("stock_type") == "STOCK":
                    stock_codes.append(code)
        else:
            print(f"获取名称失败: {market_prefix} {codes}")
    return name_map, stock_codes


# =========================================================
# 板块信息拉取
# =========================================================
def fetch_all_stock_plates(symbols, quote_ctx):
    """获取所有股票的板块信息，返回 DataFrame"""
    print("获取股票名称...")
    name_map, stock_codes = fetch_stock_basicinfo_map(symbols, quote_ctx)
    print(f"正股 {len(stock_codes)} 只（排除非正股 {len(symbols) - len(stock_codes)} 只）")

    all_rows = []
    batch_size = 200
    for batch_start in range(0, len(stock_codes), batch_size):
        batch = stock_codes[batch_start:batch_start + batch_size]
        print(f"板块批次 [{batch_start + 1}..{min(batch_start + batch_size, len(stock_codes))}/{len(stock_codes)}]")

        ret, data = quote_ctx.get_owner_plate(batch)
        if ret == RET_OK and data is not None and not data.empty:
            for _, row in data.iterrows():
                code = row.get("code", "")
                all_rows.append({
                    "code": code,
                    "stock_name": name_map.get(code, ""),
                    "plate_code": row.get("plate_code", ""),
                    "plate_name": row.get("plate_name", ""),
                    "plate_type": row.get("plate_type", ""),
                    "update_time": datetime.now()
                })

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates().sort_values(["code", "plate_type"]).reset_index(drop=True)

    # 聚合：排除OTHER，每只股票一行，板块名和类型逗号拼接
    agg = df[df["plate_type"] != "OTHER"] \
        .groupby(["code", "stock_name"], sort=False).agg(
            plates=("plate_name", lambda x: ",".join(x)),
            plate_type_list=("plate_type", lambda x: ",".join(x.unique())),
            update_time=("update_time", "first")
        ).reset_index()
    agg = agg.sort_values("code").reset_index(drop=True)

    return agg


# =========================================================
# 保存板块信息
# =========================================================
def save_stock_plates(df):
    """保存板块信息到 data/stocks_plates.parquet"""
    if df.empty:
        print("无板块数据，跳过保存")
        return
    path = os.path.join(DATA_DIR, "stocks_plates.parquet")
    df.to_parquet(path, index=False)
    print(f"板块信息保存完成: {path} 共 {len(df)} 条")


# =========================================================
# 板块同步主流程
# =========================================================
def run_plate_sync():
    """同步所有股票的板块信息"""
    symbols = load_symbols(SYMBOL_FILE)
    stock_symbols = [c for c in symbols if not c.startswith("CC.")]
    if not stock_symbols:
        print("无可同步板块信息的标的，跳过")
        return
    print(f"\n开始同步 {len(stock_symbols)} 只股票的板块信息...")
    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        df = fetch_all_stock_plates(stock_symbols, quote_ctx)
        save_stock_plates(df)
    finally:
        quote_ctx.close()


# =========================================================
# 下载主流程
# =========================================================
def run_download():

    init_symbols_file(SYMBOL_FILE)

    symbols = load_symbols(SYMBOL_FILE)

    end_str = datetime.now().strftime("%Y-%m-%d")

    non_crypto = [c for c in symbols if not c.startswith("CC.")]
    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111) if non_crypto else None

    for i, code in enumerate(symbols, 1):

        print("\n================================================")
        print(f"{i}/{len(symbols)} 下载: {code}")
        print("================================================")

        start = get_start_date(code)

        start_str = start.strftime("%Y-%m-%d")

        print("开始日期:", start_str)
        print("结束日期:", end_str)

        if code.startswith("CC."):
            print("  使用 Binance 数据源")
            df = fetch_binance_weekly(code, start_str, end_str)
        else:
            df = fetch_weekly(code, start_str, end_str, quote_ctx)

        save_data(df, code)

    if quote_ctx is not None:
        quote_ctx.close()


# =========================================================
# HA计算
# =========================================================
def calc_heikin_ashi(df):

    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

    ha_open = np.zeros(len(df))

    ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2

    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2

    ha_dir = np.where(ha_close > ha_open, 1, -1)

    return ha_close, ha_open, ha_dir


# =========================================================
# 指标计算
# =========================================================
def calc_signal(df, ma_len):

    df = df.copy()

    ha_close, _, _ = calc_heikin_ashi(df)

    df["ha_close"] = ha_close

    df["ma"] = df["ha_close"].rolling(ma_len, min_periods=ma_len).mean()

    # MA相等时 = -1
    df["dir"] = np.where(df["ma"] > df["ma"].shift(1), 1, -1)

    df["buy"] = (df["dir"] == 1) & (df["dir"].shift(1) == -1)

    df["sell"] = (df["dir"] == -1) & (df["dir"].shift(1) == 1)

    return df


# =========================================================
# 获取最近信号
# =========================================================
def get_last_signal_info(df):

    today = pd.Timestamp.today().normalize()

    # BUY
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

    # SELL
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

    return (
        buy_time,
        buy_close,
        buy_days,
        sell_time,
        sell_close,
        sell_days
    )


# =========================================================
# 指标扫描
# =========================================================
def run_scan():

    symbols = load_symbols(SYMBOL_FILE)

    results = []

    for code in symbols:

        path = os.path.join(DATA_DIR, f"{code}_1w.parquet")

        if not os.path.exists(path):

            print(code, "无数据")

            continue

        df = pd.read_parquet(path)

        df = df.sort_values("datetime").reset_index(drop=True)

        if len(df) < max(MA_LIST) + 5:

            print(code, "数据不足")

            continue

        for ma in MA_LIST:

            df_tmp = calc_signal(df, ma)

            last = df_tmp.iloc[-1]

            signal = "NONE"

            if last["buy"]:
                signal = "BUY"

            elif last["sell"]:
                signal = "SELL"

            (
                buy_time,
                buy_close,
                buy_days,
                sell_time,
                sell_close,
                sell_days
            ) = get_last_signal_info(df_tmp)

            results.append({

                "symbol": code,
                "ma": ma,
                "datetime": last["datetime"],
                "close": round(last["close"], 2),
                "ha_close": round(last["ha_close"], 2),
                "ma_value": round(last["ma"], 2),
                "dir": int(last["dir"]),
                "signal": signal,

                "buy_signal_time": buy_time,
                "buy_signal_close": buy_close,
                "buy_signal_days": buy_days,

                "sell_signal_time": sell_time,
                "sell_signal_close": sell_close,
                "sell_signal_days": sell_days
            })

    # DataFrame

    rename_map = {

        "symbol": "股票代码",
        "ma": "均线周期",
        "datetime": "时间",
        "close": "收盘价",
        "ha_close": "HA收盘价",
        "ma_value": "HA均线值",
        "dir": "趋势方向",
        "signal": "信号",

        "buy_signal_time": "买入信号时间",
        "buy_signal_close": "买入信号收盘价",
        "buy_signal_days": "距离买入信号已过天数",

        "sell_signal_time": "卖出信号时间",
        "sell_signal_close": "卖出信号收盘价",
        "sell_signal_days": "距离卖出信号已过天数"
    }

    df_result = pd.DataFrame(results)

    signal_df = df_result[df_result["signal"] != "NONE"].copy()

    # 控制台输出
    print("\n==============================")
    print("最新指标提醒")
    print("==============================")

    if len(signal_df) > 0:
        print(signal_df)
    else:
        print("无最新信号")

    # Excel输出
    # 中文版本（用于展示）
    df_all_cn = df_result.rename(columns=rename_map)
    df_signal_cn = signal_df.rename(columns=rename_map)

    # Excel输出
    excel_path = os.path.join(
        RESULT_DIR,
        f"scan_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        # 英文原始数据（方便回测/调试）
        df_result.to_excel(writer, sheet_name="all_raw", index=False)
        signal_df.to_excel(writer, sheet_name="signal_raw", index=False)

        # 中文展示数据（给人看）
        df_all_cn.to_excel(writer, sheet_name="all", index=False)
        df_signal_cn.to_excel(writer, sheet_name="signal", index=False)

    print("\n结果保存:", excel_path)


# =========================================================
# 主入口
# =========================================================
if __name__ == "__main__":

    # 下载数据
    run_download()

    # 板块信息同步
    run_plate_sync()

    # 指标扫描
    run_scan()