import os
import time
import requests
import urllib3
import pandas as pd

from collections import deque
from datetime import datetime
from futu import OpenQuoteContext, KLType, AuType, RET_OK

# 关闭 Binance SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =========================================================
# 配置
# =========================================================
DATA_DIR = "data"
SYMBOL_FILE = "symbols.csv"

os.makedirs(DATA_DIR, exist_ok=True)


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

    # Binance SSL 兼容：使用自定义 session
    sess = requests.Session()
    sess.verify = False
    sess.headers.update({"User-Agent": "Mozilla/5.0"})

    while current_start < end_ms:
        params = {
            "symbol": binance_symbol,
            "interval": "1w",
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        try:
            resp = sess.get(base_url, params=params, timeout=15)
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

    # 补充无板块数据的标的（如ETF、未返回板块的普通股票），确保名称能进入 stocks_plates.parquet
    found_codes = {r["code"] for r in all_rows}
    for code in name_map:
        if code not in found_codes:
            all_rows.append({
                "code": code,
                "stock_name": name_map.get(code, ""),
                "plate_code": "",
                "plate_name": "",
                "plate_type": "",
                "update_time": datetime.now()
            })

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

    all_dfs = []

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
        if not df.empty:
            all_dfs.append(df)

    if quote_ctx is not None:
        quote_ctx.close()

    # 合并所有股票数据输出总表
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.rename(columns={"time_key": "datetime"})
        combined["datetime"] = pd.to_datetime(combined["datetime"])
        combined = combined.drop_duplicates(["code", "datetime"]).sort_values(["code", "datetime"]).reset_index(drop=True)

        # 合并板块信息
        plates_path = os.path.join(DATA_DIR, "stocks_plates.parquet")
        if os.path.exists(plates_path):
            plates_df = pd.read_parquet(plates_path)[["code", "plates", "plate_type_list"]]
            combined = combined.merge(plates_df, on="code", how="left")

        # 列名统一：name → stock_name
        if "name" in combined.columns:
            combined = combined.rename(columns={"name": "stock_name"})

        # 列排序：单股文件字段在前，附加字段在后
        base_cols = ["code", "datetime", "open", "high", "low", "close", "volume", "turnover"]
        extra_cols = [c for c in ["stock_name", "plates", "plate_type_list"] if c in combined.columns]
        keep_cols = base_cols + extra_cols
        combined = combined[[c for c in keep_cols if c in combined.columns]]
        out_path = os.path.join(DATA_DIR, "all_1w.parquet")
        combined.to_parquet(out_path, index=False)
        print(f"\n总表保存完成: {out_path} 共 {len(combined)} 条")
    else:
        print("\n无数据，跳过总表保存")



# =========================================================
# 主入口
# =========================================================
if __name__ == "__main__":

    # 下载数据
    run_download()

    # 板块信息同步
    run_plate_sync()