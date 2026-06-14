import os
import sys
import time
import argparse
import logging
from tqdm import tqdm
import requests
import urllib3
import pandas as pd
import baostock as bs

import contextlib
from collections import deque
from datetime import datetime
import duckdb
from futu import OpenQuoteContext, KLType, AuType, RET_OK

# 关闭杂项日志
logging.getLogger("futu").setLevel(logging.ERROR)
logging.getLogger("baostock").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =========================================================
# 配置
# =========================================================
DATA_DIR = "data_uscncc"
SYMBOL_FILE = "symbols/symbols.csv"
DEFAULT_MARKET = "US,CC"   # 默认市场: all / US / CN / CC / US,CC
DEFAULT_KTYPE = "week,day"      # 默认K线周期: week(周K) / day(日K) / all(全部) / week,day(逗号拼接)
DEFAULT_API = "futu"  # 美股数据源: futu(富途) / stooq-local(本地全量数据包)

# ktype → 子目录名 / 文件后缀 映射
KTYPE_DIR_MAP = {"week": "1w", "day": "1d"}
KTYPE_SUFFIX_MAP = {"week": "1w", "day": "1d"}

os.makedirs(DATA_DIR, exist_ok=True)
for sub in ["us", "cn", "cc"]:
    os.makedirs(os.path.join(DATA_DIR, sub), exist_ok=True)
for ktype_dir in ["1w", "1d"]:
    for sub in ["us", "cn", "cc"]:
        os.makedirs(os.path.join(DATA_DIR, ktype_dir, sub), exist_ok=True)


# =========================================================
# 富途 API 限速（60次/30秒）
# =========================================================
REQUEST_LIMIT = 60
WINDOW_SECONDS = 30

request_times = deque()  # 默认全局队列（兼容旧调用）


def wait_rate_limit(queue=None):
    if queue is None:
        queue = request_times
    now = time.time()
    while queue and now - queue[0] > WINDOW_SECONDS:
        queue.popleft()

    if len(queue) >= REQUEST_LIMIT:
        sleep_time = WINDOW_SECONDS - (now - queue[0]) + 0.5
        sleep_time = max(0, sleep_time)

        for _ in tqdm(range(int(sleep_time), 0, -1), desc="  限频倒计时", unit="s", leave=False):
            time.sleep(1)

    queue.append(time.time())


# =========================================================
# 股票列表
# =========================================================
def init_symbols_file(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame({
            "code": ["US.AAPL", "US.TSLA", "SH.600036", "CC.BTCUSDT"],
            "market": ["US", "US", "CN", "CC"],
        }).to_csv(path, index=False)


def load_symbols(path):
    df = pd.read_csv(path)
    return df["code"].dropna().tolist()


def get_market(code):
    """根据 code 前缀返回市场分类: us / cn / cc"""
    if code.startswith("CC."):
        return "cc"
    if code.startswith(("SH.", "SZ.")):
        return "cn"
    if code.startswith("US."):
        return "us"
    raise ValueError(f"未知代码前缀: {code}")


MARKET_LABEL = {"us": "美股", "cn": "A股", "cc": "加密货币"}


# =========================================================
# 统一请求开始日期（各数据源自会按上市日期截断）
# =========================================================
def get_start_date(code):
    return datetime(2000, 1, 3)

def get_start_date_by_ktype(ktype):
    """根据周期返回起始日期：周线、日线均为最近19年。"""
    return datetime(datetime.now().year - 19, 1, 3)


# =========================================================
# 下载数据（富途 US）
# =========================================================
def fetch_futu_data(code, start_str, end_str, quote_ctx, ktype="week", rate_limit_queue=None):
    ktype_map = {"week": KLType.K_WEEK, "day": KLType.K_DAY}
    futu_ktype = ktype_map.get(ktype, KLType.K_WEEK)
    all_data = []
    page_req_key = None
    last_max_time = None
    retry = 0

    while True:
        # 仅首页请求限频（使用独立队列）
        if page_req_key is None:
            wait_rate_limit(rate_limit_queue)

        ret, data, page_req_key = quote_ctx.request_history_kline(
            code=code, start=start_str, end=end_str,
            ktype=futu_ktype, autype=AuType.QFQ,
            page_req_key=page_req_key
        )

        if ret != RET_OK:
            retry += 1
            tqdm.write(f"{code} 请求失败，重试 {retry}/3")
            if retry >= 3:
                tqdm.write(f"{code} 下载失败")
                break
            time.sleep(2)
            continue

        retry = 0
        if data is None or len(data) == 0:
            break

        # 防止分页重复
        current_max = data["time_key"].max()
        if last_max_time is not None and current_max <= last_max_time:
            break
        last_max_time = current_max
        all_data.append(data)

        if not page_req_key:
            break

    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


# =========================================================
# Binance API（加密货币，不限频）
# =========================================================
def fetch_binance_data(code, start_str, end_str, ktype="week"):
    """
    从 Binance 获取 K 线数据（支持日/周）。
    """
    symbol_map = {
        "CC.BTCUSDT": "BTCUSDT",
        "CC.ETHUSDT": "ETHUSDT",
        "CC.SOLUSDT": "SOLUSDT",
    }
    binance_symbol = symbol_map.get(code)
    if binance_symbol is None:
        print(f"{code} 不支持 Binance 数据源")
        return pd.DataFrame()

    interval = {"week": "1w", "day": "1d"}.get(ktype, "1w")

    base_url = "https://api.binance.com/api/v3/klines"
    start_ms = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_str).timestamp() * 1000)

    all_rows = []
    current_start = start_ms

    sess = requests.Session()
    sess.verify = False
    sess.headers.update({"User-Agent": "Mozilla/5.0"})

    while current_start < end_ms:
        params = {
            "symbol": binance_symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        try:
            resp = sess.get(base_url, params=params, timeout=15)
            resp.raise_for_status()
            klines = resp.json()
        except Exception as e:
            tqdm.write(f"Binance 请求失败: {e}")
            break

        if not klines:
            break

        for k in klines:
            all_rows.append({
                "code": code,
                "time_key": pd.Timestamp(k[0], unit="ms"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "turnover": float(k[7]),
            })

        current_start = klines[-1][6] + 1
        if len(klines) < 1000:
            break

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


# =========================================================
# Stooq 本地数据（US 股票，从全量历史数据包读取）
# =========================================================
_STOOQ_LOCAL_DIR = os.path.join(DATA_DIR, "data", "daily", "us")
_STOOQ_MARKETS = ["nasdaq stocks", "nasdaq etfs", "nyse stocks", "nyse etfs", "nysemkt stocks", "nysemkt etfs"]

def fetch_stooq_local_data(code, start_str, end_str, ktype="week"):
    """从本地 Stooq 全量数据包读取 K 线数据"""
    _search = code.replace("US.", "").lower().replace(".", "-")
    _path = None
    for _mkt in _STOOQ_MARKETS:
        _mkt_dir = os.path.join(_STOOQ_LOCAL_DIR, _mkt)
        if not os.path.isdir(_mkt_dir):
            continue
        # ETFs: 文件直接放在市场目录下
        _candidate = os.path.join(_mkt_dir, f"{_search}.us.txt")
        if os.path.exists(_candidate):
            _path = _candidate
            break
        # Stocks: 文件放在数字子目录中
        for _sub in sorted(os.listdir(_mkt_dir)):
            _sub_dir = os.path.join(_mkt_dir, _sub)
            if not os.path.isdir(_sub_dir):
                continue
            _candidate = os.path.join(_sub_dir, f"{_search}.us.txt")
            if os.path.exists(_candidate):
                _path = _candidate
                break
        if _path:
            break
    if _path is None:
        return pd.DataFrame()

    try:
        df = pd.read_csv(_path)
    except Exception as e:
        print(f"  Stooq本地 {code}: 读取失败 {e}")
        return pd.DataFrame()

    df.columns = [c.strip("<>") for c in df.columns]  # 去掉 < >
    df = df.rename(columns={
        "DATE": "time_key", "OPEN": "open", "HIGH": "high",
        "LOW": "low", "CLOSE": "close", "VOL": "volume",
    })
    df["time_key"] = pd.to_datetime(df["time_key"].astype(str), format="%Y%m%d")
    df["code"] = code
    df["turnover"] = 0
    df = df[["code", "time_key", "open", "high", "low", "close", "volume", "turnover"]]
    df = df.sort_values("time_key").reset_index(drop=True)

    # 按起始日期截断（先截断再保存/聚合）
    _start_ts = pd.Timestamp(start_str)
    df = df[df["time_key"] >= _start_ts]

    # 周线：日线 → 周线聚合
    if ktype == "week":
        df = df.set_index("time_key")
        _agg = {
            "code": "first", "open": "first",
            "high": "max", "low": "min",
            "close": "last", "volume": "sum", "turnover": "sum",
        }
        df = df.resample("W-FRI").agg(_agg).dropna(subset=["close"]).reset_index()
        df["time_key"] = df["time_key"] - pd.Timedelta(days=4)

    return df


# =========================================================
# Baostock 数据（CN 股票）
# =========================================================
def fetch_cn_data(code, start_str, end_str, ktype="week"):
    """
    从 Baostock 获取 A 股 K 线数据（支持日/周，前复权）。
    code 格式: SH.600036 或 SZ.000001
    """
    # baostock 格式: sh.600036
    bs_code = code.lower()
    frequency = {"week": "w", "day": "d"}.get(ktype, "w")
    adjustflag = "2"  # 前复权

    with contextlib.redirect_stdout(None):
        lg = bs.login()
    if lg.error_code != "0":
        print(f"  Baostock 登录失败: {lg.error_msg}")
        return pd.DataFrame()

    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume",
            start_date=start_str,
            end_date=end_str,
            frequency=frequency,
            adjustflag=adjustflag,
        )
        if rs.error_code != "0":
            print(f"  Baostock 查询失败: {rs.error_msg}")
            return pd.DataFrame()

        rows = []
        while rs.next():
            row = rs.get_row_data()
            if not row or row[0] == "":
                continue
            date_str, o, h, l, c, v = row
            rows.append({
                "code": code,
                "time_key": pd.Timestamp(date_str),
                "open": float(o) if o else 0.0,
                "high": float(h) if h else 0.0,
                "low": float(l) if l else 0.0,
                "close": float(c) if c else 0.0,
                "volume": float(v) if v else 0.0,
                "turnover": 0.0,
            })
    finally:
        with contextlib.redirect_stdout(None):
            bs.logout()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


# =========================================================
# 保存数据
# =========================================================
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
    name_map, stock_codes = fetch_stock_basicinfo_map(symbols, quote_ctx)

    all_rows = []
    batch_size = 200
    _batches = list(range(0, len(stock_codes), batch_size))
    for batch_start in tqdm(_batches, desc="  板块信息", unit="batch"):
        batch = stock_codes[batch_start:batch_start + batch_size]

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

    # 补充无板块数据的标的（如ETF、未返回板块的普通股票）
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
# Baostock 行业分类（CN 股票）
# =========================================================
def fetch_cn_stock_industry(cn_symbols):
    """从 Baostock 获取 A 股行业分类"""
    rows = []
    with contextlib.redirect_stdout(None):
        lg = bs.login()
    if lg.error_code != "0":
        print(f"  Baostock 登录失败: {lg.error_msg}")
        return pd.DataFrame()

    try:
        for code in cn_symbols:
            bs_code = code.lower()
            rs = bs.query_stock_industry(bs_code)
            if rs.error_code == "0" and rs.next():
                row_data = rs.get_row_data()
                # row_data: [更新日期, 股票代码, 股票名称, 行业名称, 行业分类]
                if len(row_data) >= 4 and row_data[3]:
                    rows.append({
                        "code": code,
                        "plate_name": row_data[3],
                        "plate_type": "INDUSTRY",
                    })
    finally:
        with contextlib.redirect_stdout(None):
            bs.logout()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    print(f"  Baostock: 获取 {len(df)} 只 A 股行业分类")
    return df


# =========================================================
# 板块同步主流程（直接返回 {code: plates_str}）
# =========================================================
def run_plate_sync(selected_markets=None):
    """同步板块/行业信息，返回 {code: plates_str}"""
    if selected_markets is None:
        selected_markets = ["us", "cn", "cc"]

    symbols = load_symbols(SYMBOL_FILE)
    symbols = [c for c in symbols if get_market(c) in selected_markets]
    if not symbols:
        print("无可同步板块信息的标的，跳过")
        return {}

    plates_map = {}

    # US: 富途板块数据
    us_symbols = [c for c in symbols if get_market(c) == "us"]
    if us_symbols:
        quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
        try:
            df_us = fetch_all_stock_plates(us_symbols, quote_ctx)
            if not df_us.empty:
                for _, row in df_us.iterrows():
                    val = row.get("plates", "")
                    if val:
                        plates_map[row["code"]] = val
        finally:
            quote_ctx.close()

    # CN: Baostock 行业数据
    cn_symbols = [c for c in symbols if get_market(c) == "cn"]
    if cn_symbols:
        df_cn = fetch_cn_stock_industry(cn_symbols)
        if not df_cn.empty:
            for _, row in df_cn.iterrows():
                val = row.get("plate_name", "")
                if val:
                    plates_map[row["code"]] = val

    return plates_map


def import_stooq_all_to_db():
    """导入 Stooq 全量美股数据到 DuckDB stooq_local_all_us_stocks 表，含技术指标计算"""
    import glob as _glob

    _stooq_dir = os.path.join(DATA_DIR, "data", "daily", "us")
    if not os.path.isdir(_stooq_dir):
        print(f"  Stooq 数据目录不存在: {_stooq_dir}")
        return

    _all_files = _glob.glob(os.path.join(_stooq_dir, "**", "*.txt"), recursive=True)
    if not _all_files:
        print("  Stooq 无数据文件，跳过")
        return

    print(f"  Stooq 全量导入: {len(_all_files)} 个文件...")
    _db_path = os.path.join(os.path.dirname(__file__), "database", "market.duckdb")
    _con = duckdb.connect(_db_path)
    for _drop in ["DROP VIEW IF EXISTS stooq_local_all_us_stocks", "DROP TABLE IF EXISTS stooq_local_all_us_stocks"]:
        try: _con.execute(_drop)
        except: pass
    _con.execute("""
        CREATE TABLE stooq_local_all_us_stocks (
            code        VARCHAR,
            datetime    DATE,
            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            volume      DOUBLE,
            ktype       VARCHAR,
            type        VARCHAR,
            market      VARCHAR,
            turnover_amount DOUBLE,
            avg_turnover_60d DOUBLE,
            pct_chg_60d DOUBLE,
            PRIMARY KEY (code, datetime)
        )
    """)
    _total_rows = 0
    _errors = 0
    for _mkt_dir in sorted(_glob.glob(os.path.join(_stooq_dir, "*"))):
        if not os.path.isdir(_mkt_dir):
            continue
        _mkt_name = os.path.basename(_mkt_dir)
        _pattern = os.path.join(_mkt_dir, "**", "*.txt")
        _files = _glob.glob(_pattern, recursive=True)
        if not _files:
            continue

        _type = "etf" if "etfs" in _mkt_name else "stock"
        print(f"    [{_mkt_name}] {len(_files)} 个文件...", end=" ", flush=True)
        try:
            _con.execute(f"""
                INSERT OR REPLACE INTO stooq_local_all_us_stocks (code, datetime, open, high, low, close, volume, ktype, type, market, turnover_amount)
                SELECT
                    'US.' || replace(replace("<TICKER>", '.US', ''), '-', '.') AS code,
                    strptime("<DATE>"::VARCHAR, '%Y%m%d')::DATE AS datetime,
                    "<OPEN>"::DOUBLE AS open,
                    "<HIGH>"::DOUBLE AS high,
                    "<LOW>"::DOUBLE AS low,
                    "<CLOSE>"::DOUBLE AS close,
                    "<VOL>"::DOUBLE AS volume,
                    '1D' AS ktype,
                    '{_type}' AS type,
                    'us' AS market,
                    "<CLOSE>"::DOUBLE * "<VOL>"::DOUBLE AS turnover_amount
                FROM read_csv_auto('{_pattern}', header=true, union_by_name=true)
                ORDER BY code, datetime
            """)
            _rows = _con.execute("SELECT count(*) FROM stooq_local_all_us_stocks").fetchone()[0] - _total_rows
            _total_rows += _rows
            print(f"{_rows:,} 行")
        except Exception as e:
            _errors += 1
            print(f"失败: {e}")

    # 全局排序（code 升序, datetime 降序）
    print("    全局排序...", end=" ", flush=True)
    _con.execute("""
        CREATE TABLE stooq_tmp AS
        SELECT * FROM stooq_local_all_us_stocks
        ORDER BY code ASC, datetime DESC
    """)
    _con.execute("DROP TABLE stooq_local_all_us_stocks")
    _con.execute("ALTER TABLE stooq_tmp RENAME TO stooq_local_all_us_stocks")
    print("完成")

    # 计算技术指标（仅 avg_turnover_60d、pct_chg_60d）
    print("    计算技术指标...", end=" ", flush=True)
    _con.execute("""
        UPDATE stooq_local_all_us_stocks t
        SET
            avg_turnover_60d = w.a60,
            pct_chg_60d      = w.c60
        FROM (
            SELECT code, datetime,
                CASE WHEN COUNT(turnover_amount) OVER w60 >= 60 THEN AVG(turnover_amount) OVER w60 END AS a60,
                (close - LAG(close, 60) OVER w) / NULLIF(LAG(close, 60) OVER w, 0) AS c60
            FROM stooq_local_all_us_stocks
            WINDOW w   AS (PARTITION BY code ORDER BY datetime),
                   w60 AS (PARTITION BY code ORDER BY datetime ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
        ) w
        WHERE t.code = w.code AND t.datetime = w.datetime
    """)

    # 覆盖写入排名表
    print("    生成排名变动表...", end=" ", flush=True)
    _con.execute("DROP TABLE IF EXISTS top_turnover_stock_rank")
    _con.execute("""
        CREATE TABLE top_turnover_stock_rank AS
        SELECT rank,
               prev_rank - rank AS rank_change,
               CAST(prev_rank - rank AS DOUBLE) / NULLIF(prev_rank, 0) AS rank_change_pct,
               CONCAT(ROUND(CAST(prev_rank - rank AS DOUBLE) / NULLIF(prev_rank, 0) * 100, 2), '%') AS rank_change_pct_display,
               code, datetime, open, high, low, close, volume, ktype, type, market,
               turnover_amount, avg_turnover_60d, pct_chg_60d,
               CONCAT(ROUND(pct_chg_60d * 100, 2), '%') AS pct_chg_60d_display
        FROM (
            SELECT rank,
                   LAG(rank) OVER (PARTITION BY code ORDER BY datetime) AS prev_rank,
                   code, datetime, open, high, low, close, volume, ktype, type, market,
                   turnover_amount, avg_turnover_60d, pct_chg_60d
            FROM (
                SELECT ROW_NUMBER() OVER (PARTITION BY datetime ORDER BY avg_turnover_60d DESC) AS rank,
                       code, datetime, open, high, low, close, volume, ktype, type, market,
                       turnover_amount, avg_turnover_60d, pct_chg_60d
                FROM stooq_local_all_us_stocks
                WHERE (type IS NULL OR type = 'stock')
                  AND avg_turnover_60d IS NOT NULL
            ) r
        ) t
        ORDER BY datetime DESC, avg_turnover_60d DESC
    """)
    _con.execute("DROP TABLE IF EXISTS top_turnover_etf_rank")
    _con.execute("""
        CREATE TABLE top_turnover_etf_rank AS
        SELECT rank,
               prev_rank - rank AS rank_change,
               CAST(prev_rank - rank AS DOUBLE) / NULLIF(prev_rank, 0) AS rank_change_pct,
               CONCAT(ROUND(CAST(prev_rank - rank AS DOUBLE) / NULLIF(prev_rank, 0) * 100, 2), '%') AS rank_change_pct_display,
               code, datetime, open, high, low, close, volume, ktype, type, market,
               turnover_amount, avg_turnover_60d, pct_chg_60d,
               CONCAT(ROUND(pct_chg_60d * 100, 2), '%') AS pct_chg_60d_display
        FROM (
            SELECT rank,
                   LAG(rank) OVER (PARTITION BY code ORDER BY datetime) AS prev_rank,
                   code, datetime, open, high, low, close, volume, ktype, type, market,
                   turnover_amount, avg_turnover_60d, pct_chg_60d
            FROM (
                SELECT ROW_NUMBER() OVER (PARTITION BY datetime ORDER BY avg_turnover_60d DESC) AS rank,
                       code, datetime, open, high, low, close, volume, ktype, type, market,
                       turnover_amount, avg_turnover_60d, pct_chg_60d
                FROM stooq_local_all_us_stocks
                WHERE type = 'etf'
                  AND avg_turnover_60d IS NOT NULL
            ) r
        ) t
        ORDER BY datetime DESC, avg_turnover_60d DESC
    """)
    print("完成")
    _con.close()
    print(f"  Stooq 导入完成: {_total_rows:,} 行, {_errors} 错误")


# Moomoo OpenD 配置（默认端口 11112）
MOOMOO_HOST = "127.0.0.1"
MOOMOO_PORT = 11112


def _get_quota_info(host="127.0.0.1", port=11111):
    """查询 OpenD 实例的历史 K 线额度使用明细"""
    from futu import OpenQuoteContext, RET_OK
    ctx = OpenQuoteContext(host=host, port=port)
    try:
        ret, result = ctx.get_history_kl_quota(get_detail=True)
        if ret != RET_OK:
            return 0, 0, set()
        # result 是元组 (used_quota, remain_quota, detail_list)
        used = int(result[0])
        remain = int(result[1])
        detail = set()
        for item in result[2]:
            _code = str(item.get("code", ""))
            if _code:
                detail.add(_code)
        return used, remain, detail
    except Exception:
        return 0, 0, set()
    finally:
        ctx.close()


def _assign_api_for_stocks(symbols, futu_detail, futu_remain, moomoo_detail, moomoo_remain):
    """根据 quota 明细分配每只股票走哪个 API

    Returns:
        dict: {code: "futu"|"moomoo"|"stooq-local"}
    """
    _api_of = {}
    _futu_used = 0
    _moomoo_used = 0
    for _c in symbols:
        if get_market(_c) != "us":
            _api_of[_c] = None  # 非 US 股票走各自数据源
        elif _c in futu_detail:
            _api_of[_c] = "futu"  # 已有记录，不扣额度
        elif _c in moomoo_detail:
            _api_of[_c] = "moomoo"
        elif _futu_used < futu_remain:
            _api_of[_c] = "futu"
            _futu_used += 1
        elif _moomoo_used < moomoo_remain:
            _api_of[_c] = "moomoo"
            _moomoo_used += 1
        else:
            _api_of[_c] = "stooq-local"
    return _api_of
def fetch_stock_names(symbols, quote_ctx):
    """预取所有标的的名称，返回 {code: name}"""
    name_map = {}
    us_codes = [c for c in symbols if get_market(c) == "us"]
    cn_codes = [c for c in symbols if get_market(c) == "cn"]
    cc_codes = [c for c in symbols if get_market(c) == "cc"]

    # US: 富途
    if us_codes and quote_ctx:
        for market_prefix in ["US", "HK"]:
            batch = [c for c in us_codes if c.startswith(market_prefix + ".")]
            if not batch:
                continue
            ret, data = quote_ctx.get_stock_basicinfo(market=market_prefix, code_list=batch)
            if ret == RET_OK and data is not None:
                for _, row in data.iterrows():
                    name_map[row["code"]] = row.get("name", "")
        pass

    # CN: baostock
    if cn_codes:
        lg = bs.login()
        if lg.error_code == "0":
            try:
                for c in cn_codes:
                    bs_code = c.lower()
                    rs = bs.query_stock_basic(bs_code)
                    if rs.next():
                        row_data = rs.get_row_data()
                        if row_data and len(row_data) > 1:
                            name_map[c] = row_data[1]
            finally:
                bs.logout()

    # CC: 直接用 symbol 显示名
    for c in cc_codes:
        raw = c.replace("CC.", "")
        name_map[c] = raw.replace("USDT", "/USDT")

    return name_map


def fetch_top_turnover_stocks(limit=100):
    """通过 Futu OpenD 获取当日成交额前 N 的美股，返回代码列表"""
    from futu import OpenQuoteContext, AccumulateFilter, StockField, SortDir, RET_OK, Market

    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        af = AccumulateFilter()
        af.stock_field = StockField.TURNOVER
        af.is_no_filter = False
        af.filter_min = 1
        af.sort = SortDir.DESCEND

        ret, data = quote_ctx.get_stock_filter(Market.US, [af], begin=0, num=limit)
        if ret != RET_OK:
            print(f"get_stock_filter 失败: {data}")
            return []

        _, _, stock_list = data
        codes = [str(getattr(item, "stock_code", "")) for item in stock_list if getattr(item, "stock_code", "")]
        if not codes:
            print("get_stock_filter 返回空列表")
            return []

        print(f"  美股成交额排名: {len(codes)} 只")
        return codes
    finally:
        quote_ctx.close()


def fetch_cn_top_turnover(limit=100):
    """通过 Futu OpenD 获取沪深主板当日成交额前 N 的股票，返回代码列表"""
    from futu import OpenQuoteContext, AccumulateFilter, StockField, SortDir, RET_OK, Market

    def _is_excluded(code):
        return code.startswith("SH.688") or code.startswith(("SZ.300", "SZ.301")) or code.startswith("BJ.")

    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    all_codes = []
    fetch_num = max(limit * 3, 60)
    try:
        for market_name, market in [("SH", Market.SH), ("SZ", Market.SZ)]:
            af = AccumulateFilter()
            af.stock_field = StockField.TURNOVER
            af.is_no_filter = False
            af.filter_min = 1
            af.sort = SortDir.DESCEND
            ret, result = quote_ctx.get_stock_filter(market, [af], begin=0, num=fetch_num)
            if ret != RET_OK:
                continue
            if not (isinstance(result, tuple) and len(result) >= 3):
                continue
            for s in result[2]:
                code = str(getattr(s, "stock_code", ""))
                if code and not _is_excluded(code):
                    all_codes.append(code)
    finally:
        quote_ctx.close()
    # 去重 + 取前 limit
    seen = set()
    codes = []
    for c in all_codes:
        if c not in seen:
            seen.add(c)
            codes.append(c)
        if len(codes) >= limit:
            break
    print(f"  A股成交额排名: {len(codes)} 只")
    return codes


def _sync_watchlist_db(us_realtime_codes=None, cn_realtime_codes=None):
    """从排名表 + 实时排名结果 + symbols.csv 合并写入 watchlist 表

    Args:
        us_realtime_codes: fetch_top_turnover_stocks() 返回的 US 代码列表
        cn_realtime_codes: fetch_cn_top_turnover() 返回的 CN 代码列表
    """
    try:
        _db_path = os.path.join(os.path.dirname(__file__), "database", "market.duckdb")
        _con = duckdb.connect(_db_path)

        # 1. 各来源收集代码
        _sources = {}  # code → set of source strings

        # a. symbols.csv → 手动添加
        _csv_path = os.path.join(os.path.dirname(__file__), "symbols", "symbols.csv")
        if os.path.exists(_csv_path):
            _symbols = pd.read_csv(_csv_path)
            for _, _row in _symbols.iterrows():
                _c = str(_row["code"]).strip()
                _sources.setdefault(_c, set()).add("手动添加")

        # b. top_turnover_stock_rank 最新日期 rank ≤ 200
        try:
            for _r in _con.execute("""
                SELECT DISTINCT code FROM top_turnover_stock_rank
                WHERE datetime = (SELECT MAX(datetime) FROM top_turnover_stock_rank)
                  AND rank <= 200
            """).fetchall():
                _sources.setdefault(str(_r[0]), set()).add("60日成交额排名")
        except Exception:
            pass

        # c. top_turnover_etf_rank 最新日期 rank ≤ 10
        try:
            for _r in _con.execute("""
                SELECT DISTINCT code FROM top_turnover_etf_rank
                WHERE datetime = (SELECT MAX(datetime) FROM top_turnover_etf_rank)
                  AND rank <= 10
            """).fetchall():
                _sources.setdefault(str(_r[0]), set()).add("ETF成交额排名")
        except Exception:
            pass

        # d. 实时成交额排名
        if us_realtime_codes:
            for _c in us_realtime_codes:
                _sources.setdefault(_c, set()).add("实时成交额排名")
        if cn_realtime_codes:
            for _c in cn_realtime_codes:
                _sources.setdefault(_c, set()).add("实时成交额排名")

        # 2. 确定市场
        _market_of = {}
        for _c in _sources:
            if _c.startswith("CC."):
                _market_of[_c] = "cc"
            elif _c.startswith(("SH.", "SZ.")):
                _market_of[_c] = "cn"
            elif _c.startswith("US."):
                _market_of[_c] = "us"

        # 3. 按市场排序写入（us → cc → cn）
        _mkt_order = {"us": 0, "cc": 1, "cn": 2}
        _con.execute('DELETE FROM watchlist')
        for _c in sorted(_sources.keys(), key=lambda c: (_mkt_order.get(_market_of.get(c, ""), 9), c)):
            _m = _market_of.get(_c)
            if not _m:
                continue
            _con.execute(
                'INSERT INTO watchlist(code, market, source, created_at) VALUES (?, ?, ?, ?)',
                [_c, _m, ",".join(sorted(_sources[_c])), datetime.now()],
            )
        _con.close()
        print(f"  watchlist 已同步: {len(_sources)} 只")
    except Exception as e:
        print(f"  watchlist 同步失败: {e}")


def _load_symbols_from_watchlist(api="all"):
    """从 DuckDB watchlist 表加载股票列表。

    api="all": US 股票取全部
    api="futu": US 股票只取来源含"手动添加"的（遗留兼容）
    """
    _db_path = os.path.join(os.path.dirname(__file__), "database", "market.duckdb")
    if not os.path.exists(_db_path):
        return None
    try:
        _con = duckdb.connect(_db_path, read_only=True)
        _us_filter = ""
        if api == "futu":
            _us_filter = "AND source LIKE '%手动添加%'"
        _rows = _con.execute(f"""
            SELECT DISTINCT code FROM watchlist
            WHERE market = 'us' {_us_filter}
            UNION ALL
            SELECT DISTINCT code FROM watchlist WHERE market = 'cn'
            UNION ALL
            SELECT DISTINCT code FROM watchlist WHERE market = 'cc'
        """).fetchall()
        _con.close()
        _codes = [str(r[0]) for r in _rows]
        if _codes:
            print(f"  watchlist 加载: {len(_codes)} 只")
        return _codes
    except Exception:
        return None


def run_download(ktype="week", selected_markets=None):

    if selected_markets is None:
        selected_markets = ["us", "cn", "cc"]

    init_symbols_file(SYMBOL_FILE)

    # 从 watchlist 表加载股票（不再按 api 过滤）
    _symbols = _load_symbols_from_watchlist(api="all")
    if _symbols is None:
        _symbols = load_symbols(SYMBOL_FILE)

    symbols = [c for c in _symbols if get_market(c) in selected_markets]
    if not symbols:
        print(f"无匹配的标的 (市场: {selected_markets})")
        return

    end_str = datetime.now().strftime("%Y-%m-%d")
    us_codes = [c for c in symbols if get_market(c) == "us"]

    # ---- 根据 quota 分配 US 股票走哪个 API ----
    _api_of = {}
    futu_ctx = None
    moomoo_ctx = None
    futu_rl = deque()    # 独立限速队列
    moomoo_rl = deque()
    if us_codes:
        _, futu_remain, futu_detail = _get_quota_info("127.0.0.1", 11111)
        _, moomoo_remain, moomoo_detail = _get_quota_info(MOOMOO_HOST, MOOMOO_PORT)
        _api_of = _assign_api_for_stocks(us_codes, futu_detail, futu_remain, moomoo_detail, moomoo_remain)
        _futu_list = [c for c, a in _api_of.items() if a == "futu"]
        _moomoo_list = [c for c, a in _api_of.items() if a == "moomoo"]
        if _futu_list:
            futu_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
        if _moomoo_list:
            moomoo_ctx = OpenQuoteContext(host=MOOMOO_HOST, port=MOOMOO_PORT)
        print(f"  API 分配: futu={len(_futu_list)}, moomoo={len(_moomoo_list)}, stooq={sum(1 for a in _api_of.values() if a=='stooq-local')}")

    # 预取股票名称
    name_ctx = futu_ctx or moomoo_ctx
    name_map = fetch_stock_names(symbols, name_ctx)

    all_dfs = []
    _ok = _fail = 0
    _failed_codes = []
    _src_map = {}

    for code in tqdm(symbols, desc=f"{ktype}下载", unit="stock"):
        start = get_start_date_by_ktype(ktype)
        start_str = start.strftime("%Y-%m-%d")
        market = get_market(code)

        if market == "cc":
            df = fetch_binance_data(code, start_str, end_str, ktype)
            _src_map[code] = "binance"
        elif market == "cn":
            df = fetch_cn_data(code, start_str, end_str, ktype)
            _src_map[code] = "baostock"
        elif market == "us":
            _use_api = _api_of.get(code, "stooq-local")
            if _use_api in ("futu", "moomoo"):
                _ctx = futu_ctx if _use_api == "futu" else moomoo_ctx
                _rl = futu_rl if _use_api == "futu" else moomoo_rl
                df = fetch_futu_data(code, start_str, end_str, _ctx, ktype, rate_limit_queue=_rl) if _ctx else pd.DataFrame()
                _src_map[code] = _use_api
            else:
                df = fetch_stooq_local_data(code, start_str, end_str, ktype)
                _src_map[code] = "stooq"
        else:
            df = pd.DataFrame()

        if not df.empty:
            all_dfs.append(df)
            _ok += 1
        else:
            _fail += 1
            _failed_codes.append(code)

    if futu_ctx:
        futu_ctx.close()
    if moomoo_ctx:
        moomoo_ctx.close()

    if _fail:
        print(f"  {ktype} 下载完成: {_ok} 成功, {_fail} 失败 — {'; '.join(_failed_codes)}")
    else:
        print(f"  {ktype} 下载完成: {_ok} 成功")

    # 合并写入 DuckDB（覆盖 + 全局排序）
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.rename(columns={"time_key": "datetime"})
        combined["datetime"] = pd.to_datetime(combined["datetime"])  # 不再 normalize / 转 DATE
        combined = combined.drop_duplicates(["code", "datetime"]).sort_values(["code", "datetime"]).reset_index(drop=True)
        combined["market"] = combined["code"].apply(lambda c: MARKET_LABEL.get(get_market(c), get_market(c)))
        combined["source"] = combined["code"].map(_src_map)
        combined["turnover_amount"] = (combined["close"] * combined["volume"]).round(2)

        _kt_name = {"week": "1w", "day": "1d"}.get(ktype, ktype)
        _tbl = f"klines_{_kt_name}"
        _db_path = os.path.join(os.path.dirname(__file__), "database", "market.duckdb")
        try:
            _con = duckdb.connect(_db_path)
            # 清空旧数据（首次运行表可能不存在）
            try: _con.execute(f"DELETE FROM {_tbl}")
            except: pass
            _con.execute(f"ALTER TABLE {_tbl} ADD COLUMN IF NOT EXISTS created_at TIMESTAMP")
            _con.execute(f"ALTER TABLE {_tbl} ADD COLUMN IF NOT EXISTS source VARCHAR")
            _con.execute(f"ALTER TABLE {_tbl} ADD COLUMN IF NOT EXISTS turnover_amount DOUBLE")
            _con.execute("CREATE OR REPLACE TEMP TABLE _tmp AS SELECT * FROM combined")
            _con.execute(f"""
                INSERT INTO {_tbl} (code, datetime, open, high, low, close, volume, turnover, market, ktype, source, turnover_amount, created_at)
                SELECT code, datetime, open, high, low, close, volume, turnover, market, '{_kt_name}', source, turnover_amount, CURRENT_TIMESTAMP FROM _tmp
            """)
            # 全局排序（code ASC, datetime DESC）
            _con.execute(f"""
                CREATE TABLE {_tbl}_sorted_tmp AS
                SELECT * FROM {_tbl}
                ORDER BY code ASC, datetime DESC
            """)
            _con.execute(f"DROP TABLE {_tbl}")
            _con.execute(f"ALTER TABLE {_tbl}_sorted_tmp RENAME TO {_tbl}")
            _con.close()
            print(f"  {_tbl}: {len(combined)} 行写入 + 排序")
        except Exception as e:
            print(f"  DuckDB 写入失败: {e}")
    else:
        print("\n无数据，跳过写入")



# =========================================================
# 主入口
# =========================================================
def run_download_all(selected_markets=None, skip_week=False, skip_day=False):
    """分阶段下载周线、日线数据。"""
    ktypes = []
    if not skip_week:
        ktypes.append("week")
    if not skip_day:
        ktypes.append("day")

    for ktype in ktypes:
        run_download(ktype=ktype, selected_markets=selected_markets)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="多市场 K 线数据下载")
    parser.add_argument("--ktype", default=DEFAULT_KTYPE,
                        help="K线周期: week(周K) / day(日K) / all(全部) / week,day(逗号拼接, 默认: week,day)")
    parser.add_argument("--market", default=DEFAULT_MARKET,
                        help=f"市场: US / CN / CC / US,CN / all (默认: {DEFAULT_MARKET})")
    parser.add_argument("--top-turnover", type=int, nargs="?", const=100, default=100,
                        help="获取成交额前 N 的美股列表 (默认 N=100, 设为0跳过)")
    parser.add_argument("--top-turnover-cn", type=int, nargs="?", const=100, default=100,
                        help="获取成交额前 N 的沪深主板股票列表 (默认 N=100, 设为0跳过)")
    parser.add_argument("--only-turnover", action="store_true",
                        help="只执行 Stooq 导入和成交额排名，不下载K线数据")
    parser.add_argument("--skip-week", action="store_true", help="跳过周线下载（已弃用，用 --ktype 替代）")
    parser.add_argument("--skip-day", action="store_true", help="跳过日线下载（已弃用，用 --ktype 替代）")
    args = parser.parse_args()

    # 解析市场参数
    if args.market.lower() == "all":
        selected_markets = ["us", "cn", "cc"]
    else:
        selected_markets = [m.strip().lower() for m in args.market.split(",")]

    # ── 1. Stooq 导入（仅 US 市场，每次自动运行）─────────────
    if "us" in selected_markets:
        import_stooq_all_to_db()

    # ── 2. 成交额排名 + watchlist 同步 ─────────────────────────
    _us_realtime = []
    _cn_realtime = []
    if args.top_turnover and "us" in selected_markets:
        _us_realtime = fetch_top_turnover_stocks(limit=args.top_turnover) or []
    if args.top_turnover_cn and "cn" in selected_markets:
        _cn_realtime = fetch_cn_top_turnover(limit=args.top_turnover_cn) or []
    _sync_watchlist_db(us_realtime_codes=_us_realtime, cn_realtime_codes=_cn_realtime)

    # --only-turnover：不下载K线数据和板块
    if args.only_turnover:
        sys.exit(0)

    # ── 3. 下载 K 线 ────────────────────────────────────────────
    # 解析 ktype
    _ktypes = []
    for _k in args.ktype.lower().replace("，", ",").split(","):
        _k = _k.strip()
        if _k == "all":
            _ktypes = ["week", "day"]
            break
        if _k in ("week", "day") and _k not in _ktypes:
            _ktypes.append(_k)
    if not _ktypes:
        _ktypes = ["week", "day"]
    if args.skip_week and "week" in _ktypes:
        _ktypes.remove("week")
    if args.skip_day and "day" in _ktypes:
        _ktypes.remove("day")

    _all_start = time.time()
    run_download_all(selected_markets=selected_markets,
                     skip_week="week" not in _ktypes,
                     skip_day="day" not in _ktypes)

    # ── 4. 板块信息同步 ──────────────────────────────────────────
    plates_map = run_plate_sync(selected_markets=selected_markets)
    if plates_map:
        try:
            import duckdb
            _db_path = os.path.join(os.path.dirname(__file__), "database", "market.duckdb")
            _con = duckdb.connect(_db_path)
            _con.execute("DELETE FROM plates")
            for _c, _p in plates_map.items():
                _con.execute(
                    'INSERT INTO plates(code, plates) VALUES (?, ?) ON CONFLICT (code) DO UPDATE SET plates = ?',
                    [_c, _p, _p],
                )
            _con.close()
            print(f"  plates: {len(plates_map)} 只写入")
        except Exception as e:
            print(f"  plates 写入失败: {e}")
    _elapsed = time.time() - _all_start
    _min = int(_elapsed // 60)
    _sec = int(_elapsed % 60)
    print(f"全部完成，总耗时 {_min}分{_sec}秒")