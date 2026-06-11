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

import threading
import contextlib
from collections import deque
from datetime import datetime
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
DEFAULT_MARKET = "US"   # 默认市场: all / US / CN / CC / US,CC
DEFAULT_KTYPE = "week,day"      # 默认K线周期: week(周K) / day(日K) / 60m(60分钟K) / all(全部) / week,day(逗号拼接)
DEFAULT_API = "stooq-local"  # 美股数据源: futu/ibkr/yahoo/stooq-local(本地全量数据包)

# ktype → 子目录名 / 文件后缀 映射
KTYPE_DIR_MAP = {"week": "1w", "day": "1d", "60m": "60m"}
KTYPE_SUFFIX_MAP = {"week": "1w", "day": "1d", "60m": "60m"}

os.makedirs(DATA_DIR, exist_ok=True)
for sub in ["us", "cn", "cc"]:
    os.makedirs(os.path.join(DATA_DIR, sub), exist_ok=True)
for ktype_dir in ["1w", "1d", "60m"]:
    for sub in ["us", "cn", "cc"]:
        os.makedirs(os.path.join(DATA_DIR, ktype_dir, sub), exist_ok=True)


# =========================================================
# 富途 API 限速（60次/30秒）
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

        for _ in tqdm(range(int(sleep_time), 0, -1), desc="  限频倒计时", unit="s", leave=False):
            time.sleep(1)

    request_times.append(time.time())


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
    """根据周期返回起始日期：周线全量，日线20年，60分钟2年。"""
    today = datetime.now()
    if ktype == "day":
        return datetime(today.year - 20, 1, 3)
    elif ktype == "60m":
        return datetime(today.year - 2, 1, 3)
    return datetime(2000, 1, 3)


# =========================================================
# 下载数据（富途 US）
# =========================================================
def fetch_futu_data(code, start_str, end_str, quote_ctx, ktype="week"):
    ktype_map = {"week": KLType.K_WEEK, "day": KLType.K_DAY, "60m": KLType.K_60M}
    futu_ktype = ktype_map.get(ktype, KLType.K_WEEK)
    all_data = []
    page_req_key = None
    last_max_time = None
    retry = 0

    while True:
        # 仅首页请求限频
        if page_req_key is None:
            wait_rate_limit()

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

    interval = {"week": "1w", "day": "1d", "60m": "1h"}.get(ktype, "1w")

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
# Stooq 数据（US 股票，免费 HTTP API）
# =========================================================
def fetch_stooq_data(code, start_str, end_str, ktype="week"):
    """从 Stooq 获取美股 K 线数据（免费，无需注册）"""
    symbol = code.replace("US.", "").lower()
    interval = {"week": "w", "day": "d", "60m": "d"}.get(ktype, "w")
    url = f"https://stooq.com/q/d/l/?s={symbol}.us&i={interval}"

    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"  Stooq {symbol}: {e}")
        return pd.DataFrame()

    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    df = df.rename(columns={
        "Date": "time_key", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["time_key"] = pd.to_datetime(df["time_key"])
    df["code"] = code
    df["turnover"] = 0
    df = df.sort_values("time_key").reset_index(drop=True)
    # 按 start 截断
    df = df[df["time_key"] >= pd.Timestamp(start_str)]
    return df


# =========================================================
# Stooq 本地数据（US 股票，从全量历史数据包读取）
# =========================================================
_STOOQ_LOCAL_DIR = os.path.join(DATA_DIR, "data", "daily", "us")
_STOOQ_MARKETS = ["nasdaq stocks", "nasdaq etfs", "nyse stocks", "nyse etfs", "nysemkt stocks", "nysemkt etfs"]

def fetch_stooq_local_data(code, start_str, end_str, ktype="week", _save_daily_path=None):
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

    # 保存日线（周线聚合时使用，列名统一为 datetime）
    if _save_daily_path and not df.empty:
        os.makedirs(os.path.dirname(_save_daily_path), exist_ok=True)
        df.rename(columns={"time_key": "datetime"}).to_parquet(_save_daily_path, index=False)

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
# Yahoo Finance 数据（US/CC 股票，免费）
# =========================================================
_YAHOO_CACHE = {}  # {ktype: {symbol: DataFrame}}

def fetch_yahoo_data(code, start_str, end_str, ktype="week"):
    """从 Yahoo Finance 获取 K 线数据（免费，前复权）"""
    symbol = code.replace("US.", "").replace("CC.", "")
    # 检查缓存
    _ck = _YAHOO_CACHE.get(ktype, {})
    if symbol in _ck:
        df = _ck[symbol]
    else:
        return pd.DataFrame()  # 不在缓存中，由外层批量下载

    df = df.reset_index()
    # yfinance 多股票返回 MultiIndex 列，取该股票
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, axis=1, level=1).reset_index()
    df.columns = [c.lower() for c in df.columns]
    # 列名统一
    _date_col = "date" if "date" in df.columns else "datetime"
    df = df.rename(columns={_date_col: "time_key"})
    df["time_key"] = pd.to_datetime(df["time_key"])
    df["code"] = code
    df["turnover"] = 0
    return df


def _yahoo_batch_download(codes, start_str, end_str, ktype="week"):
    """批量下载 Yahoo Finance 数据（大幅减少请求次数）"""
    interval = {"week": "1wk", "day": "1d", "60m": "60m"}.get(ktype, "1wk")
    symbols = [c.replace("US.", "").replace("CC.", "") for c in codes]
    if not symbols:
        return

    for i in range(0, len(symbols), 5):  # 每批 5 只
        batch = symbols[i:i + 5]
        for retry in range(5):
            try:
                import yfinance as yf
                df = yf.download(batch, start=start_str, end=end_str,
                                 interval=interval, auto_adjust=True, group_by="ticker")
                if df.empty:
                    break
                # 存入缓存
                if ktype not in _YAHOO_CACHE:
                    _YAHOO_CACHE[ktype] = {}
                for s in batch:
                    if isinstance(df.columns, pd.MultiIndex) and s in df.columns.levels[1]:
                        _YAHOO_CACHE[ktype][s] = df.xs(s, axis=1, level=1)
                break
            except Exception as e:
                if "Rate limited" in str(e):
                    _wait = 30 * (retry + 1)
                    tqdm.write(f"  Yahoo 限频，等待 {_wait} 秒（第 {retry+1} 次重试）...")
                    time.sleep(_wait)
                else:
                    tqdm.write(f"  Yahoo batch 错误: {e}")
                    break
        time.sleep(60)  # 批次间间隔 60 秒


# =========================================================
# IBKR 数据（US 股票，替代 Futu）
# =========================================================
class _IBKRApp:
    """IBKR API 事件驱动客户端包装器"""
    def __init__(self):
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper
        self.data = []
        self._req_id = 1
        self._connected = threading.Event()
        self._done = threading.Event()
        self._error = None

        class _App(EWrapper, EClient):
            def __init__(self, outer):
                EWrapper.__init__(self)
                EClient.__init__(self, self)
                self.outer = outer

            def nextValidId(self, orderId):
                self.outer._connected.set()

            def historicalData(self, reqId, bar):
                self.outer.data.append(bar)

            def historicalDataEnd(self, reqId, start, end):
                self.outer._done.set()

            def error(self, reqId, code, msg, *args):
                if code not in (2104, 2106, 2158):  # 忽略连接状态信息
                    self.outer._error = f"[{code}] {msg}"

        self.app = _App(self)

    def connect(self, host="127.0.0.1", port=4001, client_id=1):
        self.app.connect(host, port, client_id)
        import threading as _thr
        _thr.Thread(target=self.app.run, daemon=True).start()
        if not self._connected.wait(timeout=5):
            self.app.disconnect()
            raise ConnectionError("IBKR 连接超时")

    def reset(self):
        """请求之间复位，不清除连接"""
        self._done.clear()
        self.data.clear()
        self._error = None

    def req_historical_data(self, symbol, end_str, duration, bar_size):
        from ibapi.contract import Contract
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        end_dt = ""  # 空=当前最新数据
        self._done.clear()
        self.data.clear()
        # 周线不支持 ADJUSTED_LAST，日线/60分可用
        _what = "ADJUSTED_LAST" if bar_size in ("1 day", "1 hour") else "TRADES"
        self.app.reqHistoricalData(
            self._req_id, contract, end_dt, duration, bar_size,
            _what, 1, 1, False, [],
        )
        if not self._done.wait(timeout=30):
            raise TimeoutError(f"IBKR {symbol} 历史数据请求超时")

    def disconnect(self):
        self.app.disconnect()


def fetch_ibkr_data(code, start_str, end_str, ktype="week", _save_daily_path=None, ibkr_app=None):
    """从 IBKR 获取美股 K 线数据（需启动 TWS 或 IB Gateway）

    周线：下载日线再聚合成周线（绕开 IBKR 周线不支持 ADJUSTED_LAST 的限制）
         并可选将原始日线保存到 _save_daily_path，避免重复下载
    日线/60分：直接请求对应周期

    传 ibkr_app 可复用连接（推荐），不传则每次新建。
    """
    symbol = code.replace("US.", "")
    _is_week = (ktype == "week")
    # 周线实际请求日线数据，再聚合
    ibkr_bar = {"week": "1 day", "day": "1 day", "60m": "1 hour"}.get(ktype, "1 day")
    # 根据 start_str 计算 duration（与 get_start_date_by_ktype 一致）
    _start = pd.Timestamp(start_str)
    _end = pd.Timestamp(end_str)
    _total_days = (_end - _start).days
    if _total_days >= 730:
        ibkr_duration = f"{_total_days // 365 + 1} Y"
    elif _total_days >= 60:
        ibkr_duration = f"{_total_days // 30 + 1} M"
    else:
        ibkr_duration = f"{_total_days + 1} D"

    _own_app = ibkr_app is None
    app = ibkr_app or _IBKRApp()
    if _own_app:
        app.connect()
        app.reset()
    else:
        app.reset()
    try:
        app.req_historical_data(symbol, end_str, ibkr_duration, ibkr_bar)
    except (ConnectionError, TimeoutError) as e:
        print(f"  IBKR {symbol}: {e}")
        return pd.DataFrame()
    finally:
        if _own_app:
            app.disconnect()

    if not app.data:
        return pd.DataFrame()

    rows = []
    for bar in app.data:
        dt = pd.Timestamp(bar.date)
        rows.append({
            "code": code,
            "time_key": dt,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "turnover": 0,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 周线：先聚合再截断
    if _is_week:
        _agg = {
            "code": "first", "open": "first",
            "high": "max", "low": "min",
            "close": "last", "volume": "sum", "turnover": "sum",
        }
        df_weekly = df.set_index("time_key").resample("W-FRI").agg(_agg)
        df_weekly = df_weekly.dropna(subset=["close"]).reset_index()
        df_weekly["time_key"] = df_weekly["time_key"] - pd.Timedelta(days=4)

        # 按各自起始日期截断
        _day_start = pd.Timestamp(get_start_date_by_ktype("day"))
        _week_start = pd.Timestamp(start_str)  # 2000-01-03
        df = df[df["time_key"] >= _day_start]      # 日线截断
        df_weekly = df_weekly[df_weekly["time_key"] >= _week_start]  # 周线截断

        # 保存日线
        if _save_daily_path:
            os.makedirs(os.path.dirname(_save_daily_path), exist_ok=True)
            df.to_parquet(_save_daily_path, index=False)

        return df_weekly

    # 非周线（日线/60分）：直接按 start_str 截断
    _start_ts = pd.Timestamp(start_str)
    df = df[df["time_key"] >= _start_ts]
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
    frequency = {"week": "w", "day": "d", "60m": "60"}.get(ktype, "w")
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
def save_data(df, code, ktype="week", name_map=None):

    if df.empty:
        return

    market = get_market(code)
    suffix = KTYPE_SUFFIX_MAP.get(ktype, f"1{ktype[0]}")
    sub_dir = os.path.join(DATA_DIR, KTYPE_DIR_MAP.get(ktype, ""), market)
    os.makedirs(sub_dir, exist_ok=True)
    file_path = os.path.join(sub_dir, f"{code}_{suffix}.parquet")

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

    # 补充股票名称和市场
    stock_name = (name_map or {}).get(code, "")
    if stock_name:
        df["stock_name"] = stock_name
    df["market"] = MARKET_LABEL.get(market, market)

    df = (
        df
        .drop_duplicates(["code", "datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    df.to_parquet(file_path, index=False)

    pass  # 保存成功，日志由上层汇总


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


def add_plates_to_parquets(ktype, plates_map):
    """为已下载的 K 线 parquet 补充 plates 字段（无板块则留空）"""
    suffix = KTYPE_SUFFIX_MAP.get(ktype, f"1{ktype[0]}")
    ktype_dir = KTYPE_DIR_MAP.get(ktype, "")
    files = []
    for market in ["us", "cn", "cc"]:
        dir_path = os.path.join(DATA_DIR, ktype_dir, market)
        if not os.path.exists(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if not fname.endswith(f"_{suffix}.parquet"):
                continue
            files.append(os.path.join(dir_path, fname))

    _ok = _skip = 0
    for path in tqdm(files, desc=f"  补充板块({ktype})", unit="file"):
        df = pd.read_parquet(path)
        if "plates" in df.columns:
            _skip += 1
            continue
        code = df["code"].iloc[0]
        df["plates"] = plates_map.get(code, "")
        df.to_parquet(path, index=False)
        if df["plates"].iloc[0]:
            _ok += 1
        else:
            _skip += 1
    # 总表也补上
    all_path = os.path.join(DATA_DIR, ktype_dir, f"all_{suffix}.parquet")
    if os.path.exists(all_path):
        df = pd.read_parquet(all_path)
        if "plates" not in df.columns:
            df["plates"] = df["code"].map(plates_map).fillna("")
            df.to_parquet(all_path, index=False)


# =========================================================
# 下载主流程
# =========================================================
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


def fetch_top_turnover_stocks(limit=200):
    """通过 Futu OpenD 获取当日成交额前 N 的美股，保存到 symbols/top_turnover_{YYYYMMDD}.csv"""
    from futu import OpenQuoteContext, AccumulateFilter, StockField, SortDir, RET_OK, Market

    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        # 按成交额降序排列
        af = AccumulateFilter()
        af.stock_field = StockField.TURNOVER
        af.is_no_filter = False
        af.filter_min = 1
        af.sort = SortDir.DESCEND

        ret, data = quote_ctx.get_stock_filter(Market.US, [af], begin=0, num=limit)
        if ret != RET_OK:
            print(f"get_stock_filter 失败: {data}")
            return

        _, _, stock_list = data
        codes = [str(getattr(item, "stock_code", "")) for item in stock_list if getattr(item, "stock_code", "")]
        if not codes:
            print("get_stock_filter 返回空列表")
            return

        # 用市场快照补全价格和成交额数据
        ret2, snapshots = quote_ctx.get_market_snapshot(codes)
        snap_map = {}
        if ret2 == RET_OK and snapshots is not None and not snapshots.empty:
            for _, row in snapshots.iterrows():
                snap_map[str(row.get("code", ""))] = row

        records = []
        for code in tqdm(codes, desc="  成交额排名", unit="stock"):
            name = ""
            price = 0.0
            turnover = 0.0
            snap = snap_map.get(code)
            if snap is not None:
                name = str(snap.get("name", "") or "")
                price = float(snap.get("last_price", 0) or 0)
                turnover = float(snap.get("turnover", 0) or 0)
            turnover_val = round(turnover / 1e8, 2)
            records.append({
                "排名": 0,
                "代码": code,
                "名称": name,
                "最新价": price,
                "成交额(亿元)": turnover_val,
            })

        df = pd.DataFrame(records)
        df = df.sort_values("成交额(亿元)", ascending=False).reset_index(drop=True)
        df["排名"] = range(1, len(df) + 1)
        date_str = datetime.now().strftime("%Y%m%d")
        out_path = os.path.join("symbols", f"top_turnover_{date_str}.csv")
        os.makedirs("symbols", exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path
    finally:
        quote_ctx.close()


def run_download(ktype="week", selected_markets=None, api="futu"):

    if selected_markets is None:
        selected_markets = ["us", "cn", "cc"]

    init_symbols_file(SYMBOL_FILE)

    symbols = load_symbols(SYMBOL_FILE)

    # 按市场过滤
    symbols = [c for c in symbols if get_market(c) in selected_markets]
    if not symbols:
        print(f"无匹配的标的 (市场: {selected_markets})")
        return

    end_str = datetime.now().strftime("%Y-%m-%d")
    start_str = get_start_date_by_ktype(ktype).strftime("%Y-%m-%d")

    us_codes = [c for c in symbols if get_market(c) == "us"]
    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111) if (us_codes and api == "futu") else None

    name_map = fetch_stock_names(symbols, quote_ctx)

    # Yahoo 批量预下载
    _yahoo_codes = [c for c in symbols if get_market(c) in ("us", "cc")] if api == "yahoo" else []
    if _yahoo_codes:
        print("  Yahoo 批量下载中...")
        _yahoo_batch_download(_yahoo_codes, start_str, end_str, ktype)

    # IBKR 连接（复用，跨所有股票）
    _ibkr_app = _IBKRApp() if (api == "ibkr" and [c for c in symbols if get_market(c) == "us"]) else None
    if _ibkr_app:
        _ibkr_app.connect()
        print("  IBKR 连接成功")

    all_dfs = []
    _ok = _fail = 0
    _failed_codes = []

    for code in tqdm(symbols, desc=f"{ktype}下载", unit="stock"):
        name = name_map.get(code, "")
        start = get_start_date_by_ktype(ktype)
        start_str = start.strftime("%Y-%m-%d")
        market = get_market(code)

        if market == "cc":
            df = fetch_binance_data(code, start_str, end_str, ktype)
        elif market == "cn":
            df = fetch_cn_data(code, start_str, end_str, ktype)
        elif api == "stooq-local":
            start_str = "2000-01-03"  # Stooq 全量数据，日/周都从 2000 开始
            if ktype == "day":
                _prev = os.path.join(DATA_DIR, "1d", market, f"{code}_1d.parquet")
                if os.path.exists(_prev):
                    _ok += 1
                    continue
            _daily_path = os.path.join(DATA_DIR, "1d", market, f"{code}_1d.parquet") if ktype == "week" else None
            df = fetch_stooq_local_data(code, start_str, end_str, ktype, _save_daily_path=_daily_path)
        elif api == "yahoo":
            df = fetch_yahoo_data(code, start_str, end_str, ktype)
        elif api == "ibkr":
            if ktype == "day":
                _prev_path = os.path.join(DATA_DIR, "1d", market, f"{code}_1d.parquet")
                if os.path.exists(_prev_path):
                    _ok += 1
                    continue
                df = fetch_ibkr_data(code, start_str, end_str, ktype, ibkr_app=_ibkr_app)
            elif ktype == "week":
                _daily_path = os.path.join(DATA_DIR, "1d", market, f"{code}_1d.parquet")
                df = fetch_ibkr_data(code, start_str, end_str, ktype, _save_daily_path=_daily_path, ibkr_app=_ibkr_app)
            else:
                df = fetch_ibkr_data(code, start_str, end_str, ktype, ibkr_app=_ibkr_app)
        else:
            df = fetch_futu_data(code, start_str, end_str, quote_ctx, ktype) if quote_ctx else pd.DataFrame()

        save_data(df, code, ktype, name_map)
        if not df.empty:
            all_dfs.append(df)
            _ok += 1
        else:
            _fail += 1
            _failed_codes.append(code)
    if _fail:
        print(f"  {ktype} 下载完成: {_ok} 成功, {_fail} 失败 — {'; '.join(_failed_codes)}")
    else:
        print(f"  {ktype} 下载完成: {_ok} 成功")

    if quote_ctx is not None:
        quote_ctx.close()
    if _ibkr_app:
        _ibkr_app.disconnect()

    # 合并总表（按市场分别保存）
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.rename(columns={"time_key": "datetime"})
        combined["datetime"] = pd.to_datetime(combined["datetime"])
        combined = combined.drop_duplicates(["code", "datetime"]).sort_values(["code", "datetime"]).reset_index(drop=True)

        suffix = KTYPE_SUFFIX_MAP.get(ktype, f"1{ktype[0]}")
        base_cols = ["code", "market", "datetime", "open", "high", "low", "close", "volume", "turnover"]
        combined = combined[[c for c in base_cols if c in combined.columns]]
        out_path = os.path.join(DATA_DIR, KTYPE_DIR_MAP.get(ktype, ""), f"all_{suffix}.parquet")
        combined.to_parquet(out_path, index=False)
    else:
        print("\n无数据，跳过总表保存")



# =========================================================
# 主入口
# =========================================================
def run_download_all(selected_markets=None, skip_week=False, skip_day=False, skip_60m=False, api="futu"):
    """分阶段下载周线、日线、60分钟数据。"""
    ktypes = []
    if not skip_week:
        ktypes.append("week")
    if not skip_day:
        ktypes.append("day")
    if not skip_60m:
        ktypes.append("60m")

    for ktype in ktypes:
        run_download(ktype=ktype, selected_markets=selected_markets, api=api)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="多市场 K 线数据下载")
    parser.add_argument("--ktype", default=DEFAULT_KTYPE,
                        help="K线周期: week(周K) / day(日K) / 60m(60分钟K) / all(全部) / week,day(逗号拼接, 默认: all)")
    parser.add_argument("--market", default=DEFAULT_MARKET,
                        help=f"市场: US / CN / CC / US,CN / all (默认: {DEFAULT_MARKET})")
    parser.add_argument("--api", choices=["futu", "ibkr", "yahoo", "stooq-local"], default=DEFAULT_API,
                        help="美股数据源: futu(富途OpenD, 默认) / ibkr(IB TWS/Gateway)")
    parser.add_argument("--top-turnover", type=int, nargs="?", const=200, default=200,
                        help="获取成交额前 N 的美股列表并保存到 symbols/ (默认 N=200, 设为0跳过)")
    parser.add_argument("--skip-week", action="store_true", help="跳过周线下载（已弃用，用 --ktype 替代）")
    parser.add_argument("--skip-day", action="store_true", help="跳过日线下载（已弃用，用 --ktype 替代）")
    parser.add_argument("--skip-60m", action="store_true", help="跳过60分钟下载（已弃用，用 --ktype 替代）")
    args = parser.parse_args()

    # 获取成交额排名（默认运行，设为 --top-turnover 0 跳过）
    if args.top_turnover:
        fetch_top_turnover_stocks(limit=args.top_turnover)

    # 解析市场参数
    if args.market.lower() == "all":
        selected_markets = ["us", "cn", "cc"]
    else:
        selected_markets = [m.strip().lower() for m in args.market.split(",")]

    # 解析 ktype（支持逗号拼接，兼容旧版 skip 参数）
    _ktypes = []
    for _k in args.ktype.lower().replace("，", ",").split(","):
        _k = _k.strip()
        if _k == "all":
            _ktypes = ["week", "day", "60m"]
            break
        if _k in ("week", "day", "60m") and _k not in _ktypes:
            _ktypes.append(_k)
    if not _ktypes:
        _ktypes = ["week", "day", "60m"]
    # 旧版 skip 参数覆盖
    if args.skip_week and "week" in _ktypes:
        _ktypes.remove("week")
    if args.skip_day and "day" in _ktypes:
        _ktypes.remove("day")
    if args.skip_60m and "60m" in _ktypes:
        _ktypes.remove("60m")

    _all_start = time.time()
    # 分阶段下载 K 线数据
    run_download_all(selected_markets=selected_markets,
                     skip_week="week" not in _ktypes, api=args.api,
                     skip_day="day" not in _ktypes,
                     skip_60m="60m" not in _ktypes)

    # 再同步板块/行业信息，并补写到各周期 K 线 parquet
    plates_map = run_plate_sync(selected_markets=selected_markets)
    for _kt in KTYPE_DIR_MAP:
        add_plates_to_parquets(_kt, plates_map)
    _elapsed = time.time() - _all_start
    _min = int(_elapsed // 60)
    _sec = int(_elapsed % 60)
    print(f"全部完成，总耗时 {_min}分{_sec}秒")