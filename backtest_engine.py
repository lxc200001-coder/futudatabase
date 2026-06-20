#!/usr/bin/env python3
"""
backtest_engine.py - 逐K线策略回测引擎（独立于现有框架）
=========================================================
将逐K线回测统计写入 backtest_stats 表，不修改现有回测框架。

用法:
    python backtest_engine.py --ktype 1w --ma-start 2 --ma-end 61 --trade-mode close
    python backtest_engine.py --ktype 1d --ma-start 2 --ma-end 181 --slippage 0.001
    python backtest_engine.py --ktype 1w,1d --all  # 周线+日线全量运行
"""

import os
import sys
import argparse
import time
import concurrent.futures
from datetime import datetime
from tqdm import tqdm

import duckdb
import pandas as pd
import numpy as np
from numba import njit

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "market.duckdb")

INITIAL_CASH = 10000
FEE_RATE = 0.001
SLIPPAGE = 0.0
TRADE_MODE = "close"   # close / open
MA_MODE = "continuous" # continuous / jump
DEFAULT_KTYPE = "1w"   # 默认ktype: 1w(周K) / 1d(日K) / all(两者全部) / 1w,1d(逗号拼接)
DEFAULT_MARKET = "CC" # 默认market: all / US / CN / CC / US,CC
WINDOW_START_DATE = "2000-01-03"

# =========================================================
# Numba 加速：账户状态逐K线计算
# =========================================================

@njit
def _numba_account_loop(closes, trade_actions, trade_prices, n, initial_cash, slippage, fee_rate):
    """@njit 逐K线计算账户状态。

    trade_actions: 0=无, 1=开多, 2=平多
    trade_prices: 成交价，NaN 表示无交易
    返回所有账户数组。
    """
    available_cash = np.full(n, initial_cash, dtype=np.float64)
    held_shares = np.zeros(n, dtype=np.int64)
    trade_shares_arr = np.zeros(n, dtype=np.int64)
    comm_arr = np.zeros(n, dtype=np.float64)
    slip_arr = np.zeros(n, dtype=np.float64)
    account_value = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if i > 0:
            available_cash[i] = available_cash[i - 1]
            held_shares[i] = held_shares[i - 1]

        ta = trade_actions[i]
        tp = trade_prices[i]

        if ta == 1 and not np.isnan(tp) and tp > 0:  # 开多
            sh = int(available_cash[i] / (tp * (1 + slippage + fee_rate)))
            if sh > 0:
                sc = sh * tp * slippage
                cm = sh * tp * fee_rate
                slip_arr[i] = sc
                comm_arr[i] = cm
                available_cash[i] -= sh * tp + sc + cm
                held_shares[i] += sh
                trade_shares_arr[i] = sh

        elif ta == 2 and not np.isnan(tp) and held_shares[i] > 0:  # 平多
            sh = held_shares[i]
            sc = sh * tp * slippage
            cm = sh * tp * fee_rate
            slip_arr[i] = sc
            comm_arr[i] = cm
            available_cash[i] += sh * tp - sc - cm
            held_shares[i] = 0
            trade_shares_arr[i] = sh

        account_value[i] = available_cash[i] + held_shares[i] * closes[i]

    return available_cash, held_shares, trade_shares_arr, comm_arr, slip_arr, account_value


# =========================================================
# 核心计算
# =========================================================

def process_stock(df, ma_len, trade_mode, slippage, fee_rate, ktype):
    """对已加载的K线DataFrame计算单MA回测统计，返回逐K线结果。"""
    if df is None or df.empty:
        return pd.DataFrame()

    n = len(df)
    rows = []
    closes = df["close"].values.astype(np.float64)

    # ── ha_close ──
    ha_close = (df["open"].values + df["high"].values + df["low"].values + closes) / 4.0

    # ── ha_ma_value ── (手算 rolling 避免 pandas 开销)
    ha_ma_val = np.full(n, np.nan, dtype=np.float64)
    for i in range(ma_len - 1, n):
        ha_ma_val[i] = ha_close[i - ma_len + 1:i + 1].mean()

    # ── trend_direction / signal ──
    direction = np.full(n, "空头", dtype=object)
    for i in range(1, n):
        if not np.isnan(ha_ma_val[i]) and not np.isnan(ha_ma_val[i - 1]):
            direction[i] = "多头" if ha_ma_val[i] > ha_ma_val[i - 1] else "空头"

    signal = np.full(n, "", dtype=object)
    for i in range(1, n):
        if direction[i] == "多头" and direction[i - 1] == "空头":
            signal[i] = "买入"
        elif direction[i] == "多头" and direction[i - 1] == "多头":
            signal[i] = "持有"
        elif direction[i] == "空头" and direction[i - 1] == "多头":
            signal[i] = "卖出"
        elif direction[i] == "空头" and direction[i - 1] == "空头":
            signal[i] = "等待"
    if n > 0:
        signal[0] = "等待"

    # ── trade_action / trade_price (编码为 int/float 供 numba 使用) ──
    trade_actions = np.zeros(n, dtype=np.int64)   # 0=无, 1=开多, 2=平多
    trade_prices = np.full(n, np.nan, dtype=np.float64)

    if trade_mode == "close":
        for i in range(n):
            if signal[i] == "买入":
                trade_actions[i] = 1
                trade_prices[i] = closes[i]
            elif signal[i] == "卖出":
                trade_actions[i] = 2
                trade_prices[i] = closes[i]
    else:  # open
        opens_arr = df["open"].values.astype(np.float64)
        for i in range(1, n):
            if signal[i - 1] == "买入":
                trade_actions[i] = 1
                trade_prices[i] = opens_arr[i]
            elif signal[i - 1] == "卖出":
                trade_actions[i] = 2
                trade_prices[i] = opens_arr[i]

    # ── @njit 账户状态计算 ──
    (available_cash_arr, held_shares_arr, trade_shares_arr,
     commission_arr, slippage_arr, account_value_arr) = _numba_account_loop(
        closes, trade_actions, trade_prices, n, INITIAL_CASH, slippage, fee_rate
    )

    # ── 变动指标（numpy 向量化） ──
    acc_change = np.zeros(n, dtype=np.float64)
    acc_change_pct = np.zeros(n, dtype=np.float64)
    acc_change[1:] = account_value_arr[1:] - account_value_arr[:-1]
    acc_change_pct[1:] = np.divide(acc_change[1:], account_value_arr[:-1],
                                   out=np.zeros_like(acc_change[1:]),
                                   where=account_value_arr[:-1] != 0)

    change_init = account_value_arr - INITIAL_CASH
    change_init_pct = np.divide(change_init, INITIAL_CASH,
                                out=np.zeros_like(change_init),
                                where=INITIAL_CASH != 0)

    # ── 构建结果（从 int trade_actions 恢复中文字段） ──
    ta_labels = np.full(n, None, dtype=object)
    tp_vals = np.full(n, None, dtype=object)
    for i in range(n):
        if trade_actions[i] == 1:
            ta_labels[i] = "开多"
            tp_vals[i] = float(trade_prices[i]) if not np.isnan(trade_prices[i]) else None
        elif trade_actions[i] == 2:
            ta_labels[i] = "平多"
            tp_vals[i] = float(trade_prices[i]) if not np.isnan(trade_prices[i]) else None

    # ── 构建结果 ──
    for i in range(n):
        rows.append({
            "code": str(df["code"].iloc[i]),
            "stock_name": str(df["stock_name"].iloc[i]) if "stock_name" in df.columns else "",
            "market": str(df["market"].iloc[i]) if "market" in df.columns else "",
            "ktype": ktype,
            "datetime": df["datetime"].iloc[i],
            "open": float(df["open"].iloc[i]),
            "high": float(df["high"].iloc[i]),
            "low": float(df["low"].iloc[i]),
            "close": float(closes[i]),
            "volume": float(df["volume"].iloc[i]),
            "turnover": float(df["turnover"].iloc[i]) if "turnover" in df.columns else 0.0,
            "turnover_amount": float(df["turnover_amount"].iloc[i]) if "turnover_amount" in df.columns else 0.0,
            "source": str(df["source"].iloc[i]) if "source" in df.columns else "",
            "ha_close": float(ha_close[i]),
            "ma_len": ma_len,
            "ha_ma_value": float(ha_ma_val[i]) if not np.isnan(ha_ma_val[i]) else None,
            "trend_direction": direction[i],
            "signal": signal[i],
            "trade_action": str(ta_labels[i]) if ta_labels[i] is not None else None,
            "trade_price": float(tp_vals[i]) if tp_vals[i] is not None else None,
            "available_cash": float(available_cash_arr[i]),
            "trade_shares": int(trade_shares_arr[i]),
            "slippage": float(slippage_arr[i]),
            "commission": float(commission_arr[i]),
            "held_shares": int(held_shares_arr[i]),
            "account_value": float(account_value_arr[i]),
            "account_value_change": float(acc_change[i]),
            "account_value_change_pct": float(acc_change_pct[i]),
            "change_from_initial": float(change_init[i]),
            "change_from_initial_pct": float(change_init_pct[i]),
            "created_at": pd.Timestamp.now(),
        })

    return pd.DataFrame(rows)


def generate_windows(step_months, end_date=None):
    """根据步长生成固定窗口列表，末窗停在最后一个完整步长边界。"""
    start = pd.Timestamp(WINDOW_START_DATE)
    if end_date is None:
        end_date = pd.Timestamp.now()
    else:
        end_date = pd.Timestamp(end_date)
    windows = []
    cur = start + pd.DateOffset(months=step_months)
    while cur <= end_date:
        windows.append((start, cur))
        cur += pd.DateOffset(months=step_months)
    return windows


def run_stock(code, ktype, ma_range, windows, trade_mode, slippage, fee_rate):
    """对一只股票加载K线，按窗口运行所有 MA 参数。"""
    _kt = {"1w": "1w", "1d": "1d"}.get(ktype, ktype)
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        df_k = con.execute(f"""
            SELECT * FROM klines_{_kt}
            WHERE code = ? ORDER BY datetime
        """, [code]).fetchdf()
    finally:
        con.close()

    if df_k.empty:
        return pd.DataFrame()

    # 补齐缺失列
    for c in ["code", "stock_name", "market", "ktype", "datetime",
              "open", "high", "low", "close", "volume", "turnover",
              "turnover_amount", "source"]:
        if c not in df_k.columns:
            df_k[c] = "" if c in ("code", "stock_name", "market", "ktype", "source") else 0.0

    all_dfs = []
    for ws, we in windows:
        window_label = f"{ws.date()}~{we.date()}"
        df_w = df_k[(pd.to_datetime(df_k["datetime"]) >= ws) &
                    (pd.to_datetime(df_k["datetime"]) < we)]
        if df_w.empty:
            continue

        for ma in ma_range:
            df = process_stock(df_w, ma, trade_mode, slippage, fee_rate, ktype)
            if not df.empty:
                df["window_label"] = window_label
                all_dfs.append(df)

    if all_dfs:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", FutureWarning)
            return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


def _worker_stock(code, ktype, windows, ma_range, trade_mode, slippage, fee_rate):
    """工作进程：计算一只股票的所有MA+窗口，返回 (code, df, err)"""
    for _attempt in range(5):
        try:
            df = run_stock(code, ktype, ma_range, windows, trade_mode, slippage, fee_rate)
            return code, df, None
        except Exception as e:
            _err = str(e)
            if "另一个程序正在使用" in _err or "Cannot open file" in _err:
                time.sleep(1 * (_attempt + 1))  # 退避重试
                continue
            return code, None, _err
    return code, None, "多次重试后仍无法访问数据库"


def run_backtest(ktype="1w", ma_list=None, ma_start=2, ma_end=61, ma_step=1,
                 trade_mode="close", slippage=0.0, fee_rate=0.001, markets=None):
    """主入口：对所有股票并行回测并写入 backtest_stats 表。"""
    ma_range = ma_list if ma_list is not None else list(range(ma_start, ma_end, ma_step))

    _step_map = {"1w": 12, "1d": 6}
    step_months = _step_map.get(ktype, 12)

    # 获取股票列表
    con = duckdb.connect(DB_PATH, read_only=True)
    codes = [str(r[0]) for r in con.execute(
        "SELECT DISTINCT code FROM watchlist ORDER BY code"
    ).fetchall()]
    con.close()
    if markets:
        _pfx = []
        for _m in markets.upper().split(","):
            _m = _m.strip()
            if _m in ("ALL", "US"): _pfx.append("US.")
            if _m in ("ALL", "CN"): _pfx.extend(("SH.", "SZ."))
            if _m in ("ALL", "CC"): _pfx.append("CC.")
        if _pfx: codes = [c for c in codes if any(c.startswith(p) for p in _pfx)]
    if not codes: print("  watchlist 为空"); return

    # 全市场K线最大日期
    _kt = {"1w": "1w", "1d": "1d"}.get(ktype, ktype)
    con2 = duckdb.connect(DB_PATH, read_only=True)
    _max_dt = con2.execute(f"SELECT MAX(datetime) FROM klines_{_kt}").fetchone()[0]
    con2.close()
    if _max_dt is None: print("  无K线数据"); return
    windows = generate_windows(step_months, end_date=pd.Timestamp(_max_dt))

    _n_workers = max(1, os.cpu_count() - 1)
    total_ma = len(ma_range)
    total_rows = 0
    print(f"  并行: {_n_workers}进程 | 股票: {len(codes)} | 窗口: {len(windows)} | MA: {total_ma}")

    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=_n_workers) as executor:
        futures = {executor.submit(_worker_stock, code, ktype, windows, ma_range,
                                   trade_mode, slippage, fee_rate): code for code in codes}
        for future in concurrent.futures.as_completed(futures):
            try:
                _code, df, err = future.result()
            except Exception as e:
                print(f"\n  进程异常: {e}")
                continue
            if err:
                print(f"\n  {_code} 失败: {err}")
                continue
            if df is not None and not df.empty:
                all_results.append((_code, df))

    # 所有子进程结束 → 统一写入（避免锁冲突）
    for _code, df in all_results:
        try:
            _cw = duckdb.connect(DB_PATH)
            _cw.execute("DELETE FROM backtest_stats WHERE code = ? AND ktype = ?", [_code, ktype])
            _cw.execute("CREATE OR REPLACE TEMP TABLE _tmp AS SELECT * FROM df")
            _cw.execute("""
                INSERT INTO backtest_stats (
                    code, stock_name, market, ktype, window_label, datetime,
                    open, high, low, close, volume, turnover, turnover_amount, source,
                    ha_close, ma_len, ha_ma_value, trend_direction, signal,
                    trade_action, trade_price, available_cash, trade_shares,
                    slippage, commission, held_shares, account_value,
                    account_value_change, account_value_change_pct,
                    change_from_initial, change_from_initial_pct, created_at
                )
                SELECT
                    code, stock_name, market, ktype, window_label, datetime,
                    open, high, low, close, volume, turnover, turnover_amount, source,
                    ha_close, ma_len, ha_ma_value, trend_direction, signal,
                    trade_action, trade_price, available_cash, trade_shares,
                    slippage, commission, held_shares, account_value,
                    account_value_change, account_value_change_pct,
                    change_from_initial, change_from_initial_pct, created_at
                FROM _tmp
            """)
            _cw.close()
            total_rows += len(df)
            print(f"  {_code}: {len(df)} 行")
        except Exception as e:
            print(f"\n  {_code} 写入失败: {e}")

    print(f"\n完成: {total_rows:,} 行写入 backtest_stats")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="逐K线策略回测引擎")
    parser.add_argument("--ktype", default=DEFAULT_KTYPE,
                        help=f"K线周期: 1w / 1d / 1w,1d / all (默认: {DEFAULT_KTYPE})")
    parser.add_argument("--market", default=DEFAULT_MARKET,
                        help=f"市场: US / CN / CC / US,CC / all (默认: {DEFAULT_MARKET})")
    parser.add_argument("--ma-start", type=int, default=2,
                        help="MA起始值 (默认: 2)")
    parser.add_argument("--ma-end", type=int, default=61,
                        help="MA结束值 (默认: 61, 日线建议 181)")
    parser.add_argument("--ma-mode", choices=["continuous", "jump"], default="continuous",
                        help="MA序列类型: continuous=连续, jump=跳跃(仅日线, step=2)")
    parser.add_argument("--ma-step", type=int, default=1,
                        help="MA步长 (默认: 1, 仅 --ma-start/end 手动模式生效)")
    parser.add_argument("--trade-mode", choices=["close", "open"], default=TRADE_MODE,
                        help="成交方式: close=收盘价成交, open=下根开盘价成交")
    parser.add_argument("--slippage", type=float, default=SLIPPAGE,
                        help="滑点比例 (默认 0.0)")
    parser.add_argument("--fee-rate", type=float, default=FEE_RATE,
                        help="佣金比例 (默认 0.001)")
    args = parser.parse_args()

    # 应用前置配置项
    MA_MODE = args.ma_mode
    TRADE_MODE = args.trade_mode
    SLIPPAGE = args.slippage
    FEE_RATE = args.fee_rate

    # 生成 MA 序列（沿用 backtest_uscncc.py 逻辑）
    _ma_map = {
        "1w": list(range(2, 61)),                          # 周线: 2..60 (固定 step=1)
        "1d": list(range(2, 181, 2 if MA_MODE == "jump" else 1)),  # 日线
    }
    if args.ma_mode == "jump":
        _ma_map["1d"] = list(range(2, 181, 2))  # 日线 jump: 2..180 step=2

    # 解析 ktype
    _raw = args.ktype.lower().replace("，", ",").split(",")
    _ktypes = []
    for _k in _raw:
        _k = _k.strip()
        if _k == "all":
            _ktypes = ["1w", "1d"]; break
        if _k in ("1w", "1d") and _k not in _ktypes:
            _ktypes.append(_k)
    if not _ktypes:
        _ktypes = ["1w"]

    print(f"K线周期: {','.join(_ktypes)} | MA 序列: {_ma_map[_ktypes[0]][:3]}...~{_ma_map[_ktypes[0]][-1]} ({len(_ma_map[_ktypes[0]])}个) | "
          f"成交方式: {args.trade_mode} | 滑点: {args.slippage} | 佣金: {args.fee_rate}")

    t0 = time.time()
    for _kt in _ktypes:
        _ma_list = _ma_map[_kt]
        run_backtest(
            ktype=_kt, ma_list=_ma_list,
            trade_mode=args.trade_mode,
            slippage=args.slippage,
            fee_rate=args.fee_rate,
            markets=args.market,
        )
    print(f"总耗时: {time.time() - t0:.0f}s")
