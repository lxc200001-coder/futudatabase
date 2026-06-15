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
from datetime import datetime

import duckdb
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "market.duckdb")

INITIAL_CASH = 10000
FEE_RATE = 0.001
SLIPPAGE = 0.0
TRADE_MODE = "close"  # close / open

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

    # ── ha_ma_value ── (手算rolling避免pandas开销)
    ha_ma_val = np.full(n, np.nan, dtype=float)
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
    # 第一根K线
    if n > 0:
        signal[0] = "等待"

    # ── trade_action / trade_price ──
    trade_action = np.full(n, None, dtype=object)
    trade_price = np.full(n, np.nan, dtype=float)

    if trade_mode == "close":
        for i in range(n):
            if signal[i] == "买入":
                trade_action[i] = "开多"
                trade_price[i] = closes[i]
            elif signal[i] == "卖出":
                trade_action[i] = "平多"
                trade_price[i] = closes[i]
    else:  # open
        for i in range(1, n):
            if signal[i - 1] == "买入":
                trade_action[i] = "开多"
                trade_price[i] = float(df["open"].iloc[i])
            elif signal[i - 1] == "卖出":
                trade_action[i] = "平多"
                trade_price[i] = float(df["open"].iloc[i])

    # ── 账户状态逐K线计算 ──
    available_cash_arr = np.full(n, INITIAL_CASH, dtype=float)
    held_shares_arr = np.zeros(n, dtype=int)
    trade_shares_arr = np.zeros(n, dtype=int)
    commission_arr = np.zeros(n, dtype=float)
    slippage_arr = np.zeros(n, dtype=float)
    account_value_arr = np.zeros(n, dtype=float)

    for i in range(n):
        if i > 0:
            available_cash_arr[i] = available_cash_arr[i - 1]
            held_shares_arr[i] = held_shares_arr[i - 1]

        ta = trade_action[i]
        tp = trade_price[i]

        if ta == "开多" and not np.isnan(tp) and tp > 0:
            sh = int(available_cash_arr[i] / (tp * (1 + slippage + fee_rate)))
            if sh > 0:
                cost_slip = sh * tp * slippage
                cost_comm = sh * tp * fee_rate
                slippage_arr[i] = cost_slip
                commission_arr[i] = cost_comm
                available_cash_arr[i] -= sh * tp + cost_slip + cost_comm
                held_shares_arr[i] += sh
                trade_shares_arr[i] = sh

        elif ta == "平多" and not np.isnan(tp) and held_shares_arr[i] > 0:
            sh = held_shares_arr[i]
            cost_slip = sh * tp * slippage
            cost_comm = sh * tp * fee_rate
            slippage_arr[i] = cost_slip
            commission_arr[i] = cost_comm
            available_cash_arr[i] += sh * tp - cost_slip - cost_comm
            held_shares_arr[i] = 0
            trade_shares_arr[i] = sh

        account_value_arr[i] = available_cash_arr[i] + held_shares_arr[i] * closes[i]

    # ── 变动指标 ──
    acc_change = np.zeros(n, dtype=float)
    acc_change_pct = np.zeros(n, dtype=float)
    change_init = np.zeros(n, dtype=float)
    change_init_pct = np.zeros(n, dtype=float)

    for i in range(1, n):
        acc_change[i] = account_value_arr[i] - account_value_arr[i - 1]
        acc_change_pct[i] = acc_change[i] / account_value_arr[i - 1] if account_value_arr[i - 1] != 0 else 0.0
    for i in range(n):
        change_init[i] = account_value_arr[i] - INITIAL_CASH
        change_init_pct[i] = change_init[i] / INITIAL_CASH if INITIAL_CASH != 0 else 0.0

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
            "trade_action": str(trade_action[i]) if trade_action[i] is not None else None,
            "trade_price": float(trade_price[i]) if not np.isnan(trade_price[i]) else None,
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


def run_stock(code, ktype, ma_range, trade_mode, slippage, fee_rate):
    """对一只股票加载K线，运行所有 MA 参数。"""
    # 加载一次K线数据
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
    for ma in ma_range:
        df = process_stock(df_k, ma, trade_mode, slippage, fee_rate, ktype)
        if not df.empty:
            all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


def run_backtest(ktype="1w", ma_start=2, ma_end=61, ma_step=1,
                 trade_mode="close", slippage=0.0, fee_rate=0.001):
    """主入口：对所有股票运行回测并写入 backtest_stats 表。"""
    ma_range = list(range(ma_start, ma_end, ma_step))

    # 获取股票列表
    con = duckdb.connect(DB_PATH, read_only=True)
    codes = [str(r[0]) for r in con.execute(
        "SELECT DISTINCT code FROM watchlist ORDER BY code"
    ).fetchall()]
    con.close()

    if not codes:
        print("  watchlist 为空")
        return

    total_ma = len(ma_range)
    total_rows = 0
    con_w = duckdb.connect(DB_PATH)

    for i, code in enumerate(codes, 1):
        print(f"  [{i}/{len(codes)}] {code} ({total_ma} MA)...", end=" ", flush=True)
        t0 = time.time()
        df = run_stock(code, ktype, ma_range, trade_mode, slippage, fee_rate)
        if df.empty:
            print("跳过")
            continue

        try:
            con_w.execute("DELETE FROM backtest_stats WHERE code = ? AND ktype = ?", [code, ktype])
        except Exception:
            pass
        try:
            con_w.execute("CREATE OR REPLACE TEMP TABLE _tmp AS SELECT * FROM df")
            con_w.execute("""
                INSERT INTO backtest_stats (
                    code, stock_name, market, ktype, datetime,
                    open, high, low, close, volume, turnover, turnover_amount, source,
                    ha_close, ma_len, ha_ma_value, trend_direction, signal,
                    trade_action, trade_price, available_cash, trade_shares,
                    slippage, commission, held_shares, account_value,
                    account_value_change, account_value_change_pct,
                    change_from_initial, change_from_initial_pct, created_at
                )
                SELECT
                    code, stock_name, market, ktype, datetime,
                    open, high, low, close, volume, turnover, turnover_amount, source,
                    ha_close, ma_len, ha_ma_value, trend_direction, signal,
                    trade_action, trade_price, available_cash, trade_shares,
                    slippage, commission, held_shares, account_value,
                    account_value_change, account_value_change_pct,
                    change_from_initial, change_from_initial_pct, created_at
                FROM _tmp
            """)
            elapsed = time.time() - t0
            print(f"{len(df)} 行 ({elapsed:.1f}s)")
            total_rows += len(df)
        except Exception as e:
            print(f"写入失败: {e}")

    con_w.close()
    print(f"\n完成: {total_rows:,} 行写入 backtest_stats")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="逐K线策略回测引擎")
    parser.add_argument("--ktype", default="1w",
                        help="K线周期: 1w / 1d / 1w,1d / all (默认: 1w)")
    parser.add_argument("--ma-start", type=int, default=2,
                        help="MA起始值 (默认: 2)")
    parser.add_argument("--ma-end", type=int, default=61,
                        help="MA结束值 (默认: 61, 日线建议 181)")
    parser.add_argument("--ma-step", type=int, default=1,
                        help="MA步长 (默认: 1)")
    parser.add_argument("--trade-mode", choices=["close", "open"], default=TRADE_MODE,
                        help="成交方式: close=收盘价成交, open=下根开盘价成交")
    parser.add_argument("--slippage", type=float, default=SLIPPAGE,
                        help="滑点比例 (默认 0.0)")
    parser.add_argument("--fee-rate", type=float, default=FEE_RATE,
                        help="佣金比例 (默认 0.001)")
    args = parser.parse_args()

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

    print(f"K线周期: {','.join(_ktypes)} | MA: {args.ma_start}~{args.ma_end} step={args.ma_step} | "
          f"成交方式: {args.trade_mode} | 滑点: {args.slippage} | 佣金: {args.fee_rate}")

    t0 = time.time()
    for _kt in _ktypes:
        run_backtest(
            ktype=_kt,
            ma_start=args.ma_start,
            ma_end=args.ma_end,
            ma_step=args.ma_step,
            trade_mode=args.trade_mode,
            slippage=args.slippage,
            fee_rate=args.fee_rate,
        )
    print(f"总耗时: {time.time() - t0:.0f}s")
