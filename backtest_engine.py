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
SLIPPAGE = 0.001
TRADE_MODE = "close"   # close / open
MA_MODE = "continuous" # continuous / jump
DEFAULT_KTYPE = "1w"   # 默认ktype: 1w(周K) / 1d(日K) / all(两者全部) / 1w,1d(逗号拼接)
DEFAULT_MARKET = "US,CC" # 默认market: all / US / CN / CC / US,CC
WINDOW_START_DATE = "2000-01-03"

# =========================================================
# Numba 加速：账户状态逐K线计算
# =========================================================

@njit
def _numba_account_loop(closes, trade_actions, trade_prices, n, initial_cash, slippage, fee_rate, allow_fractional=False):
    """@njit 逐K线计算账户状态。

    trade_actions: 0=无, 1=开多, 2=平多
    trade_prices: 成交价，NaN 表示无交易
    allow_fractional: 是否允许碎股（加密货币用）
    返回所有账户数组。
    """
    available_cash = np.full(n, initial_cash, dtype=np.float64)
    held_shares = np.zeros(n, dtype=np.float64)
    trade_shares_arr = np.zeros(n, dtype=np.float64)
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
            tp_slip = tp * (1 + slippage)
            sh = available_cash[i] / (tp_slip * (1 + fee_rate))
            if not allow_fractional:
                sh = int(sh)
            else:
                sh = np.floor(sh * 10000) / 10000
                if sh < 0.0001:
                    sh = 0.0
            if sh > 0:
                sc = sh * tp_slip * slippage / (1 + slippage)
                cm = sh * tp_slip * fee_rate
                available_cash[i] -= sh * tp_slip + cm
                slip_arr[i] = sc
                comm_arr[i] = cm
                held_shares[i] += sh
                trade_shares_arr[i] = sh

        elif ta == 2 and not np.isnan(tp) and held_shares[i] > 0:  # 平多
            sh = held_shares[i]
            tp_slip = tp * (1 - slippage)
            sc = sh * tp_slip * slippage / (1 - slippage)
            cm = sh * tp_slip * fee_rate
            slip_arr[i] = sc
            comm_arr[i] = cm
            available_cash[i] += sh * tp_slip - cm
            held_shares[i] = 0
            trade_shares_arr[i] = sh

        account_value[i] = available_cash[i] + held_shares[i] * closes[i]

    return available_cash, held_shares, trade_shares_arr, comm_arr, slip_arr, account_value


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
    # 如果末窗未覆盖到 end_date，追加一个完整步长窗口
    if not windows or windows[-1][1] < end_date:
        windows.append((start, cur))
    return windows


def run_stock(code, ktype, ma_range, windows, trade_mode, slippage, fee_rate):
    """对一只股票加载K线，先算全量 MA 再按窗口切片（减少96%冗余）。"""
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
    for c in ["code", "stock_name", "market", "ktype", "datetime",
              "open", "high", "low", "close", "volume", "turnover",
              "turnover_amount", "source"]:
        if c not in df_k.columns:
            df_k[c] = "" if c in ("code", "stock_name", "market", "ktype", "source") else 0.0

    n = len(df_k)
    closes = df_k["close"].values.astype(np.float64)
    ha_close = (df_k["open"].values + df_k["high"].values + df_k["low"].values + closes) / 4.0
    _is_cc = "加密货币" in str(df_k.get("market", pd.Series([""])).iloc[0])
    opens_arr = df_k["open"].values.astype(np.float64)

    # 全量计算每个 MA 的信号数组（避免对每个窗口重复计算）
    ma_cache = {}  # ma → {ha_ma_val, direction, signal, trade_actions, trade_prices}
    for ma in ma_range:
        # ha_ma_val (全量)
        ha_ma_val = np.full(n, np.nan, dtype=np.float64)
        for i in range(ma - 1, n):
            ha_ma_val[i] = ha_close[i - ma + 1:i + 1].mean()

        # direction (全量)
        direction = np.full(n, "空头", dtype=object)
        for i in range(1, n):
            if not np.isnan(ha_ma_val[i]) and not np.isnan(ha_ma_val[i - 1]):
                direction[i] = "多头" if ha_ma_val[i] > ha_ma_val[i - 1] else "空头"

        # signal (全量)
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

        # trade_actions / trade_prices (全量)
        trade_actions = np.zeros(n, dtype=np.int64)
        trade_prices = np.full(n, np.nan, dtype=np.float64)
        if trade_mode == "close":
            for i in range(n):
                if signal[i] == "买入":
                    trade_actions[i] = 1; trade_prices[i] = closes[i]
                elif signal[i] == "卖出":
                    trade_actions[i] = 2; trade_prices[i] = closes[i]
        else:
            for i in range(1, n):
                if signal[i - 1] == "买入":
                    trade_actions[i] = 1; trade_prices[i] = opens_arr[i]
                elif signal[i - 1] == "卖出":
                    trade_actions[i] = 2; trade_prices[i] = opens_arr[i]

        # 全量 numba（一次算完，所有窗口共用）
        (ac_arr, hs_arr, ts_arr, _, _, av_arr) = _numba_account_loop(
            closes, trade_actions, trade_prices, n, INITIAL_CASH, slippage, fee_rate,
            allow_fractional=_is_cc
        )
        ma_cache[ma] = (ha_ma_val, direction, signal, trade_actions, trade_prices,
                        ac_arr, hs_arr, ts_arr, av_arr)

    # 预计算每根K线所属的所有窗口标签
    _w_labels = [f"{ws.date()}~{we.date()}" for ws, we in windows]
    _dt_arr = pd.to_datetime(df_k["datetime"]).values
    _wl_cache = {}
    for _i, _dt in enumerate(_dt_arr):
        _belongs = [_w_labels[_j] for _j in range(len(windows)) if windows[_j][1] >= _dt]
        _wl_cache[_i] = (",".join(_belongs), len(_belongs))

    # 仅末窗切片构建结果（window_label 标注所有所属窗口）
    ws, we = windows[-1]
    mask = (pd.to_datetime(df_k["datetime"]) >= ws) & \
           (pd.to_datetime(df_k["datetime"]) <= we)
    all_dfs = []
    if mask.any():
        for ma in ma_range:
            (ha_ma_val, direction, signal, trade_actions, trade_prices,
             ac_arr, hs_arr, ts_arr, av_arr) = ma_cache[ma]
            df = _build_slice_rows(df_k, mask, ma, ktype, ha_ma_val, direction, signal,
                                   trade_actions, trade_prices, ha_close, closes,
                                   ac_arr, hs_arr, ts_arr, av_arr,
                                   slippage, fee_rate, trade_mode=trade_mode, wl_cache=_wl_cache)
            if df is not None and not df.empty:
                all_dfs.append(df)

    if all_dfs:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", FutureWarning)
            return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()

def _build_slice_rows(df, mask, ma_len, ktype, ha_ma_val, direction, signal,
                      trade_actions, trade_prices, ha_close, closes,
                      ac_arr, hs_arr, ts_arr, av_arr, slippage, fee_rate,
                      trade_mode="close", wl_cache=None):
    """对切片后的预计算结果构建行（不跑 numba）。"""
    idx = np.where(mask.values)[0]
    if len(idx) == 0:
        return None

    # 切片（用 idx 索引，保持日期对齐）
    c_sl = closes[idx]
    ha_close_sl = ha_close[idx]
    ha_ma_sl = ha_ma_val[idx]
    dir_sl = direction[idx]
    sig_sl = signal[idx]

    # 变动指标（基于切片后的 av_arr）
    n_sl = len(idx)
    av_sl = av_arr[idx]
    ac_sl = ac_arr[idx]
    hs_sl = hs_arr[idx]
    ts_sl = ts_arr[idx]

    acc_chg = np.zeros(n_sl, dtype=np.float64)
    acc_chg_pct = np.zeros(n_sl, dtype=np.float64)
    acc_chg[1:] = av_sl[1:] - av_sl[:-1]
    acc_chg_pct[1:] = np.divide(acc_chg[1:], av_sl[:-1], out=np.zeros_like(acc_chg[1:]),
                                where=av_sl[:-1] != 0)
    chg_init = av_sl - INITIAL_CASH
    chg_init_pct = np.divide(chg_init, INITIAL_CASH, out=np.zeros_like(chg_init),
                             where=INITIAL_CASH != 0)

    # 中文字段映射
    ta_lbl = np.full(n_sl, None, dtype=object)
    tp_val = np.full(n_sl, None, dtype=object)
    tp_slip_val = np.full(n_sl, None, dtype=object)
    for j in range(n_sl):
        ta = trade_actions[idx[j]]
        tp = trade_prices[idx[j]]
        if ta == 1:
            ta_lbl[j] = "开多"
            if not np.isnan(tp):
                tp_val[j] = float(tp); tp_slip_val[j] = float(tp * (1 + slippage))
        elif ta == 2:
            ta_lbl[j] = "平多"
            if not np.isnan(tp):
                tp_val[j] = float(tp); tp_slip_val[j] = float(tp * (1 - slippage))

    # 交易编号 / 交易状态
    trade_id_arr = np.full(n_sl, None, dtype=object)
    trade_status_arr = np.full(n_sl, None, dtype=object)
    _tid = 0
    for j in range(n_sl):
        ta = trade_actions[idx[j]]
        hs = hs_sl[j]
        if ta == 1:
            _tid += 1
            trade_id_arr[j] = _tid
            trade_status_arr[j] = "持仓中"
        elif ta == 2:
            trade_id_arr[j] = _tid if _tid > 0 else None
            trade_status_arr[j] = "已平仓"
        elif hs > 0:
            trade_id_arr[j] = _tid if _tid > 0 else None
            trade_status_arr[j] = "持仓中"

    # 虚拟平仓（持仓中逐K线模拟平仓）
    _vp_price = np.full(n_sl, None, dtype=object)
    _vp_price_slip = np.full(n_sl, None, dtype=object)
    _vp_shares = np.full(n_sl, None, dtype=object)
    _vp_slip_cost = np.full(n_sl, None, dtype=object)
    _vp_amt = np.full(n_sl, None, dtype=object)
    _vp_comm = np.full(n_sl, None, dtype=object)
    _vp_pnl = np.full(n_sl, None, dtype=object)
    _cash_bt = np.full(n_sl, None, dtype=object)  # 交易前可用现金
    _entry_tp_slip = None
    _entry_comm = None
    _virtual_cash = INITIAL_CASH
    for j in range(n_sl):
        ta = trade_actions[idx[j]]
        tp = trade_prices[idx[j]]
        hs = hs_sl[j]
        ts_val = ts_sl[j]

        if ta == 1 and not np.isnan(tp):  # 开多
            _entry_tp_slip = float(tp * (1 + slippage))
            _entry_comm = float(_entry_tp_slip * ts_val * fee_rate) if ts_val > 0 else 0.0

        # 虚拟平仓计算（持仓中/hs>0 或 平多行都算）
        _eff_hs = hs if hs > 0 else (ts_val if ta == 2 else 0)
        if _entry_tp_slip is not None and _eff_hs > 0:
            vp = c_sl[j] if trade_mode == "close" else float(df["open"].iloc[idx[j]])
            vp_slip = vp * (1 - slippage)
            vp_slip_amt = vp_slip * _eff_hs
            vp_comm = vp_slip_amt * fee_rate
            vp_pnl = (vp_slip - _entry_tp_slip) * _eff_hs - vp_comm - (_entry_comm or 0)
            _vp_price[j] = float(vp)
            _vp_price_slip[j] = float(vp_slip)
            _vp_shares[j] = float(_eff_hs)
            _vp_slip_cost[j] = float(abs(vp_slip - vp) * _eff_hs)
            _vp_amt[j] = float(vp_slip_amt)
            _vp_comm[j] = float(vp_comm)
            _vp_pnl[j] = float(vp_pnl)

        # 交易前可用现金（持仓期间不变，平仓后加上已实现盈亏）
        _cash_bt[j] = float(_virtual_cash)
        if ta == 2 and _vp_pnl[j] is not None:
            _virtual_cash += float(_vp_pnl[j])

        if ta == 2:  # 平多：清零（在虚拟平仓计算之后）
            _entry_tp_slip = None
            _entry_comm = None

    rows = []
    for j in range(n_sl):
        i = idx[j]
        rows.append({
            "code": str(df["code"].iloc[i]), "stock_name": str(df["stock_name"].iloc[i]) if "stock_name" in df.columns else "",
            "market": str(df["market"].iloc[i]) if "market" in df.columns else "", "ktype": ktype,
            "window_label": wl_cache[i][0] if wl_cache is not None else "",
            "window_count": wl_cache[i][1] if wl_cache is not None else 0,
            "datetime": df["datetime"].iloc[i],
            "open": float(df["open"].iloc[i]), "high": float(df["high"].iloc[i]), "low": float(df["low"].iloc[i]),
            "close": float(c_sl[j]), "volume": float(df["volume"].iloc[i]),
            "turnover": float(df["turnover"].iloc[i]) if "turnover" in df.columns else 0.0,
            "turnover_amount": float(df["turnover_amount"].iloc[i]) if "turnover_amount" in df.columns else 0.0,
            "source": str(df["source"].iloc[i]) if "source" in df.columns else "",
            "ha_close": float(ha_close_sl[j]), "ma_len": ma_len,
            "ha_ma_value": float(ha_ma_sl[j]) if not np.isnan(ha_ma_sl[j]) else None,
            "trend_direction": dir_sl[j], "signal": sig_sl[j],
            "trade_id": int(trade_id_arr[j]) if trade_id_arr[j] is not None else None,
            "trade_action": str(ta_lbl[j]) if ta_lbl[j] is not None else None,
            "trade_price": float(tp_val[j]) if tp_val[j] is not None else None,
            "trade_price_after_slippage": float(tp_slip_val[j]) if tp_slip_val[j] is not None else None,
            "trade_shares": float(ts_sl[j]),
            "slippage": float(abs((tp_slip_val[j] - tp_val[j]) * ts_sl[j])) if tp_val[j] is not None and ts_sl[j] > 0 else 0.0,
            "trade_amount": float(tp_slip_val[j] * ts_sl[j]) if tp_slip_val[j] is not None and ts_sl[j] > 0 else None,
            "commission": float(tp_slip_val[j] * ts_sl[j] * fee_rate) if tp_slip_val[j] is not None and ts_sl[j] > 0 else None,
            "actual_trade_amount": float(tp_slip_val[j] * ts_sl[j] * (1 + fee_rate)) if tp_slip_val[j] is not None and ts_sl[j] > 0 else None,
            "available_cash": float(ac_sl[j]),
            "held_shares": float(hs_sl[j]),
            "trade_status": str(trade_status_arr[j]) if trade_status_arr[j] is not None else None,
            "close_price": float(_vp_price[j]) if _vp_price[j] is not None else None,
            "close_price_after_slippage": float(_vp_price_slip[j]) if _vp_price_slip[j] is not None else None,
            "close_shares": float(_vp_shares[j]) if _vp_shares[j] is not None else None,
            "close_slippage": float(_vp_slip_cost[j]) if _vp_slip_cost[j] is not None else None,
            "close_trade_amount": float(_vp_amt[j]) if _vp_amt[j] is not None else None,
            "close_commission": float(_vp_comm[j]) if _vp_comm[j] is not None else None,
            "close_actual_trade_amount": float(_vp_amt[j] + _vp_comm[j]) if _vp_amt[j] is not None else None,
            "close_pnl": float(_vp_pnl[j]) if _vp_pnl[j] is not None else None,
            "close_type": "真实平仓" if trade_status_arr[j] == "已平仓" else ("虚拟平仓" if trade_status_arr[j] == "持仓中" else None),
            "close_pnl_type": None if ta_lbl[j] == "开多" else ("盈利" if (_vp_pnl[j] is not None and _vp_pnl[j] > 0) else ("亏损" if _vp_pnl[j] is not None else None)),
            "cash_before_trade": float(_cash_bt[j]) if _cash_bt[j] is not None else None,
            "cash_after_trade": float(_cash_bt[j] + (_vp_pnl[j] or 0)) if _cash_bt[j] is not None else None,
            "account_value": float(av_sl[j]),
            "account_value_change": float(acc_chg[j]),
            "account_value_change_pct": float(acc_chg_pct[j]),
            "change_from_initial": float(chg_init[j]),
            "change_from_initial_pct": float(chg_init_pct[j]),
            "created_at": pd.Timestamp.now(),
        })

    return pd.DataFrame(rows)


def _worker_stock(code, ktype, windows, ma_range, trade_mode, slippage, fee_rate):
    """工作进程：计算一只股票的所有MA+窗口，保存到临时 parquet 返回路径。"""
    try:
        df = run_stock(code, ktype, ma_range, windows, trade_mode, slippage, fee_rate)
        if df is None or df.empty:
            return code, None, None
        _tmp = os.path.join(PROJECT_ROOT, "results_uscncc", f"_tmp_{code.replace('.','_')}_{ktype}.parquet")
        os.makedirs(os.path.dirname(_tmp), exist_ok=True)
        df.to_parquet(_tmp, index=False)
        return code, _tmp, None
    except Exception as e:
        return code, None, str(e)


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

    # 提前清空旧数据
    _cw = duckdb.connect(DB_PATH)
    try: _cw.execute("DELETE FROM backtest_stats")
    except: pass
    _cw.close()

    _tmp_files = []
    _cols = ["code","stock_name","market","ktype","window_label","window_count","datetime",
        "open","high","low","close","volume","turnover","turnover_amount","source",
        "ha_close","ma_len","ha_ma_value","trend_direction","signal",
        "trade_id","trade_action","trade_price","trade_price_after_slippage",
        "trade_shares","slippage","trade_amount","commission","actual_trade_amount",
        "available_cash","held_shares","trade_status",
        "close_price","close_price_after_slippage","close_shares","close_slippage",
        "close_trade_amount","close_commission","close_actual_trade_amount","close_pnl","close_type","close_pnl_type",
        "cash_before_trade","cash_after_trade",
        "account_value","account_value_change","account_value_change_pct",
        "change_from_initial","change_from_initial_pct",
        "created_at"]
    _sel = ",".join(_cols)

    # 子进程全部结束后再统一写入（避免多进程锁冲突）
    with concurrent.futures.ProcessPoolExecutor(max_workers=_n_workers) as executor:
        futures = {executor.submit(_worker_stock, code, ktype, windows, ma_range,
                                   trade_mode, slippage, fee_rate): code for code in codes}
        with tqdm(total=len(futures), desc="  回测", unit="stock") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    _code, _path, err = future.result()
                except Exception as e:
                    print(f"\n  进程异常: {e}")
                    pbar.update(1); continue
                if err:
                    print(f"\n  {_code} 失败: {err}")
                    pbar.update(1); continue
                if not _path:
                    pbar.update(1); continue
                # 记录临时文件路径
                _tmp_files.append(_path)
                pbar.update(1)

    # 所有子进程结束→DuckDB 原生批量读 parquet（比逐行快 100 倍）
    if _tmp_files:
        print("  写入中...", end=" ", flush=True)
        _cw = duckdb.connect(DB_PATH)
        _plist = ",".join(f"'{p}'" for p in _tmp_files)
        _cw.execute(f"""
            INSERT INTO backtest_stats ({_sel})
            SELECT {_sel} FROM read_parquet([{_plist}])
        """)
        total_rows = _cw.execute("SELECT count(*) FROM backtest_stats").fetchone()[0]
        _cw.close()
        print(f"  {len(_tmp_files)} 个 parquet 写入完成 ({total_rows:,} 行)")
        for _p in _tmp_files:
            try: os.remove(_p)
            except: pass

    # 全局排序（设置临时目录避免 OOM）
    if total_rows > 0:
        print("  排序中...", end=" ", flush=True)
        _tmp_dir = os.path.join(PROJECT_ROOT, "results_uscncc", "_duckdb_tmp")
        os.makedirs(_tmp_dir, exist_ok=True)
        _cw = duckdb.connect(DB_PATH)
        _cw.execute(f"SET temp_directory = '{_tmp_dir}'")
        _cw.execute(f"""
            CREATE TABLE backtest_stats_sorted AS
            SELECT * FROM backtest_stats
            ORDER BY code ASC, ktype ASC, window_label ASC, ma_len ASC, datetime ASC
        """)
        _cw.execute("DROP TABLE backtest_stats")
        _cw.execute("ALTER TABLE backtest_stats_sorted RENAME TO backtest_stats")
        _cw.close()

    # 派生交易记录表
    if total_rows > 0:
        print("  派生交易记录...", end=" ", flush=True)
        try:
            _cw = duckdb.connect(DB_PATH)
            _cw.execute("DELETE FROM backtest_trades")
            _cw.execute("""
                INSERT INTO backtest_trades (
                    code, stock_name, market, ktype, window_label, datetime,
                    ma_len, trade_id, trade_action, trade_price_after_slippage,
                    trade_shares, slippage, trade_amount, commission,
                    actual_trade_amount, trade_status, close_pnl, close_type,
                    cash_before_trade, cash_after_trade, available_cash, close_pnl_type, created_at
                )
                SELECT
                    code, stock_name, market, ktype, w, datetime,
                    ma_len, trade_id, trade_action, trade_price_after_slippage,
                    trade_shares, slippage, trade_amount, commission,
                    actual_trade_amount, trade_status, close_pnl,
                    CASE WHEN trade_action = '平多' THEN close_type ELSE NULL END,
                    cash_before_trade, cash_after_trade, available_cash, close_pnl_type, created_at
                FROM backtest_stats,
                     UNNEST(STRING_SPLIT(window_label, ',')) AS t(w)
                WHERE trade_action IS NOT NULL
            """)
            # 为未平仓交易补虚拟平仓行
            _cw.execute("""
                INSERT INTO backtest_trades (
                    code, stock_name, market, ktype, window_label, datetime,
                    ma_len, trade_id, trade_action, trade_price_after_slippage,
                    trade_shares, slippage, trade_amount, commission,
                    actual_trade_amount, trade_status, close_pnl, close_type,
                    cash_before_trade, cash_after_trade, available_cash, close_pnl_type, created_at
                )
                SELECT s.code, s.stock_name, s.market, s.ktype, w, s.datetime,
                       s.ma_len, s.trade_id, '平多',
                       s.close_price_after_slippage,
                       s.close_shares, s.close_slippage, s.close_trade_amount, s.close_commission,
                       s.close_actual_trade_amount, s.trade_status, s.close_pnl, '虚拟平仓',
                       s.cash_before_trade, s.cash_after_trade, s.cash_after_trade, s.close_pnl_type, s.created_at
                FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY code, ktype, ma_len, trade_id ORDER BY datetime DESC
                    ) AS rn
                    FROM backtest_stats s
                    WHERE trade_status = '持仓中'
                ) s,
                     UNNEST(STRING_SPLIT(s.window_label, ',')) AS t(w)
                WHERE s.rn = 1
                  AND NOT EXISTS (
                    SELECT 1 FROM backtest_stats s2
                    WHERE s2.code = s.code AND s2.ktype = s.ktype
                      AND s2.ma_len = s.ma_len AND s2.trade_id = s.trade_id
                      AND s2.trade_action = '平多'
                      AND s2.window_label LIKE '%' || w || '%'
                  )
            """)
            # 全局排序
            _cw.execute("""
                CREATE TABLE backtest_trades_sorted AS
                SELECT * FROM backtest_trades
                ORDER BY code, ktype, window_label, ma_len, datetime
            """)
            _cw.execute("DROP TABLE backtest_trades")
            _cw.execute("ALTER TABLE backtest_trades_sorted RENAME TO backtest_trades")
            _cnt = _cw.execute("SELECT count(*) FROM backtest_trades").fetchone()[0]
            print(f"  交易记录: {_cnt:,} 行")

            # 派生策略表现表
            _cw.execute("DELETE FROM backtest_performance")
            _cw.execute("""
                INSERT INTO backtest_performance
                WITH closes AS (
                    SELECT code, ktype, ma_len, window_label,
                           MIN(datetime) AS first_dt, MAX(datetime) AS last_dt,
                           MIN(close) AS first_close, MAX(close) AS last_close
                    FROM backtest_stats
                    GROUP BY code, ktype, ma_len, window_label
                ),
                equity AS (
                    SELECT code, ktype, ma_len, window_label,
                           MIN(account_value) AS min_ac, MAX(account_value) AS max_ac,
                           (array_agg(account_value ORDER BY datetime))[-1] AS final_ac,
                           COUNT(*) AS n_bars,
                           MIN(account_value) FILTER (
                               WHERE datetime = (SELECT MIN(datetime) FROM backtest_stats s2
                                                  WHERE s2.code = s.code AND s2.ktype = s.ktype
                                                    AND s2.ma_len = s.ma_len AND s2.window_label = s.window_label)
                           ) AS first_ac
                    FROM backtest_stats s
                    GROUP BY code, ktype, ma_len, window_label
                ),
                trades_summary AS (
                    SELECT t.code, t.ktype, t.ma_len, t.window_label,
                           COUNT(DISTINCT t.trade_id) FILTER (WHERE t.trade_action = '开多') AS n_trades,
                           COUNT(*) FILTER (WHERE t.trade_action = '平多' AND t.close_pnl IS NOT NULL) AS n_closed,
                           COUNT(*) FILTER (WHERE t.trade_action = '平多' AND t.close_pnl > 0) AS n_wins,
                           COUNT(*) FILTER (WHERE t.trade_action = '平多' AND t.close_pnl < 0) AS n_losses,
                           SUM(t.close_pnl) FILTER (WHERE t.trade_action = '平多' AND t.close_pnl > 0) AS total_win,
                           SUM(t.close_pnl) FILTER (WHERE t.trade_action = '平多' AND t.close_pnl < 0) AS total_loss,
                           AVG(t.close_pnl) FILTER (WHERE t.trade_action = '平多' AND t.close_pnl > 0) AS avg_win,
                           AVG(t.close_pnl) FILTER (WHERE t.trade_action = '平多' AND t.close_pnl < 0) AS avg_loss,
                           MAX(t.close_pnl) FILTER (WHERE t.trade_action = '平多') AS max_win,
                           MIN(t.close_pnl) FILTER (WHERE t.trade_action = '平多') AS max_loss,
                           AVG(t.close_amount) FILTER (WHERE t.trade_action = '平多' AND t.close_pnl > 0) AS avg_win_amt,
                           AVG(t.close_amount) FILTER (WHERE t.trade_action = '平多' AND t.close_pnl < 0) AS avg_loss_amt,
                           SUM(t.cash_before_trade) FILTER (WHERE t.trade_action = '开多') AS total_cash_before,
                           SUM(t.close_pnl) FILTER (WHERE t.trade_action = '平多') AS total_pnl
                    FROM backtest_trades t
                    GROUP BY t.code, t.ktype, t.ma_len, t.window_label
                ),
                streaks AS (
                    SELECT code, ktype, ma_len, window_label,
                           MAX(CASE WHEN close_pnl > 0 THEN streak_len ELSE 0 END) AS max_win_streak,
                           MAX(CASE WHEN close_pnl < 0 THEN streak_len ELSE 0 END) AS max_loss_streak
                    FROM (
                        SELECT *, COUNT(*) OVER (PARTITION BY code, ktype, ma_len, window_label, grp) AS streak_len
                        FROM (
                            SELECT *,
                                   ROW_NUMBER() OVER (PARTITION BY code, ktype, ma_len, window_label ORDER BY trade_id)
                                   - ROW_NUMBER() OVER (PARTITION BY code, ktype, ma_len, window_label,
                                                         CASE WHEN close_pnl > 0 THEN 1 ELSE 0 END ORDER BY trade_id) AS grp
                            FROM backtest_trades
                            WHERE trade_action = '平多' AND close_pnl IS NOT NULL
                        ) s
                    ) s
                    GROUP BY code, ktype, ma_len, window_label
                ),
                hold_times AS (
                    SELECT o.code, o.ktype, o.ma_len, o.window_label,
                           AVG(c.datetime - o.datetime) AS avg_hold_time
                    FROM backtest_trades o
                    JOIN backtest_trades c ON o.code = c.code AND o.ktype = c.ktype
                        AND o.ma_len = c.ma_len AND o.window_label = c.window_label
                        AND o.trade_id = c.trade_id
                    WHERE o.trade_action = '开多' AND c.trade_action = '平多'
                    GROUP BY o.code, o.ktype, o.ma_len, o.window_label
                ),
                dd AS (
                    SELECT code, ktype, ma_len, window_label,
                           MAX((running_max - account_value) / running_max * 100) AS max_dd
                    FROM (
                        SELECT *,
                               MAX(account_value) OVER (PARTITION BY code, ktype, ma_len, window_label ORDER BY datetime) AS running_max
                        FROM backtest_stats
                    ) s
                    GROUP BY code, ktype, ma_len, window_label
                ),
                returns_calc AS (
                    SELECT code, ktype, ma_len, window_label,
                           (final_ac / first_ac - 1) * 100 AS total_ret,
                           EXTRACT(EPOCH FROM (last_dt - first_dt)) / 86400 / 365.0 AS years,
                           last_close / NULLIF(first_close, 0) * 100 AS buy_hold,
                           first_ac AS init_cash,
                           final_ac AS final_cash
                    FROM equity JOIN closes USING (code, ktype, ma_len, window_label)
                )
                SELECT
                    r.code, '', '', '', r.window_label, r.ma_len,
                    r.total_ret,
                    CASE WHEN r.years > 0 THEN (POWER(r.final_cash / NULLIF(r.init_cash, 0), 1.0 / r.years) - 1) * 100 ELSE 0 END AS cagr,
                    r.buy_hold - 100 AS buy_hold_return,
                    r.total_ret - (r.buy_hold - 100) AS excess_return,
                    CASE WHEN ts.n_trades > 0 THEN ts.total_pnl / NULLIF(ts.total_cash_before, 0) * 100 ELSE 0 END AS avg_trade_return,
                    COALESCE(dd.max_dd, 0) AS max_drawdown,
                    0 AS sharpe_ratio,
                    0 AS calmar_ratio,
                    COALESCE(ts.n_trades, 0) AS trade_count,
                    CASE WHEN ts.n_closed > 0 THEN ts.n_wins * 100.0 / ts.n_closed ELSE 0 END AS win_rate,
                    CASE WHEN ts.total_loss < 0 THEN ts.total_win / ABS(ts.total_loss) ELSE 0 END AS profit_factor,
                    CASE WHEN ts.avg_loss > 0 THEN ABS(ts.avg_win) / ABS(ts.avg_loss) ELSE 0 END AS payoff_ratio,
                    COALESCE(ts.avg_win, 0) AS avg_win,
                    COALESCE(ts.avg_win_amt, 0) AS avg_win_pct,
                    COALESCE(ts.avg_loss, 0) AS avg_loss,
                    COALESCE(ts.avg_loss_amt, 0) AS avg_loss_pct,
                    COALESCE(ts.max_win, 0) AS max_win,
                    COALESCE(ts.max_loss, 0) AS max_loss,
                    COALESCE(st.max_win_streak, 0) AS max_win_streak,
                    COALESCE(st.max_loss_streak, 0) AS max_loss_streak,
                    EXTRACT(EPOCH FROM ht.avg_hold_time) / 86400 AS avg_hold_days,
                    EXTRACT(EPOCH FROM ht.avg_hold_time) / 86400 / 7 AS avg_hold_bars,
                    r.init_cash, r.final_cash, CURRENT_TIMESTAMP
                FROM returns_calc r
                LEFT JOIN trades_summary ts ON r.code = ts.code AND r.ktype = ts.ktype AND r.ma_len = ts.ma_len AND r.window_label = ts.window_label
                LEFT JOIN streaks st ON r.code = st.code AND r.ktype = st.ktype AND r.ma_len = st.ma_len AND r.window_label = st.window_label
                LEFT JOIN hold_times ht ON r.code = ht.code AND r.ktype = ht.ktype AND r.ma_len = ht.ma_len AND r.window_label = ht.window_label
                LEFT JOIN dd ON r.code = dd.code AND r.ktype = dd.ktype AND r.ma_len = dd.ma_len AND r.window_label = dd.window_label
            """)
            _cw.close()
            print(f"  策略表现: 已生成")
        except Exception as e:
            print(f"  派生失败: {e}")
            _cw.close()

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
