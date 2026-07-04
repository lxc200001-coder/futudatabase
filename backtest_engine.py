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
import bottleneck as bn
from numba import njit

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "market.duckdb")

INITIAL_CASH = 10000
FEE_RATE = 0.001
SLIPPAGE = 0.000
TRADE_MODE = "close"   # close / open
MA_MODE = "continuous" # continuous / jump
DEFAULT_KTYPE = "1w"   # 默认ktype: 1w(周K) / 1d(日K) / all(两者全部) / 1w,1d(逗号拼接)
DEFAULT_MARKET = "US,CC" # 默认market: all / US / CN / CC / US,CC
WINDOW_START_DATE = "2000-01-03"

# =========================================================
# Numba 加速：账户状态逐K线计算
# =========================================================

@njit
def _numba_account_loop(closes, trade_actions, trade_prices, n, initial_cash, slippage, fee_rate,
                         allow_fractional=False, initial_shares=0.0):
    """@njit 逐K线计算账户状态。

    trade_actions: 0=无, 1=开多, 2=平多
    trade_prices: 成交价，NaN 表示无交易
    allow_fractional: 是否允许碎股（加密货币用）
    initial_shares: 首根K线持有的股数（WF跨窗口接续用）
    返回所有账户数组。
    """
    available_cash = np.full(n, initial_cash, dtype=np.float64)
    held_shares = np.full(n, initial_shares, dtype=np.float64)
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


@njit
def _calc_perf_numba(av_arr, closes, vp_pnl, vp_amt, vp_price_slip, vp_shares,
                      ta_code, trade_id, dt_days):
    """@njit 一次性计算策略表现指标。

    ta_code: 0=无, 1=开多, 2=平多
    trade_id: 交易编号，0=无效
    dt_days: datetime 的天数（用于持仓天数）
    """
    n = len(av_arr)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0)

    first_ac = av_arr[0]; final_ac = av_arr[-1]
    total_ret = (final_ac / first_ac - 1) if first_ac > 0 else 0.0
    first_close = closes[0]; last_close = closes[-1]
    buy_hold = (last_close / first_close - 1) if first_close > 0 else 0.0

    # Sharpe（一次循环计算均值和方差）
    ret_sum = 0.0; ret_sum2 = 0.0; ret_cnt = 0
    for i in range(1, n):
        r = av_arr[i] / av_arr[i-1] - 1
        if not np.isnan(r) and not np.isinf(r):
            ret_sum += r; ret_sum2 += r * r; ret_cnt += 1
    if ret_cnt > 1:
        ret_avg = ret_sum / ret_cnt
        ret_var = (ret_sum2 - ret_sum * ret_sum / ret_cnt) / (ret_cnt - 1)
        ret_std = np.sqrt(ret_var) if ret_var > 0 else 0.0
        sharpe = ret_avg / ret_std * np.sqrt(252) if ret_std > 1e-10 else 0.0
    else:
        sharpe = 0.0

    # 最大回撤
    running_max = av_arr[0]; max_dd = 0.0
    for i in range(1, n):
        if av_arr[i] > running_max: running_max = av_arr[i]
        dd = (running_max - av_arr[i]) / running_max if running_max > 0 else 0.0
        if dd > max_dd: max_dd = dd

    # 交易统计 / 连续盈亏 / 持仓天数（一次循环）
    n_closed = 0; n_wins = 0; n_losses = 0
    total_win = 0.0; total_loss = 0.0; max_win = 0.0; max_loss = 0.0
    total_cash_before = 0.0; max_tid = 0
    max_ws = 0; max_ls = 0; cur_stk = 0; cur_sgn = 0
    max_tid_local = 0
    for j in range(n):
        ta = ta_code[j]; tid = trade_id[j]
        if ta == 1:
            total_cash_before += vp_shares[j] * vp_price_slip[j]
        if ta == 2:
            pnl = vp_pnl[j]
            if not np.isnan(pnl):
                n_closed += 1
                if pnl > 0:
                    n_wins += 1; total_win += pnl
                    if pnl > max_win: max_win = pnl
                    s = 1
                else:
                    n_losses += 1; total_loss += pnl
                    if pnl < max_loss: max_loss = pnl
                    s = -1
                # 连续盈亏
                if s == cur_sgn:
                    cur_stk += 1
                else:
                    if cur_sgn == 1 and cur_stk > max_ws: max_ws = cur_stk
                    elif cur_sgn == -1 and cur_stk > max_ls: max_ls = cur_stk
                    cur_stk = 1; cur_sgn = s
        if tid > max_tid_local: max_tid_local = tid

    if cur_sgn == 1 and cur_stk > max_ws: max_ws = cur_stk
    elif cur_sgn == -1 and cur_stk > max_ls: max_ls = cur_stk

    # 持仓天数（trade_id 索引）
    max_tid_local = max(max_tid_local, 1)
    oi = np.zeros(max_tid_local + 1, dtype=np.int64)
    od = np.zeros(max_tid_local + 1, dtype=np.int64)
    hb = 0; hd = 0; hc = 0
    for j in range(n):
        tid = trade_id[j]
        if tid <= 0: continue
        if ta_code[j] == 1 and oi[tid] == 0:
            oi[tid] = j; od[tid] = dt_days[j]
        elif ta_code[j] == 2 and oi[tid] > 0:
            hb += j - oi[tid]; hd += dt_days[j] - od[tid]; hc += 1
            oi[tid] = 0

    avg_hb = hb / hc if hc > 0 else 0.0
    avg_hd = hd / hc if hc > 0 else 0.0

    return (total_ret, buy_hold, sharpe, max_dd,
            max_tid_local, n_closed, n_wins, n_losses,
            total_win, total_loss, max_win, max_loss,
            total_cash_before, max_ws, max_ls, avg_hb, avg_hd)


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
    opens_arr = df_k["open"].values.astype(np.float64)
    highs_arr = df_k["high"].values.astype(np.float64)
    lows_arr = df_k["low"].values.astype(np.float64)
    ha_close = (opens_arr + highs_arr + lows_arr + closes) / 4.0
    _is_cc = "加密货币" in str(df_k.get("market", pd.Series([""])).iloc[0])

    # 全量计算每个 MA 的信号数组（避免对每个窗口重复计算）
    ma_cache = {}  # ma → {ha_ma_val, direction, signal, trade_actions, trade_prices}
    for ma in ma_range:
        # ha_ma_val (全量，bottleneck 加速)
        ha_ma_val = bn.move_mean(ha_close, window=ma, min_count=ma)

        # direction / signal（全量，向量化）
        _up = np.zeros(n, dtype=np.int8)
        _valid = ~np.isnan(ha_ma_val)
        _up[1:] = np.where(_valid[1:] & _valid[:-1] & (ha_ma_val[1:] > ha_ma_val[:-1]), 1, 0)
        _up[:1] = 0

        direction = np.where(_up == 1, "多头", "空头")
        signal = np.full(n, "", dtype=object)
        signal[0] = "等待"
        _du = _up[1:] == 1; _dd = _up[1:] == 0
        _pu = _up[:-1] == 1; _pd = _up[:-1] == 0
        signal[1:][_du & _pd] = "买入"    # 多头←空头
        signal[1:][_du & _pu] = "持有"    # 多头→多头
        signal[1:][_dd & _pu] = "卖出"    # 空头←多头
        signal[1:][_dd & _pd] = "等待"    # 空头→空头

        # trade_actions / trade_prices（向量化）
        trade_actions = np.zeros(n, dtype=np.int64)
        trade_prices = np.full(n, np.nan, dtype=np.float64)
        _buy = signal == "买入"
        _sell = signal == "卖出"
        if trade_mode == "close":
            trade_actions[_buy] = 1; trade_prices[_buy] = closes[_buy]
            trade_actions[_sell] = 2; trade_prices[_sell] = closes[_sell]
        else:
            _buy1 = np.roll(_buy, 1); _buy1[0] = False
            _sell1 = np.roll(_sell, 1); _sell1[0] = False
            trade_actions[_buy1] = 1; trade_prices[_buy1] = opens_arr[_buy1]
            trade_actions[_sell1] = 2; trade_prices[_sell1] = opens_arr[_sell1]

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

    # 仅末窗：写入 parquet（跳过 DataFrame 和 pd.concat）
    ws, we = windows[-1]
    window_label = f"{ws.date()}~{we.date()}"
    mask = (pd.to_datetime(df_k["datetime"]) >= ws) & \
           (pd.to_datetime(df_k["datetime"]) <= we)
    _stats_path = None
    all_perf = []
    if mask.any():
        _tmp_base = os.path.join(PROJECT_ROOT, "results_uscncc", f"_tmp_{code.replace('.','_')}_{ktype}")
        os.makedirs(os.path.dirname(_tmp_base), exist_ok=True)
        for ma in ma_range:
            (ha_ma_val, direction, signal, trade_actions, trade_prices,
             ac_arr, hs_arr, ts_arr, av_arr) = ma_cache[ma]
            _p = f"{_tmp_base}_{ma}.parquet"
            _tp = f"{_tmp_base}_{ma}_trades.parquet"
            path_or_df, _perf = _build_slice_rows(
                df_k, mask, ma, ktype, ha_ma_val, direction, signal,
                trade_actions, trade_prices, ha_close, closes,
                ac_arr, hs_arr, ts_arr, av_arr,
                slippage, fee_rate, trade_mode=trade_mode, wl_cache=_wl_cache,
                parquet_path=_p, trades_parquet_path=_tp)
            if path_or_df is not None:
                _stats_path = _p

    # 遍历所有窗口收集 perf
    _sn = str(df_k["stock_name"].iloc[0]) if "stock_name" in df_k.columns else ""
    _mkt = str(df_k["market"].iloc[0]) if "market" in df_k.columns else ""
    for ws, we in windows:
        _wl = f"{ws.date()}~{we.date()}"
        _mask = (pd.to_datetime(df_k["datetime"]) >= ws) & \
                (pd.to_datetime(df_k["datetime"]) <= we)
        if not _mask.any():
            continue
        for ma in ma_range:
            (ha_ma_val, direction, signal, trade_actions, trade_prices,
             ac_arr, hs_arr, ts_arr, av_arr) = ma_cache[ma]
            _, _perf = _build_slice_rows(df_k, _mask, ma, ktype, ha_ma_val, direction, signal,
                                          trade_actions, trade_prices, ha_close, closes,
                                          ac_arr, hs_arr, ts_arr, av_arr,
                                          slippage, fee_rate, trade_mode=trade_mode, wl_cache=_wl_cache, perf_only=True)
            if _perf:
                all_perf.append({**{"code": code, "stock_name": _sn, "market": _mkt, "ktype": ktype, "ma_len": ma, "window_label": _wl}, **_perf})

    if _stats_path:
        return _stats_path, pd.DataFrame(all_perf) if all_perf else pd.DataFrame()
    return None, pd.DataFrame()

def _calc_strategy_score(cagr, sharpe_ratio, max_drawdown, profit_factor, win_rate, trade_count):
    """计算策略综合评分（与 backtest_uscncc.py 中 calc_score_row 逻辑一致）。"""
    def _normalize(x, lo, hi):
        if x is None: return 0
        if hi == lo: return 0
        x = max(lo, min(x, hi))
        return (x - lo) / (hi - lo)

    cagr_score = _normalize(cagr, 0, 0.30) * 100
    sharpe_score = _normalize(sharpe_ratio, 0, 2) * 100
    dd_score = (1 - _normalize(max_drawdown, 0, 0.50)) * 100
    pf_score = _normalize(profit_factor, 1, 3) * 100
    win_score = _normalize(win_rate, 0.30, 0.80) * 100
    trade_score = _normalize(min(trade_count, 100), 10, 100) * 100

    return (
        cagr_score * 0.30 +
        sharpe_score * 0.25 +
        dd_score * 0.20 +
        pf_score * 0.15 +
        win_score * 0.05 +
        trade_score * 0.05
    )


def build_score_matrix(summary_rows):
    """从 window_label 回测汇总行构建评分明细和排名透视表（与 backtest_uscncc.py 逻辑一致）。"""
    df = pd.DataFrame(summary_rows)

    # 排除评分全为 0 的 window_label
    valid = df.groupby("window_label")["strategy_score"].transform("max") > 0
    df = df[valid]

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 评分明细透视
    score_pivot = df.pivot_table(
        index=["code", "ktype", "ma_len"],
        columns="window_label",
        values="strategy_score",
        aggfunc="first"
    )
    score_pivot = score_pivot[sorted(score_pivot.columns)]
    score_pivot = score_pivot.reset_index()

    # 评分排名（每股票每 window_label 内独立排名）
    df["window_rank"] = df.groupby(["code", "window_label"])["strategy_score"].rank(ascending=False, method="min")

    rank_pivot = df.pivot_table(
        index=["code", "ktype", "ma_len"],
        columns="window_label",
        values="window_rank",
        aggfunc="first"
    )
    rank_pivot = rank_pivot[sorted(rank_pivot.columns)]
    rank_pivot = rank_pivot.reset_index()

    return score_pivot, rank_pivot


def build_window_stability(summary_rows):
    """对每个累积 window_label 阶段计算参数稳定性（与 backtest_uscncc.py 逻辑一致，支持多股票 + ma_len）。"""
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return pd.DataFrame()

    # 排除评分全为 0 的 window_label
    valid = df.groupby("window_label")["strategy_score"].transform("max") > 0
    df = df[valid]
    if df.empty:
        return pd.DataFrame()

    windows = sorted(df["window_label"].unique())
    if not windows:
        return pd.DataFrame()

    # 每 window_label 内按 (code, window_label) 组排名
    df["window_rank"] = df.groupby(["code", "window_label"])["strategy_score"].rank(ascending=False, method="min")

    def _norm(series, higher_is_better=True):
        lo, hi = series.min(), series.max()
        if hi == lo:
            return pd.Series(0.5, index=series.index)
        return (series - lo) / (hi - lo) if higher_is_better else (hi - series) / (hi - lo)

    all_stages = []
    for i, w in enumerate(windows):
        stage_df = df[df["window_label"].isin(windows[:i + 1])]

        stats = stage_df.groupby(["code", "ma_len"]).agg(
            ktype=("ktype", "first"),
            window_count=("window_label", "nunique"),
            win_window_count=("cagr", lambda x: (x > 0).sum()),
            avg_score_rank=("window_rank", "mean"),
            rank_first_count=("window_rank", lambda x: (x == 1).sum()),
            rank_top3_pct=("window_rank", lambda x: (x <= 3).sum() / max(len(x), 1)),
            score_rank_std=("window_rank", "std"),
            avg_cagr=("cagr", "mean"),
            cagr_std=("cagr", "std"),
        ).reset_index()

        stats["win_window_pct"] = stats["win_window_count"] / stats["window_count"]
        stats["avg_cagr"] = stats["avg_cagr"]
        stats["cagr_std"] = stats["cagr_std"].fillna(0)
        stats["score_rank_std"] = stats["score_rank_std"].fillna(0)

        # 归一化加权评分
        avg_rank_n = _norm(stats["avg_score_rank"], higher_is_better=False)
        top3_n = _norm(stats["rank_top3_pct"], higher_is_better=True)
        std_n = _norm(stats["score_rank_std"], higher_is_better=False)
        cagr_n = _norm(stats["avg_cagr"], higher_is_better=True)
        win_rate_n = _norm(stats["win_window_pct"], higher_is_better=True)
        cagr_std_n = _norm(stats["cagr_std"], higher_is_better=False)

        stats["stability_score"] = (
            0.20 * avg_rank_n + 0.10 * top3_n + 0.15 * std_n
            + 0.25 * cagr_n + 0.20 * win_rate_n + 0.10 * cagr_std_n
        )

        stats.insert(0, "window_label", w)
        all_stages.append(stats)

    result = pd.concat(all_stages, ignore_index=True)
    result = result.sort_values(["code", "window_label", "ma_len"]).reset_index(drop=True)

    # 标记每股票每 window_label 的最优参数
    result["is_best"] = ""
    idx = (
        result
        .sort_values(["stability_score", "score_rank_std", "avg_cagr"],
                      ascending=[False, True, False])
        .groupby(["code", "window_label"], sort=False)
        .head(1)
        .index
    )
    result.loc[idx, "is_best"] = "最优"
    return result


def _calc_perf(av_arr, closes, first_dt, last_dt, ktype, df, idx, n_sl,
               _vp_pnl, _vp_shares, _vp_amt, _vp_price_slip,
               trade_id_arr, ta_lbl, ta_code=None):
    """从内存 arrays 计算全部策略表现指标（使用 numba 加速）。"""
    first_ac, final_ac = float(av_arr[0]), float(av_arr[-1])
    years = max((last_dt - first_dt).days / 365.0, 1 / 365.0)

    # 准备 ta_code（优先使用传入的 int 数组）
    if ta_code is None:
        ta_code = np.where(ta_lbl == "开多", 1, np.where(ta_lbl == "平多", 2, 0)).astype(np.int64)

    # 准备 trade_id（object → int64，None → 0）
    _tid_arr = np.array([int(t) if t is not None else 0 for t in trade_id_arr], dtype=np.int64)

    # 准备 dt_days（datetime → int64 days since epoch）
    _dt_vals = df["datetime"].values[idx]
    _dt_days = np.array([int(t.astype("datetime64[D]").astype(np.int64)) for t in _dt_vals], dtype=np.int64)

    # 准备 vp_* 数组（object → float64，None → NaN）
    _vp_p = np.array([float(v) if v is not None else np.nan for v in _vp_pnl], dtype=np.float64)

    # 调用 numba 一次循环完成全部统计
    (total_ret, buy_hold, sharpe, max_dd,
     trade_count, n_closed, n_wins, n_losses,
     total_win, total_loss, max_win, max_loss,
     total_cash_before, max_win_streak, max_loss_streak,
     avg_hold_bars, avg_hold_days) = _calc_perf_numba(
        av_arr.astype(np.float64), closes.astype(np.float64),
        _vp_p, np.full(n_sl, np.nan, dtype=np.float64),
        np.array([float(v) if v is not None else np.nan for v in _vp_price_slip], dtype=np.float64),
        np.array([float(v) if v is not None else 0.0 for v in _vp_shares], dtype=np.float64),
        ta_code, _tid_arr, _dt_days)

    cagr = ((final_ac / first_ac) ** (1.0 / years) - 1) if first_ac > 0 else 0.0
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    # 派生统计（numba 返回原始值，在此派生）
    n_closed = int(n_closed); n_wins = int(n_wins); n_losses = int(n_losses)
    trade_count = int(trade_count)
    win_rate = (n_wins / n_closed) if n_closed > 0 else 0
    profit_factor = total_win / abs(total_loss) if total_loss < 0 else 0
    avg_win = (total_win / n_wins) if n_wins else 0
    avg_loss = (total_loss / n_losses) if n_losses else 0.0
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    avg_trade_return = ((total_win + total_loss) / total_cash_before) if total_cash_before > 0 and n_closed > 0 else 0
    # avg_win_amt / avg_loss_amt（从 _vp_amt 计算）
    _close_amts = [float(_vp_amt[j]) for j in range(n_sl) if ta_lbl[j] == "平多" and _vp_pnl[j] is not None]
    _close_pnls2 = [float(_vp_pnl[j]) for j in range(n_sl) if ta_lbl[j] == "平多" and _vp_pnl[j] is not None]
    _n_c2 = len(_close_pnls2)
    avg_win_amt = (sum(_close_amts[i] for i in range(_n_c2) if _close_pnls2[i] > 0) / n_wins) if n_wins else 0
    avg_loss_amt = (sum(_close_amts[i] for i in range(_n_c2) if _close_pnls2[i] < 0) / n_losses) if n_losses else 0

    strategy_score = round(_calc_strategy_score(
        cagr, sharpe, max_dd, profit_factor, win_rate, trade_count
    ), 2)

    return {
        "total_return": round(total_ret, 4), "cagr": round(cagr, 4),
        "buy_hold_return": round(buy_hold, 4), "excess_return": round(total_ret - buy_hold, 4),
        "avg_trade_return": round(avg_trade_return, 4),
        "max_drawdown": round(max_dd, 4), "sharpe_ratio": round(sharpe, 4),
        "calmar_ratio": round(calmar, 4),
        "trade_count": trade_count, "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4), "payoff_ratio": round(payoff_ratio, 4),
        "strategy_score": strategy_score,
        "avg_win": round(avg_win, 4), "avg_win_pct": round(avg_win_amt, 4),
        "avg_loss": round(avg_loss, 4), "avg_loss_pct": round(avg_loss_amt, 4),
        "max_win": round(max_win, 4), "max_loss": round(max_loss, 4),
        "max_win_streak": max_win_streak, "max_loss_streak": max_loss_streak,
        "avg_hold_days": round(avg_hold_days, 2), "avg_hold_bars": round(avg_hold_bars, 2),
        "initial_cash": float(first_ac), "final_cash": float(final_ac),
    }


def _build_slice_rows(df, mask, ma_len, ktype, ha_ma_val, direction, signal,
                      trade_actions, trade_prices, ha_close, closes,
                      ac_arr, hs_arr, ts_arr, av_arr, slippage, fee_rate,
                      trade_mode="close", wl_cache=None, perf_only=False, signal_exec=None,
                      parquet_path=None, trades_parquet_path=None):
    """对切片后的预计算结果构建行（不跑 numba）。
    signal_exec: WF 传入的信号执行状态；为 None 时自动设为 '执行'。"""
    idx = np.where(mask.values)[0]
    if len(idx) == 0:
        return None, None

    # 切片（用 idx 索引，保持日期对齐）
    c_sl = closes[idx]
    ha_close_sl = ha_close[idx]
    ha_ma_sl = ha_ma_val[idx]
    dir_sl = direction[idx]
    sig_sl = signal[idx]

    # 变动指标（基于切片后的 av_arr）
    n_sl = len(idx)

    # signal_exec: 传入时为预计算数组；为 None 时自动设为 '执行'
    if signal_exec is None:
        sig_exec_sl = np.full(n_sl, None, dtype=object)
        sig_exec_sl[(sig_sl == "买入") | (sig_sl == "卖出")] = "执行"
    elif isinstance(signal_exec, np.ndarray):
        sig_exec_sl = signal_exec[idx]
    else:
        sig_exec_sl = np.array(signal_exec, dtype=object)
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

    # 中文字段映射（向量化）
    _ta_sl = trade_actions[idx]
    _tp_sl = trade_prices[idx]
    ta_lbl = np.full(n_sl, None, dtype=object)
    tp_val = np.full(n_sl, None, dtype=object)
    tp_slip_val = np.full(n_sl, None, dtype=object)
    _m1 = _ta_sl == 1; _m2 = _ta_sl == 2
    ta_lbl[_m1] = "开多"; ta_lbl[_m2] = "平多"
    _n1 = _m1 & ~np.isnan(_tp_sl); _n2 = _m2 & ~np.isnan(_tp_sl)
    tp_val[_n1] = _tp_sl[_n1]; tp_slip_val[_n1] = _tp_sl[_n1] * (1 + slippage)
    tp_val[_n2] = _tp_sl[_n2]; tp_slip_val[_n2] = _tp_sl[_n2] * (1 - slippage)

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

    # 列式构造（替代逐行 dict，快 10-50 倍）
    _has_sn = "stock_name" in df.columns
    _has_mkt = "market" in df.columns
    _has_tv = "turnover" in df.columns
    _has_tv_amt = "turnover_amount" in df.columns
    _has_src = "source" in df.columns
    _ts_nonzero = ts_sl > 0
    _ts_g0 = _ts_nonzero
    # 预提取 df 列到 ndarray（免除 .iloc 和重复 str() 调用）
    _idx_s = idx  # 局部引用加速
    _c_arr = df["code"].values[_idx_s].astype(str)
    _sn_arr = df["stock_name"].values[_idx_s].astype(str) if _has_sn else np.full(n_sl, "", dtype=object)
    _mkt_arr = df["market"].values[_idx_s].astype(str) if _has_mkt else np.full(n_sl, "", dtype=object)
    _dt_arr = df["datetime"].values[_idx_s]
    _o_arr = df["open"].values[_idx_s].astype(np.float64)
    _h_arr = df["high"].values[_idx_s].astype(np.float64)
    _l_arr = df["low"].values[_idx_s].astype(np.float64)
    _v_arr = df["volume"].values[_idx_s].astype(np.float64)
    _tv_arr = df["turnover"].values[_idx_s].astype(np.float64) if _has_tv else None
    _tva_arr = df["turnover_amount"].values[_idx_s].astype(np.float64) if _has_tv_amt else None
    _src_arr = df["source"].values[_idx_s].astype(str) if _has_src else np.full(n_sl, "", dtype=object)
    # wl_cache 预提取
    _wl0 = np.array([wl_cache[i][0] if wl_cache is not None else "" for i in _idx_s], dtype=object)
    _wl1 = np.array([wl_cache[i][1] if wl_cache is not None else 0 for i in _idx_s], dtype=np.int64)

    # 布尔数组（ndarray 比较）
    _closed = trade_status_arr == "已平仓"
    _holding = trade_status_arr == "持仓中"
    _vp_valid = np.array([_vp_pnl[j] is not None for j in range(n_sl)], dtype=bool)
    _vp_gt0 = np.array([_vp_pnl[j] is not None and _vp_pnl[j] > 0 for j in range(n_sl)], dtype=bool)

    # 策略表现
    _first_dt = df["datetime"].iloc[idx[0]]
    _last_dt = df["datetime"].iloc[idx[-1]]
    _perf = _calc_perf(av_sl, c_sl, _first_dt, _last_dt, ktype, df, idx, n_sl,
                       _vp_pnl, _vp_shares, _vp_amt, _vp_price_slip,
                       trade_id_arr, ta_lbl)

    if perf_only:
        return None, _perf

    # 预处理带条件的列（避免 list comprehension）
    _ha_ma_out = np.full(n_sl, None, dtype=object)
    _m_ok = ~np.isnan(ha_ma_sl)
    _ha_ma_out[_m_ok] = ha_ma_sl[_m_ok]

    _tid_out = np.full(n_sl, None, dtype=object)
    _tid_ok = trade_id_arr != np.array(None)
    _tid_out[_tid_ok] = [int(x) for x in trade_id_arr[_tid_ok]]

    _ta_out = np.full(n_sl, None, dtype=object)
    _ta_ok = ta_lbl != np.array(None)
    _ta_out[_ta_ok] = [str(x) for x in ta_lbl[_ta_ok]]

    _tp_out = np.full(n_sl, np.nan, dtype=np.float64)
    _tp_ok = tp_val != np.array(None)
    _tp_out[_tp_ok] = [float(x) for x in tp_val[_tp_ok]]

    _tps_out = np.full(n_sl, np.nan, dtype=np.float64)
    _tps_ok = tp_slip_val != np.array(None)
    _tps_out[_tps_ok] = [float(x) for x in tp_slip_val[_tps_ok]]

    _ts_out = np.full(n_sl, None, dtype=object)
    _ts_ok = trade_status_arr != np.array(None)
    _ts_out[_ts_ok] = [str(x) for x in trade_status_arr[_ts_ok]]

    _vp_p_out = np.full(n_sl, np.nan, dtype=np.float64)
    _vp_p_ok = _vp_price != np.array(None)
    _vp_p_out[_vp_p_ok] = [float(x) for x in _vp_price[_vp_p_ok]]

    _vp_ps_out = np.full(n_sl, np.nan, dtype=np.float64)
    _vp_ps_ok = _vp_price_slip != np.array(None)
    _vp_ps_out[_vp_ps_ok] = [float(x) for x in _vp_price_slip[_vp_ps_ok]]

    _vp_sh_out = np.full(n_sl, np.nan, dtype=np.float64)
    _vp_sh_ok = _vp_shares != np.array(None)
    _vp_sh_out[_vp_sh_ok] = [float(x) for x in _vp_shares[_vp_sh_ok]]

    _vp_sc_out = np.full(n_sl, np.nan, dtype=np.float64)
    _vp_sc_ok = _vp_slip_cost != np.array(None)
    _vp_sc_out[_vp_sc_ok] = [float(x) for x in _vp_slip_cost[_vp_sc_ok]]

    _vp_a_out = np.full(n_sl, np.nan, dtype=np.float64)
    _vp_a_ok = _vp_amt != np.array(None)
    _vp_a_out[_vp_a_ok] = [float(x) for x in _vp_amt[_vp_a_ok]]

    _vp_c_out = np.full(n_sl, np.nan, dtype=np.float64)
    _vp_c_ok = _vp_comm != np.array(None)
    _vp_c_out[_vp_c_ok] = [float(x) for x in _vp_comm[_vp_c_ok]]

    _vp_pnl_out = np.full(n_sl, np.nan, dtype=np.float64)
    _vp_pnl_ok = _vp_pnl != np.array(None)
    _vp_pnl_out[_vp_pnl_ok] = [float(x) for x in _vp_pnl[_vp_pnl_ok]]

    _cb_out = np.full(n_sl, np.nan, dtype=np.float64)
    _cb_ok = _cash_bt != np.array(None)
    _cb_out[_cb_ok] = [float(x) for x in _cash_bt[_cb_ok]]

    # slippage 列（特殊复合逻辑）
    _slip_v = np.zeros(n_sl, dtype=np.float64)
    _slip_ok = (tp_val != np.array(None)) & _ts_nonzero
    _slip_v[_slip_ok] = np.abs((_tps_out[_slip_ok] - _tp_out[_slip_ok]) * ts_sl[_slip_ok])

    # trade_amount / commission / actual_trade_amount
    _trad_amt = np.full(n_sl, np.nan, dtype=np.float64)
    _comm_amt = np.full(n_sl, np.nan, dtype=np.float64)
    _act_amt = np.full(n_sl, np.nan, dtype=np.float64)
    _ta_mask = _ts_g0 & (_vp_ps_out != np.array(None))
    _trad_amt[_ta_mask] = _vp_ps_out[_ta_mask] * ts_sl[_ta_mask]
    _comm_amt[_ta_mask] = _trad_amt[_ta_mask] * fee_rate
    _act_amt[_ta_mask] = _trad_amt[_ta_mask] * (1 + fee_rate)

    # close_type / close_pnl_type（字符串条件列）
    _ct = np.full(n_sl, None, dtype=object)
    _ct[_closed] = "真实平仓"
    _ct[~_closed & _holding] = "虚拟平仓"

    _cpt = np.full(n_sl, None, dtype=object)
    _cpt[_vp_gt0] = "盈利"
    _cpt[~_vp_gt0 & _vp_valid] = "亏损"

    if parquet_path:
        import pyarrow as pa
        import pyarrow.parquet as pq
        _now = pd.Timestamp.now()
        _ca2 = np.where(_cb_ok, _cb_out + np.where(_vp_pnl_ok, _vp_pnl_out, 0), np.nan)
        pq.write_table(pa.table({
            "code": _c_arr, "stock_name": _sn_arr, "market": _mkt_arr,
            "ktype": pa.array([ktype] * n_sl, type=pa.string()),
            "window_label": _wl0.astype(str), "window_count": _wl1,
            "datetime": _dt_arr,
            "open": _o_arr, "high": _h_arr, "low": _l_arr, "close": c_sl,
            "volume": _v_arr,
            "turnover": _tv_arr if _tv_arr is not None else np.zeros(n_sl),
            "turnover_amount": _tva_arr if _tva_arr is not None else np.zeros(n_sl),
            "source": _src_arr,
            "ha_close": ha_close_sl,
            "ma_len": pa.array([ma_len] * n_sl, type=pa.int64()),
            "ha_ma_value": _ha_ma_out,
            "trend_direction": dir_sl, "signal": sig_sl, "signal_exec": sig_exec_sl,
            "trade_id": _tid_out, "trade_action": _ta_out,
            "trade_price": _tp_out, "trade_price_after_slippage": _tps_out,
            "trade_shares": ts_sl,
            "slippage": _slip_v,
            "trade_amount": _trad_amt, "commission": _comm_amt, "actual_trade_amount": _act_amt,
            "available_cash": ac_sl, "held_shares": hs_sl, "trade_status": _ts_out,
            "close_price": _vp_p_out, "close_price_after_slippage": _vp_ps_out,
            "close_shares": _vp_sh_out, "close_slippage": _vp_sc_out,
            "close_trade_amount": _vp_a_out, "close_commission": _vp_c_out,
            "close_actual_trade_amount": _vp_a_out + _vp_c_out,
            "close_pnl": _vp_pnl_out,
            "close_type": _ct, "close_pnl_type": _cpt,
            "cash_before_trade": _cb_out,
            "cash_after_trade": _ca2,
            "account_value": av_sl,
            "account_value_change": acc_chg,
            "account_value_change_pct": acc_chg_pct,
            "change_from_initial": chg_init,
            "change_from_initial_pct": chg_init_pct,
            "created_at": pa.array([_now] * n_sl, type=pa.timestamp("us")),
        }), parquet_path)
        # 写交易记录 parquet（trade_action 非空行）
        if trades_parquet_path:
            _tm = _ta_out != np.array(None)
            _tc = np.array([str(x) for x in _ta_out[_tm]], dtype=object)
            _ti = np.array([int(x) for x in _tid_out[_tm]], dtype=np.int64)
            _ts2 = np.array([str(x) for x in _ts_out[_tm]], dtype=object)
            _tc2 = np.array([str(x) for x in _ct[_tm]], dtype=object)
            _tcp2 = np.array([str(x) for x in _cpt[_tm]], dtype=object)
            _tcaf = np.where(_cb_ok[_tm], _cb_out[_tm] + np.where(_vp_pnl_ok[_tm], _vp_pnl_out[_tm], 0), np.nan)
            pq.write_table(pa.table({
                "code": _c_arr[_tm], "stock_name": _sn_arr[_tm], "market": _mkt_arr[_tm],
                "ktype": pa.array([ktype] * _tm.sum(), type=pa.string()),
                "window_label": _wl0[_tm].astype(str), "datetime": _dt_arr[_tm],
                "ma_len": pa.array([ma_len] * _tm.sum(), type=pa.int64()),
                "trade_id": _ti, "trade_action": _tc,
                "trade_price_after_slippage": _tps_out[_tm],
                "trade_shares": ts_sl[_tm], "slippage": _slip_v[_tm],
                "trade_amount": _trad_amt[_tm], "commission": _comm_amt[_tm],
                "actual_trade_amount": _act_amt[_tm],
                "trade_status": _ts2, "close_pnl": _vp_pnl_out[_tm],
                "close_type": _tc2, "close_pnl_type": _tcp2,
                "cash_before_trade": _cb_out[_tm], "cash_after_trade": _tcaf,
                "available_cash": _cb_out[_tm],
                "created_at": pa.array([_now] * _tm.sum(), type=pa.timestamp("us")),
            }), trades_parquet_path)
        return parquet_path, _perf
    return pd.DataFrame({
        "code": _c_arr, "stock_name": _sn_arr, "market": _mkt_arr,
        "ktype": ktype,
        "window_label": _wl0, "window_count": _wl1,
        "datetime": _dt_arr,
        "open": _o_arr, "high": _h_arr, "low": _l_arr, "close": c_sl,
        "volume": _v_arr,
        "turnover": _tv_arr if _tv_arr is not None else np.zeros(n_sl),
        "turnover_amount": _tva_arr if _tva_arr is not None else np.zeros(n_sl),
        "source": _src_arr,
        "ha_close": ha_close_sl,
        "ma_len": ma_len, "ha_ma_value": _ha_ma_out,
        "trend_direction": dir_sl, "signal": sig_sl, "signal_exec": sig_exec_sl,
        "trade_id": _tid_out, "trade_action": _ta_out,
        "trade_price": _tp_out, "trade_price_after_slippage": _tps_out,
        "trade_shares": ts_sl,
        "slippage": _slip_v,
        "trade_amount": _trad_amt, "commission": _comm_amt, "actual_trade_amount": _act_amt,
        "available_cash": ac_sl, "held_shares": hs_sl, "trade_status": _ts_out,
        "close_price": _vp_p_out, "close_price_after_slippage": _vp_ps_out,
        "close_shares": _vp_sh_out, "close_slippage": _vp_sc_out,
        "close_trade_amount": _vp_a_out, "close_commission": _vp_c_out,
        "close_actual_trade_amount": _vp_a_out + _vp_c_out,
        "close_pnl": _vp_pnl_out,
        "close_type": _ct, "close_pnl_type": _cpt,
        "cash_before_trade": _cb_out,
        "cash_after_trade": np.where(_cb_ok, _cb_out + np.where(_vp_pnl_ok, _vp_pnl_out, 0), np.nan),
        "account_value": av_sl,
        "account_value_change": acc_chg,
        "account_value_change_pct": acc_chg_pct,
        "change_from_initial": chg_init,
        "change_from_initial_pct": chg_init_pct,
        "created_at": pd.Timestamp.now(),
    }).astype({
        "code": "category", "stock_name": "category", "market": "category",
        "trend_direction": "category", "signal": "category", "signal_exec": "category",
        "trade_status": "category", "close_type": "category", "close_pnl_type": "category",
    }), _perf


def _worker_stock(code, ktype, windows, ma_range, trade_mode, slippage, fee_rate):
    """工作进程：计算一只股票的所有MA+窗口。返回 (code, stats_path, perf_path, err)"""
    try:
        _stats_path, df_perf = run_stock(code, ktype, ma_range, windows, trade_mode, slippage, fee_rate)
        if _stats_path is None:
            return code, None, None, None, None
        _tmp = os.path.join(PROJECT_ROOT, "results_uscncc", f"_tmp_{code.replace('.','_')}_{ktype}")
        os.makedirs(os.path.dirname(_tmp), exist_ok=True)
        _p_stats = _tmp + "_stats.parquet"
        _p_trades = _tmp + "_trades.parquet"
        _p_perf = _tmp + "_perf.parquet"
        import pyarrow as pa; import pyarrow.parquet as pq
        for _p_out, _sfx in [(_p_stats, ""), (_p_trades, "_trades")]:
            _tbls, _fs = [], []
            for ma in ma_range:
                _f = f"{_tmp}_{ma}{_sfx}.parquet"
                if os.path.exists(_f):
                    _fs.append(_f); _tbls.append(pq.read_table(_f, memory_map=False))
            for _f in _fs:
                try: os.remove(_f)
                except: pass
            if _tbls:
                pq.write_table(pa.concat_tables(_tbls), _p_out)
        if not df_perf.empty:
            df_perf.to_parquet(_p_perf, index=False)
        return code, _p_stats, _p_trades, _p_perf, None
    except Exception as e:
        return code, None, None, None, None, str(e)


def run_stock_walkforward(code, ktype, windows, ma_range, trade_mode, slippage, fee_rate, wf_plan):
    """Walk Forward 回测：用上个窗口的 is_best MA 作为当前窗口的交易参数。"""
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
        return pd.DataFrame(), pd.DataFrame()
    for c in ["code", "stock_name", "market", "ktype", "datetime",
              "open", "high", "low", "close", "volume", "turnover",
              "turnover_amount", "source"]:
        if c not in df_k.columns:
            df_k[c] = "" if c in ("code", "stock_name", "market", "ktype", "source") else 0.0

    n = len(df_k)
    closes = df_k["close"].values.astype(np.float64)
    opens_arr = df_k["open"].values.astype(np.float64)
    highs_arr = df_k["high"].values.astype(np.float64)
    lows_arr = df_k["low"].values.astype(np.float64)
    ha_close = (opens_arr + highs_arr + lows_arr + closes) / 4.0
    _is_cc = "加密货币" in str(df_k.get("market", pd.Series([""])).iloc[0])

    # 构建 WF 映射：{window_label → best_ma}
    stock_wf = {row["window_label"]: row["best_ma"] for _, row in wf_plan.iterrows()
                if row["code"] == code}
    wf_mas = set(stock_wf.values())

    # 找到第一个有有效 is_best 的 WF 段的起始日期
    _wf_start = None
    for i in range(1, len(windows)):
        prev_wl = f"{windows[i-1][0].date()}~{windows[i-1][1].date()}"
        if stock_wf.get(prev_wl) is not None:
            _wf_start = windows[i-1][1] + pd.Timedelta(days=1)
            break
    if _wf_start is None:
        return pd.DataFrame(), pd.DataFrame()
    wf_mask = (pd.to_datetime(df_k["datetime"]) >= _wf_start) & \
               (pd.to_datetime(df_k["datetime"]) <= windows[-1][1])
    if not wf_mask.any():
        return pd.DataFrame(), pd.DataFrame()
    wf_idx = np.where(wf_mask.values)[0]
    n_wf = len(wf_idx)

    _sn = str(df_k["stock_name"].iloc[0]) if "stock_name" in df_k.columns else ""
    _mkt = str(df_k["market"].iloc[0]) if "market" in df_k.columns else ""

    # 构建 ma_cache（仅计算 WF 需要用到的 MA）
    _sn = str(df_k["stock_name"].iloc[0]) if "stock_name" in df_k.columns else ""
    _mkt = str(df_k["market"].iloc[0]) if "market" in df_k.columns else ""
    ma_cache = {}
    for ma in wf_mas:
        ha_ma_val = bn.move_mean(ha_close, window=ma, min_count=ma)
        _up = np.zeros(n, dtype=np.int8)
        _valid = ~np.isnan(ha_ma_val)
        _up[1:] = np.where(_valid[1:] & _valid[:-1] & (ha_ma_val[1:] > ha_ma_val[:-1]), 1, 0)
        direction = np.where(_up == 1, "多头", "空头")
        signal = np.full(n, "", dtype=object)
        signal[0] = "等待"
        _du = _up[1:] == 1; _dd = _up[1:] == 0
        _pu = _up[:-1] == 1; _pd = _up[:-1] == 0
        signal[1:][_du & _pd] = "买入"
        signal[1:][_du & _pu] = "持有"
        signal[1:][_dd & _pu] = "卖出"
        signal[1:][_dd & _pd] = "等待"
        trade_actions = np.zeros(n, dtype=np.int64)
        trade_prices = np.full(n, np.nan, dtype=np.float64)
        _buy = signal == "买入"
        _sell = signal == "卖出"
        if trade_mode == "close":
            trade_actions[_buy] = 1; trade_prices[_buy] = closes[_buy]
            trade_actions[_sell] = 2; trade_prices[_sell] = closes[_sell]
        else:
            _buy1 = np.roll(_buy, 1); _buy1[0] = False
            _sell1 = np.roll(_sell, 1); _sell1[0] = False
            trade_actions[_buy1] = 1; trade_prices[_buy1] = opens_arr[_buy1]
            trade_actions[_sell1] = 2; trade_prices[_sell1] = opens_arr[_sell1]
        ma_cache[ma] = (ha_ma_val, direction, signal, trade_actions, trade_prices)

    # 初始化全量数组
    full_ma = np.zeros(n, dtype=np.int64)
    full_ta = np.zeros(n, dtype=np.int64)
    full_tp = np.full(n, np.nan, dtype=np.float64)
    full_ha_ma = np.full(n, np.nan, dtype=np.float64)
    full_dir = np.full(n, "", dtype=object)
    full_sig = np.full(n, "", dtype=object)
    full_sig_exec = np.full(n, None, dtype=object)
    full_ha_close = np.full(n, np.nan, dtype=np.float64)

    # ha_close 直接用已计算的 numpy 数组切片（不查 DB）
    full_ha_close[wf_idx] = ha_close[wf_idx]

    # 逐段 numpy 切片填充 WF 数组
    _wf_labels_seen = []
    for i in range(1, len(windows)):
        seg_start = windows[i-1][1]
        seg_end = windows[i][1]
        seg_mask = (pd.to_datetime(df_k["datetime"]) > seg_start) &                     (pd.to_datetime(df_k["datetime"]) <= seg_end)
        seg_idx = np.where(seg_mask.values)[0]
        if not len(seg_idx):
            continue

        prev_wl = f"{windows[i-1][0].date()}~{windows[i-1][1].date()}"
        wf_ma = stock_wf.get(prev_wl)
        if wf_ma is None:
            continue

        _wl = f"{(seg_start + pd.Timedelta(days=1)).date()}~{seg_end.date()}"
        _wf_labels_seen.append(_wl)

        # numpy 切片批量赋值（比 per-bar dict 查找快）
        ha_ma_val, direction, signal, trade_actions, trade_prices = ma_cache[wf_ma]
        full_ma[seg_idx] = wf_ma
        full_ha_ma[seg_idx] = ha_ma_val[seg_idx]
        full_dir[seg_idx] = direction[seg_idx]
        full_sig[seg_idx] = signal[seg_idx]
        full_ta[seg_idx] = trade_actions[seg_idx]
        full_tp[seg_idx] = trade_prices[seg_idx]

    # 一次连续信号冲突检测（所有 WF 段一次过）
    has_position = False
    for ii in wf_idx:
        sig = full_sig[ii]
        if sig == "买入":
            if has_position:
                full_sig_exec[ii] = "忽略"
                full_ta[ii] = 0
            else:
                full_sig_exec[ii] = "执行"
                has_position = True
        elif sig == "卖出":
            if has_position:
                full_sig_exec[ii] = "执行"
                has_position = False
            else:
                full_sig_exec[ii] = "忽略"
                full_ta[ii] = 0

    if not _wf_labels_seen:
        return pd.DataFrame(), pd.DataFrame()

    # 一次 numba 从 INITIAL_CASH 开始
    (ac_arr, hs_arr, ts_arr, _, _, av_arr) = _numba_account_loop(
        closes, full_ta, full_tp, n, INITIAL_CASH, slippage, fee_rate,
        allow_fractional=_is_cc
    )

    # 用 _build_slice_rows 构建 DataFrame
    _wf_label = f"{_wf_labels_seen[0].split('~')[0]}~{_wf_labels_seen[-1].split('~')[-1]}"
    _wf_count = len(_wf_labels_seen)
    _wl_cache = {_i: (_wf_label, _wf_count) for _i in range(n)}

    wf_df, _ = _build_slice_rows(
        df_k, wf_mask, 0, ktype,
        full_ha_ma, full_dir, full_sig,
        full_ta, full_tp, full_ha_close, closes,
        ac_arr, hs_arr, ts_arr, av_arr,
        slippage, fee_rate, trade_mode=trade_mode,
        wl_cache=_wl_cache
    )

    if wf_df is not None and not wf_df.empty:
        wf_df["ma_len"] = [int(full_ma[i]) for i in wf_idx]
        wf_df["signal_exec"] = [full_sig_exec[i] for i in wf_idx]

        # 构建总 performance
        av_sl = av_arr[wf_idx]
        c_sl = closes[wf_idx]
        ta_lbl_wf = np.array([None if pd.isna(v) else str(v) for v in wf_df["trade_action"].values], dtype=object)
        tid_wf = np.array([int(v) if not pd.isna(v) else None for v in wf_df["trade_id"].values], dtype=object)

        _first_dt = df_k["datetime"].iloc[wf_idx[0]]
        _last_dt = df_k["datetime"].iloc[wf_idx[-1]]
        total_perf = _calc_perf(
            av_sl, c_sl, _first_dt, _last_dt, ktype, df_k, wf_idx, n_wf,
            wf_df["close_pnl"].values if "close_pnl" in wf_df.columns else np.full(n_wf, None),
            wf_df["close_shares"].values if "close_shares" in wf_df.columns else np.full(n_wf, None),
            wf_df["close_trade_amount"].values if "close_trade_amount" in wf_df.columns else np.full(n_wf, None),
            wf_df["close_price_after_slippage"].values if "close_price_after_slippage" in wf_df.columns else np.full(n_wf, None),
            tid_wf, ta_lbl_wf)

        perf_row = {**{"code": code, "stock_name": _sn, "market": _mkt,
                       "ktype": ktype, "window_label": _wf_label}, **total_perf}
        return wf_df, pd.DataFrame([perf_row])

    return pd.DataFrame(), pd.DataFrame()


def _worker_stock_walkforward(code, ktype, windows, ma_range, trade_mode, slippage, fee_rate, wf_plan):
    """WF 工作进程。返回 (code, stats_path, perf_path, err)"""
    try:
        df_stats, df_perf = run_stock_walkforward(code, ktype, windows, ma_range, trade_mode, slippage, fee_rate, wf_plan)
        if df_stats is None or df_stats.empty:
            return code, None, None, None, None
        _tmp = os.path.join(PROJECT_ROOT, "results_uscncc", f"_tmp_wf_{code.replace('.','_')}_{ktype}")
        os.makedirs(os.path.dirname(_tmp), exist_ok=True)
        _p_stats = _tmp + "_stats.parquet"
        _p_perf = _tmp + "_perf.parquet"
        df_stats.to_parquet(_p_stats, index=False)
        if not df_perf.empty:
            df_perf.to_parquet(_p_perf, index=False)
        return code, _p_stats, _p_trades, _p_perf, None
    except Exception as e:
        return code, None, None, None, None, str(e)


def run_backtest(ktype="1w", ma_list=None, ma_start=2, ma_end=61, ma_step=1,
                 trade_mode="close", slippage=0.0, fee_rate=0.001, markets=None):
    """主入口：对所有股票并行回测并写入 backtest_stats 表。"""
    ma_range = ma_list if ma_list is not None else list(range(ma_start, ma_end, ma_step))

    print(f"\n{'='*60}")
    print(f"  【主回测 {ktype}】")
    print(f"{'='*60}")

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
    try: _cw.execute("DROP TABLE IF EXISTS backtest_stats")
    except: pass
    _cw.close()

    _tmp_files = []
    _cols = ["code","stock_name","market","ktype","window_label","window_count","datetime",
        "open","high","low","close","volume","turnover","turnover_amount","source",
        "ha_close","ma_len","ha_ma_value","trend_direction","signal","signal_exec",
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

    _tmp_perf_files = []
    _tmp_trades_files = []
    _failures = []  # 收集失败信息，回测结束后统一输出
    # 子进程全部结束后再统一写入
    with concurrent.futures.ProcessPoolExecutor(max_workers=_n_workers) as executor:
        futures = {executor.submit(_worker_stock, code, ktype, windows, ma_range,
                                   trade_mode, slippage, fee_rate): code for code in codes}
        with tqdm(total=len(futures), desc="  回测", unit="stock") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    _code, _path, _trades_path, _perf_path, err = future.result()
                except Exception as e:
                    _failures.append(("进程异常", str(e)))
                    pbar.update(1); continue
                if err:
                    _failures.append((_code, err))
                    pbar.update(1); continue
                if not _path:
                    pbar.update(1); continue
                _tmp_files.append(_path)
                if _trades_path:
                    _tmp_trades_files.append(_trades_path)
                if _perf_path:
                    _tmp_perf_files.append(_perf_path)
                pbar.update(1)

    # 回测结束，统一输出失败信息
    for _code, _err in _failures:
        print(f"  {_code} 失败: {_err}")

    # 所有子进程结束→DuckDB 原生批量读 parquet + 排序（一步完成）
    if _tmp_files:
        _t0 = time.time()
        print("  写入中...", end=" ", flush=True)
        _cw = duckdb.connect(DB_PATH)
        _cw.execute("DROP TABLE IF EXISTS backtest_stats")
        _plist = ",".join(f"'{p}'" for p in _tmp_files)
        _cw.execute(f"""
            CREATE TABLE backtest_stats AS
            SELECT {_sel} FROM read_parquet([{_plist}])
            ORDER BY code ASC, ktype ASC, window_label ASC, ma_len ASC, datetime ASC
        """)
        total_rows = _cw.execute("SELECT count(*) FROM backtest_stats").fetchone()[0]
        # 建索引加速后续 JOIN
        _cw.execute("CREATE INDEX IF NOT EXISTS idx_bt_code_ktype_ml_tid ON backtest_stats(code, ktype, ma_len, trade_id)")
        _cw.close()
        print(f"  {len(_tmp_files)} 个 parquet 写入完成 ({total_rows:,} 行) ({time.time()-_t0:.0f}s)")
        for _p in _tmp_files:
            try: os.remove(_p)
            except: pass

    # 写入交易记录表（从回测阶段直接生成，无需派生）
    if _tmp_trades_files:
        _tt0 = time.time()
        _cw = duckdb.connect(DB_PATH)
        _cw.execute("DELETE FROM backtest_trades")
        _tlist = ",".join(f"'{p}'" for p in _tmp_trades_files)
        _cw.execute(f"""
            INSERT INTO backtest_trades
            SELECT code, stock_name, market, ktype, window_label, datetime,
                   ma_len, trade_id, trade_action, trade_price_after_slippage,
                   trade_shares, slippage, trade_amount, commission,
                   actual_trade_amount, trade_status, close_pnl, close_type,
                   close_pnl_type,
                   cash_before_trade, cash_after_trade, available_cash, created_at
            FROM read_parquet([{_tlist}])
        """)
        _cnt_t = _cw.execute("SELECT count(*) FROM backtest_trades").fetchone()[0]
        _cw.close()
        for _p in _tmp_trades_files:
            try: os.remove(_p)
            except: pass
        print(f"  交易记录: {_cnt_t:,} 行 ({time.time()-_tt0:.0f}s)")

    # 写入策略表现表
    if _tmp_perf_files:
        try:
            _cw = duckdb.connect(DB_PATH)
            _cw.execute("DELETE FROM backtest_performance")
            _plist2 = ",".join(f"'{p}'" for p in _tmp_perf_files)
            _cw.execute(f"""
                INSERT INTO backtest_performance
                SELECT code, stock_name, market, ktype, window_label, ma_len,
                       total_return, cagr, buy_hold_return, excess_return, avg_trade_return,
                       strategy_score,
                       max_drawdown, sharpe_ratio, calmar_ratio,
                       trade_count, win_rate, profit_factor, payoff_ratio,
                       avg_win, avg_win_pct, avg_loss, avg_loss_pct,
                       max_win, max_loss, max_win_streak, max_loss_streak,
                       avg_hold_days, avg_hold_bars,
                       initial_cash, final_cash, CURRENT_TIMESTAMP
                FROM read_parquet([{_plist2}])
            """)
            _cw.close()
            for _p in _tmp_perf_files:
                try: os.remove(_p)
                except: pass
        except Exception as e:
            print(f"  策略表现写入失败: {e}")

    # 构建 strategy_score 明细 / 排名 / 稳定性表（从 backtest_performance 查询后写入）
    try:
        _qc = duckdb.connect(DB_PATH)
        _perf_df = _qc.execute("""
            SELECT code, ktype, ma_len, window_label, strategy_score, cagr
            FROM backtest_performance
        """).fetchdf()
        if not _perf_df.empty:
            _score_pivot, _rank_pivot = build_score_matrix(_perf_df.to_dict("records"))

            if not _score_pivot.empty:
                # 透视表 melt 回长格式 → 评分明细
                _score_long = _score_pivot.melt(
                    id_vars=["code", "ktype", "ma_len"],
                    var_name="window_label", value_name="strategy_score"
                ).dropna(subset=["strategy_score"])
                _score_long["created_at"] = datetime.now()

                _qc.execute("DELETE FROM strategy_score_detail")
                _qc.register("_s", _score_long)
                _qc.execute("""
                    INSERT INTO strategy_score_detail
                    SELECT code, ktype, ma_len, window_label, strategy_score, created_at
                    FROM _s
                """)

            if not _rank_pivot.empty:
                _rank_long = _rank_pivot.melt(
                    id_vars=["code", "ktype", "ma_len"],
                    var_name="window_label", value_name="window_rank"
                ).dropna(subset=["window_rank"])
                _rank_long["created_at"] = datetime.now()

                _qc.execute("DELETE FROM strategy_score_rank")
                _qc.register("_r", _rank_long)
                _qc.execute("""
                    INSERT INTO strategy_score_rank
                    SELECT code, ktype, ma_len, window_label, window_rank, created_at
                    FROM _r
                """)

            # 参数稳定性分析
            _stab_df = build_window_stability(_perf_df.to_dict("records"))
            if not _stab_df.empty:
                _stab_df["created_at"] = datetime.now()
                _qc.execute("DELETE FROM strategy_score_stability")
                _qc.register("_t", _stab_df)
                _qc.execute("""
                    INSERT INTO strategy_score_stability
                    SELECT code, ktype, ma_len, window_label, window_count,
                           win_window_count, win_window_pct, avg_cagr, cagr_std,
                           rank_top3_pct, avg_score_rank, score_rank_std, rank_first_count,
                           stability_score, is_best, created_at
                    FROM _t
                """)
        _qc.close()
    except Exception as e:
        print(f"  strategy_score 透视表构建失败: {e}")

    print(f"  ──────────────────────────────────────")
    print(f"  完成: {total_rows:,} 行写入 backtest_stats")

    # =========================================================
    # Walk Forward 回测（使用 strategy_score_stability 的 is_best 计划）
    # =========================================================
    if _tmp_perf_files:
        try:
            _qc = duckdb.connect(DB_PATH)
            _wf_plan = _qc.execute("""
                SELECT code, window_label, ma_len AS best_ma
                FROM strategy_score_stability
                WHERE is_best = '最优'
                ORDER BY code, window_label
            """).fetchdf()
            _qc.close()
        except Exception as e:
            print(f"  WF 计划读取失败: {e}")
            _wf_plan = pd.DataFrame()

        if not _wf_plan.empty and len(windows) >= 2:
            print(f"\n  {'─'*50}")
            print(f"  【Walk Forward 回测 {ktype}】")
            print(f"  {'─'*50}")
            # 提前清空旧数据
            _cw = duckdb.connect(DB_PATH)
            for _wt in ["backtest_stats_walkforward", "backtest_trades_walkforward",
                        "backtest_performance_walkforward"]:
                try: _cw.execute(f"DELETE FROM {_wt}")
                except: pass
            _cw.close()

            _wf_cols = ["code","stock_name","market","ktype","window_label","window_count","datetime",
                "open","high","low","close","volume","turnover","turnover_amount","source",
                "ha_close","ma_len","ha_ma_value","trend_direction","signal","signal_exec",
                "trade_id","trade_action","trade_price","trade_price_after_slippage",
                "trade_shares","slippage","trade_amount","commission","actual_trade_amount",
                "available_cash","held_shares","trade_status",
                "close_price","close_price_after_slippage","close_shares","close_slippage",
                "close_trade_amount","close_commission","close_actual_trade_amount","close_pnl","close_type","close_pnl_type",
                "cash_before_trade","cash_after_trade",
                "account_value","account_value_change","account_value_change_pct",
                "change_from_initial","change_from_initial_pct",
                "created_at"]
            _wf_sel = ",".join(_wf_cols)

            _tmp_wf_files = []
            _tmp_wf_perf = []
            _n_workers = max(1, os.cpu_count() - 1)
            with concurrent.futures.ProcessPoolExecutor(max_workers=_n_workers) as executor:
                _wf_codes = _wf_plan["code"].unique()
                _futures = {executor.submit(_worker_stock_walkforward, code, ktype, windows, ma_range,
                                           trade_mode, slippage, fee_rate, _wf_plan): code for code in _wf_codes}
                with tqdm(total=len(_futures), desc="  WF 回测", unit="stock") as _wf_pbar:
                    for _future in concurrent.futures.as_completed(_futures):
                        try:
                            _code, _path, _perf_path, err = _future.result()
                        except Exception as e:
                            _wf_pbar.update(1); continue
                        if err or not _path:
                            _wf_pbar.update(1); continue
                        _tmp_wf_files.append(_path)
                        if _perf_path:
                            _tmp_wf_perf.append(_perf_path)
                        _wf_pbar.update(1)

            # 批量写入
            if _tmp_wf_files:
                _cw = duckdb.connect(DB_PATH)
                _plist = ",".join(f"'{p}'" for p in _tmp_wf_files)
                _cw.execute(f"""
                    INSERT INTO backtest_stats_walkforward ({_wf_sel})
                    SELECT {_wf_sel} FROM read_parquet([{_plist}])
                """)
                _cw.close()
                for _p in _tmp_wf_files:
                    try: os.remove(_p)
                    except: pass

            if _tmp_wf_perf:
                try:
                    _cw = duckdb.connect(DB_PATH)
                    _plist2 = ",".join(f"'{p}'" for p in _tmp_wf_perf)
                    _cw.execute(f"""
                        INSERT INTO backtest_performance_walkforward
                        SELECT code, stock_name, market, ktype, window_label,
                               total_return, cagr, buy_hold_return, excess_return, avg_trade_return,
                               strategy_score,
                               max_drawdown, sharpe_ratio, calmar_ratio,
                               trade_count, win_rate, profit_factor, payoff_ratio,
                               avg_win, avg_win_pct, avg_loss, avg_loss_pct,
                               max_win, max_loss, max_win_streak, max_loss_streak,
                               avg_hold_days, avg_hold_bars,
                               initial_cash, final_cash, CURRENT_TIMESTAMP
                        FROM read_parquet([{_plist2}])
                    """)
                    _cw.close()
                    for _p in _tmp_wf_perf:
                        try: os.remove(_p)
                        except: pass
                except Exception as e:
                    print(f"  WF 策略表现写入失败: {e}")

            # 派生 WF 交易记录
            _cw = duckdb.connect(DB_PATH)
            try:
                _cw.execute("DELETE FROM backtest_trades_walkforward")
                _cw.execute("""
                    INSERT INTO backtest_trades_walkforward
                    SELECT code, stock_name, market, ktype, w, datetime,
                           ma_len, trade_id, trade_action, trade_price_after_slippage,
                           trade_shares, slippage, trade_amount, commission,
                           actual_trade_amount, trade_status, close_pnl, close_type,
                           close_pnl_type,
                           cash_before_trade, cash_after_trade, available_cash, created_at
                    FROM backtest_stats_walkforward,
                         UNNEST(STRING_SPLIT(window_label, ',')) AS t(w)
                    WHERE trade_action IS NOT NULL
                """)
                # 持仓未平仓补虚拟平仓
                _cw.execute("""
                    INSERT INTO backtest_trades_walkforward
                    SELECT s.code, s.stock_name, s.market, s.ktype, s.ow, s.datetime,
                           s.ma_len, s.trade_id, '平多',
                           s.close_price_after_slippage,
                           s.close_shares, s.close_slippage, s.close_trade_amount, s.close_commission,
                           s.close_actual_trade_amount, s.trade_status, s.close_pnl, '虚拟平仓',
                           s.close_pnl_type,
                           s.cash_before_trade, s.cash_after_trade, s.cash_after_trade, s.created_at
                    FROM (
                        SELECT oww.w AS ow, s.*, ROW_NUMBER() OVER (
                            PARTITION BY oww.code, oww.ktype, oww.ma_len, oww.trade_id, oww.w
                            ORDER BY s.datetime DESC
                        ) AS rn
                        FROM (
                            SELECT DISTINCT o.code, o.ktype, o.ma_len, o.trade_id, t.w
                            FROM backtest_stats_walkforward o,
                                 UNNEST(STRING_SPLIT(o.window_label, ',')) AS t(w)
                            WHERE o.trade_action = '开多'
                        ) oww
                        JOIN backtest_stats_walkforward s ON s.code = oww.code AND s.ktype = oww.ktype
                            AND s.ma_len = oww.ma_len AND s.trade_id = oww.trade_id
                            AND s.trade_status = '持仓中'
                            AND s.datetime <= STRPTIME(SPLIT_PART(oww.w, '~', 2), '%Y-%m-%d')
                        WHERE NOT EXISTS (
                            SELECT 1 FROM backtest_stats_walkforward s2
                            WHERE s2.code = oww.code AND s2.ktype = oww.ktype
                              AND s2.trade_id = oww.trade_id
                              AND s2.trade_action = '平多'
                              AND s2.window_label LIKE '%' || oww.w || '%'
                        )
                    ) s
                    WHERE s.rn = 1
                """)
            except Exception as e:
                print(f"  WF 交易记录派生失败: {e}")
            _cw.close()
            print(f"  {'─'*50}")


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
