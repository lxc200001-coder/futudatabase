# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import numpy as np
import pandas as pd

"""
HA-MA 均线方向策略（掘金量化终端版）

策略逻辑（源自回测框架 2windows_param_scan_numba.py）：
  1. 计算 Heikin-Ashi 收盘价（平滑价格噪声）
  2. 计算 HA 收盘价的 N 周期均线（参数通过参数扫描确定最优值）
  3. 均线上升时持有多头，均线下降时空仓
  4. 均线方向变化时触发开平仓（次日开盘价成交）
"""


def init(context):
    # =========================================================
    # TODO: 策略参数（根据参数扫描结果修改）
    # =========================================================
    context.sym_list = [
        gm_symbol("SH.510300"),    # 沪深300 ETF
        gm_symbol("SH.600036"),    # 招商银行
    ]
    context.ma_len = 30                               # 均线周期（从参数扫描获取最优值）
    context.frequency = "1d"                          # K线周期：1d=日线, 1w=周线
    context.bar_count = context.ma_len + 120          # 缓存长度（需 > ma_len，为 HA 预热留余量）
    context.alloc_per_sym = 10000                     # 每只标的分配资金

    # =========================================================
    # 订阅行情
    # =========================================================
    for sym in context.sym_list:
        subscribe(sym, context.frequency, context.bar_count)

    print(f"[策略初始化] 标的: {context.sym_list}")
    print(f"  均线周期: {context.ma_len}")
    print(f"  K线周期: {context.frequency}")
    print(f"  每只分配资金: {context.alloc_per_sym}")


def on_bar(context, bars):
    """每根 K 线触发一次"""

    for sym in context.sym_list:
        # 每个标的独立计算
        _process_symbol(context, sym)


def _process_symbol(context, sym):
    """处理单个标的的 HA-MA 信号"""
    data = context.data(
        sym, context.frequency, context.bar_count,
        fields="eob,open,high,low,close,volume"
    )
    if data is None or len(data) < context.ma_len + 5:
        return

    df = pd.DataFrame(data).sort_values("eob").reset_index(drop=True)

    # 计算 Heikin-Ashi 收盘价
    ha_close = calc_ha_close(df)

    # 计算均线
    ma_series = pd.Series(ha_close).rolling(context.ma_len, min_periods=context.ma_len).mean()

    # 判断方向变化（取最完整的最新值）
    idx = -1 if pd.notna(ma_series.iloc[-1]) else -2
    if pd.isna(ma_series.iloc[idx]) or pd.isna(ma_series.iloc[idx - 1]):
        return

    dir_curr = 1 if ma_series.iloc[idx] > ma_series.iloc[idx - 1] else -1
    dir_prev = 1 if ma_series.iloc[idx - 1] > ma_series.iloc[idx - 2] else -1

    # 获取当前持仓
    pos = context.account().position(symbol=sym, side=PositionSide_Long)
    has_position = pos is not None

    # 最新价和时间
    latest_data = df.iloc[-1]
    latest_close = latest_data["close"]
    latest_time = latest_data["eob"]

    # ---- 买入信号：MA 从下降转为上升 ----
    if dir_curr == 1 and dir_prev == -1 and not has_position:
        order_target_value(
            sym, context.alloc_per_sym,
            order_type=OrderType_Limit,
            position_side=PositionSide_Long,
            price=latest_close,
        )
        print(f"[买入] {sym}  {latest_time}  价格={latest_close:.2f}  MA={ma_series.iloc[idx]:.2f}")

    # ---- 卖出信号：MA 从上升转为下降 ----
    elif dir_curr == -1 and dir_prev == 1 and has_position:
        order_target_value(
            sym, 0,
            order_type=OrderType_Limit,
            position_side=PositionSide_Long,
            price=latest_close,
        )
        print(f"[卖出] {sym}  {latest_time}  价格={latest_close:.2f}  MA={ma_series.iloc[idx]:.2f}")


# =========================================================
# 辅助函数
# =========================================================

def gm_symbol(code):
    """将富途代码格式转为掘金量化格式。
    示例: SH.510300 → SHSE.510300,  SZ.159915 → SZSE.159915
    美股和港股格式不变（US.AAPL → US.AAPL, HK.00700 → HK.00700）
    """
    prefix_map = {"SH.": "SHSE.", "SZ.": "SZSE."}
    for old, new in prefix_map.items():
        if code.startswith(old):
            return code.replace(old, new, 1)
    return code


def calc_ha_close(df):
    """计算 Heikin-Ashi 收盘价 = (O + H + L + C) / 4"""
    return (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0


# =========================================================
# 策略入口
# =========================================================
if __name__ == "__main__":
    run(
        strategy_id="3014a373-5603-11f1-aa74-a8a159d70d0e",          # TODO: 填入掘金量化策略ID
        filename="main.py",
        token="c2bccaea52b00b0a4af3ab417371c7e857fc17a5",                       # TODO: 填入掘金量化token
        mode=MODE_BACKTEST,
        backtest_start_time="2000-01-01 00:00:00",
        backtest_end_time="2024-12-31 00:00:00",
        backtest_adjust=ADJUST_NONE,
        backtest_initial_cash=20000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.0001,
    )
