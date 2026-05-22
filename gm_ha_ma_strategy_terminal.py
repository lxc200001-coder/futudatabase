# coding=utf-8
from gm.api import *
import numpy as np
import pandas as pd

"""
HA-MA 均线方向策略（掘金量化终端版）
"""


def init(context):
    context.sym_list = [
        gm_symbol("SH.510300"),    # 沪深300 ETF
        gm_symbol("SH.600036"),    # 招商银行
    ]
    context.ma_len = 30
    context.frequency = "1d"
    context.bar_count = context.ma_len + 120
    context.alloc_per_sym = 10000

    for sym in context.sym_list:
        subscribe(sym, context.frequency, context.bar_count)

    print(f"[策略初始化] 标的: {context.sym_list}")
    print(f"  均线周期: {context.ma_len}")
    print(f"  K线周期: {context.frequency}")
    print(f"  每只分配资金: {context.alloc_per_sym}")


def on_bar(context, bars):
    for sym in context.sym_list:
        _process_symbol(context, sym)


def _process_symbol(context, sym):
    data = context.data(
        sym, context.frequency, context.bar_count,
        fields="eob,open,high,low,close,volume"
    )
    if data is None or len(data) < context.ma_len + 5:
        return

    df = pd.DataFrame(data).sort_values("eob").reset_index(drop=True)

    ha_close = calc_ha_close(df)
    ma_series = pd.Series(ha_close).rolling(context.ma_len, min_periods=context.ma_len).mean()

    idx = -1 if pd.notna(ma_series.iloc[-1]) else -2
    if pd.isna(ma_series.iloc[idx]) or pd.isna(ma_series.iloc[idx - 1]):
        return

    dir_curr = 1 if ma_series.iloc[idx] > ma_series.iloc[idx - 1] else -1
    dir_prev = 1 if ma_series.iloc[idx - 1] > ma_series.iloc[idx - 2] else -1

    pos = context.account().position(symbol=sym, side=PositionSide_Long)
    has_position = pos is not None

    latest_data = df.iloc[-1]
    latest_close = latest_data["close"]
    latest_time = latest_data["eob"]

    if dir_curr == 1 and dir_prev == -1 and not has_position:
        order_target_value(
            sym, context.alloc_per_sym,
            order_type=OrderType_Limit,
            position_side=PositionSide_Long,
            price=latest_close,
        )
        print(f"[买入] {sym}  {latest_time}  价格={latest_close:.2f}  MA={ma_series.iloc[idx]:.2f}")

    elif dir_curr == -1 and dir_prev == 1 and has_position:
        order_target_value(
            sym, 0,
            order_type=OrderType_Limit,
            position_side=PositionSide_Long,
            price=latest_close,
        )
        print(f"[卖出] {sym}  {latest_time}  价格={latest_close:.2f}  MA={ma_series.iloc[idx]:.2f}")


def gm_symbol(code):
    prefix_map = {"SH.": "SHSE.", "SZ.": "SZSE."}
    for old, new in prefix_map.items():
        if code.startswith(old):
            return code.replace(old, new, 1)
    return code


def calc_ha_close(df):
    return (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
