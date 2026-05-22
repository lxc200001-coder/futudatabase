# coding=utf-8
"""
从掘金量化导出历史K线数据，供本地回测框架使用。

用法：
    python export_gm_data.py

依赖：pip install gm pandas pyarrow
"""
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# =========================================================
# 配置
# =========================================================
SYMBOLS = [
    "SHSE.510300",  # 沪深300 ETF
    "SHSE.600036",  # 招商银行
    "SZSE.159915",  # 创业板 ETF
    "SZSE.000001",  # 平安银行
]
FREQUENCY = "1d"
START_TIME = "2010-01-01 00:00:00"
END_TIME = "2024-12-31 00:00:00"
OUTPUT_DIR = r"c:\pythonstock\futudatabase\data_gm"

STRATEGY_ID = "3014a373-5603-11f1-aa74-a8a159d70d0e"
TOKEN = "c2bccaea52b00b0a4af3ab417371c7e857fc17a5"


# =========================================================
# 数据导出策略（只导出、不交易）
# =========================================================
def init(context):
    context.sym_list = SYMBOLS
    context.frequency = FREQUENCY
    context.bar_count = 8000
    context.bar_idx = 0

    for sym in context.sym_list:
        subscribe(sym, context.frequency, context.bar_count)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[开始] 导出 {START_TIME} → {END_TIME}")


def on_bar(context, bars):
    context.bar_idx += 1
    bar_date = bars[-1]["eob"]

    # 每 200 根 K 线保存一次进度；最后一根（接近 2024 年底）必须保存
    near_end = bar_date.replace(tzinfo=None) >= datetime(2024, 12, 1)
    if not near_end and context.bar_idx % 200 != 0:
        return

    for sym in context.sym_list:
        data = context.data(
            sym, context.frequency, context.bar_count,
            fields="eob,open,high,low,close,volume"
        )
        if data is None or len(data) == 0:
            continue

        df = pd.DataFrame(data).sort_values("eob").reset_index(drop=True)
        df["symbol"] = sym

        fpath = os.path.join(OUTPUT_DIR, f"{sym}.parquet")
        df.to_parquet(fpath, index=False)

    flag = " [最终]" if near_end else ""
    print(f"[保存] {bar_date}  {len(df)} 根K线{flag}")


# =========================================================
# 运行
# =========================================================
if __name__ == "__main__":
    from gm.api import *

    # 注册当前模块，避免 run() 内部循环导入
    sys.modules["export_gm_data"] = sys.modules["__main__"]

    run(
        strategy_id=STRATEGY_ID,
        filename="export_gm_data",
        token=TOKEN,
        mode=MODE_BACKTEST,
        backtest_start_time=START_TIME,
        backtest_end_time=END_TIME,
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=100000,
        backtest_commission_ratio=0,
        backtest_slippage_ratio=0,
    )
