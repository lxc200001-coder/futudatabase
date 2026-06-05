"""
获取上证板块、深证板块成交额前 30 的股票
数据来源：富途 OpenD（需先启动 OpenD）
输出格式：与 symbols/top_turnover_*.csv 一致
"""
import os
import sys
import time
import pandas as pd
from datetime import datetime
from futu import OpenQuoteContext, AccumulateFilter, StockField, SortDir, RET_OK, Market


def _is_excluded(code):
    """判断是否属于科创板/创业板/北交所"""
    # SH.688xxx → 科创板
    # SZ.300xxx / SZ.301xxx → 创业板
    # BJ.8xxxxx / BJ.4xxxxx → 北交所
    if code.startswith("SH.688"):
        return True
    if code.startswith(("SZ.300", "SZ.301")):
        return True
    if code.startswith("BJ."):
        return True
    return False


def fetch_cn_top_turnover(limit=30):
    """通过 Futu OpenD 获取沪深主板当日成交额前 N 的股票"""
    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    all_records = []
    fetch_num = max(limit * 3, 60)  # 多取一些，过滤后仍有足够数据

    try:
        for market_name, market in [("SH", Market.SH), ("SZ", Market.SZ)]:
            af = AccumulateFilter()
            af.stock_field = StockField.TURNOVER
            af.is_no_filter = False
            af.filter_min = 1
            af.sort = SortDir.DESCEND

            print(f"[{market_name}] 开始筛选...")
            ret, result = quote_ctx.get_stock_filter(market, [af], begin=0, num=fetch_num)
            if ret != RET_OK:
                print(f"[{market_name}] get_stock_filter 失败: {result}")
                continue
            # SH/SZ 返回 (is_last_page, total_count, stock_list)
            if isinstance(result, tuple) and len(result) >= 3:
                is_last, total, stock_list = result[:3]
                print(f"[{market_name}] 共 {total} 只，获取到 {len(stock_list)} 只")
                codes = [s.stock_code for s in stock_list]
            elif isinstance(result, pd.DataFrame):
                codes = result["code"].tolist()
            else:
                print(f"[{market_name}] 返回格式异常: {type(result).__name__}")
                continue

            print(f"[{market_name}] 筛选到 {len(codes)} 只（过滤前）")

            for i, code in enumerate(codes):
                if _is_excluded(code):
                    continue
                if i > 0 and i % 10 == 0:
                    time.sleep(0.5)
                ret2, snap = quote_ctx.get_market_snapshot([code])
                if ret2 == RET_OK and snap is not None and not snap.empty:
                    row = snap.iloc[0]
                    name = str(row.get("code_name", row.get("name", "")))
                    price = float(row.get("last_price", row.get("close_price", 0)) or 0)
                    turnover = float(row.get("turnover", 0) or 0)
                    all_records.append({
                        "排名": 0,
                        "代码": code,
                        "名称": name,
                        "最新价": round(price, 2),
                        "成交额(亿元)": round(turnover / 1e8, 2),
                    })
                else:
                    print(f"  {code} 快照获取失败")

    finally:
        quote_ctx.close()

    if not all_records:
        print("未获取到任何数据")
        return

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["代码"])
    df = df.sort_values("成交额(亿元)", ascending=False).reset_index(drop=True)
    df = df.head(limit)
    df["排名"] = range(1, len(df) + 1)

    date_str = datetime.now().strftime("%Y%m%d")
    out_path = os.path.join("symbols", f"top_turnover_cn_{date_str}.csv")
    os.makedirs("symbols", exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n已保存: {out_path}")
    print(f"共 {len(df)} 只股票，成交额前 3:")
    for _, r in df.head(3).iterrows():
        print(f"  {r['排名']}. {r['代码']} {r['名称']} {r['成交额(亿元)']}亿")
    return out_path


if __name__ == "__main__":
    fetch_cn_top_turnover(limit=30)
