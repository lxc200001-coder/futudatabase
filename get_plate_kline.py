import os
import time
import pandas as pd

from collections import deque
from datetime import datetime
from futu import OpenQuoteContext, KLType, AuType, RET_OK
from futu.common.constant import Plate

# =========================================================
# 配置
# =========================================================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# =========================================================
# API限速
# =========================================================
REQUEST_LIMIT = 60
WINDOW_SECONDS = 30

request_times = deque()


def wait_rate_limit():

    now = time.time()
    while request_times and now - request_times[0] > WINDOW_SECONDS:
        request_times.popleft()

    if len(request_times) >= REQUEST_LIMIT:
        sleep_time = max(0, WINDOW_SECONDS - (now - request_times[0]) + 0.5)
        print(f"\n触发限频，开始倒计时 {sleep_time:.1f} 秒")
        start = time.time()
        while True:
            remaining = sleep_time - (time.time() - start)
            if remaining <= 0:
                break
            print(f"\r剩余等待: {remaining:.1f} 秒", end="", flush=True)
            time.sleep(0.2)
        print("\r剩余等待: 0.0 秒，继续执行        ")

    request_times.append(time.time())


# =========================================================
# 下载周线数据
# =========================================================
def fetch_weekly(code, start_str, end_str, quote_ctx):

    all_data = []
    page_req_key = None
    last_max_time = None
    retry = 0

    while True:
        if page_req_key is None:
            wait_rate_limit()

        ret, data, page_req_key = quote_ctx.request_history_kline(
            code=code,
            start=start_str,
            end=end_str,
            ktype=KLType.K_WEEK,
            autype=AuType.QFQ,
            page_req_key=page_req_key
        )

        if ret != RET_OK:
            retry += 1
            print(code, f"请求失败，重试 {retry}/3")
            if retry >= 3:
                print(code, "下载失败")
                break
            time.sleep(2)
            continue

        retry = 0

        if data is None or len(data) == 0:
            break

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
# 获取美股所有板块列表
# =========================================================
def fetch_plate_list():
    """获取美股所有板块列表，返回 DataFrame"""
    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    all_plates = []
    try:
        for plate_class, class_name in [(Plate.INDUSTRY, "行业"), (Plate.CONCEPT, "概念")]:
            wait_rate_limit()
            ret, data = quote_ctx.get_plate_list(market="US", plate_class=plate_class)
            if ret == RET_OK and data is not None and not data.empty:
                data["plate_class"] = class_name
                all_plates.append(data)
                print(f"获取板块列表: US {class_name} {len(data)} 个")
    finally:
        quote_ctx.close()

    if not all_plates:
        return pd.DataFrame()
    df = pd.concat(all_plates, ignore_index=True)
    df = df.drop_duplicates("code").reset_index(drop=True)
    return df


# =========================================================
# 下载所有美股板块的周 K 线
# =========================================================
def run_plate_kline_download():
    """下载所有美股板块的周 K 线"""
    end_str = datetime.now().strftime("%Y-%m-%d")
    start_str = "2010-01-01"
    out_path = os.path.join(DATA_DIR, "plates_kline.parquet")

    plates = fetch_plate_list()
    if plates.empty:
        print("无板块列表，跳过")
        return
    print(f"开始下载 {len(plates)} 个板块的周 K 线...")

    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    all_bars = []
    try:
        for i, (_, row) in enumerate(plates.iterrows(), 1):
            code = row["code"]
            print(f"K线 [{i}/{len(plates)}] {code} {row.get('plate_name', '')}")

            df = fetch_weekly(code, start_str, end_str, quote_ctx)
            if not df.empty:
                df["plate_code"] = code
                df["plate_name"] = row.get("plate_name", "")
                df["plate_class"] = row.get("plate_class", "")
                all_bars.append(df)

        if all_bars:
            result = pd.concat(all_bars, ignore_index=True)
            result = result.sort_values(["plate_code", "time_key"]).reset_index(drop=True)
            result.to_parquet(out_path, index=False)
            print(f"板块 K 线保存完成: {out_path} 共 {len(result)} 条")
        else:
            print("无板块 K 线数据")
    finally:
        quote_ctx.close()


# =========================================================
# 主入口
# =========================================================
if __name__ == "__main__":
    run_plate_kline_download()
