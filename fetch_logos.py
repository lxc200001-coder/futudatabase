import os
import json
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SYMBOL_FILE = "symbols.csv"
LOGO_DIR = "logos"
FINNHUB_TOKEN = "d87eun1r01ql0hslgkigd87eun1r01ql0hslgkj0"
os.makedirs(LOGO_DIR, exist_ok=True)


def _get_logo_url(ticker):
    """Query Finnhub profile2 API to get the logo URL for a ticker."""
    url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={FINNHUB_TOKEN}"
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return None
    data = r.json()
    return data.get("logo") or None


def _download_logo(code):
    """Download logo for a US stock via Finnhub. Returns (code, success, path)."""
    ticker = code.replace("US.", "", 1)
    out_path = os.path.join(LOGO_DIR, f"{ticker}.png")

    try:
        logo_url = _get_logo_url(ticker)
        if not logo_url:
            return code, False, None

        r = requests.get(logo_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if r.status_code != 200 or len(r.content) < 500:
            return code, False, None

        with open(out_path, "wb") as f:
            f.write(r.content)
        return code, True, out_path
    except Exception:
        if os.path.exists(out_path):
            os.remove(out_path)
        return code, False, None


def fetch_logos(codes=None, max_workers=5):
    if codes is None:
        symbols = pd.read_csv(SYMBOL_FILE)["code"].dropna().tolist()
        codes = [s for s in symbols if s.startswith("US.")]

    results = {"found": {}, "missing": []}

    # Finnhub free tier: 60 req/min, stagger requests to stay under limit
    # Use a small delay between submissions to stay within rate limits
    import threading
    _lock = threading.Lock()
    _last_call = [0.0]

    def _rate_limited_download(code):
        with _lock:
            elapsed = time.time() - _last_call[0]
            if elapsed < 1.2:
                time.sleep(1.2 - elapsed)
            result = _download_logo(code)
            _last_call[0] = time.time()
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(_rate_limited_download, c): c for c in codes}
        with tqdm(total=len(codes), desc="下载 logo", unit="个") as pbar:
            for f in as_completed(fut_map):
                code, ok, path = f.result()
                if ok:
                    results["found"][code] = path
                else:
                    results["missing"].append(code)
                pbar.update(1)

    # Save ticker -> local path mapping
    mapping = {code.replace("US.", "", 1): path
               for code, path in results["found"].items()}
    mapping_path = os.path.join(LOGO_DIR, "logo_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, ensure_ascii=False)

    print(f"完成: {len(results['found'])} 个 logo 已下载到 {LOGO_DIR}/")
    if results["missing"]:
        print(f"未找到: {len(results['missing'])} 个 ({', '.join(results['missing'][:10])}...)")
    print(f"映射文件: {mapping_path}")
    return results


if __name__ == "__main__":
    fetch_logos()
