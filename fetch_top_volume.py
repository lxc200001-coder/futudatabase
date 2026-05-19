"""
获取币值（市值）前 20 的现货币对
数据来源：CoinGecko 免费 API
"""
import requests

PROXIES = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}

COINGECKO_URL = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=market_cap_desc"
    "&per_page=20&page=1&sparkline=false"
    "&locale=zh"
)


def main():
    resp = requests.get(COINGECKO_URL, proxies=PROXIES, timeout=15)
    data = resp.json()

    print(f"{'排名':>4} {'币种':>10} {'最新价(USD)':>16} {'市值(USD)':>20} {'24h涨跌':>10}")
    print("-" * 66)

    for i, coin in enumerate(data, 1):
        symbol = coin["symbol"].upper()
        price = coin["current_price"] or 0
        mcap = coin["market_cap"] or 0
        change = coin.get("price_change_percentage_24h") or 0
        print(f"{i:>4} {symbol:>10} {price:>16.8f} {mcap:>20.0f} {change:>+9.2f}%")


if __name__ == "__main__":
    main()
