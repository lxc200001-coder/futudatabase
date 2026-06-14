"""
DuckDB 数据库初始化
====================
创建/更新表结构、导入排名数据。

用法:
    python database/init_db.py                              # 创建表 + 导入数据
    python database/init_db.py --reset                       # 清空重建（改列类型/删表时用）
    python database/init_db.py --query "SELECT * FROM top_stocks"
    python database/init_db.py --list                        # 列出所有表

说明:
    新增表/列/注释 → 直接运行 init_db.py，无需 reset
    修改列类型/删除表 → 需要 --reset（会清空数据）
    --reset 会自动重建所有 klines、watchlist 等表
"""
import os
import argparse
from datetime import datetime

import duckdb
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "market.duckdb")


def update_watchlist(con):
    """合并 symbols.csv + 排名变动表去重后写入 watchlist 表"""
    import pandas as pd

    _sources = {}  # code → set of source

    # 1. symbols.csv → 手动添加
    _symbol_path = os.path.join(PROJECT_ROOT, "symbols", "symbols.csv")
    if os.path.exists(_symbol_path):
        for c in pd.read_csv(_symbol_path)["code"].dropna():
            _sources.setdefault(c.strip(), set()).add("手动添加")

    # 2. top_turnover_stock_rank 最新日期 rank ≤ 200
    try:
        for row in con.execute("""
            SELECT DISTINCT code FROM top_turnover_stock_rank
            WHERE datetime = (SELECT MAX(datetime) FROM top_turnover_stock_rank)
              AND rank <= 200
        """).fetchall():
            _sources.setdefault(str(row[0]), set()).add("60日成交额排名")
    except Exception:
        pass

    # 3. top_turnover_etf_rank 最新日期 rank ≤ 10
    try:
        for row in con.execute("""
            SELECT DISTINCT code FROM top_turnover_etf_rank
            WHERE datetime = (SELECT MAX(datetime) FROM top_turnover_etf_rank)
              AND rank <= 10
        """).fetchall():
            _sources.setdefault(str(row[0]), set()).add("ETF成交额排名")
    except Exception:
        pass

    # 4. 确定市场
    _market_of = {}
    for code in _sources:
        if code.startswith("CC."):
            _market_of[code] = "cc"
        elif code.startswith(("SH.", "SZ.")):
            _market_of[code] = "cn"
        elif code.startswith("US."):
            _market_of[code] = "us"

    # 5. 按市场排序写入（us → cc → cn）
    _mkt_order = {"us": 0, "cc": 1, "cn": 2}
    con.execute("DELETE FROM watchlist")
    for code in sorted(_sources.keys(), key=lambda c: (_mkt_order.get(_market_of.get(c, ""), 9), c)):
        mkt = _market_of.get(code)
        if not mkt:
            continue
        con.execute(
            'INSERT INTO watchlist(code, market, source, created_at) VALUES (?, ?, ?, ?)',
            [code, mkt, ",".join(sorted(_sources[code])), datetime.now()],
        )
    print(f"  watchlist: {len(_sources)} 只")


def create_tables(con):
    """建表"""
    con.execute("DROP TABLE IF EXISTS watchlist")
    con.execute("""
        CREATE TABLE watchlist (
            code        VARCHAR,
            market      VARCHAR,
            source      VARCHAR,
            created_at  TIMESTAMP
        )
    """)
    con.execute("COMMENT ON COLUMN watchlist.code IS '股票代码'")
    con.execute("COMMENT ON COLUMN watchlist.market IS '市场: us/cn/cc'")
    con.execute("COMMENT ON COLUMN watchlist.source IS '来源: 手动添加/60日成交额排名/ETF成交额排名/实时成交额排名(逗号拼接)'")
    con.execute("COMMENT ON COLUMN watchlist.created_at IS '添加时间(精确到秒)'")

    # K 线数据表（按周期分表）
    for _kt, _kt_desc in [("1d", "日线"), ("1w", "周线")]:
        con.execute(f"DROP TABLE IF EXISTS klines_{_kt}")
        con.execute(f"""
            CREATE TABLE klines_{_kt} (
                code        VARCHAR,
                stock_name  VARCHAR,
                market      VARCHAR,
                ktype       VARCHAR,
                datetime    TIMESTAMP,
                open        DOUBLE,
                high        DOUBLE,
                low         DOUBLE,
                close       DOUBLE,
                volume      DOUBLE,
                turnover    DOUBLE,
                turnover_amount DOUBLE,
                source      VARCHAR,
                created_at  TIMESTAMP
            )
        """)
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.code IS '股票代码'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.stock_name IS '股票名称'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.market IS '市场: us/cn/cc'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.ktype IS 'K线周期: 1D/1W'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.datetime IS 'K线时间(原始格式)'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.open IS '开盘价'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.high IS '最高价'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.low IS '最低价'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.close IS '收盘价'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.volume IS '成交量'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.turnover IS '成交额(数据源原生)'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.turnover_amount IS '估算成交额(收盘价×成交量)'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.source IS '数据来源: futu/moomoo/stooq/binance/baostock'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.created_at IS '入库时间'")

    con.execute("""
        CREATE TABLE IF NOT EXISTS plates (
            code        VARCHAR PRIMARY KEY,
            stock_name  VARCHAR,
            market      VARCHAR,
            plates      VARCHAR
        )
    """)
    con.execute("COMMENT ON COLUMN plates.code IS '股票代码'")
    con.execute("COMMENT ON COLUMN plates.stock_name IS '股票名称'")
    con.execute("COMMENT ON COLUMN plates.market IS '市场: us/cn/cc'")
    con.execute("COMMENT ON COLUMN plates.plates IS '所属板块(逗号分隔)'")

    # 表描述
    con.execute("COMMENT ON TABLE watchlist IS '监控标的库(合并rank表+实时排名+symbols.csv)'")
    con.execute("COMMENT ON TABLE plates IS '板块/行业信息'")
    con.execute("""
        CREATE TABLE IF NOT EXISTS stooq_local_all_us_stocks (
            code        VARCHAR,
            market      VARCHAR,
            ktype       VARCHAR,
            datetime    TIMESTAMP,
            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            volume      DOUBLE,
            turnover    DOUBLE,
            turnover_amount DOUBLE,
            type        VARCHAR,
            avg_turnover_60d DOUBLE,
            pct_chg_60d DOUBLE,
            source      VARCHAR,
            created_at  TIMESTAMP
        )
    """)
    con.execute("ALTER TABLE stooq_local_all_us_stocks ADD COLUMN IF NOT EXISTS type VARCHAR")
    con.execute("ALTER TABLE stooq_local_all_us_stocks ADD COLUMN IF NOT EXISTS market VARCHAR")
    con.execute("COMMENT ON TABLE stooq_local_all_us_stocks IS 'Stooq全量美股日线数据'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.code IS '股票代码'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.market IS '市场: us'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.ktype IS 'K线周期: 1D'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.datetime IS '日期'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.open IS '开盘价'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.high IS '最高价'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.low IS '最低价'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.close IS '收盘价'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.volume IS '成交量'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.turnover IS '成交额(数据源原生)'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.turnover_amount IS '估算成交额(收盘价×成交量)'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.type IS '股票类型: stock/etf'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.avg_turnover_60d IS '60日均成交额'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.pct_chg_60d IS '60日涨跌幅'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.source IS '数据来源'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.created_at IS '入库时间'")
    for _kt, _desc in [("1d", "日线K线数据"), ("1w", "周线K线数据")]:
        con.execute(f"COMMENT ON TABLE klines_{_kt} IS '{_desc}'")

    for _kt in ["1d", "1w"]:
        con.execute(f"""
            CREATE OR REPLACE VIEW klines_{_kt}_sorted AS
            SELECT * FROM klines_{_kt} ORDER BY code, datetime
        """)



def create_views(con):
    """基于 stooq_local_all_us_stocks 创建分析视图"""
    con.execute("""
        CREATE OR REPLACE VIEW v_stooq_all_sorted AS
        SELECT * FROM stooq_local_all_us_stocks
        ORDER BY code, datetime
    """)
    for _drop in ["DROP VIEW IF EXISTS top_turnover_stock_rank", "DROP TABLE IF EXISTS top_turnover_stock_rank"]:
        try: con.execute(_drop)
        except: pass
    con.execute("""
        CREATE OR REPLACE TABLE top_turnover_stock_rank AS
        SELECT rank,
               prev_rank - rank AS rank_change,
               CAST(prev_rank - rank AS DOUBLE) / NULLIF(prev_rank, 0) AS rank_change_pct,
               CONCAT(ROUND(CAST(prev_rank - rank AS DOUBLE) / NULLIF(prev_rank, 0) * 100, 2), '%') AS rank_change_pct_display,
               code, datetime, open, high, low, close, volume, ktype, type, market,
               turnover_amount, avg_turnover_60d, pct_chg_60d,
               CONCAT(ROUND(pct_chg_60d * 100, 2), '%') AS pct_chg_60d_display
        FROM (
            SELECT rank,
                   LAG(rank) OVER (PARTITION BY code ORDER BY datetime) AS prev_rank,
                   code, datetime, open, high, low, close, volume, ktype, type, market,
                   turnover_amount, avg_turnover_60d, pct_chg_60d
            FROM (
                SELECT ROW_NUMBER() OVER (PARTITION BY datetime ORDER BY avg_turnover_60d DESC) AS rank,
                       code, datetime, open, high, low, close, volume, ktype, type, market,
                       turnover_amount, avg_turnover_60d, pct_chg_60d
                FROM stooq_local_all_us_stocks
                WHERE (type IS NULL OR type = 'stock')
                  AND avg_turnover_60d IS NOT NULL
            ) r
        ) t
        ORDER BY datetime DESC, avg_turnover_60d DESC
    """)
    con.execute("COMMENT ON TABLE top_turnover_stock_rank IS '60日均成交额排名变动(stock): 全历史每日rank + rank_change + rank_change_pct'")

    for _drop in ["DROP VIEW IF EXISTS top_turnover_etf_rank", "DROP TABLE IF EXISTS top_turnover_etf_rank"]:
        try: con.execute(_drop)
        except: pass
    con.execute("""
        CREATE OR REPLACE TABLE top_turnover_etf_rank AS
        SELECT rank,
               prev_rank - rank AS rank_change,
               CAST(prev_rank - rank AS DOUBLE) / NULLIF(prev_rank, 0) AS rank_change_pct,
               CONCAT(ROUND(CAST(prev_rank - rank AS DOUBLE) / NULLIF(prev_rank, 0) * 100, 2), '%') AS rank_change_pct_display,
               code, datetime, open, high, low, close, volume, ktype, type, market,
               turnover_amount, avg_turnover_60d, pct_chg_60d,
               CONCAT(ROUND(pct_chg_60d * 100, 2), '%') AS pct_chg_60d_display
        FROM (
            SELECT rank,
                   LAG(rank) OVER (PARTITION BY code ORDER BY datetime) AS prev_rank,
                   code, datetime, open, high, low, close, volume, ktype, type, market,
                   turnover_amount, avg_turnover_60d, pct_chg_60d
            FROM (
                SELECT ROW_NUMBER() OVER (PARTITION BY datetime ORDER BY avg_turnover_60d DESC) AS rank,
                       code, datetime, open, high, low, close, volume, ktype, type, market,
                       turnover_amount, avg_turnover_60d, pct_chg_60d
                FROM stooq_local_all_us_stocks
                WHERE type = 'etf'
                  AND avg_turnover_60d IS NOT NULL
            ) r
        ) t
        ORDER BY datetime DESC, avg_turnover_60d DESC
    """)
    con.execute("COMMENT ON TABLE top_turnover_etf_rank IS '60日均成交额排名变动(etf): 全历史每日rank + rank_change + rank_change_pct'")
    con.execute("""
        CREATE OR REPLACE VIEW top_gainers_200 AS
        SELECT code, datetime, open, high, low, close, volume, ktype, type, market,
               turnover_amount, avg_turnover_60d, pct_chg_60d,
               CONCAT(ROUND(pct_chg_60d * 100, 2), '%') AS pct_chg_60d_display
        FROM stooq_local_all_us_stocks
        WHERE datetime = (SELECT MAX(datetime) FROM stooq_local_all_us_stocks)
          AND pct_chg_60d IS NOT NULL
        ORDER BY pct_chg_60d DESC
        LIMIT 200
    """)
    # 视图中文描述
    con.execute("COMMENT ON VIEW v_stooq_all_sorted IS 'Stooq全量美股(按代码日期排序)'")
    con.execute("COMMENT ON VIEW top_gainers_200 IS '最新日期60日涨跌幅TOP200'")
    for _kt, _desc in [("1d", "日线"), ("1w", "周线")]:
        con.execute(f"COMMENT ON VIEW klines_{_kt}_sorted IS '{_desc}K线数据(按代码日期排序)'")


def list_all(con):
    """列出所有表和视图"""
    names = ["watchlist",
             "klines_1d", "klines_1w",
             "v_klines_1d", "v_klines_1w"]
    for name in names:
        _type = "T" if "rankings" in name else "V"
        try:
            cnt = con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
            print(f"  [{_type}] {name}: {cnt}")
        except Exception:
            pass


def run_query(con, query):
    try:
        result = con.execute(query)
        if result.description:
            df = result.fetchdf()
            print(df.to_string(max_rows=50))
        else:
            print("查询执行成功")
    except Exception as e:
        print(f"查询失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="执行 SQL 查询")
    parser.add_argument("--list", action="store_true", help="列出所有视图")
    parser.add_argument("--reset", action="store_true", help="重建")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = duckdb.connect(DB_PATH)

    if args.reset:
        for tbl in ["watchlist", "turnover_rankings", "stooq_local_all_us_stocks",
                     "klines_1d", "klines_1w", "klines_60m",
                     "klines_1d_sorted", "klines_1w_sorted", "klines_60m_sorted",
                     "v_klines_1d", "v_klines_1w", "v_klines_60m",
                     "top_stocks", "top_stocks_all",
                     "top_turnover_200", "top_turnover_etf_50", "top_turnover_stock_200",
                     "v_backtest_1w", "v_backtest_1d",
                     "v_scores_1w", "v_scores_1d",
                     "v_stability_1w", "v_stability_1d",
                     "klines", "trades"]:
            try: conn.execute(f'DROP TABLE IF EXISTS "{tbl}"')
            except: pass
            try: conn.execute(f'DROP VIEW IF EXISTS "{tbl}"')
            except: pass

    create_tables(conn)
    update_watchlist(conn)
    create_views(conn)
    conn.commit()

    print(f"\n数据库: {DB_PATH}")
    list_all(conn)

    if args.query:
        print(f"\n=== 查询: {args.query} ===")
        run_query(conn, args.query)

    conn.close()
