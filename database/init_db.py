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
    --reset 会自动重建所有 klines、watchlist、top_stocks 等表
"""
import os
import glob
import argparse
from datetime import datetime

import duckdb
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "market.duckdb")


def update_watchlist(con):
    """合并 symbols.csv + top_stocks 去重后写入 watchlist 表"""
    import pandas as pd

    # 1. 读取 symbols.csv → 手动添加
    _manual = set()
    _symbol_path = os.path.join(PROJECT_ROOT, "symbols", "symbols.csv")
    if os.path.exists(_symbol_path):
        for c in pd.read_csv(_symbol_path)["code"].dropna():
            _manual.add(c.strip())

    # 2. 读取 top_stocks → 成交额排名
    _ranked = set()
    try:
        for row in con.execute("SELECT DISTINCT code FROM top_stocks").fetchall():
            _ranked.add(str(row[0]))
    except Exception:
        pass

    # 3. 合并所有代码并确定来源
    _all_codes = _manual | _ranked
    _market_of = {}
    for code in _all_codes:
        if code.startswith("CC."):
            _market_of[code] = "cc"
        elif code.startswith(("SH.", "SZ.")):
            _market_of[code] = "cn"
        elif code.startswith("US."):
            _market_of[code] = "us"

    # 4. 按市场排序写入（us → cc → cn，同市场内按代码升序）
    _mkt_order = {"us": 0, "cc": 1, "cn": 2}
    con.execute("DELETE FROM watchlist")
    for code in sorted(_all_codes, key=lambda c: (_mkt_order.get(_market_of.get(c, ""), 9), c)):
        mkt = _market_of.get(code)
        if not mkt:
            continue
        src = []
        if code in _manual:
            src.append("手动添加")
        if code in _ranked:
            src.append("成交额排名")
        con.execute(
            'INSERT INTO watchlist(code, market, source, created_at) VALUES (?, ?, ?, ?)',
            [code, mkt, ",".join(src), datetime.now()],
        )
    print(f"  watchlist: {len(_all_codes)} 只")


def create_tables(con):
    """建表"""
    con.execute("""
        CREATE TABLE IF NOT EXISTS top_stocks (
            market      VARCHAR,
            code        VARCHAR,
            rank        INTEGER,
            name        VARCHAR,
            price       DOUBLE,
            turnover    DOUBLE,
            date        DATE,
            created_at  TIMESTAMP,
            PRIMARY KEY (market, code, date)
        )
    """)
    for _col, _desc in [
        ("market", "市场: us/cn"), ("code", "股票代码"), ("rank", "成交额排名"),
        ("name", "股票名称"), ("price", "最新价"), ("turnover", "成交额(亿元)"),
        ("date", "数据日期"), ("created_at", "添加时间(精确到秒)"),
    ]:
        con.execute(f"COMMENT ON COLUMN top_stocks.{_col} IS '{_desc}'")

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
    con.execute("COMMENT ON COLUMN watchlist.source IS '来源: 手动添加/成交额排名/手动添加,成交额排名'")
    con.execute("COMMENT ON COLUMN watchlist.created_at IS '添加时间(精确到秒)'")

    # K 线数据表（按周期分表）
    for _kt, _kt_desc in [("1d", "日线"), ("1w", "周线"), ("60m", "60分钟")]:
        con.execute(f"DROP TABLE IF EXISTS klines_{_kt}")
        con.execute(f"""
            CREATE TABLE klines_{_kt} (
                code        VARCHAR,
                datetime    DATE,
                open        DOUBLE,
                high        DOUBLE,
                low         DOUBLE,
                close       DOUBLE,
                volume      DOUBLE,
                turnover    DOUBLE,
                market      VARCHAR,
                ktype       VARCHAR,
                source      VARCHAR,
                turnover_amount DOUBLE,
                created_at  TIMESTAMP,
                PRIMARY KEY (code, datetime)
            )
        """)
        con.execute(f"ALTER TABLE klines_{_kt} ADD COLUMN IF NOT EXISTS created_at TIMESTAMP")
        con.execute(f"ALTER TABLE klines_{_kt} ADD COLUMN IF NOT EXISTS source VARCHAR")
        con.execute(f"ALTER TABLE klines_{_kt} ADD COLUMN IF NOT EXISTS turnover_amount DOUBLE")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.code IS '股票代码'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.created_at IS '添加时间(精确到秒)'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.datetime IS 'K线日期'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.open IS '开盘价'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.high IS '最高价'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.low IS '最低价'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.close IS '收盘价'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.volume IS '成交量'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.market IS '市场: us/cn/cc'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.ktype IS 'K线周期: 1D/1W/60m'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.source IS '数据来源: futu/stooq/baostock/binance'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.turnover IS '成交额(数据源原生)'")
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.turnover_amount IS '估算成交额(收盘价×成交量)'")

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
    con.execute("COMMENT ON TABLE top_stocks IS '成交额排名标的库'")
    con.execute("COMMENT ON TABLE watchlist IS '监控标的库(合并symbols.csv+top_stocks)'")
    con.execute("COMMENT ON TABLE plates IS '板块/行业信息'")
    con.execute("""
        CREATE TABLE IF NOT EXISTS stooq_local_all_us_stocks (
            code        VARCHAR,
            datetime    DATE,
            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            volume      DOUBLE,
            ktype       VARCHAR,
            type        VARCHAR,
            market      VARCHAR,
            turnover_amount DOUBLE,
            avg_turnover_5d DOUBLE,
            avg_turnover_10d DOUBLE,
            avg_turnover_20d DOUBLE,
            avg_turnover_60d DOUBLE,
            pct_chg_5d DOUBLE,
            pct_chg_10d DOUBLE,
            pct_chg_20d DOUBLE,
            pct_chg_60d DOUBLE,
            pct_chg_120d DOUBLE,
            pct_chg_250d DOUBLE,
            pct_chg_ytd DOUBLE,
            PRIMARY KEY (code, datetime)
        )
    """)
    con.execute("ALTER TABLE stooq_local_all_us_stocks ADD COLUMN IF NOT EXISTS type VARCHAR")
    con.execute("ALTER TABLE stooq_local_all_us_stocks ADD COLUMN IF NOT EXISTS market VARCHAR")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.ktype IS 'K线周期: 1D'")
    con.execute("COMMENT ON TABLE stooq_local_all_us_stocks IS 'Stooq全量美股日线数据'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.turnover_amount IS '估算成交额(收盘价×成交量)'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.type IS '股票类型: stock/etf'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.market IS '市场: us'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.avg_turnover_5d IS '5日均成交额'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.avg_turnover_10d IS '10日均成交额'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.avg_turnover_20d IS '20日均成交额'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.avg_turnover_60d IS '60日均成交额'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.pct_chg_5d IS '5日涨跌幅'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.pct_chg_10d IS '10日涨跌幅'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.pct_chg_20d IS '20日涨跌幅'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.pct_chg_60d IS '60日涨跌幅'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.pct_chg_120d IS '120日涨跌幅'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.pct_chg_250d IS '250日涨跌幅'")
    con.execute("COMMENT ON COLUMN stooq_local_all_us_stocks.pct_chg_ytd IS '年初至今涨跌幅'")
    for _kt, _desc in [("1d", "日线K线数据"), ("1w", "周线K线数据"), ("60m", "60分钟K线数据")]:
        con.execute(f"COMMENT ON TABLE klines_{_kt} IS '{_desc}'")

    for _kt in ["1d", "1w", "60m"]:
        con.execute(f"""
            CREATE OR REPLACE VIEW klines_{_kt}_sorted AS
            SELECT * FROM klines_{_kt} ORDER BY code, datetime
        """)

    con.execute("""
        CREATE OR REPLACE VIEW top_stocks_all AS
        SELECT * FROM top_stocks
        ORDER BY market ASC, rank ASC, date DESC
        LIMIT 1000000
    """)


def import_turnover(con):
    """导入成交额排名到表中"""
    csv_dir = os.path.join(PROJECT_ROOT, "symbols")
    for mkt_suffix in ["", "_cn"]:
        files = sorted(glob.glob(os.path.join(csv_dir, f"top_turnover{mkt_suffix}_*.csv")))
        for f in files:
            date_str = os.path.basename(f).split("_")[-1].replace(".csv", "")
            mkt = "cn" if mkt_suffix else "us"
            try:
                dt = datetime.strptime(date_str, "%Y%m%d").date()
            except ValueError:
                continue
            df = pd.read_csv(f)
            df["date"] = dt
            df["market"] = mkt
            con.execute("CREATE OR REPLACE TEMP TABLE _tmp AS SELECT * FROM df")
            con.execute(f"DELETE FROM top_stocks WHERE market = '{mkt}'")
            con.execute('''
                INSERT INTO top_stocks (market, code, rank, name, price, turnover, date, created_at)
                SELECT market, "代码", "排名", "名称", "最新价", "成交额(亿元)", date, CURRENT_TIMESTAMP FROM _tmp
            ''')


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
    con.execute("COMMENT ON VIEW top_stocks_all IS '成交额排名(按市场排名日期排序)'")
    for _kt, _desc in [("1d", "日线"), ("1w", "周线"), ("60m", "60分钟")]:
        con.execute(f"COMMENT ON VIEW klines_{_kt}_sorted IS '{_desc}K线数据(按代码日期排序)'")


def list_all(con):
    """列出所有表和视图"""
    names = ["watchlist", "top_stocks",
             "klines_1d", "klines_1w", "klines_60m",
             "v_klines_1d", "v_klines_1w", "v_klines_60m"]
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
        for tbl in ["watchlist", "top_stocks", "turnover_rankings", "stooq_local_all_us_stocks",
                     "klines_1d", "klines_1w", "klines_60m",
                     "klines_1d_sorted", "klines_1w_sorted", "klines_60m_sorted",
                     "v_klines_1d", "v_klines_1w", "v_klines_60m",
                     "v_backtest_1w", "v_backtest_1d",
                     "v_scores_1w", "v_scores_1d",
                     "v_stability_1w", "v_stability_1d",
                     "klines", "trades"]:
            try: conn.execute(f'DROP TABLE IF EXISTS "{tbl}"')
            except: pass
            try: conn.execute(f'DROP VIEW IF EXISTS "{tbl}"')
            except: pass

    create_tables(conn)
    import_turnover(conn)
    update_watchlist(conn)
    create_views(conn)
    conn.commit()

    print(f"\n数据库: {DB_PATH}")
    list_all(conn)

    if args.query:
        print(f"\n=== 查询: {args.query} ===")
        run_query(conn, args.query)

    conn.close()
