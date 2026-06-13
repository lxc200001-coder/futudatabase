"""
DuckDB 数据库初始化
====================
通过视图直接查询 parquet 文件，无需导入。

用法:
    python database/init_db.py                          # 创建视图
    python database/init_db.py --query "SELECT count(*) FROM klines"
    python database/init_db.py --list                   # 列出所有视图
"""
import os
import glob
import argparse
from datetime import datetime

import duckdb
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "market.duckdb")


def create_tables(con):
    """建表（成交额排名需要实际表，其他用视图）"""
    con.execute("""
        CREATE TABLE IF NOT EXISTS turnover_rankings (
            date            DATE NOT NULL,
            code            VARCHAR NOT NULL,
            rank            INTEGER,
            name            VARCHAR,
            last_price      DOUBLE,
            turnover_amt    DOUBLE,
            market          VARCHAR
        )
    """)
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
            con.execute('''
                INSERT INTO turnover_rankings
                SELECT date, "代码", "排名", "名称", "最新价", "成交额(亿元)", market FROM _tmp
            ''')
            con.execute(f"DELETE FROM top_stocks WHERE market = '{mkt}'")
            con.execute('''
                INSERT INTO top_stocks (market, code, rank, name, price, turnover, date, created_at)
                SELECT market, "代码", "排名", "名称", "最新价", "成交额(亿元)", date, CURRENT_TIMESTAMP FROM _tmp
            ''')


def create_views(con):
    """基于 parquet 文件创建视图"""

    # 1. K 线数据
    _kline_parts = []
    for ktype_dir, ktype_tag in [("1d", "1D"), ("1w", "1W"), ("60m", "60m")]:
        pattern = os.path.join(PROJECT_ROOT, "data_uscncc", ktype_dir, "**", "*.parquet")
        if glob.glob(pattern, recursive=True):
            view_name = f"v_klines_{ktype_tag.lower()}"
            con.execute(f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT *, '{ktype_tag}' AS ktype
                FROM read_parquet('{pattern}', union_by_name=true)
            """)
            _kline_parts.append(view_name)
    if _kline_parts:
        _union = " UNION ALL BY NAME ".join(f"SELECT * FROM {v}" for v in _kline_parts)
        con.execute(f"CREATE OR REPLACE VIEW klines AS {_union}")
        print(f"  klines ← {len(_kline_parts)} 个目录")

    # 2. 回测汇总
    for ktype_dir, ktype_tag in [("1w", "1W"), ("1d", "1D")]:
        files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "results_uscncc", ktype_dir, "*_回测汇总.parquet")))
        if files:
            con.execute(f"""
                CREATE OR REPLACE VIEW v_backtest_{ktype_tag.lower()} AS
                SELECT *, '{ktype_tag}' AS ktype FROM read_parquet('{files[-1]}')
            """)

    # 3. 策略评分矩阵
    for ktype_dir, ktype_tag in [("1w", "1W"), ("1d", "1D")]:
        files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "results_uscncc", ktype_dir, "*_策略评分明细.parquet")))
        if files:
            con.execute(f"""
                CREATE OR REPLACE VIEW v_scores_{ktype_tag.lower()} AS
                SELECT *, '{ktype_tag}' AS ktype FROM read_parquet('{files[-1]}')
            """)

    # 4. 参数稳定性
    for ktype_dir, ktype_tag in [("1w", "1W"), ("1d", "1D")]:
        files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "results_uscncc", ktype_dir, "*_全窗口参数稳定性分析.parquet")))
        if files:
            con.execute(f"""
                CREATE OR REPLACE VIEW v_stability_{ktype_tag.lower()} AS
                SELECT *, '{ktype_tag}' AS ktype FROM read_parquet('{files[-1]}')
            """)

    # 5. 交易明细
    trades_pat = os.path.join(PROJECT_ROOT, "results_uscncc", "**", "*_trades.parquet")
    if glob.glob(trades_pat, recursive=True):
        con.execute(f"""
            CREATE OR REPLACE VIEW trades AS
            SELECT * FROM read_parquet('{trades_pat}', union_by_name=true)
        """)
        print("  trades ← parquet 文件")


def list_all(con):
    """列出所有表和视图"""
    names = ["top_stocks", "turnover_rankings", "klines", "trades",
             "v_backtest_1w", "v_backtest_1d", "v_scores_1w", "v_scores_1d",
             "v_stability_1w", "v_stability_1d"]
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
        for tbl in ["top_stocks", "turnover_rankings", "klines", "trades",
                     "v_klines_1d", "v_klines_1w", "v_klines_60m",
                     "v_backtest_1w", "v_backtest_1d",
                     "v_scores_1w", "v_scores_1d",
                     "v_stability_1w", "v_stability_1d"]:
            try: conn.execute(f'DROP TABLE IF EXISTS "{tbl}"')
            except: pass
            try: conn.execute(f'DROP VIEW IF EXISTS "{tbl}"')
            except: pass

    create_tables(conn)
    import_turnover(conn)
    create_views(conn)
    conn.commit()

    print(f"\n数据库: {DB_PATH}")
    list_all(conn)

    if args.query:
        print(f"\n=== 查询: {args.query} ===")
        run_query(conn, args.query)

    conn.close()
