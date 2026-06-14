"""
导入 Stooq 全量美股数据到 DuckDB
==================================
从 data_uscncc/data/daily/us/ 读取所有 .txt 文件，
写入 stooq_local_all_us_stocks 表（含 turnover_amount 估算成交额）。

用法:
    python import_stooq_all.py
"""
import os
import time
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STOOQ_DIR = os.path.join(PROJECT_ROOT, "data_uscncc", "data", "daily", "us")
DB_PATH = os.path.join(PROJECT_ROOT, "database", "market.duckdb")


def main():
    import glob
    all_files = glob.glob(os.path.join(STOOQ_DIR, "**", "*.txt"), recursive=True)
    print(f"共扫描到 {len(all_files)} 个 Stooq 文件")

    if not all_files:
        return

    con = duckdb.connect(DB_PATH)
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
    con.execute("DELETE FROM stooq_local_all_us_stocks")

    # 分批处理（每批一个市场目录），DuckDB 直接读 CSV
    total_rows = 0
    errors = 0
    for _mkt_dir in sorted(glob.glob(os.path.join(STOOQ_DIR, "*"))):
        if not os.path.isdir(_mkt_dir):
            continue
        mkt_name = os.path.basename(_mkt_dir)
        pattern = os.path.join(_mkt_dir, "**", "*.txt")
        files = glob.glob(pattern, recursive=True)
        if not files:
            continue

        _type = "etf" if "etfs" in mkt_name else "stock"
        print(f"  [{mkt_name}] {len(files)} 个文件...", end=" ", flush=True)
        try:
            con.execute(f"""
                INSERT OR REPLACE INTO stooq_local_all_us_stocks (code, datetime, open, high, low, close, volume, ktype, type, market, turnover_amount)
                SELECT
                    'US.' || replace("<TICKER>", '.US', '') AS code,
                    strptime("<DATE>"::VARCHAR, '%Y%m%d')::DATE AS datetime,
                    "<OPEN>"::DOUBLE AS open,
                    "<HIGH>"::DOUBLE AS high,
                    "<LOW>"::DOUBLE AS low,
                    "<CLOSE>"::DOUBLE AS close,
                    "<VOL>"::DOUBLE AS volume,
                    '1D' AS ktype,
                    '{_type}' AS type,
                    'us' AS market,
                    "<CLOSE>"::DOUBLE * "<VOL>"::DOUBLE AS turnover_amount
                FROM read_csv_auto('{pattern}', header=true, union_by_name=true)
                ORDER BY code, datetime
            """)
            rows = con.execute("SELECT count(*) FROM stooq_local_all_us_stocks").fetchone()[0] - total_rows
            total_rows += rows
            print(f"{rows:,} 行")
        except Exception as e:
            errors += 1
            print(f"失败: {e}")

    # 全局排序（按 code, datetime）
    print("全局排序...", end=" ", flush=True)
    con.execute("""
        CREATE TABLE stooq_tmp AS
        SELECT * FROM stooq_local_all_us_stocks
        ORDER BY code, datetime
    """)
    con.execute("DROP TABLE stooq_local_all_us_stocks")
    con.execute("ALTER TABLE stooq_tmp RENAME TO stooq_local_all_us_stocks")
    print("完成")

    # 计算成交额均值 + 涨跌幅
    print("计算技术指标...", end=" ", flush=True)
    con.execute("""
        UPDATE stooq_local_all_us_stocks t
        SET
            avg_turnover_5d  = w.a5,
            avg_turnover_10d = w.a10,
            avg_turnover_20d = w.a20,
            avg_turnover_60d = w.a60,
            pct_chg_5d   = w.c5,
            pct_chg_10d  = w.c10,
            pct_chg_20d  = w.c20,
            pct_chg_60d  = w.c60,
            pct_chg_120d = w.c120,
            pct_chg_250d = w.c250,
            pct_chg_ytd  = w.ytd
        FROM (
            SELECT code, datetime,
                CASE WHEN COUNT(turnover_amount) OVER w5  >= 5  THEN AVG(turnover_amount) OVER w5  END AS a5,
                CASE WHEN COUNT(turnover_amount) OVER w10 >= 10 THEN AVG(turnover_amount) OVER w10 END AS a10,
                CASE WHEN COUNT(turnover_amount) OVER w20 >= 20 THEN AVG(turnover_amount) OVER w20 END AS a20,
                CASE WHEN COUNT(turnover_amount) OVER w60 >= 60 THEN AVG(turnover_amount) OVER w60 END AS a60,
                (close - LAG(close, 5)  OVER w) / NULLIF(LAG(close, 5)  OVER w, 0) AS c5,
                (close - LAG(close, 10) OVER w) / NULLIF(LAG(close, 10) OVER w, 0) AS c10,
                (close - LAG(close, 20) OVER w) / NULLIF(LAG(close, 20) OVER w, 0) AS c20,
                (close - LAG(close, 60) OVER w) / NULLIF(LAG(close, 60) OVER w, 0) AS c60,
                (close - LAG(close, 120) OVER w) / NULLIF(LAG(close, 120) OVER w, 0) AS c120,
                (close - LAG(close, 250) OVER w) / NULLIF(LAG(close, 250) OVER w, 0) AS c250,
                (close - FIRST_VALUE(close) OVER (PARTITION BY code, YEAR(datetime) ORDER BY datetime))
                    / NULLIF(FIRST_VALUE(close) OVER (PARTITION BY code, YEAR(datetime) ORDER BY datetime), 0) AS ytd
            FROM stooq_local_all_us_stocks
            WINDOW w   AS (PARTITION BY code ORDER BY datetime),
                   w5  AS (PARTITION BY code ORDER BY datetime ROWS BETWEEN 4  PRECEDING AND CURRENT ROW),
                   w10 AS (PARTITION BY code ORDER BY datetime ROWS BETWEEN 9  PRECEDING AND CURRENT ROW),
                   w20 AS (PARTITION BY code ORDER BY datetime ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
                   w60 AS (PARTITION BY code ORDER BY datetime ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
        ) w
        WHERE t.code = w.code AND t.datetime = w.datetime
    """)
    print("完成")

    con.close()
    print(f"\n导入完成: {total_rows:,} 行, {len(all_files)} 文件, {errors} 错误")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"耗时: {time.time() - t0:.0f} 秒")
