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

    # 4. 确定市场（中文）
    _market_of = {}
    _mkt_order = {}
    for code in _sources:
        if code.startswith("CC."):
            _market_of[code] = "加密货币"; _mkt_order[code] = 1
        elif code.startswith(("SH.", "SZ.")):
            _market_of[code] = "A股"; _mkt_order[code] = 2
        elif code.startswith("US."):
            _market_of[code] = "美股"; _mkt_order[code] = 0

    # 5. 按市场排序写入（美股 → 加密货币 → A股）
    con.execute("DELETE FROM watchlist")
    for code in sorted(_sources.keys(), key=lambda c: _mkt_order.get(c, 9)):
        mkt = _market_of.get(code)
        if not mkt:
            continue
        con.execute(
            'INSERT INTO watchlist(code, stock_name, market, source, created_at) VALUES (?, ?, ?, ?, ?)',
            [code, "", mkt, ",".join(sorted(_sources[code])), datetime.now()],
        )
    print(f"  watchlist: {len(_sources)} 只")


def create_tables(con):
    """建表"""
    con.execute("DROP TABLE IF EXISTS watchlist")
    con.execute("""
        CREATE TABLE watchlist (
            code        VARCHAR,
            stock_name  VARCHAR,
            market      VARCHAR,
            source      VARCHAR,
            created_at  TIMESTAMP
        )
    """)
    con.execute("COMMENT ON COLUMN watchlist.code IS '股票代码'")
    con.execute("COMMENT ON COLUMN watchlist.stock_name IS '股票名称'")
    con.execute("COMMENT ON COLUMN watchlist.market IS '市场: 美股/A股/加密货币'")
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
        con.execute(f"COMMENT ON COLUMN klines_{_kt}.market IS '市场: 美股/A股/加密货币'")
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
            plates      VARCHAR,
            created_at  TIMESTAMP
        )
    """)
    con.execute("COMMENT ON COLUMN plates.code IS '股票代码'")
    con.execute("COMMENT ON COLUMN plates.stock_name IS '股票名称'")
    con.execute("COMMENT ON COLUMN plates.market IS '市场: 美股/A股'")
    con.execute("ALTER TABLE plates ADD COLUMN IF NOT EXISTS created_at TIMESTAMP")
    con.execute("COMMENT ON COLUMN plates.plates IS '所属板块(逗号分隔)'")
    con.execute("COMMENT ON COLUMN plates.created_at IS '入库时间'")

    # ── 回测输出表 ──
    # 2. backtest_summary：回测指标
    con.execute("DROP TABLE IF EXISTS backtest_summary")
    con.execute("""
        CREATE TABLE backtest_summary (
            code            VARCHAR,
            stock_name      VARCHAR,
            market          VARCHAR,
            ktype           VARCHAR,
            ma              INTEGER,
            window_label    VARCHAR,
            effective_range VARCHAR,
            total_bars      INTEGER,
            total_return    DOUBLE,
            cagr            DOUBLE,
            buy_hold_return DOUBLE,
            excess_return   DOUBLE,
            avg_trade_return DOUBLE,
            max_drawdown    DOUBLE,
            sharpe_ratio    DOUBLE,
            calmar_ratio    DOUBLE,
            trade_count     INTEGER,
            win_rate        DOUBLE,
            profit_factor   DOUBLE,
            payoff_ratio    DOUBLE,
            avg_win         DOUBLE,
            avg_win_pct     DOUBLE,
            avg_loss        DOUBLE,
            avg_loss_pct    DOUBLE,
            max_win         DOUBLE,
            max_loss        DOUBLE,
            max_win_streak  INTEGER,
            max_loss_streak INTEGER,
            avg_hold_days   DOUBLE,
            avg_hold_bars   DOUBLE,
            initial_cash    DOUBLE,
            final_cash      DOUBLE,
            strategy_score  DOUBLE,
            created_at      TIMESTAMP
        )
    """)
    con.execute("COMMENT ON TABLE backtest_summary IS '回测指标汇总'")
    for _c, _d in [("code","股票代码"),("stock_name","股票名称"),("market","市场"),
        ("ktype","K线周期"),("ma","均线周期"),("window_label","窗口标签"),
        ("effective_range","窗口内有效数据周期"),("total_bars","窗口内有效数据K线数"),
        ("total_return","收益率"),("cagr","年化收益率"),("buy_hold_return","买入持有收益率"),
        ("excess_return","超额收益率"),("avg_trade_return","平均每笔收益率"),
        ("max_drawdown","最大回撤"),("sharpe_ratio","夏普比率"),("calmar_ratio","卡尔玛比率"),
        ("trade_count","交易次数"),("win_rate","盈利交易率"),("profit_factor","盈利因子"),
        ("payoff_ratio","盈亏比"),("avg_win","平均盈利"),("avg_win_pct","平均盈利比"),
        ("avg_loss","平均亏损"),("avg_loss_pct","平均亏损比"),("max_win","最大单笔盈利"),
        ("max_loss","最大单笔亏损"),("max_win_streak","最大连续盈利次数"),
        ("max_loss_streak","最大连续亏损次数"),("avg_hold_days","平均持仓天数"),
        ("avg_hold_bars","平均持仓K线数"),("initial_cash","初始资金"),("final_cash","最终资金"),
        ("strategy_score","策略评分")]:
        con.execute(f"COMMENT ON COLUMN backtest_summary.{_c} IS '{_d}'")

    # 3. backtest_scores：策略评分 + 排名
    con.execute("DROP TABLE IF EXISTS backtest_scores")
    con.execute("""
        CREATE TABLE backtest_scores (
            code            VARCHAR,
            stock_name      VARCHAR,
            market          VARCHAR,
            ktype           VARCHAR,
            ma              INTEGER,
            window_label    VARCHAR,
            strategy_score  DOUBLE,
            window_rank     INTEGER,
            created_at      TIMESTAMP
        )
    """)
    con.execute("COMMENT ON TABLE backtest_scores IS '策略评分与排名'")
    for _c, _d in [("code","股票代码"),("stock_name","股票名称"),("market","市场"),
        ("ktype","K线周期"),("ma","均线周期"),("window_label","窗口标签"),
        ("strategy_score","策略评分"),("window_rank","窗口内排名")]:
        con.execute(f"COMMENT ON COLUMN backtest_scores.{_c} IS '{_d}'")

    # 4. backtest_stability：参数稳定性
    con.execute("DROP TABLE IF EXISTS backtest_stability")
    con.execute("""
        CREATE TABLE backtest_stability (
            code            VARCHAR,
            stock_name      VARCHAR,
            market          VARCHAR,
            ktype           VARCHAR,
            ma              INTEGER,
            window_label    VARCHAR,
            window_count    INTEGER,
            win_window_count INTEGER,
            win_window_pct  DOUBLE,
            avg_score_rank  DOUBLE,
            rank_first_count INTEGER,
            rank_top3_pct   DOUBLE,
            score_rank_std  DOUBLE,
            avg_cagr        DOUBLE,
            cagr_std        DOUBLE,
            stability_score DOUBLE,
            is_best         VARCHAR,
            created_at      TIMESTAMP
        )
    """)
    con.execute("COMMENT ON TABLE backtest_stability IS '参数稳定性分析'")
    for _c, _d in [("code","股票代码"),("stock_name","股票名称"),("market","市场"),
        ("ktype","K线周期"),("ma","均线周期"),("window_label","窗口标签"),
        ("window_count","窗口数量"),("win_window_count","盈利窗口数量"),
        ("win_window_pct","盈利窗口占比"),("avg_score_rank","策略评分排名平均值"),
        ("rank_first_count","策略评分排名第一次数"),("rank_top3_pct","策略评分排名Top3占比"),
        ("score_rank_std","策略评分排名标准差"),("avg_cagr","年化收益率平均值"),
        ("cagr_std","年化收益率标准差"),("stability_score","参数稳定性评分"),
        ("is_best","是否最优")]:
        con.execute(f"COMMENT ON COLUMN backtest_stability.{_c} IS '{_d}'")

    # 5. backtest_signals：最新信号
    con.execute("DROP TABLE IF EXISTS backtest_signals")
    con.execute("""
        CREATE TABLE backtest_signals (
            code            VARCHAR,
            stock_name      VARCHAR,
            market          VARCHAR,
            ktype           VARCHAR,
            ma              INTEGER,
            signal_date     DATE,
            close           DOUBLE,
            ha_close        DOUBLE,
            ha_ma           DOUBLE,
            direction       INTEGER,
            signal          VARCHAR,
            signal_time     TIMESTAMP,
            signal_close    DOUBLE,
            signal_confirmed VARCHAR,
            last_signal      VARCHAR,
            last_signal_time TIMESTAMP,
            last_signal_close DOUBLE,
            bars_since_last_signal INTEGER,
            change_since_last_signal DOUBLE,
            holding_daily_return DOUBLE,
            created_at      TIMESTAMP
        )
    """)
    con.execute("COMMENT ON TABLE backtest_signals IS '最新信号'")
    for _c, _d in [("code","股票代码"),("stock_name","股票名称"),("market","市场"),
        ("ktype","K线周期"),("ma","最优均线周期"),("signal_date","信号日期"),
        ("close","收盘价"),("ha_close","HA收盘价"),("ha_ma","HA均线值"),
        ("direction","趋势方向"),("signal","最新信号"),("signal_time","最新信号时间"),
        ("signal_close","最新信号收盘价"),("signal_confirmed","最新信号确认"),
        ("last_signal","历史信号"),("last_signal_time","历史信号时间"),
        ("last_signal_close","历史信号收盘价"),
        ("bars_since_last_signal","距离历史信号已过K线数"),
        ("change_since_last_signal","距离历史信号收盘价涨跌幅"),
        ("holding_daily_return","持仓日化收益率")]:
        con.execute(f"COMMENT ON COLUMN backtest_signals.{_c} IS '{_d}'")

    # 6. backtest_stats：策略回测统计（逐K线记录）
    con.execute("DROP TABLE IF EXISTS backtest_stats")
    con.execute("""
        CREATE TABLE backtest_stats (
            code                    VARCHAR,
            stock_name              VARCHAR,
            market                  VARCHAR,
            ktype                   VARCHAR,
            window_label            VARCHAR,
            window_count            INTEGER,
            datetime                TIMESTAMP,
            open                    DOUBLE,
            high                    DOUBLE,
            low                     DOUBLE,
            close                   DOUBLE,
            volume                  DOUBLE,
            turnover                DOUBLE,
            turnover_amount         DOUBLE,
            source                  VARCHAR,
            ha_close                DOUBLE,
            ma_len                  INTEGER,
            ha_ma_value             DOUBLE,
            trend_direction         VARCHAR,
            signal                  VARCHAR,
            trade_id                INTEGER,
            trade_action            VARCHAR,
            trade_price             DOUBLE,
            trade_price_after_slippage DOUBLE,
            trade_shares            DOUBLE,
            slippage                DOUBLE,
            trade_amount            DOUBLE,
            commission              DOUBLE,
            actual_trade_amount     DOUBLE,
            available_cash          DOUBLE,
            held_shares             DOUBLE,
            trade_status            VARCHAR,
            close_price             DOUBLE,
            close_price_after_slippage DOUBLE,
            close_shares            DOUBLE,
            close_slippage          DOUBLE,
            close_amount            DOUBLE,
            close_commission        DOUBLE,
            actual_close_amount     DOUBLE,
            close_pnl               DOUBLE,
            close_type              VARCHAR,
            close_pnl_type                VARCHAR,
            cash_before_trade       DOUBLE,
            cash_after_trade        DOUBLE,
            account_value           DOUBLE,
            account_value_change    DOUBLE,
            account_value_change_pct DOUBLE,
            change_from_initial     DOUBLE,
            change_from_initial_pct DOUBLE,
            created_at              TIMESTAMP
        )
    """)
    con.execute("COMMENT ON TABLE backtest_stats IS '策略回测统计（逐K线记录）'")
    for _c, _d in [("code","股票代码"),("stock_name","股票名称"),("market","市场"),
        ("ktype","K线周期"),("window_label","窗口标签"),("window_count","窗口数量"),("datetime","K线时间"),("open","开盘价"),("high","最高价"),
        ("low","最低价"),("close","收盘价"),("volume","成交量"),("turnover","成交额(数据源原生)"),
        ("turnover_amount","估算成交额"),("source","数据来源"),
        ("ha_close","平均K线收盘价"),("ma_len","均线周期"),
        ("ha_ma_value","根据平均K线收盘价计算的均线周期值"),
        ("trend_direction","趋势方向"),("signal","信号"),
        ("trade_id","交易编号"),
        ("trade_action","交易动作"),("trade_price","交易价格"),("trade_price_after_slippage","扣除滑点后的成交价"),
        ("available_cash","可用现金"),("trade_shares","交易股数"),
        ("slippage","交易滑点"),
        ("trade_amount","交易金额(扣除滑点后的成交价×交易股数)"),
        ("commission","佣金(交易金额×佣金费率)"),
        ("actual_trade_amount","实际发生交易金额(交易金额+佣金)"),
        ("slippage","交易滑点"),
        ("held_shares","持有股数"),("trade_status","交易状态: 持仓中/已平仓"),
        ("close_price","平仓交易价格"),
        ("close_price_after_slippage","平仓扣除滑点后的交易价格"),
        ("close_shares","平仓股数"),
        ("close_slippage","平仓滑点"),
        ("close_amount","平仓交易金额"),
        ("close_commission","平仓交易佣金"),
        ("close_commission","平仓交易佣金"),
        ("actual_close_amount","实际平仓交易金额(close_price_after_slippage×close_shares+close_commission)"),
        ("close_pnl","平仓交易盈利"),
        ("close_type","平仓类型: 虚拟平仓/真实平仓"),
        ("close_pnl_type","盈亏类型: 盈利/亏损"),
        ("cash_before_trade","交易前可用现金"),
        ("cash_after_trade","交易后可用现金"),
        ("account_value","账户价值"),
        ("account_value_change","账户价值变动数"),
        ("account_value_change_pct","账户价值变动比"),
        ("change_from_initial","自初始账户价值变动数"),
        ("change_from_initial_pct","自初始账户价值变动比"),
        ("trade_id","交易编号"),
        ("trade_status","交易状态: 持仓中/已平仓")]:
        con.execute(f"COMMENT ON COLUMN backtest_stats.{_c} IS '{_d}'")

    # ── 交易记录表（从 backtest_stats 派生） ──
    con.execute("DROP TABLE IF EXISTS backtest_trades")
    con.execute("""
        CREATE TABLE backtest_trades (
            code                    VARCHAR,
            stock_name              VARCHAR,
            market                  VARCHAR,
            ktype                   VARCHAR,
            window_label            VARCHAR,
            datetime                TIMESTAMP,
            ma_len                  INTEGER,
            trade_id                INTEGER,
            trade_action            VARCHAR,
            trade_price_after_slippage DOUBLE,
            trade_shares            DOUBLE,
            slippage                DOUBLE,
            trade_amount            DOUBLE,
            commission              DOUBLE,
            actual_trade_amount     DOUBLE,
            trade_status            VARCHAR,
            close_pnl               DOUBLE,
            close_type              VARCHAR,
            close_pnl_type                VARCHAR,
            cash_before_trade       DOUBLE,
            cash_after_trade        DOUBLE,
            available_cash          DOUBLE,
            created_at              TIMESTAMP
        )
    """)
    con.execute("COMMENT ON TABLE backtest_trades IS '交易记录（仅含 trade_action 非空的行）'")
    for _c, _d in [("code","股票代码"),("stock_name","股票名称"),("market","市场"),
        ("ktype","K线周期"),("window_label","窗口标签"),("datetime","K线时间"),
        ("ma_len","均线周期"),("trade_id","交易编号"),("trade_action","交易动作"),
        ("trade_price_after_slippage","扣除滑点后的成交价"),
        ("available_cash","可用现金"),("trade_shares","交易股数"),
        ("slippage","交易滑点"),
        ("trade_amount","交易金额"),("commission","佣金"),
        ("actual_trade_amount","实际发生交易金额"),("trade_status","交易状态"),
        ("close_pnl","平仓交易盈利"),
        ("close_type","平仓类型: 虚拟平仓/真实平仓"),
        ("close_pnl_type","盈亏类型: 盈利/亏损"),
        ("cash_before_trade","交易前可用现金"),
        ("cash_after_trade","交易后可用现金")]:
        con.execute(f"COMMENT ON COLUMN backtest_trades.{_c} IS '{_d}'")

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
            created_at  TIMESTAMP,
            PRIMARY KEY (code, datetime)
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
             "v_klines_1d", "v_klines_1w",
             "backtest_trades", "backtest_summary", "backtest_scores",
             "backtest_stability", "backtest_signals", "backtest_stats"]
    for name in names:
        try:
            _type = {"BASE TABLE": "T", "VIEW": "V"}.get(
                con.execute(f"SELECT table_type FROM information_schema.tables WHERE table_name = '{name}'").fetchone()[0], "?")
        except Exception:
            _type = "?"
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
        for tbl in ["watchlist", "plates", "turnover_rankings", "stooq_local_all_us_stocks",
                     "backtest_trades", "backtest_summary", "backtest_scores",
                     "backtest_stability", "backtest_signals", "backtest_stats",
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
