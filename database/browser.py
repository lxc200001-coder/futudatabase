"""
DuckDB 数据浏览器（Streamlit）
=============================
用法:
    pip install streamlit
    streamlit run database/browser.py
"""
import os
import duckdb
import pandas as pd
import streamlit as st

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "market.duckdb")


@st.cache_resource
def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)


def load_tables(con):
    names = [
        ("klines", "V", "K 线数据"),
        ("trades", "V", "交易明细"),
        ("turnover_rankings", "T", "成交额排名"),
        ("v_backtest_1w", "V", "回测汇总(周K)"),
        ("v_backtest_1d", "V", "回测汇总(日K)"),
        ("v_scores_1w", "V", "策略评分(周K)"),
        ("v_scores_1d", "V", "策略评分(日K)"),
        ("v_stability_1w", "V", "参数稳定性(周K)"),
        ("v_stability_1d", "V", "参数稳定性(日K)"),
    ]
    return names


def run_query(con, sql, limit=100):
    try:
        result = con.execute(sql)
        if result.description:
            df = result.fetchdf()
            return df.head(limit)
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame({"错误": [str(e)]})


def main():
    st.set_page_config(page_title="DuckDB 数据浏览器", layout="wide")
    st.title("📊 DuckDB 数据浏览器")

    conn = get_conn()
    tables = load_tables(conn)

    # 侧边栏
    st.sidebar.header("表/视图")
    table_names = [f"{'📋' if t[1]=='T' else '👁'} {t[0]} ({t[2]})" for t in tables]
    selected_display = st.sidebar.radio("选择", table_names)
    selected_name = selected_display.split(" ")[1]

    # 表信息
    try:
        cnt = conn.execute(f'SELECT count(*) FROM "{selected_name}"').fetchone()[0]
        st.sidebar.info(f"行数: {cnt:,}")
    except:
        pass

    # 查询区域
    st.sidebar.header("自定义 SQL")
    custom_sql = st.sidebar.text_area(
        "输入 SQL", value=f"SELECT * FROM \"{selected_name}\" LIMIT 100",
        height=120,
    )
    limit = st.sidebar.number_input("限制行数", 10, 10000, 200)
    if st.sidebar.button("执行"):
        sql = custom_sql
    else:
        sql = f'SELECT * FROM "{selected_name}" LIMIT {limit}'

    # 分页
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"📋 {selected_name}")

    # 执行查询
    df = run_query(conn, sql, limit=limit)

    # 显示数据
    if df.empty:
        st.warning("无数据或查询出错")
    else:
        st.dataframe(df, use_container_width=True, height=500)

        # 统计信息
        with st.expander("📈 列信息"):
            col_info = []
            for c in df.columns:
                col_info.append({
                    "列名": c,
                    "类型": str(df[c].dtype),
                    "非空": df[c].notna().sum(),
                    "唯一值": df[c].nunique() if df[c].dtype != "float64" else "-",
                    "最小值": df[c].min() if df[c].dtype in ("float64", "int64") else "-",
                    "最大值": df[c].max() if df[c].dtype in ("float64", "int64") else "-",
                })
            st.dataframe(pd.DataFrame(col_info), use_container_width=True)

        # 导出
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 导出 CSV", csv, f"{selected_name}.csv", "text/csv")

    # 快速查询模板
    st.sidebar.markdown("---")
    st.sidebar.subheader("快速查询")
    quick_queries = [
        "SELECT code, count(*) as bars FROM klines GROUP BY code ORDER BY bars DESC LIMIT 10",
        'SELECT code, "年化收益率", "夏普比率", "最大回撤" FROM v_backtest_1w WHERE ma_period = 60 ORDER BY "年化收益率" DESC LIMIT 20',
        "SELECT code, count(*) as trades_count FROM trades GROUP BY code ORDER BY trades_count DESC LIMIT 10",
        'SELECT code, avg("夏普比率") as avg_sharpe FROM v_backtest_1d GROUP BY code ORDER BY avg_sharpe DESC LIMIT 20',
        "SELECT ktype, count(*) FROM klines GROUP BY ktype",
        "SELECT date, count(*) FROM turnover_rankings GROUP BY date ORDER BY date",
    ]
    for q in quick_queries:
        if st.sidebar.button(q[:50] + "...", help=q):
            st.session_state["query"] = q
            st.rerun()


if __name__ == "__main__":
    main()
