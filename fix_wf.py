import re

with open('9windows_param_scan_numba_uscncc.py', 'r', encoding='utf-8') as f:
    c = f.read()

old_start = '    wf_trades_list = []\n    wf_signal_info = None\n    wf_params_list = []'
assert old_start in c, "start not found"

# Find the old return statement to locate where the WF block ends
old_end = "                wf_signal_info[\"市场\"] = market_val"
assert old_end in c, "end marker not found"

# Build replacement for the entire section between old_start and the line after old_end
# Find positions
s = c.find(old_start)
e = c.find(old_end, s)
e = c.find('\n', e) + len('\n')

# Extract the old block
old_block = c[s:e]

new_block = '''    wf_trades_list = []
    wf_signal_info = None
    wf_params_list = []
    if window_stability_df is not None and not window_stability_df.empty and len(windows) >= 2:
        _best_rows = window_stability_df[window_stability_df["是否最优"] == "最优"]
        _win_order = sorted(windows, key=lambda x: x[0])
        _train_best = {}
        for _r in _best_rows.itertuples():
            _train_best[_r.窗口] = int(_r.均线周期)

        _test_windows = []
        for _wi in range(len(_win_order) - 1):
            _tw_start = _win_order[_wi][1]
            _tw_end = _win_order[_wi + 1][1]
            _tw_label = f"{_win_order[_wi][1].date()}~{_win_order[_wi + 1][1].date()}"
            _tma = _train_best.get(str(_win_order[_wi][0].date()) + "~" + str(_win_order[_wi][1].date()), None)
            if _tma is not None:
                _test_windows.append((_tw_start, _tw_end, _tw_label, _tma))

        _carry_cash = INITIAL_CASH
        _carry_pos = 0.0
        _carry_entry = 0.0
        for _tw_start, _tw_end, _tw_label, _tma in _test_windows:
            _tw_mask = (pd.to_datetime(df["datetime"]) >= _tw_start) & (pd.to_datetime(df["datetime"]) < _tw_end)
            _df_tw = df[_tw_mask]
            if _df_tw.empty:
                continue
            _tw_close = _df_tw["close"].values.astype(np.float64)
            _tw_datetime = _df_tw["datetime"].values
            _tw_ma = ma_cache[_tma][_tw_mask].values.astype(np.float64)

            _tw_dir = np.zeros(len(_tw_ma), dtype=np.int8)
            _tw_dir[0] = -1
            _tw_dir[1:] = np.where(_tw_ma[1:] > _tw_ma[:-1], 1, -1)
            _tw_buy = np.zeros(len(_tw_ma), dtype=np.bool_)
            _tw_buy[1:] = (_tw_dir[1:] == 1) & (_tw_dir[:-1] == -1)
            _tw_sell = np.zeros(len(_tw_ma), dtype=np.bool_)
            _tw_sell[1:] = (_tw_dir[1:] == -1) & (_tw_dir[:-1] == 1)

            _idx = _test_windows.index((_tw_start, _tw_end, _tw_label, _tma))
            if _idx > 0 and _carry_pos > 0:
                if _tw_sell[0]:
                    _carry_cash += _carry_pos * _tw_close[0]
                    _carry_pos = 0.0
                    _carry_entry = 0.0

            _wf_trades_arr, _wf_equity_arr, _wf_n, _carry_pos, _carry_entry, _carry_cash = (
                _numba_walkforward_backtest(
                    _tw_close, _tw_buy, _tw_sell, _carry_cash, FEE_RATE,
                    _carry_pos, _carry_entry,
                )
            )

            if _wf_n > 0:
                _tw_trades_df, _ = _build_trades_from_arrays(
                    code_val, market_val, _tw_datetime, _tma,
                    _wf_trades_arr, _wf_equity_arr, _wf_n,
                )
                if not _tw_trades_df.empty:
                    _tw_trades_df["测试窗口"] = _tw_label
                    wf_trades_list.append(_tw_trades_df)

            wf_params_list.append({"测试窗口": _tw_label, "均线周期": _tma})

        if wf_trades_list:
            _wf_trades_df = pd.concat(wf_trades_list, ignore_index=True, sort=False)
            _wf_out = out_file.replace(".xlsx", "_wf_trades.parquet")
            os.makedirs(os.path.dirname(_wf_out), exist_ok=True)
            _wf_trades_df.to_parquet(_wf_out)

        if _test_windows:
            _last_tw = _test_windows[-1]
            _last_tw_start, _last_tw_end, _last_tw_label, _last_tw_ma = _last_tw
            _last_mask = (pd.to_datetime(df["datetime"]) >= _last_tw_start) & (pd.to_datetime(df["datetime"]) < _last_tw_end)
            _df_last = df[_last_mask]
            if not _df_last.empty:
                _df_last = _df_last.copy()
                _df_last["ha_close"] = ha_close_full[_last_mask]
                _df_last["ma"] = ma_cache[_last_tw_ma][_last_mask]
                _df_last["dir"] = np.where(_df_last["ma"] > _df_last["ma"].shift(1), 1, -1)
                _df_last["buy"] = (_df_last["dir"] == 1) & (_df_last["dir"].shift(1) == -1)
                _df_last["sell"] = (_df_last["dir"] == -1) & (_df_last["dir"].shift(1) == 1)
                wf_signal_info = get_last_signal_info(_df_last)
                wf_signal_info["K线周期"] = BAR_INTERVAL
                wf_signal_info["均线周期"] = _last_tw_ma
                wf_signal_info["股票代码"] = code
                wf_signal_info["股票名称"] = stock_name
                wf_signal_info["所属板块"] = stock_plates
                _turnover_rank = _top_turnover_map.get(code)
                wf_signal_info["是否全市场成交额前200"] = "是" if _turnover_rank else "否"
                wf_signal_info["全市场成交额排名"] = _turnover_rank
                wf_signal_info["市场"] = market_val
                # 补充回测汇总字段
                _wf_last_trades = wf_trades_list[-1] if wf_trades_list else pd.DataFrame()
                _wf_summary = build_summary(_wf_last_trades, _last_tw_ma, _df_last, equity_arr=_wf_equity_arr)
                for _k in ["策略评分", "策略表现", "收益率", "年化收益率", "买入持有收益率", "超额收益率",
                           "平均每笔收益率", "最大回撤", "夏普比率", "卡尔玛比率", "交易次数", "盈利交易率",
                           "盈利因子", "盈亏比", "平均盈利", "平均盈利比", "平均亏损", "平均亏损比",
                           "最大单笔盈利", "最大单笔亏损", "最大连续盈利次数", "最大连续亏损次数",
                           "平均持仓天数", "初始资金", "最终资金"]:
                    if _k in _wf_summary:
                        wf_signal_info[_k] = _wf_summary[_k]
'''

c = c[:s] + new_block + c[e:]
with open('9windows_param_scan_numba_uscncc.py', 'w', encoding='utf-8') as f:
    f.write(c)

import py_compile
try:
    py_compile.compile('9windows_param_scan_numba_uscncc.py', doraise=True)
    print('OK')
except py_compile.PyCompileError as e:
    print(f'ERROR: {e}')
