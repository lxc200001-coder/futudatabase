with open('9windows_param_scan_numba_uscncc.py', 'r', encoding='utf-8') as f:
    c = f.read()

old_block = '''        _carry_cash = INITIAL_CASH
        _carry_pos = 0.0
        _carry_entry = 0.0
        _wf_dfs = []
        for _wi_idx, (_tw_start, _tw_end, _tw_label, _tma) in enumerate(_test_windows):
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

            if _wi_idx > 0 and _carry_pos > 0:
                if _tw_sell[0]:
                    _carry_cash += _carry_pos * _tw_close[0]
                    _carry_pos = 0.0
                    _carry_entry = 0.0

            _is_last = (_wi_idx == len(_test_windows) - 1)
            _wf_trades_arr, _wf_equity_arr, _wf_n, _carry_pos, _carry_entry, _carry_cash = (
                _numba_walkforward_backtest(
                    _tw_close, _tw_buy, _tw_sell, _carry_cash, FEE_RATE,
                    _carry_pos, _carry_entry, _is_last,
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
            _wf_dfs.append(_df_tw.copy())

        _wf_trades_ret = None
        if wf_trades_list:
            _wf_trades_ret = pd.concat(wf_trades_list, ignore_index=True, sort=False)'''

new_block = '''        # 合并所有测试窗口为单个连续回测
        _wf_start = _test_windows[0][0]
        _wf_end = _test_windows[-1][1]
        _wf_mask = (pd.to_datetime(df["datetime"]) >= _wf_start) & (pd.to_datetime(df["datetime"]) < _wf_end)
        _wf_df = df[_wf_mask].copy()
        if not _wf_df.empty:
            _wf_close = _wf_df["close"].values.astype(np.float64)
            _wf_datetime = _wf_df["datetime"].values
            _wf_buy = np.zeros(len(_wf_close), dtype=np.bool_)
            _wf_sell = np.zeros(len(_wf_close), dtype=np.bool_)

            for _seg_start, _seg_end, _seg_label, _seg_ma in _test_windows:
                _seg_mask = (_wf_df["datetime"] >= _seg_start) & (_wf_df["datetime"] < _seg_end)
                _seg_idx = np.where(_seg_mask.values)[0]
                if len(_seg_idx) < 2:
                    continue
                _seg_ma_vals = ma_cache[_seg_ma][_wf_mask].values[_seg_idx]
                _seg_dir = np.zeros(len(_seg_idx), dtype=np.int8)
                _seg_dir[0] = -1
                _seg_dir[1:] = np.where(_seg_ma_vals[1:] > _seg_ma_vals[:-1], 1, -1)
                _seg_buy = np.zeros(len(_seg_idx), dtype=np.bool_)
                _seg_buy[1:] = (_seg_dir[1:] == 1) & (_seg_dir[:-1] == -1)
                _wf_buy[_seg_idx] = _seg_buy
                _seg_sell = np.zeros(len(_seg_idx), dtype=np.bool_)
                _seg_sell[1:] = (_seg_dir[1:] == -1) & (_seg_dir[:-1] == 1)
                _wf_sell[_seg_idx] = _seg_sell
                wf_params_list.append({"测试窗口": _seg_label, "均线周期": _seg_ma})

            # 单次回测（_numba_backtest 自带末尾强平）
            _wf_trades_arr, _wf_equity_arr, _wf_n = _numba_backtest(
                _wf_close, _wf_buy, _wf_sell, INITIAL_CASH, FEE_RATE,
            )

            _wf_trades_ret = None
            if _wf_n > 0:
                _wf_trades_df, _ = _build_trades_from_arrays(
                    code_val, market_val, _wf_datetime, 0,
                    _wf_trades_arr, _wf_equity_arr, _wf_n,
                )
                if not _wf_trades_df.empty:
                    # 按开仓时间分配测试窗口标签
                    def _wf_seg_label(t):
                        for _sl, _ss, _se, _sm in _test_windows:
                            if _ss <= t < _se:
                                return _sl
                        return ""
                    _wf_trades_df["测试窗口"] = _wf_trades_df["开仓时间"].apply(_wf_seg_label)
                    wf_trades_list.append(_wf_trades_df)
                    _wf_trades_ret = pd.concat(wf_trades_list, ignore_index=True, sort=False)
                    _wf_dfs = [_wf_df]  # 用于合并 equity_curve'''

if old_block in c:
    c = c.replace(old_block, new_block)
    with open('9windows_param_scan_numba_uscncc.py', 'w', encoding='utf-8') as f:
        f.write(c)
    print('OK')
else:
    print('OLD BLOCK NOT FOUND')
    # Debug
    idx = c.find('_carry_cash = INITIAL_CASH')
    if idx > 0:
        print(repr(c[idx:idx+100]))
