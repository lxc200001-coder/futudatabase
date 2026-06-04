with open('9windows_param_scan_numba_uscncc.py', 'r', encoding='utf-8') as f:
    c = f.read()

# 1. Add training window label to test window tuple
old1 = '                _test_windows.append((_tw_start, _tw_end, _tw_label, _tma))'
new1 = '                _train_label = str(_win_order[_wi][0].date()) + "~" + str(_win_order[_wi][1].date())\n                _test_windows.append((_tw_start, _tw_end, _tw_label, _tma, _train_label))'
if old1 in c:
    c = c.replace(old1, new1)
    print('1. Added training label')
else:
    print('1. NOT FOUND')

# 2. Update loop unpack
old2 = '        for _tw_start, _tw_end, _tw_label, _tma in _test_windows:'
new2 = '        for _tw_start, _tw_end, _tw_label, _tma, _tw_train_label in _test_windows:'
if old2 in c:
    c = c.replace(old2, new2)
    print('2. Updated loop unpack')
else:
    print('2. NOT FOUND')

# 3. Track carry at start
old3 = '            if _df_tw.empty:\n                continue\n            _tw_close'
new3 = '            _carry_at_start = _carry_pos\n            if _df_tw.empty:\n                continue\n            _tw_close'
if old3 in c:
    c = c.replace(old3, new3)
    print('3. Added carry tracking')
else:
    print('3. NOT FOUND')

# 4. Add debug fields after trades
old4 = "            wf_params_list.append({\"测试窗口\": _tw_label, \"均线周期\": _tma})\n            _wf_dfs.append(_df_tw.copy())"
new4 = '''            wf_params_list.append({\"测试窗口\": _tw_label, \"均线周期\": _tma})
            _wf_dfs.append(_df_tw.copy())
            if _wf_n > 0:
                _tw_trades_df[\"训练窗口\"] = _tw_train_label
                _tw_trades_df[\"开窗是否来自上个窗口\"] = \"是\" if _carry_at_start > 0 else \"否\"
                _tw_trades_df[\"是否平仓\"] = _tw_trades_df[\"交易状态\"].apply(lambda x: \"是\" if \"已平仓\" in str(x) else \"否\")'''
if old4 in c:
    c = c.replace(old4, new4)
    print('4. Added debug fields')
else:
    print('4. NOT FOUND')

with open('9windows_param_scan_numba_uscncc.py', 'w', encoding='utf-8') as f:
    f.write(c)

import py_compile
try:
    py_compile.compile('9windows_param_scan_numba_uscncc.py', doraise=True)
    print('Syntax OK')
except py_compile.PyCompileError as e:
    print(f'ERROR: {e}')
