#!/usr/bin/env python3
"""
查询历史 K 线额度使用明细
"""
import sys
import os
from collections import defaultdict

_SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills", "futuapi", "scripts")
if os.path.isdir(_SCRIPT_DIR):
    sys.path.insert(0, _SCRIPT_DIR)
else:
    _ALT_DIR = os.path.normpath(os.path.join(os.path.expanduser("~"), ".claude", "skills", "futuapi", "scripts"))
    if os.path.isdir(_ALT_DIR):
        sys.path.insert(0, _ALT_DIR)
    else:
        print("错误: 未找到 futuapi common 模块，请确认 skills/futuapi/scripts 目录存在", file=sys.stderr)
        sys.exit(1)
from common import create_quote_context, check_ret, safe_close  # type: ignore


def main():
    ctx = None
    try:
        ctx = create_quote_context()
        ret, data = ctx.get_history_kl_quota(get_detail=True)
        check_ret(ret, data, ctx, "获取 K 线额度")

        # data 可能是 list 或 DataFrame，统一处理
        if hasattr(data, "iloc"):
            used_quota = int(data.iloc[0]["used_quota"])
            remain_quota = int(data.iloc[0]["remain_quota"])
            import json as _json
            detail = _json.loads(data.iloc[0]["detail_list"]) if data.iloc[0]["detail_list"] else []
        else:
            used_quota, remain_quota, detail = int(data[0]), int(data[1]), data[2] if len(data) > 2 else []

        print(f"\n历史 K 线额度:  已使用 {used_quota} / {used_quota + remain_quota}  剩余 {remain_quota}\n")

        if not detail:
            print("  (暂无明细)")
            return

        by_date = defaultdict(list)
        for item in detail:
            by_date[item["request_time"][:10]].append(item)

        for date in sorted(by_date.keys(), reverse=True):
            items = sorted(by_date[date], key=lambda x: x["code"])
            print(f"──── {date} ({len(items)} 条) ────")
            for item in items:
                print(f"  {item['code']:20s} {item['name']:20s} {item['request_time']}")
            print()

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        safe_close(ctx)


if __name__ == "__main__":
    main()
