"""编织引擎单元测试。"""

import pytest

from app.engine import (
    MAX_EXPANDED_LEAVES,
    ExpansionLimitError,
    ParseError,
    analyze_plan,
    analyze_row,
    count_leaves,
    expand_tokens,
    parse,
)


# ---------------------------------------------------------------- 解析
def test_parse_basic_leaf_consume_produce():
    nodes = parse("k5")
    assert len(nodes) == 1
    assert nodes[0].consumed == 5 and nodes[0].produced == 5

    nodes = parse("inc3")
    assert nodes[0].consumed == 1 and nodes[0].produced == 3

    nodes = parse("dec4")
    assert nodes[0].consumed == 4 and nodes[0].produced == 1


def test_parse_sequence_and_whitespace():
    nodes = parse("  k2 inc3   dec2 ")
    assert [n.type for n in nodes] == ["k", "inc", "dec"]
    assert [n.n for n in nodes] == [2, 3, 2]


def test_parse_repeat_flat():
    nodes = parse("[k2 inc1]*3")
    assert len(nodes) == 1 and nodes[0].type == "repeat" and nodes[0].n == 3
    # 一轮消耗 2+1=3，产出 2+1=3；重复三轮
    assert nodes[0].consumed == 9
    assert nodes[0].produced == 9


def test_parse_nested_repeat():
    nodes = parse("[[k2]*2 inc1]*3")
    root = nodes[0]
    assert root.type == "repeat" and root.n == 3
    assert [c.type for c in root.children] == ["repeat", "inc"]
    # 内层 [k2]*2 一轮消耗4，加 inc1 一轮消耗5、产出5，重复3轮
    assert root.consumed == 15
    assert root.produced == 15
    assert expand_tokens(nodes) == ["k2", "k2", "inc1"] * 3


def test_repeat_without_times_fails():
    with pytest.raises(ParseError) as ei:
        parse("[k2]*")
    assert "正整数" in ei.value.message


def test_empty_brackets_fails():
    with pytest.raises(ParseError) as ei:
        parse("[  ]*3")
    assert "为空" in ei.value.message


def test_unclosed_bracket_fails():
    with pytest.raises(ParseError) as ei:
        parse("k2 [k3*2")
    assert ei.value.pos == len("k2 [k3*2")


def test_bracket_without_repeat_fails():
    with pytest.raises(ParseError) as ei:
        parse("[k2]")
    assert "*M" in ei.value.message


def test_extra_closing_bracket_fails():
    with pytest.raises(ParseError) as ei:
        parse("k2]")
    assert ei.value.pos == 2


def test_missing_number_fails():
    with pytest.raises(ParseError) as ei:
        parse("k2 inc")
    assert ei.value.pos == 6  # 行尾，'inc' 之后


def test_zero_number_fails():
    with pytest.raises(ParseError):
        parse("k0")
    with pytest.raises(ParseError):
        parse("[k1]*0")


def test_unknown_keyword_fails():
    with pytest.raises(ParseError) as ei:
        parse("x3")
    assert ei.value.pos == 0


def test_empty_row_fails():
    with pytest.raises(ParseError) as ei:
        parse("   ")
    assert ei.value.pos == 0


def test_source_spans_nested():
    nodes = parse("[k2]*1")
    rep = nodes[0]
    leaf = rep.children[0]
    assert rep.start == 0 and rep.end == 6
    assert leaf.start == 1 and leaf.end == 3

    nodes = parse("[[k2]*2]*1")
    inner = nodes[0].children[0]
    leaf = inner.children[0]
    assert leaf.start == 2 and leaf.end == 4


def test_expansion_limit():
    with pytest.raises(ExpansionLimitError):
        count_leaves(parse(f"[k1]*{MAX_EXPANDED_LEAVES + 1}"))


# ---------------------------------------------------------------- 自适应重复
def test_parse_adaptive_marker():
    nodes = parse("[k2]*~4")
    rep = nodes[0]
    assert rep.type == "repeat" and rep.adaptive is True
    assert rep.target_remaining == 4 and rep.n == 0  # 解析阶段不推导，次数初始为 0
    assert rep.star_pos == 4
    assert rep.start == 0 and rep.end == 7


def test_parse_adaptive_zero_target():
    assert parse("[k2]*~0")[0].target_remaining == 0


def test_adaptive_repeats_derived_before_expansion():
    # 20 针：(20-4)/2 = 8 轮，留 4 针给 k4 正好用完
    rr = analyze_row(0, "[k2]*~4 k4", 20, False)
    assert rr.status == "ok"
    assert len(rr.adaptive_blocks) == 1
    blk = rr.adaptive_blocks[0]
    assert blk.actual_repeats == 8
    assert blk.target_remaining == 4
    assert blk.per_pass_consumed == 2
    assert rr.consumed == 20 and rr.produced == 20 and rr.remaining == 20
    assert rr.expanded_count == 9
    assert rr.expanded_preview == ["k2"] * 8 + ["k4"]


def test_adaptive_without_followup_is_unconsumed():
    # 没有后续指令时，保留下来的 N 针计入未消耗
    rr = analyze_row(0, "[k2]*~4", 20, False)
    assert rr.status == "unconsumed"
    assert rr.adaptive_blocks[0].actual_repeats == 8
    assert rr.consumed == 16 and rr.unconsumed == 4
    assert rr.remaining == 16


def test_adaptive_zero_repeats_allowed():
    # 当前剩余恰好等于目标保留针数 => 重复 0 次是合法解
    rr = analyze_row(0, "k3 [k2]*~7 k7", 10, False)
    assert rr.status == "ok"
    assert rr.adaptive_blocks[0].actual_repeats == 0
    assert rr.consumed == 10


def test_adaptive_indivisible_is_error_at_operator():
    rr = analyze_row(0, "[k2]*~5", 20, False)
    assert rr.status == "error"
    assert "整除" in rr.error
    assert rr.error_pos == 4  # '*' 的列号
    assert rr.adaptive_blocks == []


def test_adaptive_target_too_large_is_error_at_operator():
    rr = analyze_row(0, "k2 [k2]*~9", 10, False)
    assert rr.status == "error"
    assert "无法成立" in rr.error
    assert rr.error_pos == 7  # 该块 '*' 的列号


def test_adaptive_with_nested_fixed_block():
    # [[k2]*2 inc1] 每轮消耗 5、产出 5；20 针留 5 => 3 轮
    rr = analyze_row(0, "[[k2]*2 inc1]*~5 k5", 20, False)
    assert rr.status == "ok"
    assert rr.adaptive_blocks[0].actual_repeats == 3
    assert rr.expanded_preview == ["k2", "k2", "inc1"] * 3 + ["k5"]
    assert rr.consumed == 20


def test_nested_adaptive_block_is_syntax_error():
    rr = analyze_row(0, "[k2 [k1]*~2]*3", 20, False)
    assert rr.status == "error"
    assert "顶层" in rr.error
    assert rr.error_pos == 9  # 内层 '~' 的全局列号


def test_multiple_adaptive_blocks_resolved_in_sequence():
    # 20 -> 第一个块 (20-10)/2=5 轮留10 -> 第二个 (10-5)/1=5 轮留5 -> k5 用完
    rr = analyze_row(0, "[k2]*~10 [k1]*~5 k5", 20, False)
    assert rr.status == "ok"
    assert [b.actual_repeats for b in rr.adaptive_blocks] == [5, 5]


def test_adaptive_block_production_passes_to_next_row():
    # inc2 每轮耗 1 产 2：10 针留 4 => 6 轮，耗 6 产 12；下一行以 12 为可用
    results = analyze_plan(10, ["[inc2]*~4", "k12"])
    assert results[0].status == "unconsumed"
    assert results[0].adaptive_blocks[0].actual_repeats == 6
    assert results[0].remaining == 12
    assert results[1].status == "ok" and results[1].available == 12


def test_adaptive_failure_blocks_following_rows():
    results = analyze_plan(10, ["[k2]*~3", "k1"])
    assert results[0].status == "error"
    assert results[1].status == "blocked"


def test_adaptive_shortage_locates_following_leaf():
    # [k2]*~10 => (20-10)/2=5 轮留10，dec11 需要 11 针 => 短缺 1
    rr = analyze_row(0, "[k2]*~10 dec11", 20, False)
    assert rr.status == "shortage"
    assert rr.failing_instruction == "dec11"
    assert rr.shortage == 1 and rr.consumed == 10
    assert rr.adaptive_blocks[0].actual_repeats == 5


def test_adaptive_second_block_failure_reports_first_block_result():
    rr = analyze_row(0, "[k2]*~10 [k2]*~3", 20, False)
    assert rr.status == "error" and "整除" in rr.error
    assert rr.error_pos == 13  # 第二个块的 '*'
    assert len(rr.adaptive_blocks) == 1
    assert rr.adaptive_blocks[0].actual_repeats == 5


def test_adaptive_expansion_limit_error_at_operator():
    rr = analyze_row(0, "[k1]*~0", MAX_EXPANDED_LEAVES + 1, False)
    assert rr.status == "error"
    assert "上限" in rr.error
    assert rr.error_pos == 4


def test_adaptive_missing_number_fails():
    with pytest.raises(ParseError) as ei:
        parse("[k2]*~")
    assert "非负整数" in ei.value.message
    with pytest.raises(ParseError):
        parse("[k2]*~x")


def test_adaptive_to_dict_payload():
    d = analyze_row(0, "[k2]*~4 k4", 20, False).to_dict()
    blk = d["adaptive_blocks"][0]
    assert blk["actual_repeats"] == 8
    assert blk["start"] == 0 and blk["end"] == 7 and blk["star_pos"] == 4
    assert blk["target_remaining"] == 4


# ---------------------------------------------------------------- 逐行审校
def test_plan_ok_chain():
    results = analyze_plan(10, ["k10", "inc5 k5", "dec2 k8"])
    # 行1: 10 -> 10；行2: 消耗 1+5=6 产 5+5=10（未消耗4）；行3: 消耗 2+8=10 产 1+8=9
    assert [r.status for r in results] == ["ok", "unconsumed", "ok"]
    assert results[0].remaining == 10
    assert results[1].available == 10 and results[1].produced == 10
    assert results[1].unconsumed == 4
    assert results[2].available == 10 and results[2].remaining == 9


def test_shortage_locates_first_failing_instruction():
    # 可用 6 针：k2（剩4）、inc1（剩3）、dec4 需要4 针 -> 短缺 1
    rr = analyze_row(0, "k2 inc1 dec4", 6, False)
    assert rr.status == "shortage"
    assert rr.failing_instruction == "dec4"
    assert rr.failing_start == 8 and rr.failing_end == 12
    assert rr.shortage == 1
    assert rr.consumed == 3  # 只有前两条执行成功


def test_shortage_inside_repeat_points_to_source_leaf():
    # 重复块 [k3 dec2] 每轮消耗5；可用7：第一轮后剩2，第二轮 dec2 刚好...
    # k3 剩 -1 -> 第二轮的 k3 失败，短缺1；源位置指向 k3
    rr = analyze_row(0, "[k3 dec2]*2", 7, False)
    assert rr.status == "shortage"
    assert rr.failing_instruction == "k3"
    assert rr.failing_start == 1 and rr.failing_end == 3
    assert rr.shortage == 1
    assert rr.consumed == 5


def test_syntax_error_blocks_following_rows():
    results = analyze_plan(8, ["k8", "inc", "k2"])
    assert results[0].status == "ok"
    assert results[1].status == "error"
    assert results[1].error_pos == 3  # 'inc' 之后行尾
    assert results[2].status == "blocked"


def test_shortage_blocks_following_rows():
    results = analyze_plan(4, ["dec5", "k1"])
    assert results[0].status == "shortage"
    assert results[1].status == "blocked"


def test_exact_consumption_is_ok():
    rr = analyze_row(0, "[inc1 k1]*3", 6, False)  # 每轮消耗2
    assert rr.status == "ok"
    assert rr.consumed == 6 and rr.produced == 6 and rr.remaining == 6


def test_preview_truncated():
    rr = analyze_row(0, "k1 " * 300, 300, False)
    assert rr.status == "ok"
    assert rr.expanded_count == 300
    assert rr.expanded_truncated is True
    assert len(rr.expanded_preview) == 200


def test_to_dict_keys():
    rr = analyze_row(0, "k1", 1, False)
    d = rr.to_dict()
    assert d["status"] == "ok" and d["remaining"] == 1


# ---------------------------------------------------------------- 针数变化来源
def test_stitch_changes_basic_entries():
    # 起始 12 针：k5 耗5产5、inc3 耗1产3、dec2 耗2产1、k4 耗4产4 => 12 全消耗
    rr = analyze_row(0, "k5 inc3 dec2 k4", 12, False)
    assert rr.status == "ok"
    changes = rr.stitch_changes
    assert [e.text for e in changes] == ["inc3", "dec2"]
    inc, dec = changes
    assert inc.start == 3 and inc.end == 7
    assert inc.executions == 1 and inc.consumed == 1 and inc.produced == 3 and inc.net == 2
    assert dec.start == 8 and dec.end == 12
    assert dec.executions == 1 and dec.consumed == 2 and dec.produced == 1 and dec.net == -1
    assert rr.stitch_change_net == 1  # 产出 13 - 消耗 12
    assert rr.produced - rr.consumed == rr.stitch_change_net


def test_stitch_changes_same_text_different_positions_are_split():
    # 同一行两处字面相同的 inc2，必须分列
    rr = analyze_row(0, "inc2 k4 inc2", 6, False)
    assert rr.status == "ok"
    changes = rr.stitch_changes
    assert len(changes) == 2
    assert [e.text for e in changes] == ["inc2", "inc2"]
    assert changes[0].start != changes[1].start
    assert (changes[0].start, changes[0].end) == (0, 4)
    assert (changes[1].start, changes[1].end) == (8, 12)
    assert all(e.executions == 1 for e in changes)


def test_stitch_changes_inside_fixed_repeat_merged():
    # [k2 dec2]*3：dec2 同一源区间执行 3 次合并为一条
    rr = analyze_row(0, "[k2 dec2]*3", 12, False)
    assert rr.status == "ok"
    changes = rr.stitch_changes
    assert len(changes) == 1
    e = changes[0]
    assert e.text == "dec2" and (e.start, e.end) == (4, 8)
    assert e.executions == 3
    assert e.consumed == 6 and e.produced == 3 and e.net == -3
    assert rr.stitch_change_net == rr.produced - rr.consumed == -3


def test_stitch_changes_nested_repeat_multipliers_multiply():
    # [[k2 inc1]*2 dec2]*3：每轮耗 8 产 7，共耗24产21；inc1 执行 2*3=6 次，dec2 执行 3 次
    rr = analyze_row(0, "[[k2 inc1]*2 dec2]*3", 24, False)
    assert rr.status == "ok"
    by_text = {e.text: e for e in rr.stitch_changes}
    inc = by_text["inc1"]
    assert inc.executions == 6
    assert inc.consumed == 6 and inc.produced == 6 and inc.net == 0
    dec = by_text["dec2"]
    assert dec.executions == 3
    assert dec.consumed == 6 and dec.produced == 3 and dec.net == -3
    assert rr.stitch_change_net == -3
    assert rr.produced - rr.consumed == -3


def test_stitch_changes_adaptive_uses_derived_repeats():
    # 16 针：[inc2]*~6 => (16-6)/1=10 轮，耗10产20，再 k6 耗尽；inc2 执行 10 次
    rr = analyze_row(0, "[inc2]*~6 k6", 16, False)
    assert rr.status == "ok"
    assert rr.adaptive_blocks[0].actual_repeats == 10
    changes = rr.stitch_changes
    assert len(changes) == 1
    e = changes[0]
    assert e.text == "inc2" and e.executions == 10
    assert e.consumed == 10 and e.produced == 20 and e.net == 10
    assert rr.stitch_change_net == 10
    assert rr.produced == 26 and rr.consumed == 16


def test_stitch_changes_adaptive_nested_fixed_multiplied():
    # [[k2]*2 inc1]*~5 k5（20 针）=> 3 轮；inc1 执行 3 次
    rr = analyze_row(0, "[[k2]*2 inc1]*~5 k5", 20, False)
    assert rr.status == "ok"
    assert rr.adaptive_blocks[0].actual_repeats == 3
    changes = rr.stitch_changes
    assert len(changes) == 1
    e = changes[0]
    assert e.text == "inc1" and e.executions == 3
    assert e.consumed == 3 and e.produced == 3 and e.net == 0
    assert rr.stitch_change_net == 0


def test_stitch_changes_adaptive_zero_repeats_absent():
    # 自适应块推导为 0 次时，块内 inc 实际未执行，不生成来源条目
    rr = analyze_row(0, "k3 [inc1]*~7 k7", 10, False)
    assert rr.status == "ok"
    assert rr.adaptive_blocks[0].actual_repeats == 0
    assert rr.stitch_changes == []
    assert rr.stitch_change_net == 0


def test_stitch_changes_unconsumed_row_still_generated():
    # 未全部消耗的行也要生成清单：inc2 耗1产2，净变化 +1
    rr = analyze_row(0, "inc2", 10, False)
    assert rr.status == "unconsumed"
    assert len(rr.stitch_changes) == 1
    e = rr.stitch_changes[0]
    assert e.text == "inc2" and e.executions == 1
    assert e.net == 1
    assert rr.stitch_change_net == rr.produced - rr.consumed == 1


def test_stitch_changes_shortage_row_not_generated():
    # 短缺行不生成清单（失败指令不参与统计）
    rr = analyze_row(0, "inc3 dec4", 2, False)
    assert rr.status == "shortage"
    assert rr.stitch_changes == []
    assert rr.stitch_change_net == 0


def test_stitch_changes_blocked_and_error_rows_not_generated():
    results = analyze_plan(8, ["inc", "k2"])
    assert results[0].status == "error" and results[0].stitch_changes == []
    assert results[1].status == "blocked" and results[1].stitch_changes == []


def test_stitch_changes_to_dict_payload():
    rr = analyze_row(0, "[inc2]*2 dec3", 5, False)
    d = rr.to_dict()
    assert "stitch_changes" in d and "stitch_change_net" in d
    inc, dec = d["stitch_changes"]
    assert inc["type"] == "inc" and inc["text"] == "inc2"
    assert inc["executions"] == 2 and inc["consumed"] == 2
    assert inc["produced"] == 4 and inc["net"] == 2
    assert dec["type"] == "dec" and dec["executions"] == 1
    assert dec["consumed"] == 3 and dec["produced"] == 1 and dec["net"] == -2
    assert d["stitch_change_net"] == 0
