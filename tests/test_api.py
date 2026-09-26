"""API 冒烟测试。"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_analyze_ok():
    resp = client.post("/api/analyze", json={
        "start_stitches": 10,
        "rows": ["k10", "[inc1 k1]*5"],
    })
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert rows[0]["status"] == "ok"
    assert rows[0]["remaining"] == 10
    assert rows[1]["status"] == "ok"
    assert rows[1]["consumed"] == 10
    assert rows[1]["produced"] == 10


def test_analyze_shortage_payload():
    resp = client.post("/api/analyze", json={
        "start_stitches": 5,
        "rows": ["k2 dec4"],
    })
    row = resp.json()["rows"][0]
    assert row["status"] == "shortage"
    assert row["failing_instruction"] == "dec4"
    assert row["shortage"] == 1
    assert row["failing_start"] == 3 and row["failing_end"] == 7


def test_analyze_validation_error():
    resp = client.post("/api/analyze", json={"start_stitches": -1, "rows": []})
    assert resp.status_code == 422


def test_analyze_adaptive_repeat_payload():
    resp = client.post("/api/analyze", json={
        "start_stitches": 20,
        "rows": ["[k2]*~4 k4"],
    })
    row = resp.json()["rows"][0]
    assert row["status"] == "ok"
    assert len(row["adaptive_blocks"]) == 1
    blk = row["adaptive_blocks"][0]
    assert blk["actual_repeats"] == 8
    assert blk["target_remaining"] == 4
    assert blk["star_pos"] == 4 and blk["start"] == 0 and blk["end"] == 7
    assert row["expanded_count"] == 9


def test_analyze_adaptive_indivisible_error():
    resp = client.post("/api/analyze", json={
        "start_stitches": 20,
        "rows": ["[k2]*~5", "k1"],
    })
    rows = resp.json()["rows"]
    assert rows[0]["status"] == "error"
    assert rows[0]["error_pos"] == 4
    assert rows[1]["status"] == "blocked"


def test_index_served():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "编织" in resp.text


def test_analyze_stitch_changes_payload():
    # 12 针：[k2 dec2]*3 耗12产9（净变化 -3），dec2 同一区间执行 3 次合并
    resp = client.post("/api/analyze", json={
        "start_stitches": 12,
        "rows": ["[k2 dec2]*3"],
    })
    row = resp.json()["rows"][0]
    assert row["status"] == "ok"
    assert row["stitch_change_net"] == -3
    assert row["stitch_change_net"] == row["produced"] - row["consumed"]
    assert len(row["stitch_changes"]) == 1
    e = row["stitch_changes"][0]
    assert e["type"] == "dec" and e["text"] == "dec2"
    assert e["start"] == 4 and e["end"] == 8
    assert e["executions"] == 3
    assert e["consumed"] == 6 and e["produced"] == 3 and e["net"] == -3


def test_analyze_stitch_changes_adaptive_uses_derived_count():
    # 20 针：[k2 inc2]*~8 => (20-8)/3=4 轮，inc2 执行 4 次（每次净 +1），随后 k8
    resp = client.post("/api/analyze", json={
        "start_stitches": 20,
        "rows": ["[k2 inc2]*~8 k8"],
    })
    row = resp.json()["rows"][0]
    assert row["status"] == "ok"
    changes = row["stitch_changes"]
    assert len(changes) == 1
    assert changes[0]["text"] == "inc2" and changes[0]["executions"] == 4
    assert changes[0]["net"] == 4
    assert row["stitch_change_net"] == row["produced"] - row["consumed"] == 4


def test_analyze_stitch_changes_omitted_for_failed_and_blocked_rows():
    resp = client.post("/api/analyze", json={
        "start_stitches": 2,
        "rows": ["[inc2]*3", "k1"],
    })
    rows = resp.json()["rows"]
    # [inc2]*3 共需消耗 3 针：前两次 inc2 各耗 1 后可用为 0，第三次短缺；
    # 短缺行不生成清单，后续行阻断。
    assert rows[0]["status"] == "shortage"
    assert rows[0]["stitch_changes"] == []
    assert rows[0]["stitch_change_net"] == 0
    assert rows[1]["status"] == "blocked"
    assert rows[1]["stitch_changes"] == []
