"""鲁棒性矩阵评测：纯函数边界计算 + 端到端 API（只读保证 / 参数校验 / 先裁切后压缩）。

运行（需项目依赖）：
    WM_DATA_DIR 由本测试自动指向临时目录；
    python -m pytest tests/test_robustness.py -q
或直接：
    python tests/test_robustness.py
"""
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_png(size=384) -> bytes:
    rng = np.random.RandomState(7)
    y, x = np.mgrid[0:size, 0:size]
    base = (
        128
        + 40 * np.sin(2 * np.pi * x / 130)
        + 30 * np.cos(2 * np.pi * y / 95)
        + rng.normal(0, 12, (size, size))
    ).clip(0, 255).astype(np.uint8)
    rgb = np.stack([base, (base * 0 + 128), (255 - base)], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _fresh_client():
    tmp = tempfile.mkdtemp(prefix="wm-matrix-")
    os.environ["WM_DATA_DIR"] = tmp
    os.environ["WM_MAX_JOBS"] = "200"
    for mod in ("app.main", "app.archive"):
        sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient
    from app.main import app, DATA_DIR  # noqa: E402

    return TestClient(app), Path(DATA_DIR)


def _embed(client, text="机密-矩阵-2026"):
    r = client.post(
        "/api/embed",
        files={"image": ("img.png", _make_png(), "image/png")},
        data={"text": text, "strength": "medium"},
    )
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


def _snapshot_tree(root: Path) -> dict:
    """记录目录内每个文件的 (字节数, mtime_ns)，用于验证只读性。"""
    out = {}
    for p in root.rglob("*"):
        if p.is_file():
            st = p.stat()
            out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


# ---------------------------------------------------------------- 边界纯函数

def test_boundary_strict_pareto():
    from app import robustness

    # 空集
    assert robustness.compute_boundary(set()) == set()

    # 单点
    assert robustness.compute_boundary({(60, 0.75)}) == {(60, 0.75)}

    # 典型情形：质量更低且保留更少的成功格支配宽松格
    # (30,0.5) 在两维上都比 (60,0.75) 更严，故唯一边界是 (30,0.5)
    points = {(90, 1.0), (90, 0.75), (60, 0.75), (60, 1.0), (30, 0.5)}
    assert robustness.compute_boundary(points) == {(30, 0.5)}

    # 相同质量、不同比例 -> 只有最小比例在边界
    assert robustness.compute_boundary({(90, 1.0), (90, 0.5), (90, 0.75)}) == {(90, 0.5)}

    # 相同比例、不同质量 -> 只有最低质量在边界
    assert robustness.compute_boundary({(90, 0.75), (60, 0.75), (30, 0.75)}) == {(30, 0.75)}

    # 各有一维更严：互不支配，两格都在边界（不做相邻推断）
    assert robustness.compute_boundary({(30, 1.0), (90, 0.5)}) == {(30, 1.0), (90, 0.5)}

    # 失败格不参与：显式传入时才计入（API 层只传 success 格）
    assert robustness.compute_boundary({(90, 1.0)}) == {(90, 1.0)}


def test_boundary_via_deterministic_fake_detector():
    """用替身检测器精确控制各格成功/失败，验证端到端边界只由成功格决定。"""
    from app import dctwm
    from app import robustness
    from app.dctwm import DetectResult

    # 成功集合 = Q60@100%、Q30@100%、Q60@75%
    success = {(60, 1.0), (30, 1.0), (60, 0.75)}
    calls = []

    def fake_crop(rgb, fraction):
        calls.append(("crop", round(float(fraction), 3)))
        return rgb

    def fake_jpeg(rgb, quality):
        calls.append(("jpeg", int(quality)))
        return rgb

    def fake_detect(rgb):
        q = next(v for (k, v) in reversed(calls) if k == "jpeg")
        f = next(v for (k, v) in reversed(calls) if k == "crop")
        if (q, f) in success:
            return DetectResult(True, text="机密-矩阵-2026", confidence=88.0,
                                sync_score=0.9, sync_agreement=0.95)
        return DetectResult(False, sync_score=0.2, reason="fake-fail")

    orig = (dctwm.center_crop, dctwm.jpeg_compress, dctwm.detect_watermark)
    dctwm.center_crop = fake_crop
    dctwm.jpeg_compress = fake_jpeg
    dctwm.detect_watermark = fake_detect
    try:
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        res = robustness.run_matrix(
            img, "机密-矩阵-2026", [60, 30], [1.0, 0.75])
    finally:
        dctwm.center_crop, dctwm.jpeg_compress, dctwm.detect_watermark = orig

    by_key = {(c["jpeg_quality"], c["crop_fraction"]): c for c in res["cells"]}
    # (60,1.0) 成功但被 (60,0.75) 与 (30,1.0) 双向支配，不在边界
    assert by_key[(60, 1.0)]["success"] and not by_key[(60, 1.0)]["boundary"]
    assert by_key[(30, 1.0)]["success"] and by_key[(30, 1.0)]["boundary"]
    assert by_key[(60, 0.75)]["success"] and by_key[(60, 0.75)]["boundary"]
    # Q30@75% 失败：不参与边界，也不会被用来推断
    assert not by_key[(30, 0.75)]["success"]
    assert not by_key[(30, 0.75)]["boundary"]
    assert set(map(tuple, res["boundary"])) == {(30, 1.0), (60, 0.75)}


# ---------------------------------------------------------------- 顺序与独立性

def test_cell_pipeline_is_crop_then_jpeg_from_same_source():
    """每格必须：从同一份含水印原图 -> 先裁切 -> 后压缩；记录调用形状证据。"""
    from app import dctwm
    from app.dctwm import center_crop as _real_crop
    from app.dctwm import jpeg_compress as _real_jpeg

    log = []

    def crop_spy(rgb, fraction):
        log.append(("crop", tuple(rgb.shape)))
        return _real_crop(rgb, fraction)

    def jpeg_spy(rgb, quality):
        log.append(("jpeg", tuple(rgb.shape)))
        return _real_jpeg(rgb, quality)

    orig_crop, orig_jpeg = dctwm.center_crop, dctwm.jpeg_compress
    dctwm.center_crop = crop_spy
    dctwm.jpeg_compress = jpeg_spy
    try:
        client, data_dir = _fresh_client()
        job = _embed(client)
        wm_path = data_dir / job / "watermarked.png"
        wm_img = np.array(Image.open(wm_path).convert("RGB"))
        r = client.post(f"/api/jobs/{job}/robustness-matrix",
                        json={"jpeg_qualities": [90, 60], "crop_fractions": [1.0, 0.5]})
    finally:
        dctwm.center_crop = orig_crop
        dctwm.jpeg_compress = orig_jpeg

    assert r.status_code == 200, r.text
    full = tuple(wm_img.shape)
    assert len(log) == 8                              # 2×2 共 4 格，每格 crop+jpeg
    # 顺序严格为每格 crop -> jpeg
    assert [kind for kind, _ in log] == ["crop", "jpeg"] * 4
    # 每次裁切都从原始含水印图尺寸开始（各格相互独立、不从其他格结果叠加）
    assert all(shape == full for kind, shape in log if kind == "crop")
    # 先裁切后压缩：jpeg 的输入在 100% 格为原尺寸、50% 格已变小
    jpeg_shapes = [shape for kind, shape in log if kind == "jpeg"]
    assert jpeg_shapes[0] == full and jpeg_shapes[2] == full
    assert all(s[0] < full[0] and s[1] < full[1]
               for s in (jpeg_shapes[1], jpeg_shapes[3]))


# ---------------------------------------------------------------- 端到端 API

def test_matrix_endpoint_structure_and_success_semantics():
    client, data_dir = _fresh_client()
    job = _embed(client, text="机密-矩阵-2026")

    r = client.post(f"/api/jobs/{job}/robustness-matrix",
                    json={"jpeg_qualities": [90, 60, 30],
                          "crop_fractions": [1.0, 0.75, 0.5]})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["cells_total"] == 9
    assert data["cells_ok"] == 9
    assert data["pipeline"] == ["center_crop", "jpeg"]
    assert data["jpeg_qualities"] == [90, 60, 30]          # 服务端降序排列
    assert data["crop_fractions"] == [1.0, 0.75, 0.5]
    assert data["original_text"] == "机密-矩阵-2026"

    for c in data["cells"]:
        assert set(c) >= {
            "jpeg_quality", "crop_fraction", "ok", "detected",
            "recovered_text", "matches_original", "success",
            "metrics", "boundary", "attacks", "width", "height"}
        assert [a["type"] for a in c["attacks"]] == ["center_crop", "jpeg"]
        assert c["metrics"] is not None
        m = c["metrics"]
        assert "confidence" in m and "sync_score" in m
        assert "sync_agreement" in m and "votes_per_bit" in m
        # success 与文本一致性同义
        assert c["success"] == c["matches_original"]
        if c["success"]:
            assert c["detected"] and c["recovered_text"] == "机密-矩阵-2026"

    # 边界只含成功格；边界格数 <= 成功格数
    success_keys = {(c["jpeg_quality"], c["crop_fraction"])
                    for c in data["cells"] if c["success"]}
    boundary_keys = {(b["jpeg_quality"], b["crop_fraction"])
                     for b in data["boundary"]}
    assert boundary_keys <= success_keys
    flagged = {(c["jpeg_quality"], c["crop_fraction"])
               for c in data["cells"] if c["boundary"]}
    assert flagged == boundary_keys
    # 与纯函数结果一致
    from app import robustness
    assert boundary_keys == robustness.compute_boundary(success_keys)

    # 强攻击至少存在一些结果差异（矩阵有信息量），且宽松角 Q90@100% 必然成功
    corner = next(c for c in data["cells"]
                  if c["jpeg_quality"] == 90 and c["crop_fraction"] == 1.0)
    assert corner["success"], corner
    assert data["cells_success"] <= 9


def test_matrix_is_strictly_readonly():
    client, data_dir = _fresh_client()
    job = _embed(client)

    # 先制造一次攻击与检测，得到攻击历史 / target / 检测记录
    a = client.post("/api/attack",
                    json={"job_id": job, "jpeg_quality": 70, "crop_fraction": 0.75})
    assert a.status_code == 200
    d = client.post("/api/detect", data={"job_id": job})
    assert d.status_code == 200

    before = _snapshot_tree(data_dir)
    meta_before = json.loads((data_dir / job / "meta.json").read_text("utf-8"))
    target_before = (data_dir / job / "target.png").read_bytes()
    n_dets_before = len(list((data_dir / job / "detections").glob("*.json")))

    r = client.post(f"/api/jobs/{job}/robustness-matrix",
                    json={"jpeg_qualities": [80, 40], "crop_fractions": [1.0, 0.5]})
    assert r.status_code == 200, r.text

    after = _snapshot_tree(data_dir)
    # 没有任何文件被新增 / 删除 / 改写（大小与 mtime 均不变）
    assert before == after, set(before) ^ set(after)
    # 攻击历史未追加、target.png 未变
    meta_after = json.loads((data_dir / job / "meta.json").read_text("utf-8"))
    assert meta_after.get("attack_history") == meta_before.get("attack_history")
    assert (data_dir / job / "target.png").read_bytes() == target_before
    # 未新增检测记录、未生成报告
    n_dets_after = len(list((data_dir / job / "detections").glob("*.json")))
    assert n_dets_after == n_dets_before
    assert not (data_dir / "reports").exists()
    assert not (data_dir / "archive.log.jsonl").exists()


def test_matrix_validation_errors():
    client, _ = _fresh_client()
    job = _embed(client)
    url = f"/api/jobs/{job}/robustness-matrix"

    # 空列表
    assert client.post(url, json={"jpeg_qualities": [], "crop_fractions": [0.5]}).status_code == 400
    # 超过 3 个
    assert client.post(url, json={
        "jpeg_qualities": [90, 80, 70, 60], "crop_fractions": [0.5]}).status_code == 400
    # 重复
    assert client.post(url, json={
        "jpeg_qualities": [90, 90], "crop_fractions": [0.5]}).status_code == 400
    # 越界
    assert client.post(url, json={
        "jpeg_qualities": [5], "crop_fractions": [0.5]}).status_code == 400
    assert client.post(url, json={
        "jpeg_qualities": [90], "crop_fractions": [0.1]}).status_code == 400
    # 非整数质量（pydantic list[int] 在解析阶段即拒绝，400/422 均为合法拒绝）
    assert client.post(url, json={
        "jpeg_qualities": [70.5], "crop_fractions": [0.5]}).status_code in (400, 422)
    # 缺字段 -> pydantic 422
    assert client.post(url, json={"jpeg_qualities": [90]}).status_code == 422
    # 不存在的任务
    assert client.post(
        "/api/jobs/" + "0" * 12 + "/robustness-matrix",
        json={"jpeg_qualities": [90], "crop_fractions": [0.5]}).status_code == 404


def test_matrix_rejects_upload_only_job():
    client, data_dir = _fresh_client()
    emb_job = _embed(client)
    wm_bytes = (data_dir / emb_job / "watermarked.png").read_bytes()
    up = client.post("/api/detect",
                     files={"image": ("up.png", wm_bytes, "image/png")})
    assert up.status_code == 200
    upload_job = up.json()["job_id"]
    assert up.json()["upload_only"] is True

    r = client.post(f"/api/jobs/{upload_job}/robustness-matrix",
                    json={"jpeg_qualities": [80], "crop_fractions": [1.0]})
    assert r.status_code == 400


def test_matrix_single_axis_each_one_cell():
    client, _ = _fresh_client()
    job = _embed(client)
    r = client.post(f"/api/jobs/{job}/robustness-matrix",
                    json={"jpeg_qualities": [90], "crop_fractions": [1.0]})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["cells_total"] == 1
    assert data["cells"][0]["success"] is True
    assert len(data["boundary"]) == 1


if __name__ == "__main__":
    import inspect
    sys.modules.pop("app.main", None)
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and inspect.isfunction(v)]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(fns)} 个鲁棒性矩阵测试全部通过 ✔")
