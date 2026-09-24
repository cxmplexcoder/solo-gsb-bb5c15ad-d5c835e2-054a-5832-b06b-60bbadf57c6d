"""鲁棒性矩阵评测（只读）。

从**同一份含水印原图**（任务的 ``watermarked.png``）出发，对用户选择的
（JPEG 质量 × 中心裁切保留比例）笛卡尔积中的每个组合**独立**执行：

    先 center_crop（中心裁切，边界吸附 8px 网格）
    再 jpeg_compress（JPEG 重压缩）

随后调用现有检测器 ``dctwm.detect_watermark``，逐格返回恢复文本、是否与任务
原文一致以及置信度 / 同步相关峰 / 同步一致率等指标。

严格只读
========
评测全程在内存中进行：不写入任务目录、不改写 ``target.png`` / ``meta.json``
（攻击历史）、不生成检测记录与报告归档。

二维最严成功边界（Pareto 前沿）
==============================
仅以**实际成功格**（检测器恢复出的文本与任务原文**完全一致**）为样本；
不从相邻格推断，失败格 / 检测异常格一律不参与。成功格 A 不在边界上，当且仅当
存在另一成功格 B，使得 B 在两个维度上都不比 A 宽松、且至少一维更严格：

    quality(B) <= quality(A) 且 fraction(B) <= fraction(A)
    且 (quality(B) < quality(A) 或 fraction(B) < fraction(A))

边界 = 所有不被上述关系支配的成功格（能恢复原文的“最狠”组合集合）。
"""

from __future__ import annotations

import numpy as np

from . import dctwm as wm

# 与攻击接口保持一致的取值范围
QUALITY_MIN, QUALITY_MAX = 10, 100
FRACTION_MIN, FRACTION_MAX = 0.2, 1.0
MAX_POINTS_PER_AXIS = 3


# ---------------------------------------------------------------- 边界计算

def compute_boundary(success_points: set[tuple[int, float]]) -> set[tuple[int, float]]:
    """二维最严成功边界。

    输入：实际成功格 ``(jpeg_quality, crop_fraction)`` 集合。
    输出：不被任何其他成功格支配的格子集合（质量更低 / 保留比例更小即“更严”）。
    """
    points = set(success_points)
    boundary: set[tuple[int, float]] = set()
    for q, f in points:
        dominated = any(
            (oq, of) != (q, f)
            and oq <= q
            and of <= f
            and (oq < q or of < f)
            for oq, of in points
        )
        if not dominated:
            boundary.add((q, f))
    return boundary


# ---------------------------------------------------------------- 单格评测

def _cell_metrics(res: wm.DetectResult) -> dict:
    """与 /api/detect 同源的指标子集（矩阵逐格展示用）。"""
    return {
        "confidence": res.confidence,
        "sync_score": res.sync_score,
        "sync_agreement": res.sync_agreement,
        "tiles_y": res.tiles_y,
        "tiles_x": res.tiles_x,
        "phase": list(res.phase),
        "votes_per_bit": res.votes_per_bit,
        "sync_blocks": res.sync_blocks,
        "payload_blocks": res.payload_blocks,
        "marked_blocks": res.marked_blocks,
        "total_blocks": res.total_blocks,
        "marked_fraction": res.marked_fraction,
    }


def run_cell(watermarked: np.ndarray, original_text: str,
             quality: int, fraction: float) -> dict:
    """单组组合：始终从含水印原图重新出发，先裁切后压缩，再调用现有检测器。"""
    fraction = round(float(fraction), 3)
    quality = int(quality)

    attacked = wm.center_crop(watermarked, fraction)
    attacked = wm.jpeg_compress(attacked, quality)
    h, w = attacked.shape[:2]

    cell: dict = {
        "jpeg_quality": quality,
        "crop_fraction": fraction,
        "attacks": [
            {"type": "center_crop", "fraction": fraction},
            {"type": "jpeg", "quality": quality},
        ],
        "width": int(w),
        "height": int(h),
        "ok": False,
        "detected": False,
        "recovered_text": None,
        "matches_original": False,
        "success": False,
        "reason": "",
        "error": None,
        "metrics": None,
        "boundary": False,
    }

    try:
        res = wm.detect_watermark(attacked)
    except wm.WatermarkError as e:
        cell["error"] = str(e)
        return cell

    cell["ok"] = True
    cell["detected"] = bool(res.detected)
    cell["recovered_text"] = res.text if res.detected else None
    cell["matches_original"] = bool(res.detected and res.text == original_text)
    cell["success"] = cell["matches_original"]
    cell["reason"] = res.reason or ""
    cell["metrics"] = _cell_metrics(res)
    return cell


# ---------------------------------------------------------------- 矩阵评测

def run_matrix(watermarked: np.ndarray, original_text: str,
               qualities: list[int], fractions: list[float]) -> dict:
    """对质量 × 保留比例笛卡尔积逐格独立评测并计算最严成功边界。

    ``qualities`` / ``fractions`` 由调用方完成去重、排序与范围校验；
    本函数约定它们按从宽到严（数值降序）排列，结果网格同序返回。
    """
    cells: list[dict] = []
    for quality in qualities:              # 行：质量降序（宽 → 严）
        for fraction in fractions:         # 列：保留比例降序（宽 → 严）
            cells.append(run_cell(watermarked, original_text, quality, fraction))

    success_points = {
        (c["jpeg_quality"], c["crop_fraction"]) for c in cells if c["success"]
    }
    boundary = compute_boundary(success_points)
    for c in cells:
        c["boundary"] = (c["jpeg_quality"], c["crop_fraction"]) in boundary

    return {
        "cells": cells,
        "success_points": sorted(success_points),
        "boundary": sorted(boundary),
    }
