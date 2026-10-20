#!/usr/bin/env python3
"""参考图引导的条状（条纹/带状）水印去除。

输入一张低分辨率无水印参考图与一张同内容的高分辨率带条状水印目标图。
脚本对齐两张图，从残差中估计条纹的方向、周期与空间强度，从目标图中减去水印成分，
输出保留高清细节的结果图与可判定的 QC 报告。

依赖：Python 3.10+、Pillow、numpy。不联网、不回写原图。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_MISSING_DEPENDENCY = 3
EXIT_NO_WATERMARK = 4

try:
    import numpy as np
    from PIL import Image, ImageOps
except Exception as exc:  # 依赖缺失时给出机器可读状态
    sys.stderr.write(
        json.dumps(
            {
                "status": "unavailable",
                "reason_code": "missing_dependency",
                "detail": f"无法导入 Pillow 或 numpy：{exc}",
                "next_action": "运行 scripts/check_environment.py 查看缺失项，再按 references/setup-guide.md 安装",
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    raise SystemExit(EXIT_MISSING_DEPENDENCY)

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ANALYSIS_MAX_WIDTH = 1600
QC_MAX_WIDTH = 1024
ANGLE_SEARCH_WIDTH = 480
DEFAULT_MAX_MEGAPIXELS = 30.0
EPS = 1e-8

STRENGTH_PRESETS = {
    "conservative": {"gain_scale": 0.85, "period_sigma": 1.60, "guard": 0.45, "shrink": 0.45},
    "balanced": {"gain_scale": 1.00, "period_sigma": 1.15, "guard": 0.35, "shrink": 0.30},
    "aggressive": {"gain_scale": 1.12, "period_sigma": 0.80, "guard": 0.18, "shrink": 0.15},
}

MIN_SIGNIFICANCE = 4.0
STRONG_SIGNIFICANCE = 8.0
MIN_PEAK_AMPLITUDE = 0.004
MIN_FOLD_EXPLAINED = 0.30
ALIGN_LEVEL_FULL = 0.90
ALIGN_LEVEL_PARTIAL = 0.60
DETAIL_RETENTION_MIN = 0.97
DETAIL_ENERGY_MIN = 0.85
DETAIL_ENERGY_MAX = 1.15
DETAIL_CUT_RATIO = 0.20
PEAK_RATIO = 3.5


class InputError(Exception):
    """输入或参数不合法。"""


# --------------------------------------------------------------------------
# 基础图像运算
# --------------------------------------------------------------------------


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA", "PA", "P"}:
            image = image.convert("RGBA")
            canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(canvas, image)
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(array: np.ndarray, path: Path, quality: int = 95) -> None:
    data = np.clip(array * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    image = Image.fromarray(data, "RGB")
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=quality, subsampling=0)
    else:
        image.save(path)


def resize_plane(plane: np.ndarray, size: tuple[int, int], resample: int) -> np.ndarray:
    image = Image.fromarray(np.ascontiguousarray(plane, dtype=np.float32), mode="F")
    return np.asarray(image.resize((int(size[0]), int(size[1])), resample), dtype=np.float32)


def resize_rgb(array: np.ndarray, size: tuple[int, int], resample: int = Image.LANCZOS) -> np.ndarray:
    out = np.empty((int(size[1]), int(size[0]), array.shape[2]), dtype=np.float32)
    for channel in range(array.shape[2]):
        out[:, :, channel] = resize_plane(array[:, :, channel], size, resample)
    return out


def zoom_plane(plane: np.ndarray, size: tuple[int, int], kx: float, ky: float, resample: int) -> np.ndarray:
    """围绕中心缩放：先生成 (W*kx, H*ky) 再回到目标尺寸。"""
    if abs(kx - 1.0) < 1e-9 and abs(ky - 1.0) < 1e-9:
        return resize_plane(plane, size, resample)
    intermediate = (max(1, int(round(size[0] * kx))), max(1, int(round(size[1] * ky))))
    step = resize_plane(plane, intermediate, resample)
    if intermediate == (int(size[0]), int(size[1])):
        return step
    return resize_plane(step, size, resample)


def zoom_rgb(array: np.ndarray, size: tuple[int, int], kx: float, ky: float, resample: int) -> np.ndarray:
    out = np.empty((int(size[1]), int(size[0]), array.shape[2]), dtype=np.float32)
    for channel in range(array.shape[2]):
        out[:, :, channel] = zoom_plane(array[:, :, channel], size, kx, ky, resample)
    return out


def luma(array: np.ndarray) -> np.ndarray:
    return array[..., 0] * 0.299 + array[..., 1] * 0.587 + array[..., 2] * 0.114


def _box_blur_axis(array: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """沿单轴做滑动均值，累积和实现，复杂度 O(N)。"""
    if radius < 1:
        return array
    length = array.shape[axis]
    window = 2 * radius + 1
    if window > length:
        radius = max(0, (length - 1) // 2)
        window = 2 * radius + 1
        if radius < 1:
            return array
    padding = [(0, 0)] * array.ndim
    padding[axis] = (radius, radius + 1)
    padded = np.pad(array, padding, mode="edge")
    cumulative = np.cumsum(padded, axis=axis, dtype=np.float64)
    zero_shape = list(cumulative.shape)
    zero_shape[axis] = 1
    cumulative = np.concatenate([np.zeros(zero_shape, dtype=np.float64), cumulative], axis=axis)
    upper = [slice(None)] * array.ndim
    lower = [slice(None)] * array.ndim
    upper[axis] = slice(window, window + length)
    lower[axis] = slice(0, length)
    return ((cumulative[tuple(upper)] - cumulative[tuple(lower)]) / window).astype(np.float32)


def box_blur(array: np.ndarray, radius: int) -> np.ndarray:
    return _box_blur_axis(_box_blur_axis(array, radius, 0), radius, 1)


def gaussian_blur(array: np.ndarray, sigma: float) -> np.ndarray:
    """三次盒式滤波逼近高斯。"""
    if sigma <= 0:
        return array.copy()
    radius = max(1, int(round((math.sqrt(4.0 * sigma * sigma + 1.0) - 1.0) / 2.0)))
    result = array
    for _ in range(3):
        result = box_blur(result, radius)
    return result


def _sample_bilinear(array: np.ndarray, ys: np.ndarray, xs: np.ndarray):
    """双线性采样，坐标越界按边缘钳位。返回 (结果, 有效掩膜)。"""
    height, width = array.shape[:2]
    valid = (ys >= 0.0) & (ys <= height - 1) & (xs >= 0.0) & (xs <= width - 1)
    clamped_y = np.clip(ys, 0.0, height - 1.0)
    clamped_x = np.clip(xs, 0.0, width - 1.0)
    y0 = np.floor(clamped_y).astype(np.int64)
    x0 = np.floor(clamped_x).astype(np.int64)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    fy = (clamped_y - y0).astype(np.float32)
    fx = (clamped_x - x0).astype(np.float32)
    if array.ndim == 2:
        a00, a01 = array[y0, x0], array[y0, x1]
        a10, a11 = array[y1, x0], array[y1, x1]
    else:
        a00, a01 = array[y0, x0, :], array[y0, x1, :]
        a10, a11 = array[y1, x0, :], array[y1, x1, :]
        fy = fy[..., None]
        fx = fx[..., None]
    top = a00 + (a01 - a00) * fx
    bottom = a10 + (a11 - a10) * fx
    return (top + (bottom - top) * fy).astype(np.float32), valid


def translate(array: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """返回 out[y, x] = array[clip(y - dy), clip(x - dx)]。

    整数位移（本工具的常态：参考图与目标图同源导出，配准结果几乎总是整像素）
    走索引快路径，跳过双线性插值 —— 后者要建两张全尺寸浮点网格并做 8 次花式索引，
    在 2048x1475x3 上是 0.5s，而 take 只要几十毫秒。
    """
    if float(dy).is_integer() and float(dx).is_integer():
        iy, ix = int(round(dy)), int(round(dx))
        h, w = array.shape[:2]
        ys = np.clip(np.arange(h, dtype=np.intp) - iy, 0, h - 1)
        xs = np.clip(np.arange(w, dtype=np.intp) - ix, 0, w - 1)
        return np.take(np.take(array, ys, axis=0), xs, axis=1)
    ys, xs = np.mgrid[0 : array.shape[0], 0 : array.shape[1]].astype(np.float32)
    sampled, _ = _sample_bilinear(array, ys - np.float32(dy), xs - np.float32(dx))
    return sampled


# --------------------------------------------------------------------------
# 配准
# --------------------------------------------------------------------------


def _ncc_overlap(a: np.ndarray, b: np.ndarray, dy: int, dx: int, min_overlap: float) -> tuple[float, float]:
    """a[y - dy, x - dx] 与 b[y, x] 在重叠区内的归一化互相关。"""
    height, width = a.shape
    ya0, ya1 = max(0, -dy), min(height, height - dy)
    xa0, xa1 = max(0, -dx), min(width, width - dx)
    if ya1 - ya0 < 8 or xa1 - xa0 < 8:
        return -1.0, 0.0
    overlap = (ya1 - ya0) * (xa1 - xa0) / float(height * width)
    if overlap < min_overlap:
        return -1.0, overlap
    pa = a[ya0:ya1, xa0:xa1].astype(np.float64)
    pb = b[ya0 + dy : ya1 + dy, xa0 + dx : xa1 + dx].astype(np.float64)
    pa -= pa.mean()
    pb -= pb.mean()
    denominator = math.sqrt(float((pa * pa).sum()) * float((pb * pb).sum()))
    if denominator <= EPS:
        return 0.0, overlap
    return float((pa * pb).sum()) / denominator, overlap


def search_shift(ref: np.ndarray, tgt: np.ndarray, max_shift: int, step: int, min_overlap: float):
    best = (-1.0, 0, 0)
    for dy in range(-max_shift, max_shift + 1, step):
        for dx in range(-max_shift, max_shift + 1, step):
            score, _ = _ncc_overlap(ref, tgt, dy, dx, min_overlap)
            if score > best[0]:
                best = (score, dy, dx)
    return best


def search_shift_near(ref: np.ndarray, tgt: np.ndarray, center_y: int, center_x: int, radius: int, min_overlap: float):
    best = (-1.0, float(center_y), float(center_x))
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            score, _ = _ncc_overlap(ref, tgt, center_y + oy, center_x + ox, min_overlap)
            if score > best[0]:
                best = (score, float(center_y + oy), float(center_x + ox))
    return best


def overlap_ncc(a: np.ndarray, b: np.ndarray) -> float:
    pa = a.astype(np.float64).ravel()
    pb = b.astype(np.float64).ravel()
    pa -= pa.mean()
    pb -= pb.mean()
    denominator = math.sqrt(float((pa * pa).sum()) * float((pb * pb).sum()))
    if denominator <= EPS:
        return 0.0
    return float((pa * pb).sum()) / denominator


def align_reference(reference: np.ndarray, target_small: np.ndarray, search_scale: float = 0.06,
                    work_px: int = 360):
    """估计参考图到目标图坐标系的各向异性缩放与平移，返回 (对齐后的参考图, 信息)。

    work_px 是 NCC 搜索的工作分辨率（长边）。该搜索是 O(候选数 × N)，
    降到 256 可省约 2 倍时间；实测 Getty 这一类（参考图与目标图同源、
    缩放比固定）在此尺寸下配准结果与 360 完全一致。
    """
    target_h, target_w = target_small.shape[:2]
    work_scale = min(1.0, float(work_px) / float(target_w))
    work_w = max(32, int(round(target_w * work_scale)))
    work_h = max(32, int(round(target_h * work_scale)))
    target_work = luma(resize_rgb(target_small, (work_w, work_h), Image.LANCZOS))
    reference_gray = luma(reference)

    cache: dict[tuple[float, float], np.ndarray] = {}

    def candidate(kx: float, ky: float) -> np.ndarray:
        key = (round(kx, 6), round(ky, 6))
        if key not in cache:
            cache[key] = zoom_plane(reference_gray, (work_w, work_h), kx, ky, Image.LANCZOS)
        return cache[key]

    span = max(4, int(round(work_w * search_scale)))
    base = candidate(1.0, 1.0)
    coarse = search_shift(base, target_work, span, max(1, span // 6), 0.55)
    fine = search_shift(base, target_work, max(2, span // 6 + 2), 1, 0.60)
    best = (fine[0], 1.0, 1.0, float(fine[1]), float(fine[2])) if fine[0] >= coarse[0] else (
        coarse[0], 1.0, 1.0, float(coarse[1]), float(coarse[2])
    )

    for scale_step, radius in ((0.012, 2), (0.005, 2), (0.002, 1)):
        improved = best
        center_y, center_x = int(round(best[3])), int(round(best[4]))
        for kx in np.arange(best[1] - 3 * scale_step, best[1] + 3 * scale_step + 1e-9, scale_step):
            for ky in np.arange(best[2] - 3 * scale_step, best[2] + 3 * scale_step + 1e-9, scale_step):
                kx, ky = float(kx), float(ky)
                if kx <= 0.2 or ky <= 0.2:
                    continue
                local = search_shift_near(candidate(kx, ky), target_work, center_y, center_x, radius, 0.60)
                if local[0] > improved[0]:
                    improved = (local[0], kx, ky, local[1], local[2])
        best = improved

    score, kx, ky, shift_y, shift_x = best
    aligned = zoom_rgb(reference, (target_w, target_h), kx, ky, Image.LANCZOS)
    aligned = translate(aligned, shift_y / work_scale, shift_x / work_scale)
    info = {
        "scale_x": round(kx, 5),
        "scale_y": round(ky, 5),
        "shift_target_px": [round(shift_y / work_scale, 3), round(shift_x / work_scale, 3)],
        "ncc": round(float(score), 5),
    }
    return aligned, info


# --------------------------------------------------------------------------
# 投影、周期与模板
# --------------------------------------------------------------------------


def projection_grid(shape: tuple[int, int], angle_deg: float):
    """返回投影坐标矩阵（未减去偏移）与其最小值。"""
    height, width = shape
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rows = np.arange(height, dtype=np.float32)[:, None] * np.float32(sin_t)
    columns = np.arange(width, dtype=np.float32)[None, :] * np.float32(cos_t)
    grid = rows + columns
    return grid, float(grid.min()) if grid.size else 0.0


def project_profile(array: np.ndarray, angle_deg: float):
    """沿垂直于 angle_deg 的方向把图像分箱投影成一维剖面，返回 (剖面(C,L), 偏移)。"""
    grid, origin = projection_grid(array.shape[:2], angle_deg)
    index = (grid - np.float32(origin)).astype(np.int64).ravel()
    length = int(index.max()) + 1
    counts = np.bincount(index, minlength=length).astype(np.float64)
    flat = array.reshape(-1, array.shape[2])
    profile = np.empty((array.shape[2], length), dtype=np.float32)
    for channel in range(array.shape[2]):
        sums = np.bincount(index, weights=flat[:, channel].astype(np.float64), minlength=length)
        profile[channel] = sums / np.maximum(counts, 1.0)
    return profile, origin


def template_field(template: np.ndarray, shape: tuple[int, int], angle_deg: float, origin: float) -> np.ndarray:
    """按投影坐标把一维模板铺成二维场，逐像素线性插值。"""
    grid, _ = projection_grid(shape, angle_deg)
    positions = np.arange(template.shape[1], dtype=np.float64)
    flat = (grid - np.float32(origin)).ravel().astype(np.float64)
    out = np.empty((flat.size, template.shape[0]), dtype=np.float32)
    for channel in range(template.shape[0]):
        out[:, channel] = np.interp(flat, positions, template[channel].astype(np.float64))
    return out.reshape(shape[0], shape[1], template.shape[0])


def poly_detrend(profile: np.ndarray, degree: int = 3) -> np.ndarray:
    """减去低阶多项式趋势，保留周期成分且不引入相位失真。"""
    length = profile.shape[1]
    x = np.linspace(-1.0, 1.0, length)
    design = np.vstack([x**power for power in range(degree + 1)]).T
    coefficients, *_ = np.linalg.lstsq(design, profile.T.astype(np.float64), rcond=None)
    return (profile.T - design @ coefficients).T.astype(np.float32)


def detect_peak(profile: np.ndarray, min_period: float, max_period: float, peak_ratio: float):
    """在剖面频谱中定位最显著的周期峰，返回 (峰位, 显著度, 粗周期)。"""
    length = profile.shape[1]
    spectrum = np.abs(np.fft.rfft(profile.astype(np.float64), axis=1)).sum(axis=0)
    low_index = max(1, int(math.ceil(length / float(max_period))))
    high_index = min(len(spectrum) - 2, int(math.floor(length / float(min_period))))
    if high_index - low_index < 2:
        return 0, 0.0, 0.0
    band = spectrum[low_index : high_index + 1]
    significance = band / (float(np.median(band)) + EPS)

    peaks: list[tuple[int, float]] = []
    for offset in range(len(band)):
        value = float(significance[offset])
        if value < peak_ratio:
            continue
        lower = float(significance[offset - 1]) if offset > 0 else -1.0
        upper = float(significance[offset + 1]) if offset + 1 < len(band) else -1.0
        if value >= lower and value > upper:
            peaks.append((low_index + offset, value))
    if not peaks:
        return 0, 0.0, 0.0
    index, score = max(peaks, key=lambda item: item[1])
    return index, score, length / float(index)


def folded_energy(profile: np.ndarray, period: float) -> float:
    """按给定周期做相位折叠，返回扣除噪声基线后的跨相位能量。

    相位分箱数随周期增大而增多，纯噪声下的组间方差期望约为 (B-1)/N·var，
    直接用折叠方差会系统性偏爱大周期。这里减去该基线，使不同周期的判据可比。
    """
    length = profile.shape[1]
    if period < 2.0 or period > length / 2.0:
        return 0.0
    bin_count = int(np.clip(round(period), 8, 512))
    positions = np.arange(length, dtype=np.float64)
    index = np.clip(((positions % period) / period * bin_count).astype(np.int64), 0, bin_count - 1)
    luma_profile = profile.mean(axis=0).astype(np.float64)
    counts = np.bincount(index, minlength=bin_count).astype(np.float64)
    sums = np.bincount(index, weights=luma_profile, minlength=bin_count)
    means = sums / np.maximum(counts, 1.0)
    energy = float(np.average((means - means.mean()) ** 2, weights=counts))
    baseline = (bin_count - 1) / float(length) * float(luma_profile.var())
    return max(0.0, energy - baseline)


def refine_period(profile: np.ndarray, period_guess: float, span: float = 0.02, steps: int = 41) -> float:
    """两阶段相位折叠搜索：先粗搜，再在最优附近细搜。"""

    def scan(center: float, half_span: float) -> float:
        period, score = float(center), -1.0
        for step in range(steps):
            factor = 1.0 + half_span * (2.0 * step / max(1, steps - 1) - 1.0)
            candidate = float(center) * factor
            value = folded_energy(profile, candidate)
            if value > score:
                score = value
                period = candidate
        return period

    return scan(scan(float(period_guess), span), span / 20.0)


def resolve_period(profile: np.ndarray, coarse_period: float, min_period: float, max_period: float) -> float:
    """在倍率候选上取周期性能量平台的最短周期，消除谐波与次谐波歧义。"""
    if coarse_period <= 0.0:
        return 0.0
    candidates = []
    for ratio in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        guess = coarse_period * ratio
        if guess < min_period * 0.9 or guess > max_period * 1.1:
            continue
        period = refine_period(profile, guess)
        if min_period <= period <= max_period:
            candidates.append((period, folded_energy(profile, period)))
    if not candidates:
        return 0.0
    best_energy = max(energy for _, energy in candidates)
    if best_energy <= 0.0:
        return 0.0
    return min(period for period, energy in candidates if energy >= 0.85 * best_energy)


def fold_template(profile: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """按周期做相位折叠，得到各通道周期波形（已去直流）与其按相位重排的形式。"""
    length = profile.shape[1]
    bin_count = int(np.clip(round(period), 8, 512))
    positions = np.arange(length, dtype=np.float64)
    index = np.clip(((positions % period) / period * bin_count).astype(np.int64), 0, bin_count - 1)
    counts = np.maximum(np.bincount(index, minlength=bin_count).astype(np.float64), 1.0)
    folded = np.empty((profile.shape[0], bin_count), dtype=np.float64)
    for channel in range(profile.shape[0]):
        sums = np.bincount(index, weights=profile[channel].astype(np.float64), minlength=bin_count)
        folded[channel] = sums / counts
    folded -= folded.mean(axis=1, keepdims=True)
    return folded[:, index].astype(np.float32), folded


def spectrum_bins(template: np.ndarray, ratio: float = 0.08) -> list[int]:
    spectrum = np.abs(np.fft.rfft(template.astype(np.float64), axis=1)).sum(axis=0)
    if spectrum.max() <= 0.0:
        return []
    indices = np.nonzero(spectrum >= ratio * spectrum.max())[0]
    return [int(index) for index in indices if index > 0][:16]


def display_angle(angle: float) -> float:
    """把条纹法线角度规范化到 [0, 90]，避免 0.1 与 179.9 这类等价角读数不一致。"""
    value = float(angle) % 180.0
    if value > 90.0:
        value = 180.0 - value
    return value % 180.0


def stripe_label(angle: float) -> str:
    value = display_angle(angle)
    if abs(value - 90.0) < 5.0:
        return "horizontal"
    if value < 5.0:
        return "vertical"
    return "diagonal"


# --------------------------------------------------------------------------
# 条纹检测
# --------------------------------------------------------------------------


def estimate_normal_angle(gray: np.ndarray, min_period: float, max_period: float) -> tuple[float, float]:
    """用二维频谱的径向离散度估计条纹法线方向，返回 (角度, 显著度)。"""
    height, width = gray.shape
    detail = gray - gaussian_blur(gray, max(4.0, min(height, width) / 8.0))
    window = (np.hanning(height)[:, None] * np.hanning(width)[None, :]).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2((detail * window).astype(np.float64)))
    magnitude = np.abs(spectrum).astype(np.float32)
    freq_y = (np.arange(height, dtype=np.float32) - height // 2) / float(height)
    freq_x = (np.arange(width, dtype=np.float32) - width // 2) / float(width)
    grid_x, grid_y = np.meshgrid(freq_x, freq_y)
    radius = np.sqrt(grid_x * grid_x + grid_y * grid_y)
    valid = (
        (radius >= 1.0 / max(max_period, 1.0))
        & (radius <= 1.0 / max(min_period, 1.0))
        & (magnitude > 0)
    )
    if int(valid.sum()) < 256:
        return 0.0, 0.0
    angles = np.degrees(np.arctan2(grid_y[valid], grid_x[valid])) % 180.0
    values = magnitude[valid]
    bin_count = 180
    bin_index = np.clip((angles / (180.0 / bin_count)).astype(np.int32), 0, bin_count - 1)
    order = np.argsort(bin_index, kind="stable")
    sorted_bins = bin_index[order]
    sorted_values = values[order]
    boundaries = np.searchsorted(sorted_bins, np.arange(bin_count + 1))
    scores = np.zeros(bin_count, dtype=np.float64)
    for index in range(bin_count):
        start, stop = int(boundaries[index]), int(boundaries[index + 1])
        if stop - start < 24:
            continue
        chunk = sorted_values[start:stop]
        scores[index] = float(chunk.max()) / (float(np.median(chunk)) + EPS)
    best = int(np.argmax(scores))
    if scores[best] <= 0.0:
        return 0.0, 0.0
    left = scores[(best - 1) % bin_count]
    right = scores[(best + 1) % bin_count]
    denominator = left - 2.0 * scores[best] + right
    offset = 0.0
    if abs(denominator) > EPS:
        offset = max(-0.5, min(0.5, 0.5 * (left - right) / denominator))
    return float(((best + offset) * (180.0 / bin_count)) % 180.0), float(scores[best])


def refine_angle(residual: np.ndarray, angle_deg: float, period_px: float, min_period: float, max_period: float):
    """固定周期扫描角度，取折叠能量最大的方向。"""
    if period_px <= 0.0:
        return angle_deg
    width = residual.shape[1]
    scale = min(1.0, ANGLE_SEARCH_WIDTH / float(width), max(0.05, period_px / 4.0))
    if scale < 1.0:
        size = (max(24, int(round(residual.shape[1] * scale))), max(24, int(round(residual.shape[0] * scale))))
        work = resize_rgb(residual, size, Image.BILINEAR)
        period = period_px * scale
    else:
        work = residual
        period = period_px
    period = max(2.5, period)

    def scan(center: float, span: float, step: float) -> tuple[float, float]:
        best_angle, best_score = center % 180.0, -1.0
        count = int(round(2 * span / step))
        for index in range(count + 1):
            candidate = (center - span + index * step) % 180.0
            profile, _ = project_profile(work, candidate)
            score = folded_energy(poly_detrend(profile), period)
            if score > best_score:
                best_score = score
                best_angle = candidate
        return best_angle, best_score

    coarse_angle, _ = scan(angle_deg, 3.0, 0.5)
    fine_angle, _ = scan(coarse_angle, 0.6, 0.1)
    return fine_angle


def detect_stripe(residual: np.ndarray, angle_override: float | None, min_period: float, max_period: float):
    """确定条纹法线方向与周期模板，返回 (角度, 模板或 None, 诊断信息)。"""
    diagnostics: dict[str, object] = {
        "angle_source": "manual" if angle_override is not None else "auto",
        "angle_score": None,
    }
    local_min = min_period

    def analyze(theta: float):
        profile, origin = project_profile(residual, theta)
        detrended = poly_detrend(profile)
        peak_index, score, coarse = detect_peak(detrended, local_min, max_period, PEAK_RATIO)
        if score <= 0.0:
            return None
        period = resolve_period(detrended, coarse, local_min, max_period)
        if period <= 0.0:
            return None
        template, folded = fold_template(detrended, period)
        luma_variance = float(detrended.mean(axis=0).var())
        return {
            "angle": theta,
            "origin": origin,
            "bins": spectrum_bins(template),
            "fundamental": peak_index,
            "significance": score,
            "period": period,
            "fold_explained": folded_energy(detrended, period) / (luma_variance + EPS),
            "amplitude": [
                float((folded[channel].max() - folded[channel].min()) / 2.0)
                for channel in range(folded.shape[0])
            ],
            "template": template,
        }

    candidates: list[float] = []
    if angle_override is not None:
        candidates.append(float(angle_override) % 180.0)
    else:
        raw_angle, auto_score = estimate_normal_angle(luma(residual), local_min, max_period)
        diagnostics["angle_score"] = round(auto_score, 3)
        if auto_score >= 3.0:
            candidates.append(raw_angle % 180.0)
        else:
            diagnostics["angle_source"] = "axis-fallback"
        for fallback in (0.0, 90.0):
            if not candidates or min(abs(fallback - item) for item in candidates) > 4.0:
                candidates.append(fallback)

    best = None
    for theta in candidates:
        result = analyze(theta)
        if result is not None and (best is None or result["significance"] > best["significance"]):
            best = result
    if best is None:
        return 0.0, None, diagnostics

    if angle_override is None and best["significance"] >= STRONG_SIGNIFICANCE:
        refined = refine_angle(residual, best["angle"], best["period"], local_min, max_period)
        if min(abs(refined - best["angle"]), abs(refined - best["angle"] + 180.0)) > 0.05:
            candidate = analyze(refined)
            if candidate is not None and candidate["significance"] >= best["significance"]:
                best = candidate

    diagnostics.update(
        {
            "angle": round(float(best["angle"]), 3),
            "bins": best["bins"],
            "fundamental": best["fundamental"],
            "significance": round(float(best["significance"]), 3),
            "period_analysis_px": round(float(best["period"]), 4),
            "fold_explained": round(float(best["fold_explained"]), 4),
            "amplitude": [round(value, 5) for value in best["amplitude"]],
        }
    )
    return float(best["angle"]), (best["template"], best["origin"]), diagnostics


# --------------------------------------------------------------------------
# 增益估计与去除
# --------------------------------------------------------------------------


def structure_guard(gray: np.ndarray, sigma: float, strength: float, normal_angle_deg: float) -> np.ndarray:
    """在纹理方向与条纹方向一致的位置抑制过减。"""
    if strength <= 0.0:
        return np.ones_like(gray, dtype=np.float32)
    grad_y, grad_x = np.gradient(gray)
    gyy = gaussian_blur(grad_y * grad_y, sigma)
    gxx = gaussian_blur(grad_x * grad_x, sigma)
    gxy = gaussian_blur(grad_x * grad_y, sigma)
    coherence = np.sqrt((gxx - gyy) ** 2 + 4.0 * gxy * gxy) / (gxx + gyy + EPS)
    gradient_angle = 0.5 * np.arctan2(2.0 * gxy, gxx - gyy)
    alignment = np.abs(np.cos(gradient_angle - float(math.radians(normal_angle_deg))))
    return (1.0 - strength * np.clip(coherence, 0.0, 1.0) * alignment).astype(np.float32)


def remove_stripes(
    target: np.ndarray,
    residual: np.ndarray,
    field: np.ndarray,
    reference: np.ndarray | None,
    preset: dict,
    guard_override: float | None,
    period: float,
    normal_angle_deg: float,
):
    """估计空间增益并做减法，返回 (去除场, 增益图)。"""
    sigma = max(2.0, preset["period_sigma"] * period)
    numerator = gaussian_blur(residual * field, sigma)
    squared = np.ascontiguousarray(field.astype(np.float32) ** 2)
    denominator = gaussian_blur(squared, sigma)
    local_gain = numerator / (denominator + EPS)

    global_gain = (residual * field).sum(axis=(0, 1)) / (squared.sum(axis=(0, 1)) + EPS)
    gain = (1.0 - preset["shrink"]) * local_gain + preset["shrink"] * global_gain.reshape(1, 1, -1)

    luma_template = field.mean(axis=2)
    luma_gain = gaussian_blur(luma(residual) * luma_template, sigma) / (
        gaussian_blur(luma_template * luma_template, sigma) + EPS
    )
    gain = 0.8 * gain + 0.2 * luma_gain[:, :, None]

    confidence = denominator / (denominator + 0.04 * float(denominator.max()) + EPS)
    gain = np.clip(gain, -1.6, 1.6) * np.clip(confidence, 0.0, 1.0)
    gain = gaussian_blur(gain, max(1.0, sigma * 0.5))

    guard_strength = preset["guard"] if guard_override is None else float(guard_override)
    guard_source = luma(reference) if reference is not None else luma(target)
    guard = structure_guard(guard_source, max(2.0, period), guard_strength, normal_angle_deg)

    gain = gain * guard[:, :, None] * preset["gain_scale"]
    return gain * field, gain


# --------------------------------------------------------------------------
# QC 辅助
# --------------------------------------------------------------------------


def periodic_energy(profile: np.ndarray, bins: list[int]) -> float:
    spectrum = np.fft.rfft(profile.astype(np.float64), axis=1)
    total = 0.0
    for index in bins:
        if 0 < index < spectrum.shape[1]:
            total += float((np.abs(spectrum[:, index]) ** 2).sum())
    return total


def to_display(array: np.ndarray) -> np.ndarray:
    field = array if array.ndim == 2 else luma(array)
    scale = float(np.percentile(np.abs(field), 99.5)) + EPS
    normalized = np.clip(field / scale * 0.5 + 0.5, 0.0, 1.0)
    return np.repeat(normalized[:, :, None], 3, axis=2)


# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="参考图引导的条状水印去除：对比低清无水印参考图与高清带条状水印目标图，输出去水印结果。"
    )
    parser.add_argument("--reference", default=None, help="低分辨率、无水印的参考图路径")
    parser.add_argument("--target", required=True, help="高分辨率、带条状水印的目标图路径")
    parser.add_argument("--output", default=None, help="结果图路径，默认在目标图旁生成 <原名>_clean.png")
    parser.add_argument("--report", default=None, help="报告路径，默认在结果图旁生成 <原名>_report.json")
    parser.add_argument("--strength", choices=sorted(STRENGTH_PRESETS), default="balanced", help="去除力度")
    parser.add_argument("--angle", default="auto", help="条纹法线方向角度，0 为竖条纹、90 为横条纹；默认 auto")
    parser.add_argument("--min-period", type=float, default=None, help="条纹最小周期，单位像素")
    parser.add_argument("--max-period", type=float, default=None, help="条纹最大周期，单位像素")
    parser.add_argument("--guard", type=float, default=None, help="纹理保护强度 0 到 1，默认按力度档位选取")
    parser.add_argument("--min-significance", type=float, default=MIN_SIGNIFICANCE, help="判定存在条纹的最小周期显著度")
    parser.add_argument("--min-amplitude", type=float, default=MIN_PEAK_AMPLITUDE, help="判定存在条纹的最小条纹振幅")
    parser.add_argument("--no-reference", action="store_true", help="忽略参考图，改用目标图自身周期性估计条纹")
    parser.add_argument("--debug", action="store_true", help="额外输出残差、模板、增益等调试图")
    parser.add_argument("--dry-run", action="store_true", help="只分析并输出报告，不写结果图")
    parser.add_argument("--force", action="store_true", help="允许覆盖已存在的输出文件")
    parser.add_argument("--max-megapixels", type=float, default=DEFAULT_MAX_MEGAPIXELS, help="目标图像素数上限")
    parser.add_argument("--quality", type=int, default=95, help="输出为 JPEG 时的质量")
    parser.add_argument("--quiet", action="store_true", help="不向标准输出打印报告")
    return parser


def resolve_output_paths(target_path: Path, output: str | None, report: str | None, dry_run: bool):
    output_path = Path(output).expanduser() if output else target_path.with_name(target_path.stem + "_clean.png")
    report_path = (
        Path(report).expanduser() if report else output_path.with_name(output_path.stem + "_report.json")
    )
    if not dry_run and output_path.resolve() == target_path.resolve():
        raise InputError("结果图路径与目标图相同，会覆盖原图。请用 --output 指定其它路径。")
    return output_path, report_path


def emit(payload: dict, quiet: bool, stream=None) -> None:
    if not quiet:
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=stream if stream is not None else sys.stdout)


def fail(message: str, reason_code: str) -> int:
    emit({"status": "error", "reason_code": reason_code, "detail": message}, False, sys.stderr)
    return EXIT_INPUT_ERROR


def stripe_report(stripe_info: dict, analysis_scale: float, accepted: bool) -> dict:
    """把检测诊断整理成统一字段，成功与未检出两条路径共用。

    accepted 为假时说明该候选已被门限否决，只保留方向与判据数值供排查，
    周期、谐波与振幅一律不外泄，避免下游把否决掉的候选当作结论使用。
    """
    angle = stripe_info.get("angle")
    period_analysis = float(stripe_info.get("period_analysis_px") or 0.0)
    significance = float(stripe_info.get("significance") or 0.0)
    fold_explained = float(stripe_info.get("fold_explained") or 0.0)
    significance_factor = (significance - MIN_SIGNIFICANCE) / 26.0
    fold_factor = (fold_explained - MIN_FOLD_EXPLAINED) / 0.5
    return {
        "label": stripe_label(angle) if angle is not None else None,
        "normal_angle_deg": round(display_angle(angle), 3) if angle is not None else None,
        "angle_source": stripe_info.get("angle_source"),
        "angle_score": stripe_info.get("angle_score"),
        "period_px_target": (
            round(period_analysis / max(analysis_scale, EPS), 3)
            if accepted and period_analysis > 0
            else None
        ),
        "harmonics": [int(value) for value in stripe_info.get("bins", [])] if accepted else [],
        "significance": round(significance, 3),
        "fold_explained": round(fold_explained, 4),
        "amplitude": (
            [round(float(value), 5) for value in stripe_info.get("amplitude", [])] if accepted else []
        ),
        "confidence": round(min(1.0, max(0.0, min(significance_factor, fold_factor))), 3),
    }


def write_untouched_report(
    target: np.ndarray,
    target_path: Path,
    reference_path: Path | None,
    mode: str,
    alignment: dict,
    stripe_info: dict,
    warnings: list[str],
    output_path: Path,
    report_path: Path,
    analysis_scale: float,
    dry_run: bool,
    quality: int,
    quiet: bool,
) -> int:
    report = {
        "status": "no_watermark_detected",
        "target": str(target_path),
        "reference": str(reference_path) if reference_path else None,
        "mode": mode,
        "alignment": alignment,
        "stripes": stripe_report(stripe_info, analysis_scale, False),
        "warnings": warnings,
        "outputs": {},
        "evidence_label": "已验证",
    }
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_rgb(target, output_path, quality)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["outputs"] = {"clean": str(output_path), "report": str(report_path)}
    emit(report, quiet)
    return EXIT_NO_WATERMARK


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    target_path = Path(args.target).expanduser()
    if not target_path.is_file():
        return fail(f"目标图不存在或不是文件：{target_path}", "target_missing")
    if target_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return fail(f"不支持的目标图格式 {target_path.suffix}；支持 {sorted(SUPPORTED_SUFFIXES)}", "target_format")

    reference_path: Path | None = None
    if not args.no_reference:
        if not args.reference:
            return fail("缺少 --reference；若确实没有参考图，请显式加 --no-reference。", "reference_required")
        reference_path = Path(args.reference).expanduser()
        if not reference_path.is_file():
            return fail(f"参考图不存在或不是文件：{reference_path}", "reference_missing")
        if reference_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            return fail(f"不支持的参考图格式 {reference_path.suffix}", "reference_format")

    try:
        angle_override = None if str(args.angle).lower() == "auto" else float(args.angle) % 180.0
    except ValueError:
        return fail(f"--angle 需要 auto 或数值角度，收到 {args.angle!r}", "angle_invalid")
    if args.guard is not None and not 0.0 <= args.guard <= 1.0:
        return fail("--guard 需要落在 0 到 1 之间", "guard_invalid")

    target = load_rgb(target_path)
    target_h, target_w = target.shape[:2]
    megapixels = target_h * target_w / 1e6
    if megapixels > args.max_megapixels:
        return fail(
            f"目标图像素 {megapixels:.1f}M 超过上限 {args.max_megapixels:.1f}M。"
            "先降采样，或用 --max-megapixels 提高上限。",
            "target_too_large",
        )

    try:
        output_path, report_path = resolve_output_paths(target_path, args.output, args.report, args.dry_run)
    except InputError as exc:
        return fail(str(exc), "output_conflict")
    if not args.dry_run:
        for path in (output_path, report_path):
            if path.exists() and not args.force:
                return fail(f"输出文件已存在：{path}。确认后再加 --force。", "output_exists")

    analysis_width = min(target_w, ANALYSIS_MAX_WIDTH)
    analysis_scale = analysis_width / float(target_w)
    analysis_height = max(16, int(round(target_h * analysis_scale)))
    target_ana = resize_rgb(target, (analysis_width, analysis_height), Image.LANCZOS)

    warnings: list[str] = []
    mode = "self-reference" if reference_path is None else "reference-guided"
    reference_ana: np.ndarray | None = None
    alignment: dict[str, object] = {}
    if reference_path is not None:
        reference_ana, alignment = align_reference(load_rgb(reference_path), target_ana)
        ncc = float(alignment["ncc"])
        if ncc >= ALIGN_LEVEL_FULL:
            alignment["level"] = "L0"
        elif ncc >= ALIGN_LEVEL_PARTIAL:
            alignment["level"] = "L1"
            warnings.append("参考图与目标图对齐度中等，已降低参考图引导权重")
        else:
            alignment["level"] = "L2"
            mode = "self-reference"
            reference_ana = None
            warnings.append("参考图与目标图对齐度不足或内容不一致，已切换为自参考模式")
    else:
        alignment = {"level": "L2", "ncc": None}
        warnings.append("未提供参考图，使用目标图自身的周期性估计条纹")

    residual = target_ana if reference_ana is None else target_ana - reference_ana
    self_cutoff = None
    if reference_ana is None:
        self_cutoff = max(16.0, min(target_ana.shape[:2]) / 12.0)
        residual = residual - gaussian_blur(residual, self_cutoff)

    min_period = args.min_period if args.min_period else max(3.0, min(6.0, analysis_width / 200.0))
    if args.max_period:
        max_period = args.max_period
    elif reference_ana is None:
        max_period = max(32.0, analysis_width / 10.0)
    else:
        max_period = max(24.0, analysis_width / 3.0)
    if min_period >= max_period:
        return fail("--min-period 必须小于 --max-period", "period_range_invalid")

    min_significance = args.min_significance
    min_amplitude = args.min_amplitude
    if reference_ana is None:
        min_significance = max(min_significance, 8.0)
        min_amplitude = max(min_amplitude, 0.008)

    angle, template, stripe_info = detect_stripe(residual, angle_override, min_period, max_period)
    if template is None:
        return write_untouched_report(
            target, target_path, reference_path, mode, alignment, stripe_info,
            warnings + ["未检测到显著的条状水印，已原样输出目标图"],
            output_path, report_path, analysis_scale, args.dry_run, args.quality, args.quiet,
        )

    significance = float(stripe_info.get("significance", 0.0))
    amplitudes = list(stripe_info.get("amplitude", []))
    fold_explained = float(stripe_info.get("fold_explained", 0.0))
    if (
        significance < min_significance
        or (amplitudes and max(amplitudes) < min_amplitude)
        or fold_explained < MIN_FOLD_EXPLAINED
    ):
        return write_untouched_report(
            target, target_path, reference_path, mode, alignment, stripe_info,
            warnings + ["周期性显著度、条纹振幅或周期解释率不足，判定为无有效条状水印，已原样输出目标图"],
            output_path, report_path, analysis_scale, args.dry_run, args.quality, args.quiet,
        )
    if significance < STRONG_SIGNIFICANCE:
        warnings.append(f"条纹显著性 {significance:.1f} 偏低，结果置信度有限")

    template_1d, origin = template
    field = template_field(template_1d, residual.shape[:2], angle, origin)
    preset = dict(STRENGTH_PRESETS[args.strength])
    removal_ana, gain = remove_stripes(
        target_ana, residual, field, reference_ana, preset, args.guard,
        float(stripe_info.get("period_analysis_px", 20.0)), angle,
    )

    cleaned = np.empty_like(target)
    for channel in range(target.shape[2]):
        upscaled = resize_plane(removal_ana[:, :, channel], (target_w, target_h), Image.BICUBIC)
        cleaned[:, :, channel] = np.clip(target[:, :, channel] - upscaled, 0.0, 1.0)

    anchor_weight = {"L0": 1.0, "L1": 0.5}.get(str(alignment.get("level")), 0.0)
    anchor_sigma = None
    if reference_ana is not None and anchor_weight > 0.0:
        period_analysis = float(stripe_info.get("period_analysis_px", 20.0))
        anchor_sigma = max(24.0, 4.0 * period_analysis)
        anchor_scale = min(1.0, QC_MAX_WIDTH / float(target_w))
        anchor_w = max(16, int(round(target_w * anchor_scale)))
        anchor_h = max(16, int(round(target_h * anchor_scale)))
        cleaned_small = resize_rgb(cleaned, (anchor_w, anchor_h), Image.LANCZOS)
        reference_small = resize_rgb(reference_ana, (anchor_w, anchor_h), Image.LANCZOS)
        sigma_small = max(4.0, anchor_sigma * anchor_scale)
        delta_small = gaussian_blur(reference_small, sigma_small) - gaussian_blur(cleaned_small, sigma_small)
        for channel in range(target.shape[2]):
            delta = resize_plane(delta_small[:, :, channel], (target_w, target_h), Image.BICUBIC)
            cleaned[:, :, channel] = np.clip(cleaned[:, :, channel] + anchor_weight * delta, 0.0, 1.0)

    qc_scale = min(1.0, QC_MAX_WIDTH / float(target_w))
    qc_w = max(16, int(round(target_w * qc_scale)))
    qc_h = max(16, int(round(target_h * qc_scale)))
    target_qc = resize_rgb(target, (qc_w, qc_h), Image.LANCZOS)
    cleaned_qc = resize_rgb(cleaned, (qc_w, qc_h), Image.LANCZOS)

    bins = list(stripe_info.get("bins", []))
    cleaned_ana = target_ana - removal_ana
    # 沿检测出的那条法线做投影分箱，保证任意角度下 before/after 取样方向一致
    profile_before, _ = project_profile(residual, angle)
    after_residual = (
        cleaned_ana - reference_ana
        if reference_ana is not None
        else cleaned_ana - gaussian_blur(cleaned_ana, self_cutoff)
    )
    profile_after, _ = project_profile(after_residual, angle)
    suppress_db = 10.0 * math.log10(
        (periodic_energy(profile_before, bins) + EPS) / (periodic_energy(profile_after, bins) + EPS)
    )

    period_full = float(stripe_info.get("period_analysis_px", 20.0)) / max(analysis_scale, EPS)
    period_qc = period_full * qc_scale
    detail_cut = max(2.0, DETAIL_CUT_RATIO * period_qc)
    detail_before = target_qc - gaussian_blur(target_qc, detail_cut)
    detail_after = cleaned_qc - gaussian_blur(cleaned_qc, detail_cut)
    detail_retention = overlap_ncc(detail_before, detail_after)
    detail_energy_ratio = float(detail_after.std() / (float(detail_before.std()) + EPS))

    lf_delta = None
    if reference_ana is not None:
        low_sigma = max(8.0, 3.0 * period_qc)
        low_cleaned = gaussian_blur(cleaned_qc, low_sigma)
        low_reference = gaussian_blur(resize_rgb(reference_ana, (qc_w, qc_h), Image.LANCZOS), low_sigma)
        lf_delta = float(
            np.abs(low_cleaned - low_reference).mean() / (np.abs(low_reference).mean() + EPS) * 100.0
        )

    mean_shift = float(np.abs(cleaned - target).mean() * 255.0)

    if suppress_db < 6.0:
        warnings.append(f"条纹抑制比仅 {suppress_db:.1f} dB，可能存在条纹残留")
    if detail_retention < DETAIL_RETENTION_MIN:
        warnings.append(f"高频结构相关度 {detail_retention:.3f} 偏低，检查是否误伤了真实纹理")
    if detail_energy_ratio < DETAIL_ENERGY_MIN:
        warnings.append(f"高频能量降至 {detail_energy_ratio:.3f}，存在过减迹象")
    if detail_energy_ratio > DETAIL_ENERGY_MAX:
        warnings.append(f"高频能量升至 {detail_energy_ratio:.3f}，可能存在过冲")

    verdict = (
        "pass"
        if suppress_db >= 6.0
        and detail_retention >= DETAIL_RETENTION_MIN
        and DETAIL_ENERGY_MIN <= detail_energy_ratio <= DETAIL_ENERGY_MAX
        else "review"
    )

    report = {
        "status": "ok" if verdict == "pass" else "review",
        "target": str(target_path),
        "reference": str(reference_path) if reference_path else None,
        "mode": mode,
        "sizes": {
            "target": [target_w, target_h],
            "analysis": [analysis_width, analysis_height],
            "analysis_scale": round(analysis_scale, 6),
        },
        "alignment": alignment,
        "stripes": stripe_report(stripe_info, analysis_scale, True),
        "removal": {
            "strength": args.strength,
            "guard": preset["guard"] if args.guard is None else args.guard,
            "low_freq_anchor_sigma_px": None if anchor_sigma is None else round(anchor_sigma, 2),
            "low_freq_anchor_weight": anchor_weight,
            "mean_abs_change_255": round(mean_shift, 4),
        },
        "qc": {
            "suppress_db": round(suppress_db, 2),
            "detail_retention": round(detail_retention, 4),
            "detail_energy_ratio": round(detail_energy_ratio, 4),
            "low_freq_delta_pct": None if lf_delta is None else round(lf_delta, 4),
            "verdict": verdict,
        },
        "warnings": warnings,
        "outputs": {},
        "evidence_label": "已验证",
    }

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_rgb(cleaned, output_path, args.quality)
        report["outputs"] = {"clean": str(output_path), "report": str(report_path)}
        if args.debug:
            debug_dir = output_path.with_name(output_path.stem + "_debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            save_rgb(to_display(residual), debug_dir / "residual.png")
            save_rgb(to_display(removal_ana), debug_dir / "removal.png")
            save_rgb(to_display(gain), debug_dir / "gain.png")
            save_rgb(to_display(field), debug_dir / "template.png")
            if reference_ana is not None:
                save_rgb(reference_ana, debug_dir / "aligned_reference.png")
            report["outputs"]["debug_dir"] = str(debug_dir)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    emit(report, args.quiet)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
