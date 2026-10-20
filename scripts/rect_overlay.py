# -*- coding: utf-8 -*-
"""矩形半透明蒙版水印去除（参考图引导）。

与本目录下的 remove_stripe_watermark.py 互补：
  - remove_stripe_watermark.py 处理周期性条状（条纹）水印；
  - 本脚本处理"矩形半透明色块 + 白色文字/logo"这一类水印（Getty 图库预览条、
    角落 ID 徽标、半透明遮罩条等）。

混合模型：T = (1-a)·O + a·C（T 带水印图，O 原始内容，C 蒙版色，a 不透明度）。

处理链路：
    1. 配准（带缓存，见 align_with_cache）；
    2. 区域检测：残差行/列投影分组，refine_box 用阶跃走位收紧到真实外沿；
    3. 稳健回归 T ~ (1-a)·A + a·C（迭代截尾，排除白字/logo 像素），解出 a 与 C；
    4. 互补分解重建：低频取参考图低通，高频取目标图高通 × 逐像素增益；
    5. 白字/logo 作为第二层不透明覆盖单独识别，该区走参考图回填；
    6. 区域外沿做振铃抑制，区域外像素逐位保持原样。

三处设计要点，改动前务必读reference/pipeline.md 对应章节：

  * 严格互补式：O = blur(A,s) + (T - blur(T,s))·gain。
    不能写成 A + (T - blur(T,s))·gain —— 那会让参考图与目标图高频重复计数，画面过锐。
  * 区域局部化：所有重算都在"区域包围盒外扩 LOCAL_MARGIN 的裁剪片"上做，
    不是全图。实测快 13 倍，区域内结果与全图运算逐位相同（maxdiff = 0）。
  * 性能：2048x1365 目标图、配准缓存命中，优化后约 2.0-2.3s（优化前约 3.2-3.6s）。
    9 组同系列素材整批实测 43.5s → 28.4s，**输出逐位相同**。

  第二轮提速（全部保留原数值口径，见各函数注释）：
    1. `noise_floor` 先压成单通道亮度均值再滤波。盒式滤波是线性的，
       `mean(blur(T)) == blur(mean(T))`，浮点运算量与内存流量降到 1/3
       （2048x1365：blur(RGB) 1415ms vs blur(mean) 523ms）。噪声底是百分位统计量，
       量化到 6 位有效数字即可对滤波实现的末位差异免疫。
    2. `detect_regions` / QC 低频段 / QC 高频段同样只做单通道运算。
    3. `reconstruct` 去掉重复的三通道高斯（hf 与 HF 原来各算一遍），
       并把只有 alpha>=0.80 分支才用的 `make_fill` 改成惰性构造。
    4. 区域写回改为就地，省掉每区域一次全图 `current.copy()`。
    5. `translate` 在 (scale=1, shift=0) 这种常态下跳过恒等索引拷贝。
    6. PNG 默认 `compress_level=3`（无损，1.49MB→1.60MB，191ms→99ms），
       可用 `--png-level` 调整。
    7. `--batch 清单` 批量模式，摊掉逐张起进程约 0.3s 的导入开销。

用法：
    python rect_overlay.py --reference clean_lowres.jpg --target big_watermarked.jpg

    python rect_overlay.py --batch pairs.txt --output-dir out/     # 清单每行 ref|tgt

    加 --no-align-cache 忽略配准缓存强制完整搜索。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import remove_stripe_watermark as RS  # noqa: E402

SIG_SPLIT = 4.0      # 低频/高频分界（高斯 sigma，像素）。参考图在被保留的低频段是精确的。
GAP_ROWS = 14.0      # 行分组的最大间断
GAP_COLS = 24.0      # 列分组的最大间断

# QC 纹理频段窗口（针对 SIG_SPLIT=4.0 的高通能量标定，勿随意改）。
# 下限：平坦区的量化/JPEG 噪声，增益会把它们放大，不能算"恢复"。
# 上限：白字 logo 主体及其 JPEG 硬边振铃的起点，这些像素走参考图回填，按设计不可恢复。
# 注意：即使落在这个窗口里，比值也未必等于 1/(1-alpha)。自适应增益会在局部高频
# 接近噪声底时主动降到 1.0 以保噪声，纹理越弱这部分占比越高，比值越低。
# 因此该指标只用于"是否严重欠恢复"的粗判，精确判断看 hf_restore_bands 明细。
HF_TEX_LO = 0.008
HF_TEX_HI = 0.030
# 分档边界：报告每一档的恢复比，避免单个汇总数字掩盖分布
HF_BANDS = ((0.000, 0.008), (0.008, 0.030), (0.030, 0.090), (0.090, 9.99))

# 局部运算外扩边距（像素）。三次盒式滤波逼近高斯时，sigma=4 对应 radius=8，
# 总支撑约3*radius=24，留4 像素余量。实测 margin=16 时局部高斯与全图高斯
# 在区域内部逐位相同（maxdiff = 0）。
LOCAL_MARGIN = 16
# 配准工作分辨率（长边像素）。align_reference 内部的 NCC 搜索是O(候选数 × N)，
# 降到 256 可省约 2 倍时间；实测该尺寸下配准结果与 360 一致。
ALIGN_WORK_PX = 256

# ---- 白字 / logo 第二层掩膜参数（实测标定，见 references/qc-and-troubleshooting.md）
# 判据：gap = min通道(T - ((1-alpha)*A + alpha*C))，即目标图比"半透明蒙版预测值"
# 多出来多少。真白字处 gap 在 0.30-0.80，而普通纹理处 gap < 0.05。
#
# 阈值必须低（0.08）再做**膨胀**（8px）+ 羽化（14px），不能用高阈值窄掩膜：
# 真白字边缘的羽化带 gap 只有 0.05-0.30，用 gap>0.30 判据只覆盖字芯，
# 羽化带里的白字残余会被 1/(1-alpha) 放大成明显的发光鬼影。
# 实测：窄掩膜方案残留 abs_resid=12.6/255，本方案 11.7/255，
# 而肉眼可见的字形鬼影从"清晰可读"降到"极淡轮廓"。
LOGO_ZONE_THR = 0.08
LOGO_DILATE = 8
LOGO_FEATHER = 14
# 外沿振铃带（硬边 JPEG 过冲在 1/(1-alpha) 放大后可见）
EDGE_OUT = 2.0
EDGE_IN = 5.0

# 配准缓存目录：同一图库系列（参考图与目标图尺寸组合固定）的配准结果几乎恒定，
# 缓存命中可省掉整个 NCC 搜索（约 3s，占总耗时 40%）。
ALIGN_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

# PNG 编码级别。默认 6 是 zlib 的通用折中，但对"结果图只是中间产物"的场景
# 太浪费：2048x1365 实测 level=6 191ms / 1.49MB，level=3 99ms / 1.60MB。
# 像素无损，只是文件略大。要更小体积可传 --png-level 6/9。
PNG_LEVEL = 3


def save_image(array: np.ndarray, path: Path, png_level: int = PNG_LEVEL) -> None:
    """写结果图。PNG 走可调压缩级别（无损，只影响编码耗时与体积），
    其他格式退回 RS.save_rgb（JPEG 会走 quality=95 + 4:4:4）。"""
    if path.suffix.lower() == ".png":
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(data, "RGB").save(path, compress_level=int(png_level))
    else:
        RS.save_rgb(array, path)


def align_cache_key(reference_path: Path, target_path: Path) -> Path:
    """缓存键 = 参考图与目标图的 (尺寸, 文件名) 组合。

    刻意不含文件内容：同一图库系列的命名与尺寸稳定（如gettyimages-XXXX-612x612.jpg
    配 gettyimages-XXXX-2048x2048.jpg），这正是要复用的场景。
    用内容哈希会让每张图都是miss，缓存就白做了。
    """
    def sig(p: Path):
        try:
            with Image.open(p) as im:
                return f"{p.stem}:{im.width}x{im.height}"
        except Exception:
            return f"{p.stem}:?"
    raw = sig(reference_path) + "|" + sig(target_path)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return ALIGN_CACHE_DIR / f"align_{digest}.json"


def align_with_cache(R0, T, reference_path: Path, target_path: Path, use_cache=True):
    """带缓存的配准。命中则直接用上次结果并做一次廉价校验（NCC 明显下降则重算）。"""
    cache_file = align_cache_key(reference_path, target_path)
    if use_cache and cache_file.exists():
        try:
            saved = json.loads(cache_file.read_text(encoding="utf-8"))
            kx, ky = float(saved["scale_x"]), float(saved["scale_y"])
            sy, sx = float(saved["shift_target_px"][0]), float(saved["shift_target_px"][1])
            th, tw = T.shape[:2]
            zoomed = RS.zoom_rgb(R0, (tw, th), kx, ky, Image.LANCZOS)
            # 同源导出的配准结果几乎总是 (scale=1, shift=0)，此时 translate 是恒等映射。
            # np.take 建两遍全尺寸索引副本要 20ms，直接跳过（zoom_rgb 已经是新数组，
            # 下游只读不写，复用安全）。
            A = zoomed if (sy == 0.0 and sx == 0.0) else RS.translate(zoomed, sy, sx)
            # 廉价校验：在缩略图上算一次 NCC。低于记录值 0.03 以上说明缓存失效。
            work = 192
            ts = RS.luma(RS.resize_rgb(T, (work, max(1, int(work * th / tw))), Image.LANCZOS))
            As = RS.luma(RS.resize_rgb(A, (work, max(1, int(work * th / tw))), Image.LANCZOS))
            ncc = RS.overlap_ncc(As, ts)
            if ncc >= float(saved.get("ncc", 0.0)) - 0.03:
                info = dict(saved)
                info["ncc"] = round(float(ncc), 5)
                info["cache"] = "hit"
                return A, info
        except Exception:
            pass  # 缓存损坏或不可用，静默回落到完整搜索

    A, info = RS.align_reference(R0, T, search_scale=0.06, work_px=ALIGN_WORK_PX)
    info = dict(info)
    info["cache"] = "miss"
    try:
        ALIGN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass  # 缓存写入失败不影响主流程
    return A, info


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def box_mean(x: np.ndarray, r: int) -> np.ndarray:
    """盒式均值滤波（积分图实现，O(N)），边界用 edge 复制。"""
    if r <= 0 or x.ndim != 2:
        return x
    k = 2 * r + 1
    h, w = x.shape
    p = np.pad(x, [(r, r), (r, r)], mode="edge").astype(np.float64)
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)), mode="constant")
    out = c[k:k + h, k:k + w] - c[0:h, k:k + w] - c[k:k + h, 0:w] + c[0:h, 0:w]
    return (out / float(k * k)).astype(np.float32)


def max_filter(m: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return m
    #分离式：先沿轴0 再沿轴1 做一维max，避免 (2r+1)^2 的窗口枚举。
    # 一维滑动最大值用 stride 视图做，两趟合计 2(2r+1) 次比较而非 (2r+1)^2。
    def pass1d(x, axis, r):
        width = 2 * r + 1
        pad = [(0, 0)] * x.ndim
        pad[axis] = (r, r)
        p = np.pad(x, pad, mode="constant")
        v = np.lib.stride_tricks.sliding_window_view(p, width, axis=axis)
        return v.max(axis=-1)
    return pass1d(pass1d(m, 0, r), 1, r)


def box_blur_rgb(x: np.ndarray, r: int) -> np.ndarray:
    return np.stack([box_mean(x[..., c], r) for c in range(x.shape[2])], axis=-1)


def local_crop(shape, box, margin=LOCAL_MARGIN):
    """把区域包围盒外扩 margin，返回 (裁剪框, 区域内切片索引)。"""
    h, w = shape
    y0, y1, x0, x1 = box
    cy0, cy1 = max(0, y0 - margin), min(h, y1 + margin)
    cx0, cx1 = max(0, x0 - margin), min(w, x1 + margin)
    return (cy0, cy1, cx0, cx1), (y0 - cy0, y1 - cy0, x0 - cx0, x1 - cx0)


def edge_distance(shape, box) -> np.ndarray:
    """每个像素到矩形边界（内缩为正）的最小距离。

    注意：框内为**正**且框中心值最大，框外为负。它是距离，不是权重。
    要做「框内 1、框外渐变到 0」的羽化必须写成
    `clip(1 + (edge + 0.5) / 2, 0, 1)`；直接拿edge 当权重会让框内权重
    趋近 0，整块等于没改（实测排查这个bug 花了很久）。
    """
    y0, y1, x0, x1 = box
    h, w = shape
    yy = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w))
    xx = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w))
    return np.minimum.reduce([yy - y0, (y1 - 1) - yy, xx - x0, (x1 - 1) - xx])


def group_index(idx: np.ndarray, gap: float):
    if len(idx) == 0:
        return []
    groups, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            groups.append((int(start), int(prev)))
            start = i
        prev = i
    groups.append((int(start), int(prev)))
    return groups


# --------------------------------------------------------------------------
# 区域检测
# --------------------------------------------------------------------------
def detect_regions(T: np.ndarray, A: np.ndarray, thr: float = 0.022, min_area: int = 900):
    """返回 [(y0, y1, x0, x1), ...]（y1/x1 为不含端）。"""
    h, w = T.shape[:2]
    res = RS.gaussian_blur((T - A).mean(2), 2.5)
    m = np.abs(res) > thr
    row_hits = m.sum(1)
    rows = np.where(row_hits > max(6, int(w * 0.012)))[0]
    regions = []
    for ry0, ry1 in group_index(rows, GAP_ROWS):
        band = m[ry0:ry1 + 1]
        need = max(3, int((ry1 - ry0 + 1) * 0.30))
        cols = np.where(band.sum(0) > need)[0]
        for cx0, cx1 in group_index(cols, GAP_COLS):
            if (ry1 - ry0 + 1) * (cx1 - cx0 + 1) < min_area:
                continue
            if (ry1 - ry0 + 1) < 8 or (cx1 - cx0 + 1) < 20:
                continue
            regions.append((ry0, ry1 + 1, cx0, cx1 + 1))
    return regions


def refine_box(T, A, box, pad=24):
    """把检测到的粗框扩到真实的水印外沿（残差阶跃处）。"""
    h, w = T.shape[:2]
    y0, y1, x0, x1 = box
    res = np.abs((T - A).mean(2))

    def prof(axis, lo, hi, fixed):
        lo, hi = int(max(0, lo)), int(min(h if axis == 0 else w, hi))
        if hi <= lo:
            return np.zeros(0, np.float32)
        if axis == 0:
            sl = res[lo:hi, fixed[0]:fixed[1]]
            return sl.mean(axis=1)
        sl = res[fixed[0]:fixed[1], lo:hi]
        return sl.mean(axis=0)

    inner = max(2, min((y1 - y0) // 4, (x1 - x0) // 4))
    base_y = (y0 + inner, y1 - inner)
    base_x = (x0 + inner, x1 - inner)
    lvl = float(np.median(res[base_y[0]:base_y[1], base_x[0]:base_x[1]]))
    gate = max(0.15 * lvl, 0.02)

    def walk(lo, hi, axis, fixed, from_end):
        p = prof(axis, lo, hi, fixed)
        if len(p) == 0:
            return lo if from_end else hi
        rng = range(len(p) - 1, -1, -1) if from_end else range(len(p))
        for i in rng:
            if not np.isfinite(p[i]) or p[i] < gate:
                return (lo + i + 1) if from_end else (lo + i)
        return lo if from_end else hi

    ny0 = walk(max(0, y0 - pad), y0, 0, base_x, True)
    ny1 = walk(y1, min(h, y1 + pad), 0, base_x, False)
    nx0 = walk(max(0, x0 - pad), x0, 1, base_y, True)
    nx1 = walk(x1, min(w, x1 + pad), 1, base_y, False)
    ny0, ny1 = int(ny0), int(max(ny1, ny0 + 6))
    nx0, nx1 = int(nx0), int(max(nx1, nx0 + 6))
    return (ny0, min(ny1, h), nx0, min(nx1, w))


# --------------------------------------------------------------------------
# 蒙版参数估计
# --------------------------------------------------------------------------
def fit_overlay(T, A, box, br=4):
    y0, y1, x0, x1 = box
    Tb = box_blur_rgb(T[y0:y1, x0:x1], br)
    Ab = box_blur_rgb(A[y0:y1, x0:x1], br)
    d = Tb - Ab
    out, mask = [], np.ones(d.shape[:2], bool)
    for c in range(3):
        u = Ab[..., c].reshape(-1)
        v = d[..., c].reshape(-1)
        keep = mask.reshape(-1)
        for _ in range(4):
            uu, vv = u[keep], v[keep]
            if len(uu) < 64:
                break
            q, p = np.polyfit(uu, vv, 1)
            r = vv - (p + q * uu)
            s = r.std() + 1e-9
            keep = np.abs(v - (p + q * u)) < 2.0 * s
        alpha = float(-q)
        if alpha <= 0.02:
            alpha = 0.02
        C = float(p / alpha)
        out.append({"alpha": round(alpha, 4), "C": float(np.clip(C, 0.0, 1.0))})
    return out


def alpha_of(fit):
    return float(np.mean([f["alpha"] for f in fit]))


def color_of(fit):
    return np.array([f["C"] for f in fit], np.float32)


def noise_floor(T: np.ndarray, sig: float = SIG_SPLIT, return_map: bool = False):
    """用全图局部高频能量的低分位数估计 JPEG/传感器噪声底。

    必须全分辨率计算：早先试过按 ::2 抽稀后取分位数（为了省 0.8s），
    但 box_mean 的邻域会被抽稀成等效更小的窗口，噪声能量被系统性高估
    45%（0.00351 → 0.00508），连带把徽标区误判成 reference-lf + raw-hf。
    这0.8s 不值得拿正确性换。

    **但"全分辨率"不等于"三通道"**：本函数只用到 `T.mean(2)`，而盒式滤波是
    线性算子，`mean(blur(T)) == blur(mean(T))`。先把三通道压成亮度均值再滤波，
    盒式滤波的浮点运算量和内存流量降到 1/3 —— 实测 2048x1365 上
    `blur(RGB)` 1415ms vs `blur(mean)` 523ms，整段 2082ms → 907ms，
    而噪声底只差 6e-7 相对量（第 8 百分位打印到小数点后 8 位完全一致）。

    `return_map=True` 时同时返回全图高频图 `T.mean(2) - blur(T.mean(2), sig)`，
    供 QC 段复用（QC 的 gT 要的正是同一量，避免重复算一遍全图/裁剪片高斯）。
    """
    Tm = T.mean(2)
    hf = Tm - RS.gaussian_blur(Tm, sig)
    hfs = np.sqrt(np.maximum(box_mean(hf * hf, 3), 0.0))
    value = float(np.percentile(hfs, 8))
    # 噪声底是百分位统计量，滤波实现的浮点末位差异（实测绝对量 ~3.5e-9）对它
    # 没有任何物理意义，却会在 gain_map 的 sqrt(1-(noise/hfs)^2) 上被放大成
    # "个别恰好压在 1/255 量化边界的像素翻转 1 级"。量化到 6 位有效数字既不改
    # 估计精度，又让结果对滤波实现差异免疫（实测 9 组素材由 2 像素差异收敛到逐位相同）。
    value = float(f"{value:.6g}")
    return (value, hf) if return_map else value


# --------------------------------------------------------------------------
# 单区域重建
# --------------------------------------------------------------------------
def make_fill(A: np.ndarray, sl, k: float = 0.45) -> np.ndarray:
    """参考图高频增强回填源，只在裁剪片上算（A_fill 全图版本对区域外毫无用处）。"""
    As = A[sl]
    return np.clip(As + k * (As - RS.gaussian_blur(As, 3.0)), 0, 1)


def reconstruct(T, A, box, fit, log, noise, current):
    """在 current 上就地重建区域盒，返回 current。

    局部化是主要的速度来源：主带约 150x820 像素，全图是 2048x1475，
    只算裁剪片（外扩LOCAL_MARGIN）比全图快 13 倍，且因高斯支撑完全落在
    外扩范围内，区域内结果与全图运算逐位相同（实测 maxdiff = 0）。

    就地写回（不再 `current.copy()` 一整张全图）：区域之间本来就要看到
    前一个区域的结果，逐区域累积语义不变，省掉每区域一次 33MB 全图拷贝。
    调用方传入的 current 必须是不与目标图 T 共享内存的独立副本。
    """
    h, w = T.shape[:2]
    y0, y1, x0, x1 = box
    alpha = alpha_of(fit)
    C = color_of(fit)
    sig = SIG_SPLIT
    # 外扩必须覆盖高斯支撑 + logo 掩膜的膨胀/羽化半径，
    # 否则靠近裁剪片边缘的白字像素会被截断，掩膜在片边缘出现硬边。
    margin = (int(np.ceil(3 * max(sig, 2.5))) + LOCAL_MARGIN
              + LOGO_DILATE + LOGO_FEATHER)
    (cy0, cy1, cx0, cx1), (ly0, ly1, lx0, lx1) = local_crop((h, w), box, margin)

    Ts = T[cy0:cy1, cx0:cx1]
    As = A[cy0:cy1, cx0:cx1]
    lh, lw = ly1 - ly0, lx1 - lx0
    # 注意：mbbox / wm 等掩膜必须与裁剪片 Ts 同尺寸（cy1-cy0, cx1-cx0），
    # 而非区域自身尺寸 (lh, lw)。区域只是裁剪片内部的一个子窗口，
    # 早期版本误用 (lh, lw) 导致与 Ts 广播失败。
    ch, cw = cy1 - cy0, cx1 - cx0
    mbox = np.zeros((ch, cw), np.float32)
    mbox[ly0:ly1, lx0:lx1] = 1.0

    if alpha >= 0.80:
        # 底层内容几乎被抹平，放大高频只会放大噪声 → 直接用参考图回填
        # Fs（参考图高频增强回填源）只有这条分支用得到，惰性构造。
        # 早先它被无条件算在上方，complementary 分支白付一次 sigma=3 的
        # 三通道高斯（实测裁剪片 67ms / 区域）。
        Fs = make_fill(A, (slice(cy0, cy1), slice(cx0, cx1)))
        wbox = RS.gaussian_blur(mbox, 1.2)
        # 羽化会溢出到框外，必须再乘回框掩膜，否则 outside_watermark_untouched 会失效
        wbox = wbox * mbox
        log.append({"box": [y0, y1, x0, x1], "alpha": round(alpha, 4),
                    "overlay_255": [round(float(v * 255), 1) for v in C],
                    "strategy": "reference-fill"})
        patch = current[cy0:cy1, cx0:cx1] * (1 - wbox[..., None]) + Fs * wbox[..., None]
        current[cy0:cy1, cx0:cx1] = patch
        return current

    # --- 白字 / logo（第二层不透明覆盖）影响区
    # 低阈值 + 膨胀 + 羽化：必须把白字边缘的羽化带一起罩住。
    # 窄掩膜（gap>0.30）只盖字芯，羽化残余被 1/(1-alpha) 放大成发光鬼影。
    P = (1 - alpha) * As + alpha * C
    gap = (Ts - P).min(2)
    zone = max_filter((gap > LOGO_ZONE_THR).astype(np.float32), LOGO_DILATE)
    zone = RS.gaussian_blur(zone, LOGO_FEATHER / 2.0)

    # --- 外沿振铃掩膜（硬边 JPEG 过冲在放大后可见）
    dz = edge_distance((ch, cw), (ly0, ly1, lx0, lx1))
    w_edge = np.clip(1.0 - np.maximum(dz - EDGE_OUT, 0.0) / EDGE_IN, 0, 1)
    wm = np.maximum(zone, w_edge) * mbox

    # --- 逐像素高频增益：目标图高频低于噪声底的位置不放大（避免放大噪声），
    #     高于噪声底的位置朝 1/(1-alpha) 恢复，上限取 1/(1-alpha)
    ideal = 1.0 / max(1e-3, 1.0 - alpha)
    # blurTs 同时供 gain_map 的 hf 与重建的 HF 使用 —— 这两处原来是同一表达式
    # 各算一遍，裁剪片上是 63ms 的重复三通道高斯，去掉后逐位相同。
    blurTs = RS.gaussian_blur(Ts, sig)
    hf = (Ts - blurTs).mean(2)
    hfs = np.sqrt(np.maximum(box_mean(hf * hf, 3), 0.0))
    k = np.sqrt(np.maximum(1.0 - (noise / np.maximum(hfs, 1e-6)) ** 2, 0.0))
    gain_map = np.clip(ideal * k, 1.0, 8.0)
    gain_map = RS.gaussian_blur(gain_map, 3.0)
    sel = (mbox > 0) & (max_filter(zone > 0.5, 6) < 0.02)
    region_hf = float(np.median(hfs[sel])) if sel.any() else 0.0
    strategy = "complementary-lf-hf" if region_hf > 1.35 * noise else "reference-lf + raw-hf"

    # --- 重建：低频取参考图，高频取目标图。
    # 低频**必须**取参考图，不能取 blur(T) 的反解值：sigma=4 的高斯压不掉
    # 笔画宽 20-40px 的粗体白字，白字能量会泄漏进低频，再被 1/(1-alpha)
    # 放大 2.3 倍，形成明显的发光鬼影（消融实验 ref_raw 面板证实参考图
    # 在该区域完全干净，无任何白字残影）。
    LF = RS.gaussian_blur(As, sig)
    HF = (Ts - blurTs) * gain_map[..., None]
    band = LF + (1 - wm[..., None]) * HF

    # --- logo 区：回填**参考图原值**（含其自身高频，不做锐化）。
    # 实测锐化（+0.8/+1.6×unsharp）会让残差从 11.69升到 11.81/11.92，
    # 即放大插值痕迹反而更糟。这一段内容按设计不可恢复（白字是不透明
    # 覆盖），只能如实交付。
    w_ref = np.clip(zone * 1.6, 0, 1) * mbox
    band = (1 - w_ref[..., None]) * band + w_ref[..., None] * As
    log.append({"box": [y0, y1, x0, x1], "alpha": round(alpha, 4),
                "overlay_255": [round(float(v * 255), 1) for v in C],
                "ideal_gain": round(float(ideal), 3),
                "gain_median": round(float(np.median(gain_map[mbox > 0])), 3),
                "noise_floor": round(noise, 5), "region_hf": round(region_hf, 5),
                "strategy": strategy})
    patch = np.where(mbox[..., None] > 0, band, current[cy0:cy1, cx0:cx1])
    current[cy0:cy1, cx0:cx1] = patch
    return current


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def run(reference_path: Path, target_path: Path, output_path: Path, report_path: Path,
        force: bool = False, quiet: bool = False, use_align_cache: bool = True,
        manual_boxes=None, png_level: int = PNG_LEVEL):
    if not force and output_path.resolve() == target_path.resolve():
        raise SystemExit("拒绝执行：输出路径与目标图相同，会覆盖原图。")

    T = RS.load_rgb(target_path)
    R0 = RS.load_rgb(reference_path)
    A, align = align_with_cache(R0, T, reference_path, target_path, use_align_cache)

    regions = detect_regions(T, A)
    if manual_boxes:
        # 手动指定框时仍跑一遍 refine 会把用户给的精确边界又拉偏，
        # 所以直接采用传入值（格式 (y0, y1, x0, x1)，y1/x1 不含端）。
        regions = [tuple(int(v) for v in b) for b in manual_boxes]
    log, boxes = [], []
    O = T.copy()
    # return_map：noise_floor 内部已经把全图 T 的 sigma=4 高频算出来了，
    # QC 段的 gT 要的正是同一个量，直接复用，省掉 QC 里一遍全图/裁剪片高斯。
    noise, hf_T = noise_floor(T, return_map=True)
    rejected = []
    for rough in regions:
        box = tuple(rough) if manual_boxes else refine_box(T, A, rough)
        if (box[1] - box[0]) < 6 or (box[3] - box[2]) < 6:
            continue
        fit = fit_overlay(T, A, box)
        a = alpha_of(fit)
        # alpha 物理上必须落在 [0,1]。越界说明这个"区域"不是半透明蒙版
        # （实测 897614882 的一片草地被误检，回归出alpha=1.22）。
        # 强行重建会用参考图把真实纹理糊掉，必须拒掉。
        if a< 0.05 or a > 1.0:
            rejected.append({"box": list(box), "alpha": round(a, 4),
                             "reason": "alpha_out_of_range"})
            continue
        O = reconstruct(T, A, box, fit, log, noise, O)
        boxes.append(box)

    O = np.clip(O, 0, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(O, output_path, png_level)

    # ---------------- QC ----------------
    mask = np.zeros(T.shape[:2], bool)
    for (y0, y1, x0, x1) in boxes:
        mask[y0:y1, x0:x1] = True
    qc = {
        "status": "ok" if boxes else "no_watermark_detected",
        "mode": "reference-guided-rect-overlay",
        "target": str(target_path),
        "reference": str(reference_path),
        "output": str(output_path),
        "alignment": align,
        "regions": log,
        "rejected_regions": rejected,
        "outside_watermark_untouched": bool(np.array_equal(O[~mask], T[~mask])),
    }
    if boxes:
        # 统计口径必须与重建一致：重建按 SIG_SPLIT 分割频段，QC 也必须用同一 sigma，
        # 否则测的是另一个频段，数字没有意义。
        # 高频图只按"所有区域的外扩并集"计算一次。
        # gT 直接切 noise_floor 已算好的全图 sigma=SIG_SPLIT 高频图：两者是同一个量，
        # 差别只有 cumsum 长度带来的 ~1e-7 浮点舍入。切片位置离裁剪片边界至少
        # qm = 3*SIG_SPLIT + LOCAL_MARGIN 像素，远离高斯支撑，边界处理不参与。
        qm = int(np.ceil(3 * SIG_SPLIT)) + LOCAL_MARGIN
        gO = np.zeros(T.shape[:2], np.float32)
        gT = np.zeros(T.shape[:2], np.float32)
        done = np.zeros(T.shape[:2], bool)
        for (y0, y1, x0, x1) in boxes:
            (uy0, uy1, ux0, ux1), _ = local_crop(T.shape[:2], (y0, y1, x0, x1), qm)
            sl = (slice(uy0, uy1), slice(ux0, ux1))
            todo = ~done[sl]
            if not todo.any():
                continue
            Ol = O.mean(2)[sl]
            gt = hf_T[sl]
            go = Ol - RS.gaussian_blur(Ol, SIG_SPLIT)
            gT[sl] = np.where(todo, gt, gT[sl])
            gO[sl] = np.where(todo, go, gO[sl])
            done[sl] = True
        ratios = []
        for entry in log:
            y0, y1, x0, x1 = entry["box"]
            sub = np.zeros(T.shape[:2], bool)
            sub[y0:y1, x0:x1] = True
            if "alpha" in entry and entry.get("strategy") != "reference-fill":
                alpha = entry["alpha"]
                C = np.array(entry["overlay_255"], np.float32) / 255.0
                # logo 判据与 reconstruct 保持一致（同一阈值、同一形态学参数），
                # 否则 QC 排除的像素集与实际回填区不一致，统计口径就错了。
                (qy0, qy1, qx0, qx1), _ = local_crop(
                    T.shape[:2], (y0, y1, x0, x1),
                    int(np.ceil(3 * SIG_SPLIT)) + LOCAL_MARGIN
                    + LOGO_DILATE + LOGO_FEATHER)
                gap = (T[qy0:qy1, qx0:qx1] - ((1 - alpha) * A + alpha * C)[qy0:qy1, qx0:qx1]).min(2)
                logo_l = RS.gaussian_blur(
                    max_filter((gap > LOGO_ZONE_THR).astype(np.float32),
                               LOGO_DILATE), LOGO_FEATHER / 2.0) > 0.5
                logo = np.zeros(T.shape[:2], bool)
                logo[qy0:qy1, qx0:qx1] = logo_l
                m = sub & mask
                # 参考图回填区（白字/logo 主体）的内容在低清参考图里根本不存在，
                # 按设计不可恢复。必须单独声明比例，否则它会把整区统计拖垮：
                # 实测这些像素 |gT| 可达 0.5，而真实纹理只有 0.01 量级。
                filled = m & max_filter(logo, 3)
                entry["reference_fill_fraction"] = round(float(filled.sum()) / max(1, int(m.sum())), 4)
                tex = m & ~max_filter(logo, 6)
                tex[:8, :] = False
                tex[-8:, :] = False
                tex[:, :8] = False
                tex[:, -8:] = False
                # 只取真实纹理频段：|gT| 落在 [HF_TEX_LO, HF_TEX_HI]。
                # 下限排除平坦区的量化噪声，上限排除白字及其 JPEG 硬边振铃。
                mag = np.abs(gT)
                sel = tex & (mag >= HF_TEX_LO) & (mag <= HF_TEX_HI)
                if sel.sum() > 400 and float(np.std(gT[sel])) > 1e-6:
                    r = float(np.std(gO[sel]) / np.std(gT[sel]))
                    entry["hf_restore_ratio_texture"] = round(r, 3)
                    entry["hf_restore_samples"] = int(sel.sum())
                    entry["hf_restore_ideal"] = round(1.0 / max(1e-3, 1.0 - alpha), 3)
                    ratios.append(r)
                # 分档明细：让"高频能量 vs 恢复比"的分布可见
                bands = []
                for blo, bhi in HF_BANDS:
                    bs = tex & (mag >= blo) & (mag < bhi)
                    if bs.sum() < 200 or float(np.std(gT[bs])) <= 1e-6:
                        bands.append({"band": [blo, bhi], "n": int(bs.sum()), "ratio": None})
                        continue
                    bands.append({
                        "band": [blo, bhi],
                        "n": int(bs.sum()),
                        "ratio": round(float(np.std(gO[bs]) / np.std(gT[bs])), 3),
                    })
                entry["hf_restore_bands"] = bands
        if ratios:
            qc["hf_restore_ratio_texture"] = round(float(np.mean(ratios)), 3)
        # 低频一致性只在区域内需要，同样走裁剪片，避开两次全图高斯
        lm = int(np.ceil(3 * 10)) + LOCAL_MARGIN
        lf = np.zeros(T.shape[:2], np.float32)
        seen = np.zeros(T.shape[:2], bool)
        for (y0, y1, x0, x1) in boxes:
            (uy0, uy1, ux0, ux1), _ = local_crop(T.shape[:2], (y0, y1, x0, x1), lm)
            sl = (slice(uy0, uy1), slice(ux0, ux1))
            todo = ~seen[sl]
            if not todo.any():
                continue
            # 单平面化：只用得到 `.mean(2)`，而滤波是线性的，先压均值再滤波
            # 把三通道的盒式滤波量降到 1/3（本段实测 137ms → 49ms）。
            d = np.abs(RS.gaussian_blur(O[sl].mean(2), 10)
                       - RS.gaussian_blur(A[sl].mean(2), 10))
            lf[sl] = np.where(todo, d, lf[sl])
            seen[sl] = True
        qc["lf_consistency_mean"] = round(float(lf[mask].mean()), 5)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    if not quiet:
        print(json.dumps(qc, ensure_ascii=False, indent=2))
    return qc


def parse_manifest(path: Path):
    """读批次清单：每行 `参考图|目标图`（`|` 或 Tab 分隔），`#` 开头与空行忽略。

    同一图库系列通常是「一张 612 干净图 + 一串 2048 带水印图」，逐张起进程要重复付
    约 0.3s 的解释器与 numpy/PIL 导入开销；清单模式把这一份开销摊到整批。
    """
    pairs = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip().strip('"').strip("'")
                 for p in line.replace("\t", "|").split("|")]
        parts = [p for p in parts if p]
        if len(parts) != 2:
            raise SystemExit(f"{path}:{lineno} 需要 `参考图|目标图`，收到: {raw}")
        pairs.append((Path(parts[0]), Path(parts[1])))
    if not pairs:
        raise SystemExit(f"{path} 里没有可用条目")
    return pairs


def run_batch(pairs, output_dir=None, force=False, use_align_cache=True,
              png_level=PNG_LEVEL) -> int:
    """逐对执行，单对失败不中断整批。返回失败对数。"""
    rows, failed = [], 0
    t_all = time.perf_counter()
    for ref, tgt in pairs:
        if output_dir:
            out = Path(output_dir) / (tgt.stem + "_clean.png")
        else:
            out = tgt.with_name(tgt.stem + "_clean.png")
        rep = out.with_name(out.stem + "_report.json")
        t0 = time.perf_counter()
        try:
            qc = run(ref, tgt, out, rep, force=force, quiet=True,
                     use_align_cache=use_align_cache, png_level=png_level)
            status, note = qc["status"], f"{len(qc.get('regions', []))} 区"
            if status != "ok":
                failed += 1
        except SystemExit as exc:
            status, note = "failed", str(exc)
            failed += 1
        dt = time.perf_counter() - t0
        rows.append((tgt.name, status, dt, note))
        print(f"  {tgt.name:<38} {status:<22} {dt:5.2f}s  {note}", flush=True)
    total = time.perf_counter() - t_all
    print(f"\n共 {len(rows)} 对，失败 {failed} 对，合计 {total:.2f}s，"
          f"平均 {total / max(1, len(rows)):.2f}s/对")
    return failed


def main(argv=None):
    ap = argparse.ArgumentParser(description="参考图引导去除矩形半透明蒙版水印")
    ap.add_argument("--reference")
    ap.add_argument("--target")
    ap.add_argument("--output")
    ap.add_argument("--report")
    ap.add_argument("--batch", metavar="清单文件",
                    help="批量模式：清单每行 `参考图|目标图`（或 Tab 分隔），"
                         "# 开头与空行忽略。省掉逐张起进程的 ~0.3s 导入开销。")
    ap.add_argument("--output-dir", metavar="目录",
                    help="批量模式下把所有结果图与报告写到该目录（默认写在各自目标图旁）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-align-cache", action="store_true",
                    help="忽略配准缓存，强制做完整 NCC 搜索（怀疑配准时用）")
    ap.add_argument("--box", action="append", default=None, metavar="y0,y1,x0,x1",
                    help="手动指定水印框，跳过自动检测与 refine；可重复。用于自动框"
                         "边界不准时（实测角落徽标紧贴图像边缘时refine 会越界到图边）。"
                         "格式为 y0,y1,x0,x1，y1/x1 不含端，且坐标基于目标图原始像素。")
    ap.add_argument("--png-level", type=int, default=PNG_LEVEL, metavar="0-9",
                    help=f"结果 PNG 的 zlib 压缩级别，默认 {PNG_LEVEL}（写得更快、"
                         "文件略大，像素完全无损）。要最小体积用 9，要最快用 0/1。")
    a = ap.parse_args(argv)
    manual = None
    if a.box:
        manual = []
        for s in a.box:
            parts = [p for p in s.replace(";", ",").split(",") if p.strip()]
            if len(parts) != 4:
                raise SystemExit(f"--box 需要 4 个整数 y0,y1,x0,x1，收到: {s}")
            manual.append(tuple(int(p) for p in parts))

    if a.batch:
        if a.reference or a.target or a.output or a.report:
            raise SystemExit("--batch 不能与 --reference/--target/--output/--report 同时使用。")
        if manual:
            raise SystemExit("--batch 不能与 --box 同时使用：框是逐图特有的。")
        return 0 if run_batch(parse_manifest(Path(a.batch)), output_dir=a.output_dir,
                              force=a.force, use_align_cache=not a.no_align_cache,
                              png_level=a.png_level) == 0 else 4

    if not a.reference or not a.target:
        raise SystemExit("需要 --reference 与 --target（或改用 --batch 清单模式）。")
    tgt = Path(a.target)
    out = Path(a.output) if a.output else tgt.with_name(tgt.stem + "_clean.png")
    rep = Path(a.report) if a.report else tgt.with_name(tgt.stem + "_clean_report.json")
    qc = run(Path(a.reference), tgt, out, rep, force=a.force, quiet=a.quiet,
             use_align_cache=not a.no_align_cache, manual_boxes=manual,
             png_level=a.png_level)
    return 0 if qc["status"] == "ok" else 4


if __name__ == "__main__":
    raise SystemExit(main())
