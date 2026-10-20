#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B 分支（矩形蒙版）的水印区域体检与框标定工具。

它不修改任何图像，只做三件事，正好覆盖 rect_overlay.py 自动检测最容易出错的三处：

1. **定位**：三条互补的定位器，取并集 ——
   ① **α 块扫描**（主）：96px 块内截尾回归求不透明度 α，取 α 达标的块连通；
   ② **|d| 全局相对极值**：低通后按 16px 块均值取相对最强块的比例，抓主带；
   ③ **贴边条带相对基线极值**：四条边各取 64px 深的条带，与条带**自身**中位数比，
   专抓「比主带暗得多」的角落 ID 徽标 —— 实测 2278372352 的徽标 |d| 只有主带峰值的
   0.14 倍，用 ② 的全局阈值必然整块漏检（这正是「静默跳过」的一种形态）。
   定边界用一维投影阶跃（`med|d| > thr`）或 α 投影，精修到 ±1~2px；
   `rect_overlay.py` 内置的 `detect_regions` 在真实素材上会整体漏掉主带、
   或把草地/细纹理误检成水印，本工具用「相对极值 + 投影阶跃」替代绝对阈值扫描，
   实测 7 组素材上都能一次命中（含主带、徽标与失配区各归各类）。

2. **定性**：对每个候选框做**截尾线性回归**求不透明度 α（与 `rect_overlay.fit_overlay`
   同口径，先低通到 r=4），这是判断「真水印」还是「上采样失配」最硬的一个数：
   - 真水印：`T = (1-α)·A + α·C` 给出稳定 slope，实测主带 α = 0.43~0.50、R² 0.93~1.00；
   - 上采样失配（细纹理球衣、草地、看台噪点）：差分与 A 没有稳定线性关系，
     α 塌到 0.00~0.07，而 `frac` 还能有 0.6 —— **只看 frac 会大面积误判**。
   角落 ID 徽标单独一支：它是不透明白字 + 半透明底板两层，α 从 0.000 到 6.39 都出现过，
   只能用「小尺寸 + 贴图像边 + 明显亮于框外 + 框内大量超标」判。
   同时给出框内外带符号偏移与参考图同区亮度 std，用来判「纯色底图」
   （std < 1.5/255 时 α 回归必然失效，应走参考图回填）。

3. **验收**（`--clean` 传入结果图时）：逐区域做
   - **漏检扫描**：`|T - clean|` 在框内是否恒为 0 —— 恒为 0 说明这块**根本没被处理**，
     这是「自动检测静默跳过主带」的硬证据；
   - **接缝实测**：在真实合成边界上取框内最外 1 行/列 vs 框外紧邻 1 行/列，
     拆成 **DC** 与 **HF-rms** 两个量，而且 DC 要**再减掉 target 同位置的台阶**：
     `DC_introduced = step_clean − step_target`。**只有 DC_introduced ≤ 2/255
     才算没有可见接缝** —— 内容自己在框边上本来就可能有 2–15/255 的台阶，
     直接看 clean 的 DC 会把内容台阶误判成接缝；HF-rms 大往往只是
     「框内是放大的参考图、框外是锐利原图」这一方法固有差异。

4. **第三类水印：不去除**。低 α 的平铺斜纹/网格压在球衣网格、草地、人群噪点上时，
   α 只有 0.00~0.11（与纹理失配同量级），`locate_alpha` 仍会产出候选，
   但定性一律落在 `mismatch`。**这类不要传 --box** —— 硬去会连真实纹理一起抹掉，
   比留着水印更糟。判据就是 α 门限，不是"看起来像不像水印"。

用法：
    python inspect_watermark.py --reference <612干净图> --target <2048带水印图>
    python inspect_watermark.py --reference R --target T --clean T_clean.png
    python inspect_watermark.py --reference R --target T --window 1200,1500,0,2048

输出是一份纯文本体检报告；`--json PATH` 可同时落盘结构化结果。
不改图、不写报告到素材目录，所有输出只到 stdout 与 `--json` 指定路径。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rect_overlay as RO  # noqa: E402
import remove_stripe_watermark as RS  # noqa: E402

# ---------------------------------------------------------------------------
# 标定常量（都是实测值，改动前先看 references/qc-and-troubleshooting.md）
# ---------------------------------------------------------------------------
# 判「这一像素被水印改变了」的绝对门限。Getty 主带实测：干净侧 |d| 1.3~3.5/255，
# 条内 12~45/255。取 8/255 而不是历史上的 12/255 —— 压在中间调底图上的**条尾弱段**
# |d| 只有 12~16/255（实测 2286834348 的 x>1660 段），12 会把这条尾巴整段切掉；
# 8 仍然高过干净侧的 3.5 一倍以上。0.03 归一化 = 7.65/255，是同一取值。
DEFAULT_THR_255 = 8.0
# 低通半径。r=8 会把边界外扩约 8px，只用于「找极值」，不用于定边界。
LOCATE_BLUR_R = 6
# 找极值用的降采样块边长。16px 块 ≈ 主带高度的 1/9，既有统计量又不糊边界。
BLOCK = 16
# 「候选区」判据取最强块的相对比例，而不是绝对值 —— 不同素材水印强度差 3 倍以上。
REL_STRONG = 0.35
# 候选块数下限（面积过滤）。低于此的孤立强块多为纹理巧合。
MIN_BLOCKS = 3
# 合并间隔：主带被相对阈值切成两段的常见间隔是 8~16px。
MERGE_GAP = 24
# 接缝判据：**结果图在框边上的**亮度台阶（step_clean）超过此值才算接缝。
# 直接拿它判会把内容自然台阶算进去（实测徽标右边界 target 本来就有 +2.19、
# 失配框左边有 9.01）——所以先用 `SEAM_STEP_SAME` 排除「台阶前后没变」的边。
SEAM_DC_MAX = 2.0
# 「这条边不是合成边界」的判据：clean 与 target 在同一边上的台阶差小于此值，
# 说明重建没有改变该处的过渡（框偏宽，或该边落在水印之外）。
SEAM_STEP_SAME = 0.5
# 纯色底图判据：参考图同区亮度 std 低于此值 → α 回归不可靠，改用参考图回填。
FLAT_REF_STD = 1.5
# 定性判据：框内超标像素占比下限（只作旁证）。
FRAC_WATERMARK = 0.45
# 主判据：截尾线性回归出的不透明度 α（与 rect_overlay.fit_overlay 同口径，先低通）。
# 半透明蒙版给出 T = (1-α)·A + α·C，对 A 回归必然得到稳定的 slope = -α；
# 细纹理上采样失配的差分与 A 无稳定线性关系，α 会塌到 ~0。
# 实测分离度（9 组素材，故意挑难例）：
#   真主带   α = 0.434 / 0.437 / 0.498 / 0.501 / 0.524   R² 0.93~1.00
#   纹理失配 α = 0.010 / 0.066                          R² 0.95~0.99
# R² 两者都高（截尾后必然高），**不能拿 R² 当判据**，只有 α 能分开。
ALPHA_WATERMARK = 0.15
ALPHA_WEAK = 0.08
ALPHA_REF_FILL = 0.80          # rect_overlay 在 α≥0.80 时整框走参考图回填
ALPHA_ABSURD = 1.00            # α>1 物理不可能，rect_overlay 会直接拒绝该区域
# 徽标分支的独立门限。**徽标不能走 α 规则**（白字是不透明第二层：实测 α 从 0.000
# 到 6.39 都出现过），判据只能是「小尺寸 + 贴图像边 + 明显亮于周围 + 框内大量超标」。
# frac 下限取 0.55 而不是 0.85：徽标白字占框内的比例随徽标底色明暗变化，
# 1203272712 的徽标压在中灰底上，frac 只有 0.639，用 0.85 会被漏判成 mismatch，
# 用户就会被告知「不要传 --box」—— 于是徽标永远去不掉。放宽有其它门限兜底：
# 实测 7 组素材，唯一被新捕获的正是那个真徽标；两个纹理失配框的 min(h,w) ≥ 137，
# 本来就被 BADGE_MIN_DIM 挡掉，不靠 frac 区分。
BADGE_FRAC_MIN = 0.55
BADGE_MIN_DIM = 80             # 徽标 29~46px；草地/球衣失配框 ≥137px
BADGE_RATIO = 2.0              # 框内 med|d| 至少是框外的 2 倍（徽标远亮于周围）
BADGE_STD_LONG = 192           # Getty 标准徽标长边；检测常短 3~6px，要提示补足
# α 块扫描的块边长与步长。块边长要明显大于纹理的相关长度、又小于最窄的水印条
# （Getty 主带高 147px），96/48 实测在 6 组素材上都命中；stride=3 是块内抽稀，
# 因为 α 面已先做过全图 r=4 低通，抽稀不会引入新的错配。
ALPHA_BLOCK = 96
ALPHA_STRIDE = 3
# 贴边条带扫描（第三条定位器，专抓角落 ID 徽标）。
# 动机：徽标比主带暗得多 —— 实测 2278372352 的徽标区 |d| 中位 18~20/255，
# 只有主带峰值 132.8 的 0.14 倍，`locate` 用的**全局**相对阈值（0.35×peak=46.5）
# 必然把它滤掉 → 徽标整块漏检。但徽标在**它自己所在的那条贴边条带里**是显著极值，
# 所以判据要改成「相对条带自身的基线」。四个边各扫一遍，成本可忽略。
EDGE_DEPTH = 64        # 贴边条带深度；徽标高 29px（实测 29~46），64 足够且不引入过多无关内容
EDGE_MIN_LEN = 48      # 徽标条实测 150~265px；短于此的多半是纹理巧合
EDGE_MAX_LEN = 640     # 长于此就不是角落徽标了（主带是 820）
EDGE_REL = 3.0         # 相对条带自身基线的倍数
EDGE_FLOOR = 12.0      # 绝对下限（/255）：干净侧 |d| 实测 1.3~3.5，12 高 3 倍以上
# 角落徽标一定从图像的角上长出来（6 组素材实测全部 x0=0 贴左边缘）。
# 不加这条，底边上任何 48~640px 的亮内容段都会被收进来（实测 1209892659
# 会多出 x576~697 / x952~1072 两块底部内容）。32px 容差允许轻微内缩。
EDGE_CORNER_TOL = 32
# 投影剖面的分位数。用中位数会被「条内一半是干净内容」的窗口拉到干净侧；
# 75 分位在「条占窗口 ≥25%」时就落在条内，实测 4 组素材的行/列都用它收敛。
PROFILE_Q = 75.0
# 每轮精修时把交叉窗口外扩的比例。窗口太窄会看不到条的真实末端（条的 |d|
# 强弱随底图内容变化，某一窄窗里末端可能整段低于门限）。
REFINE_GROW = 0.25
REFINE_ITERS = 3


# ---------------------------------------------------------------------------
# 基础算子
# ---------------------------------------------------------------------------
def load_rgb(path: Path) -> np.ndarray:
    """读图 → float32 [0,1] RGB。带 alpha 的图先合成到白底（与主脚本一致）。"""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("RGB"), np.float32) / 255.0


def box_blur(x: np.ndarray, r: int) -> np.ndarray:
    """可分离盒式滤波（edge padding）。r=0 时原样返回。"""
    if r <= 0:
        return x.astype(np.float32, copy=False)
    k = 2 * r + 1
    c = np.cumsum(np.pad(x, ((r + 1, r), (0, 0)), mode="edge"), axis=0)
    x = (c[k:, :] - c[:-k, :]) / k
    c = np.cumsum(np.pad(x, ((0, 0), (r + 1, r)), mode="edge"), axis=1)
    return (c[:, k:] - c[:, :-k]) / k


def ds_block_mean(x: np.ndarray, f: int) -> np.ndarray:
    h, w = x.shape
    return x[: h // f * f, : w // f * f].reshape(h // f, f, w // f, f).mean(3).mean(1)


def comps(mask: np.ndarray):
    """4-连通标记（纯 python，块栅格很小所以够快）。返回 [(y0,y1,x0,x1)]，不含端。"""
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    out = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            y0 = y1 = sy
            x0 = x1 = sx
            n = 0
            while stack:
                y, x = stack.pop()
                n += 1
                y0, y1 = min(y0, y), max(y1, y)
                x0, x1 = min(x0, x), max(x1, x)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append([y0, y1 + 1, x0, x1 + 1, n])
    return out


def merge_boxes(boxes, gap=MERGE_GAP):
    """把彼此靠近的框合并（主带常被相对阈值切成两段）。"""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if (a[0] - gap < b[1] and b[0] - gap < a[1]
                        and a[2] - gap < b[3] and b[2] - gap < a[3]):
                    boxes[i] = [min(a[0], b[0]), max(a[1], b[1]),
                                min(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return boxes


# ---------------------------------------------------------------------------
# 候选定位：相对极值 → 块连通 → 合并
# ---------------------------------------------------------------------------
def locate(d255: np.ndarray, window=None, rel=REL_STRONG):
    h, w = d255.shape
    B = box_blur(d255, LOCATE_BLUR_R)
    M = ds_block_mean(B, BLOCK)
    if window:
        wy0, wy1, wx0, wx1 = window
        by0, by1 = max(0, wy0 // BLOCK), min(M.shape[0], wy1 // BLOCK + 1)
        bx0, bx1 = max(0, wx0 // BLOCK), min(M.shape[1], wx1 // BLOCK + 1)
        sub = M[by0:by1, bx0:bx1]
        if sub.size == 0:
            return [], {}
        peak = float(sub.max())
    else:
        by0 = bx0 = 0
        peak = float(M.max())
    if peak <= 0:
        return [], {"peak_255": 0.0}
    thr = rel * peak
    mask = M > thr
    if window:
        m2 = np.zeros_like(mask)
        m2[by0:by1, bx0:bx1] = mask[by0:by1, bx0:bx1]
        mask = m2
    cs = [c for c in comps(mask) if c[4] >= MIN_BLOCKS]
    # 块栅格 → 全分辨率
    boxes = [[c[0] * BLOCK, min(h, c[1] * BLOCK), c[2] * BLOCK, min(w, c[3] * BLOCK)] for c in cs]
    boxes = merge_boxes(boxes)
    boxes = [b for b in boxes if (b[1] - b[0]) >= 12 and (b[3] - b[2]) >= 12]
    boxes.sort(key=lambda b: (b[1] - b[0]) * (b[3] - b[2]), reverse=True)
    meta = {"peak_255": peak, "rel_thr_255": thr}
    return boxes, meta


# ---------------------------------------------------------------------------
# 贴边条带扫描：角落 ID 徽标（第三条定位器）
# ---------------------------------------------------------------------------
def locate_edge_badge(d255: np.ndarray, window=None):
    """在四条贴边条带里找「相对条带自身基线」显著亮的一段，作为徽标候选。

    徽标的 |d| 远低于主带（实测 0.14 倍），全局相对阈值一定会漏掉它；
    但它在自己那条 64px 条带里是显著极值，所以基线取条带自身的**中位数**。
    """
    h, w = d255.shape
    B = box_blur(d255, LOCATE_BLUR_R)
    D = EDGE_DEPTH
    if h < 2 * D or w < 2 * D:
        return [], {"edge_depth": D, "n_edge": 0}
    strips = (("bottom", B[h - D:h, :], True), ("top", B[0:D, :], True),
              ("left", B[:, 0:D], False), ("right", B[:, w - D:], False))
    out = []
    for name, S, along_x in strips:
        n = (S.shape[1] if along_x else S.shape[0]) // BLOCK
        if n < 2:
            continue
        prof = np.array([
            (S[:, i * BLOCK:(i + 1) * BLOCK] if along_x
             else S[i * BLOCK:(i + 1) * BLOCK, :]).mean() for i in range(n)])
        thr = max(EDGE_REL * float(np.median(prof)), EDGE_FLOOR)
        hit = prof > thr
        i = 0
        span = S.shape[1] if along_x else S.shape[0]
        while i < n:
            if not hit[i]:
                i += 1
                continue
            lo = hi = i
            gap = 0
            j = i + 1
            while j < n:
                if hit[j]:
                    hi, gap = j, 0
                else:
                    gap += 1
                    if gap > 1:          # 允许 1 个块的间断（白字之间的空隙）
                        break
                j += 1
            a, b = lo * BLOCK, min((hi + 1) * BLOCK, span)
            if not (EDGE_MIN_LEN <= (b - a) <= EDGE_MAX_LEN):
                i = hi + 1
                continue
            if min(a, span - b) > EDGE_CORNER_TOL:   # 必须从角上长出来
                i = hi + 1
                continue
            if along_x:
                out.append([h - D, h, a, b] if name == "bottom"
                           else [0, D, a, b])
            else:
                out.append([a, b, 0, D] if name == "left"
                           else [a, b, w - D, w])
            i = hi + 1
    if window:
        wy0, wy1, wx0, wx1 = window
        out = [b for b in out
               if not (b[1] < wy0 or b[0] > wy1 or b[3] < wx0 or b[2] > wx1)]
    return out, {"edge_depth": D, "n_edge": len(out)}


def _inter_over(box, other):
    """box 被 other 覆盖的面积比例。"""
    iy0, iy1 = max(box[0], other[0]), min(box[1], other[1])
    ix0, ix1 = max(box[2], other[2]), min(box[3], other[3])
    a = (box[1] - box[0]) * (box[3] - box[2])
    if iy1 <= iy0 or ix1 <= ix0 or a <= 0:
        return 0.0
    return (iy1 - iy0) * (ix1 - ix0) / a


# ---------------------------------------------------------------------------
# 边界精修：一维投影阶跃（互相迭代一轮即可收敛）
# ---------------------------------------------------------------------------
def _run_at(profile: np.ndarray, center: int, thr: float):
    """取包含 center 的、profile > thr 的连续段（允许中心附近 gap<=6 的间断）。"""
    n = len(profile)
    if n == 0:
        return None
    c = int(np.clip(center, 0, n - 1))
    if not (profile[c] > thr):
        # 中心不超标（框中心可能落在白字空隙），向两侧找最近超标点
        cand = [i for i in range(n) if profile[i] > thr]
        if not cand:
            return None
        c = min(cand, key=lambda i: abs(i - c))
    lo = hi = c
    gap = 0
    i = c
    while i > 0:
        i -= 1
        if profile[i] > thr:
            lo, gap = i, 0
        else:
            gap += 1
            if gap > 6:
                break
    gap = 0
    i = c
    while i < n - 1:
        i += 1
        if profile[i] > thr:
            hi, gap = i, 0
        else:
            gap += 1
            if gap > 6:
                break
    return lo, hi + 1


def refine_alpha(planes, d255, box, step=4, hw=8, rounds=4, stride=3,
                 min_alpha=None, block=ALPHA_BLOCK, thr=DEFAULT_THR_255):
    """α 投影阶跃精修：每 `step` 行/列算一条横/竖条的 α，取 α 达标段的并集。

    **为什么 α 定位来的候选必须用 α 精修，不能改用 |d| 精修。**
    实测 2278372352：底图球衣是细网格，干净区的 |d| 75 分位就有 10~16/255，
    与真蒙版条的 15~27/255 完全重叠 —— 任何 |d| 门限都会把条外的网格行一起
    吞进来（门限 8 时行段直接涨到 589~1130，整框 α 被稀释回 0.005，又变回「漏检」）。
    α 是唯一能在这种图上分开的量：条内 0.3~0.7，条外 0.00~0.07。

    三个实现要点：
    1. **条高 hw 要小（默认 ±8 行）**：条位置 v 的 α 是 [v-hw, v+hw] 的平均，
       一个中心落在条外 0.84·hw 的位置仍会因为条占了一部分而达标 —— hw 越大，
       两端各外溢越多（实测 hw=32 时 147px 的条被报成 217~222px，整体虚高 50）。
       hw=8 时端点误差约 ±10px，配合下面打印的 α 阶跃剖面足以定到 ±2px。
    2. **取达标段的并集，不取「含中心的那一段」**：条内压在低方差内容上的几行
       会让 α 掉到门限下，含中心的单段会被切断（实测 2286834348 上把 147px 的条
       切成 134×143 的一小块）。
    3. **端点不做 hw 补偿**（端点就是达标条中心的位置），补偿会反向放大误差。
    """
    Al, Sl = planes
    h, w = Al.shape
    min_alpha = ALPHA_WATERMARK if min_alpha is None else min_alpha
    y0, y1, x0, x1 = [int(v) for v in box]
    bh, bw = max(8, y1 - y0), max(8, x1 - x0)

    def scan(lo, hi, s, axis, rows):
        out = []
        for v in range(lo, hi + 1, s):
            if axis == 0:
                sb = (max(0, v - hw), min(h, v + hw), x0, x1)
            else:
                sb = (rows[0], rows[1], max(0, v - hw), min(w, v + hw))
            if sb[1] - sb[0] < 16 or sb[3] - sb[2] < 16:
                continue
            a, _ = fit_alpha_planes(Al, Sl, sb, rounds=rounds, stride=stride)
            # 双条件：只看 α 时，条外侧仍有少量条状假阳性（细纹理区也偶发高 α），
            # 而真蒙版条内是「α 高 **且** 大范围超标」。加上 frac 后左侧/右尾的
            # 虚高全部消失（实测 2286834348 的左端由 x1059 收回到 x1240 附近）。
            sub = d255[sb[0]:sb[1]:stride, sb[2]:sb[3]:stride]
            f = float((sub > thr).mean()) if sub.size else 0.0
            if a >= min_alpha and f >= FRAC_WATERMARK:
                out.append((v, a))
        return out

    def extent(vals, max_span):
        ok = [v for v, a in vals if a >= min_alpha]
        if not ok:
            return None
        first, last = min(ok), max(ok)
        if last - first > max_span:
            # 达标点太散（多半是别处的假块混进来），退回「最长连续段」
            best = cur = None
            for v, a in vals:
                if a >= min_alpha:
                    if cur is None:
                        cur = [v, v]
                    else:
                        cur[1] = v
                    if best is None or (cur[1] - cur[0]) > (best[1] - best[0]):
                        best = list(cur)
                else:
                    cur = None
            if best is None:
                return None
            first, last = best
        return first, last

    # --- 行方向
    lo = max(0, y0 - 2 * block)
    hi = min(h - 1, y1 + 2 * block)
    e = extent(scan(lo, hi, step, 0, None), 3 * bh)
    if e is None:
        ny0, ny1 = y0, y1
    else:
        e2 = extent(scan(max(lo, e[0] - step), min(hi, e[1] + step), 1, 0, None),
                    3 * bh)
        if e2 is not None:
            e = e2
        ny0, ny1 = max(0, e[0] - hw), min(h, e[1] + hw)
    rows = (ny0, ny1)
    # --- 列方向
    lo = max(0, x0 - 2 * block)
    hi = min(w - 1, x1 + 2 * block)
    e = extent(scan(lo, hi, step, 1, rows), 3 * bw)
    if e is None:
        nx0, nx1 = x0, x1
    else:
        e2 = extent(scan(max(lo, e[0] - step), min(hi, e[1] + step), 1, 1, rows),
                    3 * bw)
        if e2 is not None:
            e = e2
        nx0, nx1 = max(0, e[0] - hw), min(w, e[1] + hw)
    return (ny0, max(ny0 + 12, ny1), max(0, nx0), max(nx0 + 12, nx1))


def refine(d255: np.ndarray, box, thr=DEFAULT_THR_255, q=PROFILE_Q,
           iters=REFINE_ITERS, grow=REFINE_GROW):
    """一维投影阶跃互相迭代，把粗框收到真实水印外沿（±1~2px）。

    两个实现要点，都是踩出来的：
    1. **剖面用分位数（默认 75）而不是中位数。** 蒙版是半透明的，条的 |d| 强弱
       随底图内容变化；在某一窄窗里，条的末端可能整段低于门限，中位数剖面会
       在那里断掉（实测 2286834348 的行剖面用中位数停在 1046，真值 1076）。
       75 分位在「条占窗口 ≥25%」时就落在条内，末端不再丢。
    2. **交叉窗口每轮外扩 25%。** 只按当前框取剖面会自我固化：窄框 → 剖面窄 →
       框不长大。外扩后条的真实末端才进得了窗口。
    """
    h, w = d255.shape
    y0, y1, x0, x1 = [int(v) for v in box]
    conf = []
    for _ in range(iters):
        ext = int((x1 - x0) * grow)
        sx0, sx1 = max(0, x0 - ext), min(w, x1 + ext)
        if sx1 - sx0 >= 8:
            rp = np.percentile(d255[:, sx0:sx1], q, axis=1)
            r = _run_at(rp, (y0 + y1) // 2, thr)
            if r is None:
                conf.append("rows-fail")
            else:
                y0, y1 = max(0, r[0]), min(h, r[1])
        ext = int((y1 - y0) * grow)
        sy0, sy1 = max(0, y0 - ext), min(h, y1 + ext)
        if sy1 - sy0 >= 8:
            cp = np.percentile(d255[sy0:sy1, :], q, axis=0)
            c = _run_at(cp, (x0 + x1) // 2, thr)
            if c is None:
                conf.append("cols-fail")
            else:
                x0, x1 = max(0, c[0]), min(w, c[1])
    return (y0, y1, x0, x1), conf


def step_probe(d255: np.ndarray, box, q=PROFILE_Q, half=7):
    """打印用：四条边各自的阶跃剖面（框外 7 px → 框内 7 px，两端各取一段）。

    这是「按残差阶跃实测框」的原始数据。真水印的阶跃是 1~2px 内从外值跳到内值；
    若看到的是平滑爬升几十像素，说明门限太低或那不是水印（多半是内容渐变）。
    """
    h, w = d255.shape
    y0, y1, x0, x1 = box
    pad_y = max(2, int((x1 - x0) * 0.08))
    pad_x = max(2, int((y1 - y0) * 0.08))
    out = {}
    sx0, sx1 = min(w, x0 + pad_y), max(x0 + 1, x1 - pad_y)
    if sx1 - sx0 > 4:
        rp = np.percentile(d255[:, sx0:sx1], q, axis=1)
        head = range(max(0, y0 - half), min(h, y0 + half))
        tail = range(max(0, y1 - half), min(h, y1 + half))
        out["rows"] = ([(int(i), round(float(rp[i]), 1)) for i in head],
                       [(int(i), round(float(rp[i]), 1)) for i in tail])
        out["row_edges"] = [int(y0), int(y1)]
    sy0, sy1 = min(h, y0 + pad_x), max(y0 + 1, y1 - pad_x)
    if sy1 - sy0 > 4:
        cp = np.percentile(d255[sy0:sy1, :], q, axis=0)
        head = range(max(0, x0 - half), min(w, x0 + half))
        tail = range(max(0, x1 - half), min(w, x1 + half))
        out["cols"] = ([(int(i), round(float(cp[i]), 1)) for i in head],
                       [(int(i), round(float(cp[i]), 1)) for i in tail])
        out["col_edges"] = [int(x0), int(x1)]
    return out


def alpha_probe(planes, box, half=7, hw=4, rounds=4, stride=3):
    """打印用：四条边的 α 阶跃剖面（框外 7 → 框内 7，两端各一段）。

    **这是 α 定位来的框该看的剖面**（|d| 剖面在细纹理底图上分不开条内外）。
    真蒙版的 α 阶跃同样是 1~2px 内从 ~0.05 跳到 ~0.45；看到的是几十像素的
    缓慢爬升，说明框边没落在真实外沿，按步长手动收/放到台阶位置即可。
    """
    Al, Sl = planes
    h, w = Al.shape
    y0, y1, x0, x1 = box
    out = {}

    def ra(v):
        return fit_alpha_planes(Al, Sl, (max(0, v - hw), min(h, v + hw), x0, x1),
                               rounds=rounds, stride=stride)[0]

    def ca(v):
        return fit_alpha_planes(Al, Sl, (y0, y1, max(0, v - hw), min(w, v + hw)),
                               rounds=rounds, stride=stride)[0]

    if (y1 - y0) >= 2 * half:
        head = range(max(0, y0 - half), min(h, y0 + half))
        tail = range(max(0, y1 - half), min(h, y1 + half))
        out["rows"] = ([(int(i), round(ra(i), 3)) for i in head],
                       [(int(i), round(ra(i), 3)) for i in tail])
    if (x1 - x0) >= 2 * half:
        head = range(max(0, x0 - half), min(w, x0 + half))
        tail = range(max(0, x1 - half), min(w, x1 + half))
        out["cols"] = ([(int(i), round(ca(i), 3)) for i in head],
                       [(int(i), round(ca(i), 3)) for i in tail])
    return out


def _cover_ratio(a, b):
    """框 a 被框 b 覆盖的面积比例。"""
    iy = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    ix = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    area = max(1, (a[1] - a[0]) * (a[3] - a[2]))
    return iy * ix / area


def _fmt_prof(pack):
    (head, tail) = pack
    f = lambda ps: " ".join(f"{i}:{v}" for i, v in ps)
    return f"[{f(head)}]  |  [{f(tail)}]"


# ---------------------------------------------------------------------------
# 区域定性
# ---------------------------------------------------------------------------
def _alpha_planes(T, A, blur_r=4):
    """一次算好全图的低通亮度面与差分面，供块扫描复用（不要每块各算一遍低通）。"""
    Al = box_blur(A.mean(2), blur_r)
    Tl = box_blur(T.mean(2), blur_r)
    return Al, Tl - Al


def fit_alpha_planes(Al, Sl, box, pct=65, rounds=8, stride=1):
    y0, y1, x0, x1 = box
    u = Al[y0:y1:stride, x0:x1:stride].ravel()
    v = Sl[y0:y1:stride, x0:x1:stride].ravel()
    if u.size < 32:
        return 0.0, 0.0
    # 回归量方差不足时 slope 在数值上不可靠：平坦区（天空、纯色底板）的 A 几乎无变化，
    # 一丁点 JPEG 抖动就能拟出 α=3~9 的荒谬斜率（实测 α 面峰值 3.4~9.4，而真相是
    # 这些块根本没有蒙版）。3/255 是「还有内容结构」的下限。
    if float(u.std()) < 0.012:
        return 0.0, 0.0
    keep = np.ones(u.shape, bool)
    q = p = 0.0
    for _ in range(rounds):
        if keep.sum() < 24:
            break
        q, p = np.polyfit(u[keep], v[keep], 1)
        r = np.abs(v - (p + q * u))
        keep = r <= np.percentile(r, pct)
    if not np.isfinite(q):
        return 0.0, 0.0
    res = v - (p + q * u)
    st = float(((v - v.mean()) ** 2).sum())
    r2 = 1.0 - float((res[keep] ** 2).sum()) / st if st > 1e-12 else 0.0
    return float(max(0.0, -q)), float(max(0.0, min(1.0, r2)))


def fit_alpha(T, A, box, blur_r=4, pct=65, rounds=8, stride=1):
    """截尾最小二乘估 α（与 rect_overlay.fit_overlay 同口径，先低通降噪）。

    先对 T 与 A 都做 r=4 的盒式低通：单像素的 JPEG 抖动与插值痕迹会把回归
    拉偏，低通后剩下的才是「蒙版造成的系统性线性关系」。
    截尾（每轮丢掉残差最大的 35%）是为了排除白字/logo —— 它们是不透明的
    第二层覆盖，不属于混合模型，会把 slope 拉向 0。
    返回值 (alpha, r2)。alpha 为负按 0 处理（说明 T 与 A 反相关，不可能）。
    """
    Al, Sl = _alpha_planes(T, A, blur_r)
    return fit_alpha_planes(Al, Sl, box, pct=pct, rounds=rounds, stride=stride)


def alpha_map(planes, d255, block=ALPHA_BLOCK, hop=None, stride=3,
              rounds=4, thr=DEFAULT_THR_255):
    """逐块估 α，返回 (amap, fmap, iy0, ix0)——块栅格上的 α 与超标占比。

    这是**主定位器**：只有半透明蒙版才会给出稳定的高 α。相比之下
    「|d| 强的块」在细纹理上到处都是（PSG 网格、草地、看台），
    实测 2278372352 用 |d| 定位会把网格与淡蒙版并成 541×634 的大框，
    整框 α 被稀释到 0.005，整条真蒙版（y≈680~800，α≈0.46）被漏掉。
    因此**候选必须由 α 而不是 |d| 产生**。
    """
    Al, Sl = planes
    h, w = d255.shape
    hop = hop or max(8, block // 2)
    ny = max(1, (h - block) // hop + 1)
    nx = max(1, (w - block) // hop + 1)
    amap = np.zeros((ny, nx), np.float32)
    fmap = np.zeros((ny, nx), np.float32)
    for i in range(ny):
        for j in range(nx):
            y, x = i * hop, j * hop
            sb = (y, min(h, y + block), x, min(w, x + block))
            a, _r2 = fit_alpha_planes(Al, Sl, sb, rounds=rounds, stride=stride)
            amap[i, j] = a
            sub = d255[sb[0]:sb[1]:stride, sb[2]:sb[3]:stride]
            fmap[i, j] = float((sub > thr).mean()) if sub.size else 0.0
    return amap, fmap, hop


def locate_alpha(planes, d255, block=ALPHA_BLOCK, hop=None, stride=3, rounds=4,
                 thr=DEFAULT_THR_255):
    """α 达标的块做连通域 → 候选框（块栅格坐标乘回全分辨率并外扩一块）。"""
    amap, fmap, hop = alpha_map(planes, d255, block=block, hop=hop,
                               stride=stride, rounds=rounds, thr=thr)
    peak = float(amap.max())
    mask = (amap >= ALPHA_WATERMARK) & (amap <= ALPHA_ABSURD) & (fmap >= FRAC_WATERMARK)
    h, w = d255.shape
    boxes = []
    for c in comps(mask):
        if c[4] < 2:
            continue
        y0 = max(0, c[0] * hop)
        y1 = min(h, c[1] * hop + block)
        x0 = max(0, c[2] * hop)
        x1 = min(w, c[3] * hop + block)
        boxes.append([y0, y1, x0, x1])
    boxes = merge_boxes(boxes)
    boxes = [b for b in boxes if (b[1] - b[0]) >= 16 and (b[3] - b[2]) >= 16]
    boxes.sort(key=lambda b: (b[1] - b[0]) * (b[3] - b[2]), reverse=True)
    return boxes, {"alpha_block": block, "alpha_hop": hop, "alpha_peak": round(peak, 4)}


def measure(T, A, d255, box, thr=DEFAULT_THR_255):
    h, w = T.shape[:2]
    y0, y1, x0, x1 = box
    # 内区（去掉边界 2px，避开羽化带）
    iy0, iy1 = min(h, y0 + 2), max(y0 + 3, y1 - 2)
    ix0, ix1 = min(w, x0 + 2), max(x0 + 3, x1 - 2)
    din = d255[iy0:iy1, ix0:ix1]
    frac = float((din > thr).mean()) if din.size else 0.0
    med_in = float(np.median(din)) if din.size else 0.0
    # 外区（框外 10~40px 环带，同轴）
    oy0, oy1 = max(0, y0 - 40), max(1, y0 - 10)
    out = d255[oy0:oy1, ix0:ix1]
    if out.size == 0:
        out = d255[iy0:iy1, max(0, x0 - 40):max(1, x0 - 10)]
    med_out = float(np.median(out)) if out.size else 0.0
    # 带符号偏移的稳健中心差（旁证）
    s = (T - A).mean(2) * 255.0
    ins = s[iy0:iy1, ix0:ix1].ravel()
    outs = s[oy0:oy1, ix0:ix1].ravel() if oy1 > oy0 else ins
    ctr = lambda v: (float(np.percentile(v, 25)) + float(np.percentile(v, 75))) / 2.0
    c_in, c_out = ctr(ins), ctr(outs)
    # 参考图同区亮度 std（判纯色底图）
    ref_std = float((A[iy0:iy1, ix0:ix1].mean(2) * 255.0).std())
    ref_lf = box_blur(A[iy0:iy1, ix0:ix1].mean(2) * 255.0, 8)
    ref_range = float(np.percentile(ref_lf, 98) - np.percentile(ref_lf, 2))
    alpha, r2 = fit_alpha(T, A, box)
    return {
        "alpha": round(alpha, 4), "fit_r2": round(r2, 3),
        "frac": round(frac, 4), "med_in_255": round(med_in, 2),
        "med_out_255": round(med_out, 2),
        "center_in_255": round(c_in, 2), "center_out_255": round(c_out, 2),
        "center_delta_255": round(abs(c_in - c_out), 2),
        "ref_lum_std_255": round(ref_std, 2),
        "ref_lum_range_255": round(ref_range, 2),
    }


def classify(box, met, shape):
    """先判徽标，再按回归 α 判主带。`frac` 只作旁证。

    **角落 ID 徽标必须单独成一支，不能走 α 规则。** 徽标是「半透明底板 +
    不透明白字」两层，白字要么把 slope 拉到 0（实测 α=0.000），要么在纯色底图上
    把回归推爆（实测 α=6.39）。这两种结果都不代表「不是水印」——
    它恰恰是 SKILL.md 里「纯色底图徽标 → 参考图回填」那个已知行为。
    判据：小尺寸 + 框内大量超标 + 明显亮于框外 + 贴图像边。
    **不要用带符号的条内外中心差**：徽标里既有暗底板又有白字，符号会互相抵消
    （实测 1203272712 带符号差只有 0.4/255，而 med|d| 内 13.7 / 外 2.6）——
    这也是「有白字图例的主带」曾被误判成 mismatch 的同一个坑。

    「细纹理上采样失配」是主带这一支最大的假阳性来源：PSG 球衣的网格、草地、
    看台噪点的 |d| 可以很大（frac 到 0.63），但 α 塌到 0.01~0.07。
    这类框尺寸都很大（≥137px），靠 BADGE_MIN_DIM 与徽标分开。
    """
    h, w = box[1] - box[0], box[3] - box[2]
    H, W = shape[:2]
    a = met["alpha"]
    ratio = met["med_in_255"] / max(met["med_out_255"], 1.0)
    touches = box[0] <= 1 or box[2] <= 1 or box[1] >= H - 1 or box[3] >= W - 1
    if (min(h, w) <= BADGE_MIN_DIM and met["frac"] >= BADGE_FRAC_MIN
            and ratio >= BADGE_RATIO and touches):
        note = ""
        if met["ref_lum_std_255"] < FLAT_REF_STD:
            note = (f"；参考图同区 lum std={met['ref_lum_std_255']:.2f} < "
                    f"{FLAT_REF_STD} 是纯色底图 → α 回归必然失效，"
                    "改用「参考图回填 + 框外 5px smoothstep 羽化」")
        # Getty 标准徽标是 29×192。检测出的长边常短 3~6px（实测 189），
        # 而条的末端残留正好表现为该端 5/255 量级的接缝
        # （实测 2278372352：右端 5.43 → 补到 192 后 0.00）。
        long_side = max(h, w)
        hint = ""
        if 1 <= BADGE_STD_LONG - long_side <= 12:
            hint = (f"。**框长边 {long_side}px 比标准徽标 {BADGE_STD_LONG}px 短 "
                    f"{BADGE_STD_LONG - long_side}px** —— 条末端没被框住，"
                    "会在那一端留下 5/255 量级的可见接缝；先按标准尺寸补足再看")
        return "badge", (
            f"角落 ID 徽标：{h}x{w}px 贴图像边、框内 {met['frac']*100:.0f}% 像素超标"
            f"、比框外亮 {met['med_in_255']-met['med_out_255']:.0f}/255。"
            f"α 回归在这里不可靠（实测 0.000~6.39，白字是不透明第二层）——"
            f"rect_overlay 拟合成功会正常处理，被拒就按上面那条补丁{note}"
            "。**框两端常比实际徽标窄**（实测左漏 31px / 右 22px），"
            f"传 --box 前先看下面的列阶跃剖面确认条的真实终点{hint}")
    if a > ALPHA_ABSURD:
        return "reject", ("α=%.3f > 1，物理上不可能。这不是半透明蒙版；"
                          "rect_overlay 也会拒绝该区域并原样保留" % a)
    if a >= ALPHA_WATERMARK and met["frac"] >= FRAC_WATERMARK:
        extra = ""
        if a >= ALPHA_REF_FILL:
            extra = ("；α≥%.2f → rect_overlay 会整框走 reference-fill（回填参考图），"
                     "底层细节按设计不可恢复" % ALPHA_REF_FILL)
        return "bar", ("候选水印：α=%.3f（超阈 %.2f）且框内超标占比 %.2f → "
                       "可交给 rect_overlay --box%s。框边界仍要按下面的阶跃剖面核对"
                       % (a, ALPHA_WATERMARK, met["frac"], extra))
    if a >= ALPHA_WEAK:
        return "weak", ("弱候选：α=%.3f 落在 %.2f~%.2f 之间。可能是很淡的蒙版，"
                        "也可能是失配与噪声的混合 → 裁图目视，并核对方框内的阶跃剖面"
                        % (a, ALPHA_WEAK, ALPHA_WATERMARK))
    tail = ""
    if min(h, w) <= BADGE_MIN_DIM and touches:
        tail = ("（形状像角落徽标，但覆盖率 %.2f < %.2f 或亮度比 %.2f < %.1f —— "
                "先裁图目视：徽标底色接近周围时这两项都会退化，别只凭这个判定）"
                % (met["frac"], BADGE_FRAC_MIN, ratio, BADGE_RATIO))
    return "mismatch", ("**不是水印**：α=%.3f < %.2f，T 与 A 之间没有稳定的半透明"
                        "混合关系 → 细纹理上采样失配（球衣网格、草地、看台噪点）。"
                        "**不要传 --box**，传了只会用低清参考图把真实纹理糊掉%s"
                        % (a, ALPHA_WEAK, tail))


def subsplit(T, A, d255, box, thr=DEFAULT_THR_255, grid=3):
    """整框 α 塌掉时，在框内按 3×3 找「α 达标」的子块并取并集。

    动机（实测）：2278372352 的细纹理球衣网格与一条很淡的蒙版条连成一片，
    定位阶段把两者并成一个大框，整框回归被网格稀释成 α=0.005（判成失配），
    而真正的条其实在框内某一段（子区域回归 α=0.458）。**整框拟合对
    「大片失配里嵌一条淡蒙版」无能为力**，必须允许「只报框内有蒙版的那一块」。

    防过拟合：子块必须 α 与 frac 同时达标；并集必须真的比原框小。
    """
    y0, y1, x0, x1 = box
    if (y1 - y0) < 24 or (x1 - x0) < 24:
        return None
    hs = np.linspace(y0, y1, grid + 1).astype(int)
    ws = np.linspace(x0, x1, grid + 1).astype(int)
    hits = []
    for i in range(grid):
        for j in range(grid):
            sb = (int(hs[i]), int(hs[i + 1]), int(ws[j]), int(ws[j + 1]))
            if (sb[1] - sb[0]) < 12 or (sb[3] - sb[2]) < 12:
                continue
            m = measure(T, A, d255, sb, thr=thr)
            if m["alpha"] >= ALPHA_WATERMARK and m["frac"] >= FRAC_WATERMARK:
                hits.append(sb)
    if not hits:
        return None
    nb = (min(h[0] for h in hits), max(h[1] for h in hits),
          min(h[2] for h in hits), max(h[3] for h in hits))
    if nb == (y0, y1, x0, x1):
        return None
    return nb


# ---------------------------------------------------------------------------
# 验收：漏检扫描 + 接缝实测
# ---------------------------------------------------------------------------
def check_clean(T, clean, box, thr=DEFAULT_THR_255):
    """漏检扫描 + 接缝实测。**接缝 = 结果图在框边上留下的可见亮度台阶。**

    三个量的关系（都是实测踩出来的，缺一个就会误判）：

    | 情况 | step_target | step_clean | excess | 结论 |
    |---|---|---|---|---|
    | 该边未被改动（框偏宽） | 9.01 | 9.01 | 0.00 | 内容台阶，**不是接缝** |
    | 去掉水印自身的边（徽标底板） | +34.6 | 0.00 | 0.00 | **正确结果**，不是接缝 |
    | 重建留下电平差 | +2.19 | +6.59 | **+4.40** | **真接缝** |
    | 重建贴合良好（主带四条边） | −0.61 | −0.78 | +0.17 | 无接缝 |

    所以判据就是 `excess = |step_clean| − |step_target|`（下限 0），
    阈值同 `SEAM_DC_MAX`。这个定义同时覆盖了三种良性情形，不需要再单独判
    「这条边是不是合成边界」—— 主带四条边虽然是合成边界，但 excess≈0，
    不会被误报（早期按「该边是否被改动」做门限的版本会在这里给出错误标签）。
    另报 `processed_fraction`（框内被改动像素占比）用来量化「框偏宽」。
    """
    h, w = T.shape[:2]
    y0, y1, x0, x1 = box
    iy0, iy1 = min(h, y0 + 2), max(y0 + 3, y1 - 2)
    ix0, ix1 = min(w, x0 + 2), max(x0 + 3, x1 - 2)
    diff = np.abs(T - clean).mean(2) * 255.0
    dc = diff[iy0:iy1, ix0:ix1]
    med_removed = float(np.median(dc)) if dc.size else 0.0
    untouched = med_removed < 1.0
    processed_fraction = float((dc > 1.0).mean()) if dc.size else 0.0
    out = {"removed_median_255": round(med_removed, 2),
           "untouched_flat": bool(untouched),
           "processed_fraction": round(processed_fraction, 3)}
    if not untouched and processed_fraction < 0.85:
        out["oversize_note"] = (
            "框内只有 %.0f%% 像素被改动 → **框比真实水印宽**，"
            "要收窄就照下面的阶跃剖面找真实末端" % (processed_fraction * 100))

    lt = T.mean(2) * 255.0
    lc = clean.mean(2) * 255.0
    hf = lc - box_blur(lc, 3)
    edges = {}

    # 真实合成边界可能不在声明框上（用户可能用了另一个 --box）。逐边在框内
    # 最多 25% 的范围内找「第一处真的被改动的位置」，在那里量接缝 ——
    # 这正是「接缝必须测真实合成边界」这条教训的可执行形式。
    p = max(2, int((x1 - x0) * 0.1))
    py = max(2, int((y1 - y0) * 0.1))
    mcols = diff[:, x0 + p:x1 - p]
    mrows = diff[y0 + py:y1 - py, :]

    capy = max(1, int((y1 - y0) * 0.25))
    capx = max(1, int((x1 - x0) * 0.25))
    probes = {}
    for name, lo, hi, step in (("top", y0, y0 + capy, 1),
                               ("bottom", y1 - 1, max(y0, y1 - 1 - capy), -1)):
        rng = range(lo, hi) if step > 0 else range(lo, hi, -1)
        found = next((r for r in rng
                      if 0 <= r < h and mcols[r].size
                      and float(np.median(mcols[r])) > 1.0), None)
        if found is not None:
            probes[name] = ("row", found)
    for name, lo, hi, step in (("left", x0, x0 + capx, 1),
                               ("right", x1 - 1, max(x0, x1 - 1 - capx), -1)):
        rng = range(lo, hi) if step > 0 else range(lo, hi, -1)
        found = next((c for c in rng
                      if 0 <= c < w and mrows[:, c].size
                      and float(np.median(mrows[:, c])) > 1.0), None)
        if found is not None:
            probes[name] = ("col", found)

    def edge_rec(sl_in, sl_out):
        ci, co = lc[sl_in], lc[sl_out]
        ti, to = lt[sl_in], lt[sl_out]
        if ci.size == 0 or co.size == 0:
            return None
        step_c = float(ci.mean()) - float(co.mean())
        step_t = float(ti.mean()) - float(to.mean())
        excess = max(0.0, abs(step_c) - abs(step_t))
        if abs(step_c) <= SEAM_DC_MAX:
            note = "结果台阶本身 ≤2/255，无可见接缝"
        elif abs(step_c - step_t) <= SEAM_STEP_SAME:
            note = "台阶与 target 一致 → 是内容台阶，不是我们引入的"
        elif abs(step_c) < abs(step_t):
            note = "台阶被削弱 → 原本是水印的边，属正常"
        else:
            note = None
        return {
            "step_clean": round(step_c, 2),
            "step_target": round(step_t, 2),
            "step_change": round(step_c - step_t, 2),
            "excess": round(excess, 2),
            "HF_in_255": round(float(hf[sl_in].std()), 2),
            "HF_out_255": round(float(hf[sl_out].std()), 2),
            "_excess": excess,
            "note": note,
        }

    # 行边：真实边界那一行 vs 紧邻框外那一行，两端各留 10% 避免取到角点
    cols = slice(x0 + p, x1 - p)
    for name in ("top", "bottom"):
        if name not in probes:
            continue
        _, iy = probes[name]
        o = iy - 1 if name == "top" else iy + 1
        if not (0 <= iy < h) or not (0 <= o < h):
            continue
        rec = edge_rec((slice(iy, iy + 1), cols), (slice(o, o + 1), cols))
        if rec:
            rec["at"] = iy
            edges[name] = rec
    # 列边：上下各留 10%，避免把条端之外的竖直内容边缘算进来
    rows = slice(y0 + py, y1 - py)
    for name in ("left", "right"):
        if name not in probes:
            continue
        _, ix = probes[name]
        o = ix - 1 if name == "left" else ix + 1
        if not (0 <= ix < w) or not (0 <= o < w):
            continue
        rec = edge_rec((rows, slice(ix, ix + 1)), (rows, slice(o, o + 1)))
        if rec:
            rec["at"] = ix
            edges[name] = rec

    out["edges"] = edges
    # 真实边界与声明框不一致时点出来。小偏移 = 框这条边多算了；大偏移 = 这块的边
    # 与真正被处理的区域不是同一条（多半是两框重叠），不宜按它下结论。
    shift = {}
    for name, (axis, pos) in probes.items():
        declared = {"top": y0, "bottom": y1 - 1, "left": x0, "right": x1 - 1}[name]
        if pos != declared:
            shift[name] = pos - declared
    if shift:
        out["boundary_shift"] = shift
        small = {k: v for k, v in shift.items() if abs(v) <= 25}
        big = {k: v for k, v in shift.items() if abs(v) > 25}
        parts = []
        if small:
            parts.append("框这条边多算了 %s（接缝已按实测边界量，收窄这 %d px 再看）"
                         % ("，".join(f"{k} {v:+d}px" for k, v in small.items()),
                            max(abs(v) for v in small.values())))
        if big:
            parts.append("有 %s 的边在框内很深处才发现被改动 —— 这块的框与真正被处理的"
                         "区域不是同一条边（多半与别的区域重叠），该边的数字请以那个区域为准"
                         % "，".join(f"{k} {v:+d}px" for k, v in big.items()))
        out["boundary_note"] = "；".join(parts)
    worst = max((e["_excess"] for e in edges.values()), default=0.0)
    out["worst_excess_255"] = round(worst, 2)
    for e in edges.values():
        del e["_excess"]
    if not edges:
        out["seam_verdict"] = ("四条边在框内 25% 的范围内都找不到被改动的位置 → "
                               "框内没有合成边界（框偏宽，或这块根本没被处理）")
    elif worst <= SEAM_DC_MAX:
        out["seam_verdict"] = "未见引入的亮度台阶（超额台阶 ≤2/255）"
    else:
        bad = [nm for nm, e in edges.items() if e["excess"] > SEAM_DC_MAX]
        out["seam_verdict"] = (
            f"**接缝风险**：{'+'.join(bad)} 边引入了 {worst:.2f}/255 的亮度台阶 "
            f"（> 2/255）—— 看 step_change 与 step_target："
            f"target 小、clean 大即为重建留下的电平差，需回头看框边界")
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(ref: Path, tgt: Path, clean: Path | None, window, thr, rel, use_cache, as_json):
    q_prof = PROFILE_Q
    T = load_rgb(tgt)
    R0 = load_rgb(ref)
    clean_img = load_rgb(clean) if clean is not None else None
    if clean_img is not None and clean_img.shape[:2] != T.shape[:2]:
        # 结果图尺寸与目标图不一致时直接拒绝，避免静默按错误坐标量接缝
        print(f"结果图尺寸 {clean_img.shape[1]}x{clean_img.shape[0]} 与目标图 "
              f"{T.shape[1]}x{T.shape[0]} 不一致，无法做接缝验收（跳过 --clean）",
              file=sys.stderr)
        clean_img = None
    R0 = load_rgb(ref)
    h, w = T.shape[:2]
    A, ainfo = RO.align_with_cache(R0, T, ref, tgt, use_cache=use_cache)
    d255 = np.abs(T - A).mean(2) * 255.0

    out = {
        "reference": str(ref), "target": str(tgt),
        "target_size": [w, h],
        "alignment": {"ncc": round(float(ainfo.get("ncc", 0)), 5),
                      "scale_x": ainfo.get("scale_x"), "scale_y": ainfo.get("scale_y"),
                      "shift_target_px": ainfo.get("shift_target_px"),
                      "cache": ainfo.get("cache")},
        "threshold_255": thr, "regions": [], "warnings": [],
    }

    # 候选来自三条互补的定位器：
    #  ① α 块扫描 —— 主定位器，专抓半透明条（不会被细纹理骗到）；
    #  ② |d| 全局相对极值 —— 兜底，抓 |d| 够强的角落徽标；
    #  ③ 贴边条带相对基线极值 —— 专抓**比主带暗得多**的徽标（②会因为全局阈值漏掉它）。
    planes = _alpha_planes(T, A)
    boxes_a, meta_a = locate_alpha(planes, d255, stride=ALPHA_STRIDE)
    boxes_s, meta_s = locate(d255, window=window, rel=rel)
    boxes_e, meta_e = locate_edge_badge(d255, window=window)
    # 合并去重：同一个水印可能被多条路都找到，框位置只差十几像素
    cand = [(b, "alpha") for b in boxes_a]
    for b in boxes_s:
        if not any(max(abs(b[i] - c[i]) for i in range(4)) <= 32 for c, _s in cand):
            cand.append((b, "strong"))
    # 贴边候选只在「没被已有候选覆盖」时才加：否则 1203272712 这种
    # 已被 ② 找到的徽标会被重复报两次（两条路的框一个含 2048 端、一个不含，
    # 相互包含判据抓不住）。
    for b in boxes_e:
        if not any(_inter_over(b, c) >= 0.5 for c, _s in cand):
            cand.append((b, "edge"))
    loc = dict(meta_s, **meta_a, **meta_e)
    loc.update(n_alpha=len(boxes_a), n_strong=len(boxes_s))
    out["locate"] = loc
    if window:
        wy0, wy1, wx0, wx1 = window
        cand = [(b, s) for b, s in cand
                if not (b[1] < wy0 or b[0] > wy1 or b[3] < wx0 or b[2] > wx1)]
    if not cand:
        out["warnings"].append(
            "三条定位器都没有产出候选。要么图上确实没有水印，要么水印弱到"
            "低于纹理失配本底 —— 先裁图目视确认。")

    # 行/列参考剖面（全局），供人工核对
    seen = []
    for b, src in cand:
        if src == "alpha":
            # 边界判据取「平台 α 的 60%」而不是固定 0.15：条内压在中间调内容上的
            # 一段 α 只有 0.15~0.20（实测 2286834348 左端 x1064~1228），用固定门限
            # 会把它当作条的一部分，整框向左虚扩 ~160px。用相对门限后真实台阶
            # （α 0.02 → 0.47）成为唯一的边界依据。
            a_in = fit_alpha_planes(planes[0], planes[1], b)[0]
            rb, conf = refine_alpha(planes, d255, b,
                                    min_alpha=max(ALPHA_WATERMARK, 0.6 * a_in)), []
        else:
            rb, conf = refine(d255, b, thr=thr)
        met = measure(T, A, d255, rb, thr=thr)
        kind, verdict = classify(rb, met, (h, w))
        split_note = None
        if kind in ("mismatch", "weak", "reject"):
            sb = subsplit(T, A, d255, rb, thr=thr)
            if sb is not None:
                rb2, conf2 = refine(d255, sb, thr=thr)
                met2 = measure(T, A, d255, rb2, thr=thr)
                if met2["alpha"] >= ALPHA_WATERMARK and met2["frac"] >= FRAC_WATERMARK:
                    rb, met, conf = rb2, met2, conf2
                    kind, verdict = classify(rb, met, (h, w))
                    split_note = ("原框整体 α 不达标，已收缩到框内 α 达标的子区 —— "
                                  "细纹理失配与淡蒙版连成一片时的正常形态")
        # —— 去重必须放在 subsplit **之后**。subsplit 会把 rb 换成框内的另一个区域，
        # 甚至把 kind 从 mismatch 改成 bar（实测 1267196194 就是这样：一个 16×48 的
        # 小候选先被判成失配、收缩后落到整条主带上），放在它前面去重就会漏掉
        # 这种「先判失配、后收缩成真水印」的重复上报。
        # 判据用**面积覆盖率**而不是坐标容差 —— 实测那两个框只差 6~7px，
        # 坐标容差取 6 时刚好漏过、取 12 又可能误并相邻水印；而覆盖率达 98%。
        # 两个方向都要判（大框包小框 / 小框被大框包）。
        if any(_cover_ratio(rb, s) >= 0.85 or _cover_ratio(s, rb) >= 0.85
               for s in seen):
            continue
        seen.append(rb)
        # 嵌套去重：同一水印常被多条定位器各报一次、或被 α 扫描切出子块。
        # 只对同为 bar/badge 的框做包含判定，避免误删真正相邻的两块水印。
        if kind in ("bar", "badge"):
            same = [r for r in out["regions"] if r["kind"] == kind]
            if any(_cover_ratio(rb, r["box"]) >= 0.85 for r in same):
                continue
            out["regions"] = [r for r in out["regions"]
                              if not (r["kind"] == kind
                                      and _cover_ratio(r["box"], rb) >= 0.85)]
        i = len(out["regions"])
        rec = {
            "index": i, "raw_box": [int(v) for v in b], "box": [int(v) for v in rb],
            "h": rb[1] - rb[0], "w": rb[3] - rb[2],
            "kind": kind, "verdict": verdict, "refine_notes": conf, **met,
            "split_note": split_note,
            "box_arg": f"--box {rb[0]},{rb[1]},{rb[2]},{rb[3]}",
        }
        if met["ref_lum_std_255"] < FLAT_REF_STD:
            rec["flat_reference"] = True
            rec["note"] = ("参考图同区是纯色底图（lum std %.2f < %.1f）→ "
                           "α 回归必然失效（会给出 >1 被拒），改用「参考图回填 + 框外羽化」"
                           % (met["ref_lum_std_255"], FLAT_REF_STD))
        if clean_img is not None:
            rec["clean_check"] = check_clean(T, clean_img, rb, thr=thr)
        out["regions"].append(rec)

    if ainfo.get("ncc", 0) < 0.90:
        out["warnings"].append(
            "配准 NCC=%.3f < 0.90，残差里混有对齐误差，投影阶跃会变钝；"
            "考虑 --no-align-cache 或核对参考图是否同源。" % ainfo.get("ncc", 0))

    print("=" * 78)
    print(f"目标图 {tgt.name}  {w}x{h}")
    print(f"参考图 {ref.name}")
    al = out["alignment"]
    print(f"配准  NCC={al['ncc']}  scale=({al['scale_x']}, {al['scale_y']})  "
          f"shift={al['shift_target_px']}  cache={al['cache']}")
    print(f"判据  投影门限 {thr:.1f}/255   候选比例 {rel:.2f}×"
          f"   α块扫描 {meta_s.get('peak_255', 0):.1f}/255（{out['locate'].get('n_alpha', 0)} 个候选，"
          f"α峰值 {out['locate'].get('alpha_peak', 0):.3f}）"
          f"   贴边条带 {out['locate'].get('n_edge', 0)} 段")
    print("-" * 78)
    if not out["regions"]:
        print("未定位到候选区域。")
    for r in out["regions"]:
        y0, y1, x0, x1 = r["box"]
        print(f"[{r['index']}] {r['kind']:8s} box={r['box']}  {r['h']}x{r['w']}px")
        print(f"     α={r['alpha']:.4f} (R²={r['fit_r2']:.3f})"
              f"   框内 med|d|={r['med_in_255']:.1f} 框外 {r['med_out_255']:.1f}"
              f"   frac={r['frac']:.3f}")
        print(f"     带符号稳健中心 内 {r['center_in_255']:+.1f} / 外 {r['center_out_255']:+.1f}"
              f" 差 {r['center_delta_255']:.1f} /255"
              f"   参考图同区 lum std={r['ref_lum_std_255']:.2f}"
              f" range={r['ref_lum_range_255']:.1f}")
        print(f"     → {r['verdict']}")
        if r.get("note"):
            print(f"     ⚠ {r['note']}")
        if r.get("split_note"):
            print(f"     ⇢ {r['split_note']}")
        if r["refine_notes"]:
            print(f"     ⚠ 边界精修异常：{r['refine_notes']}（投影阶跃没找到，框沿用了粗值）")
        ap = alpha_probe(planes, r["box"])
        if "rows" in ap:
            print(f"     α 行阶跃（框外→框内，α 用 ±4 行条估；真蒙版应 1~2px 内从 ~0.05 跳到 ~0.45）:")
            print("       " + _fmt_prof(ap["rows"]))
        if "cols" in ap:
            print("     α 列阶跃:")
            print("       " + _fmt_prof(ap["cols"]))
        sp = step_probe(d255, r["box"], q=q_prof)
        if "rows" in sp:
            print("     行阶跃剖面（框外→框内，值=%.0f%%分位 |d| /255）:" % PROFILE_Q)
            print("       " + _fmt_prof(sp["rows"]))
        if "cols" in sp:
            print("     列阶跃剖面:")
            print("       " + _fmt_prof(sp["cols"]))
        if r.get("clean_check"):
            cc = r["clean_check"]
            if cc["untouched_flat"]:
                print(f"     ✗ 漏检：|T-clean| 中位仅 {cc['removed_median_255']:.2f}/255 —— "
                      "这块**根本没被处理**")
            else:
                print(f"     去水印量 |T-clean| 中位 {cc['removed_median_255']:.2f}/255（已处理）"
                      f"   框内被改动像素占比 {cc['processed_fraction']*100:.0f}%")
            if cc.get("oversize_note"):
                print(f"     ⚠ {cc['oversize_note']}")
            if cc.get("boundary_note"):
                print(f"     ⇢ {cc['boundary_note']}")
            print(f"     接缝 最大超额台阶 {cc['worst_excess_255']:.2f}/255 "
                  f"（= |clean 台阶| − |target 台阶|，在实测边界上量）→ {cc['seam_verdict']}")
            for nm, e in cc["edges"].items():
                tag = f"   {e['note']}" if e.get("note") else ""
                print(f"        {nm:6s}@{'row' if nm in ('top','bottom') else 'col'}"
                      f"{e.get('at', '?'):<5} 台阶 clean/target="
                      f"{e['step_clean']:+.2f}/{e['step_target']:+.2f}"
                      f"  超额 {e['excess']:+.2f}  "
                      f"HF 内/外={e['HF_in_255']:.2f}/{e['HF_out_255']:.2f}{tag}")
        print(f"     复现命令参数：{r['box_arg']}")
    for wn in out["warnings"]:
        print(f"⚠ {wn}")
    good = [r for r in out["regions"] if r["kind"] in ("bar", "badge")]
    print("-" * 78)
    if good:
        print("可直接粘贴的框（只含 α 达标的区域；每框一行）：")
        for r in good:
            print(f"  {r['box_arg']}    # {r['kind']} α={r['alpha']:.3f}")
        print("  ↑ rect_overlay.py 可重复传 --box，跳过自动检测与 refine。")
    else:
        print("没有 α 达标的区域 —— 图上可能没有可去除的实心半透明蒙版。")
        print("  若你确信有水印，依次检查：① 参考图是否同源（看 NCC）；")
        print("  ② 是否属于「细纹理失配」被误当水印（看上面的 mismatch 条）；")
        print("  ③ 降低 --thr 与 --rel 重跑，再裁图目视复核。")
    print("=" * 78)

    if as_json:
        Path(as_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON 报告已写入 {as_json}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="B 分支水印区域体检与框标定（只读，不改图）")
    ap.add_argument("--reference", required=True, help="612 干净参考图")
    ap.add_argument("--target", required=True, help="2048 带水印目标图")
    ap.add_argument("--clean", help="已有结果图，用于漏检扫描与接缝实测")
    ap.add_argument("--window", metavar="y0,y1,x0,x1",
                    help="只在窗口内找候选（徽标太强盖过主带时用）")
    ap.add_argument("--thr", type=float, default=DEFAULT_THR_255,
                    help=f"投影门限（/255），默认 {DEFAULT_THR_255}")
    ap.add_argument("--rel", type=float, default=REL_STRONG,
                    help=f"候选相对最强块的比例，默认 {REL_STRONG}")
    ap.add_argument("--json", metavar="PATH", help="把结构化结果写到 PATH")
    ap.add_argument("--no-align-cache", action="store_true", help="忽略配准缓存")
    a = ap.parse_args(argv)

    ref, tgt = Path(a.reference), Path(a.target)
    for p in (ref, tgt):
        if not p.exists():
            print(f"找不到文件：{p}", file=sys.stderr)
            return 2
    win = None
    if a.window:
        try:
            win = tuple(int(v) for v in a.window.split(","))
            assert len(win) == 4
        except Exception:
            print("--window 需要 y0,y1,x0,x1 四个整数", file=sys.stderr)
            return 2
    run(ref, tgt, Path(a.clean) if a.clean else None, win, a.thr, a.rel,
        not a.no_align_cache, a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
