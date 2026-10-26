"""기본 인체를 사진에 맞추기: 자세 → 전체 크기 → 부위별 실루엣 → 얼굴 기준점."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage as ndi

from .body import PARTS, Body, rotation
from .photos import FRONT_ARMPIT_ROW, FRONT_CROTCH_ROW, FRONT_SHOULDER_ROW, FRONT_UNDERBUST_ROW, View, runs, trunk_depth

TIPTOE_DEG = 62  # 까치발: 발목을 발끝 쪽으로 꺾는 각도(측면 사진 기준)
ELBOW_STRAIGHTEN_DEG = 22  # 기본 메시는 팔꿈치가 앞으로 굽어 있어서 사진처럼 편다
FIST_DEG = (70, 95, 70)  # 주먹: 손가락 마디별로 굽히는 각도
THUMB_DEG = (15, 35, 30)
MAX_BREAST_LIFT = 0.07  # 미터
Y_STEP = 0.005  # 실루엣 맞추기 높이 간격(미터)


def line_angle(points: np.ndarray) -> float:
    """(u, y) 점들을 지나는 직선이 아래쪽 수직선과 이루는 각도(바깥쪽 +, 도)."""
    u, y = points[:, 0], points[:, 1]
    slope = np.polyfit(y, u, 1)[0]  # du/dy
    return float(np.degrees(np.arctan(-slope)))


def limb_points(front: View, rows: range, outward: int, skip_center: bool) -> np.ndarray:
    """정면 사진에서 한쪽 팔/다리 덩어리의 높이별 중심점."""
    pts = []
    c = front.center
    for r in rows:
        for a, b in runs(front.mask[r]):
            if skip_center and a <= c < b:
                continue
            mid = (a + b) / 2
            if (mid - c) * outward > 0:
                pts.append((float(front.u_of(mid)), float(front.height_of(r))))
    return np.array(pts)


def limb_angle_model(p0: np.ndarray, p1: np.ndarray, outward: int) -> float:
    d = p1 - p0
    return float(np.degrees(np.arctan2(d[0] * outward, -d[1])))


def fist(body: Body, side: str) -> dict:
    """손가락을 손바닥 쪽으로 말아 주먹을 쥔다."""
    rots = {}
    h = body.head
    across = h[f"metacarpal4.{side}"] - h[f"metacarpal2.{side}"]  # 손등을 가로지르는 방향
    for f in range(2, 6):
        d = body.tail[f"finger{f}-1.{side}"] - h[f"finger{f}-1.{side}"]
        axis = np.cross(d, across)
        for k, deg in enumerate(FIST_DEG, start=1):
            rots[f"finger{f}-{k}.{side}"] = rotation(axis, deg * (1 if side == "L" else -1))
    d = body.tail[f"finger1-1.{side}"] - h[f"finger1-1.{side}"]
    axis = np.cross(d, across)
    for k, deg in enumerate(THUMB_DEG, start=1):
        rots[f"finger1-{k}.{side}"] = rotation(axis, deg * (1 if side == "L" else -1))
    return rots


def pose_from_photos(body: Body, front: View) -> dict:
    """정면 사진의 팔·다리 각도에 맞춘 뼈 회전들."""
    rots = {}
    for s in ("L", "R"):
        rots[f"foot.{s}"] = rotation([1, 0, 0], TIPTOE_DEG)
        for t in ("toe1-1", "toe2-1", "toe3-1", "toe4-1", "toe5-1"):
            rots[f"{t}.{s}"] = rotation([1, 0, 0], -TIPTOE_DEG)
        rots[f"lowerarm01.{s}"] = rotation([1, 0, 0], ELBOW_STRAIGHTEN_DEG)
        rots.update(fist(body, s))

    # 각도는 크기와 상관없으니 쉬는 자세에서 바로 비교한다
    _, mats = body.pose(rots)
    hand_top = FRONT_ARMPIT_ROW + int(0.55 * (FRONT_CROTCH_ROW - FRONT_ARMPIT_ROW))
    leg_rows = range(FRONT_CROTCH_ROW + 40, int(front.row_of(0.12)))
    for s, out in (("L", 1), ("R", -1)):
        arm = limb_points(front, range(FRONT_ARMPIT_ROW + 10, hand_top), out, skip_center=True)
        want = line_angle(arm) * out
        have = limb_angle_model(body.joint(mats, f"upperarm01.{s}"), body.joint(mats, f"wrist.{s}"), out)
        rots[f"upperarm01.{s}"] = rotation([0, 0, 1], (want - have) * out)

        leg = limb_points(front, leg_rows, out, skip_center=False)
        want = line_angle(leg) * out
        have = limb_angle_model(body.joint(mats, f"upperleg01.{s}"), body.joint(mats, f"foot.{s}"), out)
        rots[f"upperleg01.{s}"] = rotation([0, 0, 1], (want - have) * out)
    return rots


def place(verts: np.ndarray, used: np.ndarray) -> np.ndarray:
    """발끝을 바닥(y=0)에, 몸 가운데를 x=0에 놓는다."""
    v = verts.copy()
    v[:, 1] -= v[used, 1].min()
    v[:, 0] -= np.median(v[used, 0])
    return v


def model_chin(body: Body, verts: np.ndarray) -> np.ndarray:
    """턱 끝(아래턱 가장 아랫점). 쉬는 자세에서 턱뼈 끝 바로 앞쪽 정중선 꼭짓점 중 가장 낮은 것."""
    rest = body.rest
    used = body.used_vertices()
    tip = body.tail["jaw"]
    cand = used[
        (np.abs(rest[used, 0]) < 0.004)
        & (rest[used, 2] > tip[2])
        & (rest[used, 1] > tip[1] - 0.01)
        & (rest[used, 1] < tip[1] + 0.02)
    ]
    return verts[cand[np.argmin(rest[cand, 1])]]


def extents(values: np.ndarray, heights: np.ndarray, ys: np.ndarray):
    """높이 칸마다 값의 최소·최대(빈 칸은 NaN)."""
    lo = np.full(len(ys), np.nan)
    hi = np.full(len(ys), np.nan)
    j = np.round((heights - ys[0]) / Y_STEP).astype(int)
    ok = (j >= 0) & (j < len(ys))
    j, values = j[ok], values[ok]
    order = np.argsort(j)
    j, values = j[order], values[order]
    starts = np.r_[0, np.where(np.diff(j))[0] + 1]
    lo[j[starts]] = np.minimum.reduceat(values, starts)
    hi[j[starts]] = np.maximum.reduceat(values, starts)
    # 메시 고리 간격(1~2cm)보다 칸이 촘촘해서 빈 칸·덜 찬 칸이 생긴다: 이웃 칸까지 보고 바깥쪽 값을 쓴다
    have = ~np.isnan(lo)
    if have.sum() >= 2:
        idx = np.arange(len(ys))
        span = (idx >= idx[have][0]) & (idx <= idx[have][-1])
        lo_f = np.interp(idx, idx[have], lo[have])
        hi_f = np.interp(idx, idx[have], hi[have])
        lo = np.where(span, ndi.minimum_filter1d(lo_f, 5), np.nan)
        hi = np.where(span, ndi.maximum_filter1d(hi_f, 5), np.nan)
    return lo, hi


def smooth_fill(a: np.ndarray, sigma: float, default: float) -> np.ndarray:
    """NaN을 선형 보간으로 메우고 부드럽게. 전부 NaN이면 default."""
    ok = ~np.isnan(a)
    if not ok.any():
        return np.full_like(a, default)
    idx = np.arange(len(a))
    filled = np.interp(idx, idx[ok], a[ok])
    return ndi.gaussian_filter1d(filled, sigma)


def render_mask(view: View, u: np.ndarray, y: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """모델을 사진과 같은 화면(정사영)에 그린 실루엣."""
    from PIL import Image, ImageDraw

    im = Image.new("1", (view.w, view.h), 0)
    draw = ImageDraw.Draw(im)
    cols, rows = view.col_of(u), view.row_of(y)
    for f in faces:
        draw.polygon(list(zip(cols[f], rows[f])), fill=1)
    return np.array(im, bool)


def keypoint_rows(mask: np.ndarray, center_col: int) -> dict:
    """실루엣에서 가랑이·겨드랑이 행을 찾는다(사진과 모델에 똑같이 쓴다)."""
    rows = np.where(mask.any(1))[0]
    top, bottom = rows[0], rows[-1]
    mid = top + (bottom - top) // 2
    # 가랑이: 몸 가운데 열이 처음으로 비는 행(위에서 아래로)
    crotch = next(r for r in range(top + (bottom - top) // 3, bottom) if not mask[r, center_col])
    # 겨드랑이: 가랑이에서 위로 올라가며 팔이 몸통에서 떨어져 있는 마지막 행
    armpit = crotch
    for r in range(crotch - 10, top, -1):
        if len(runs(mask[r])) >= 3:
            armpit = r
        elif armpit != crotch and r < armpit - 5:
            break
    return {"crotch": crotch, "armpit": armpit, "mid": mid}


def remap_heights(v: np.ndarray, src: list[float], dst: list[float]) -> np.ndarray:
    """높이를 구간별로 늘이고 줄인다(마지막 기준점 위는 그대로 평행이동)."""
    out = v.copy()
    y = v[:, 1]
    below = y <= src[-1]
    out[below, 1] = np.interp(y[below], src, dst)
    out[~below, 1] = y[~below] + (dst[-1] - src[-1])
    return out


class SliceFit:
    """높이별 범위 맞추기: 모델 [lo, hi] → 사진 [lo, hi].

    가운데를 기준으로 양쪽 절반을 따로 늘인다(split=True). 깊이 방향에 쓰면
    가슴·엉덩이처럼 한쪽만 튀어나온 모양이 그대로 산다.
    """

    def __init__(self, ys, m_lo, m_hi, p_lo, p_hi, valid, anchors=(), split=False, sigma=3):
        ok = valid & ~np.isnan(m_lo) & ~np.isnan(p_lo) & (m_hi - m_lo > 0.01)
        m_mid, p_mid = (m_lo + m_hi) / 2, (p_lo + p_hi) / 2
        if split:
            s_hi = (p_hi - p_mid) / np.maximum(m_hi - m_mid, 1e-6)
            s_lo = (p_mid - p_lo) / np.maximum(m_mid - m_lo, 1e-6)
        else:
            s_hi = s_lo = (p_hi - p_lo) / np.maximum(m_hi - m_lo, 1e-6)
        nan = np.full(len(ys), np.nan)
        s_hi, s_lo = np.where(ok, s_hi, nan), np.where(ok, s_lo, nan)
        shift = np.where(ok, p_mid - m_mid, nan)
        m_mid = np.where(ok, m_mid, nan)
        for y, val in anchors:  # 맞출 데이터가 없는 높이에 고정값(예: 목은 그대로)
            j = int(np.clip(round((y - ys[0]) / Y_STEP), 0, len(ys) - 1))
            if np.isnan(s_hi[j]):
                s_hi[j] = s_lo[j] = val
                shift[j] = 0.0
        self.ys = ys
        self.s_hi = smooth_fill(np.clip(s_hi, 0.5, 1.8), sigma, 1.0)
        self.s_lo = smooth_fill(np.clip(s_lo, 0.5, 1.8), sigma, 1.0)
        self.shift = smooth_fill(shift, sigma, 0.0)
        self.mid = smooth_fill(m_mid, sigma, 0.0)

    def apply(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        c = np.interp(y, self.ys, self.mid)
        s = np.where(u >= c, np.interp(y, self.ys, self.s_hi), np.interp(y, self.ys, self.s_lo))
        return c + np.interp(y, self.ys, self.shift) + (u - c) * s


def breast_mass_weight(body: Body, v: np.ndarray, used: np.ndarray, radius: float = 0.16, flat: float = 0.4) -> np.ndarray:
    """가슴 덩어리를 통째로 옮길 때 쓰는 가중치: 반경 flat*radius 안은 평평하게 1, radius에서 0.

    스키닝 가중치처럼 꼭지에서 뾰족한 가중치로 옮기면 원뿔 모양이 되고,
    줄어드는 폭이 좁으면 옮긴 만큼 가장자리 면이 늘어나 각이 진다."""
    skin_w = body.weights[:, [i for i, b in enumerate(body.bones) if b.startswith("breast")]].sum(1)
    w = np.zeros(len(v))
    for side in (1, -1):
        core = used[(skin_w[used] > 0.5) & (np.sign(v[used, 0]) == side)]
        c = v[core].mean(0)
        d = np.linalg.norm((v - c) * [1, 1, 0.5], axis=1)  # 앞뒤로는 넉넉히(가슴 안쪽 흉곽까지 같이)
        t = np.clip((radius - d) / ((1 - flat) * radius), 0, 1)
        w = np.maximum(w, t * t * (3 - 2 * t) * (np.sign(v[:, 0]) == side))
    return w


def head_landmarks(body: Body, v: np.ndarray, used: np.ndarray) -> dict:
    head_w = body.part_weights[:, PARTS.index("head")]
    eyes = {}
    for side, group in (("eye_l", 0), ("eye_r", 1)):
        idx = np.unique(body.faces[body.is_eye])
        # 눈알 두 개를 x 부호로 나눈다(캐릭터 왼쪽 = +x)
        sel = idx[(v[idx, 0] > 0) == (side == "eye_l")]
        eyes[side] = v[sel].mean(0)
    m = np.zeros(len(v), bool)
    m[used] = True
    center = m & (head_w > 0.9) & (np.abs(v[:, 0]) < 0.006)
    nose = v[np.where(center)[0][np.argmax(v[center, 2])]]
    # 입: 윗입술·아랫입술 근육 뼈 끝의 가운데(쉬는 자세 기준)에 가장 가까운 정중선 꼭짓점
    mouth_rest = (body.tail["oris05"] + body.tail["oris01"]) / 2
    mid = np.where(center)[0]
    mouth = v[mid[np.argmin(np.linalg.norm(body.rest[mid] - mouth_rest, axis=1))]]
    return {**eyes, "nose": nose, "mouth": mouth, "chin": model_chin(body, v)}


def fit_head(body: Body, v: np.ndarray, used: np.ndarray, front_lm: dict, side_lms: list[dict]) -> np.ndarray:
    """눈·코·턱 기준점이 사진과 겹치게 머리를 옮기고 늘인다(목은 가중치로 자연스럽게 이어짐)."""
    m = head_landmarks(body, v, used)
    side = {k: np.mean([s[k] for s in side_lms], axis=0) for k in ("eye", "nose", "chin")}  # (z, y)
    p_eye_mid = np.array([(front_lm["eye_l"][0] + front_lm["eye_r"][0]) / 2, front_lm["eye_l"][1], side["eye"][0]])
    m_eye_mid = (m["eye_l"] + m["eye_r"]) / 2
    sx = (front_lm["eye_l"][0] - front_lm["eye_r"][0]) / (m["eye_l"][0] - m["eye_r"][0])
    sy = (front_lm["eye_l"][1] - front_lm["chin"][1]) / (m_eye_mid[1] - m["chin"][1])
    sz = (sx + sy) / 2  # 측면 기준점 간 거리는 몇 cm라 오차가 커서 앞뒤 크기는 평균으로
    s = np.array([sx, sy, sz])
    target = p_eye_mid + (v - m_eye_mid) * s
    # 입 높이: 눈~턱 사이를 구간별로 늘이고 줄여 입이 사진의 입에 오게
    m_mouth = p_eye_mid[1] + (m["mouth"][1] - m_eye_mid[1]) * sy
    d = front_lm["mouth"][1] - m_mouth
    target[:, 1] += np.interp(target[:, 1], [front_lm["chin"][1], m_mouth, p_eye_mid[1]], [0, d, 0], left=0, right=0)
    # 앞뒤 위치는 코끝으로 맞춘다(모델 눈 기준점은 눈알 중심이라 사진의 눈 표면보다 1cm쯤 안쪽)
    target[:, 2] += side["nose"][0] - (p_eye_mid[2] + (m["nose"][2] - m_eye_mid[2]) * sz)
    w = body.part_weights[:, PARTS.index("head")][:, None]
    return v + w * (target - v)


def fit_to_photos(body: Body, front: View, sides: list[View], front_lm: dict) -> np.ndarray:
    used = body.used_vertices()
    faces = body.faces[~body.is_eye]

    # 1) 자세, 바닥, 전체 크기(턱 높이 기준: 머리카락 높이는 사진마다 달라 믿기 어렵다)
    rots = pose_from_photos(body, front)
    v, mats = body.pose(rots)
    v = place(v, used)
    v *= front_lm["chin"][1] / model_chin(body, v)[1]

    # 2) 몸 비율: 가랑이·겨드랑이·턱 높이를 사진에 맞춰 세로로 늘이고 줄인다
    kp_m = keypoint_rows(render_mask(front, v[:, 0], v[:, 1], faces), int(round(float(front.col_of(0)))))
    kp_p = keypoint_rows(front.mask, int(round(front.center)))
    chin_m = model_chin(body, v)[1]
    src = [0.0, float(front.height_of(kp_m["crotch"])), float(front.height_of(kp_m["armpit"])), chin_m]
    dst = [0.0, float(front.height_of(kp_p["crotch"])), float(front.height_of(kp_p["armpit"])), front_lm["chin"][1]]
    v = remap_heights(v, src, dst)
    y_crotch, y_armpit = dst[1], dst[2]

    # 가슴: 사진은 브라로 받친 높은 가슴이라 기본 메시보다 아랫선이 높다. 모양은 두고 통째로 올린다
    breast_w = breast_mass_weight(body, v, used)
    core = used[breast_w[used] > 0.9]
    lift = float(np.clip(front.height_of(FRONT_UNDERBUST_ROW) - v[core, 1].min(), 0, MAX_BREAST_LIFT))
    v[:, 1] += breast_w * lift
    y_shoulder = float(front.height_of(FRONT_SHOULDER_ROW))

    # 3) 부위별 높이 단면 맞추기(정면 = 좌우 폭, 측면 = 앞뒤 깊이)
    ys = np.arange(0.0, front_lm["chin"][1], Y_STEP)
    part = np.argmax(body.part_weights, axis=1)
    in_use = np.zeros(len(v), bool)
    in_use[used] = True
    neck = [(y_shoulder + 0.04, 1.0)]
    zf_all, zb_all = zip(*(trunk_depth(front, s, ys) for s in sides))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 사람이 없는 높이는 전부 NaN
        zf, zb = np.nanmean(zf_all, 0), np.nanmean(zb_all, 0)

    def model_ext(p, axis):
        sel = in_use & (part == PARTS.index(p))
        return extents(v[sel, axis], v[sel, 1], ys)

    def photo_ext(view, hints):
        lo, hi = np.full(len(ys), np.nan), np.full(len(ys), np.nan)
        for j, y in enumerate(ys):
            if np.isnan(hints[j]):
                continue
            r = view.run_at(y, hints[j])
            if r:
                lo[j], hi[j] = r
        return lo, hi

    fits = {}
    m_lo, m_hi = model_ext("trunk", 0)
    p_lo, p_hi = photo_ext(front, np.zeros(len(ys)))
    fits["trunk", 0] = SliceFit(ys, m_lo, m_hi, p_lo, p_hi, (ys > y_crotch + 0.01) & (ys < y_armpit - 0.005), neck)
    m_lo, m_hi = model_ext("trunk", 2)
    # 깊이는 목까지(측면 사진에선 목이 머리카락에 덜 가려진다)
    # 가슴 구간은 높이별로 늘이면 원뿔처럼 뾰족해진다: 건너뛰고(위아래에서 보간) 아래에서 가슴만 따로 맞춘다
    y_underbust = float(front.height_of(FRONT_UNDERBUST_ROW))
    bust_band = (ys > y_underbust - 0.015) & (ys < y_armpit + 0.03)
    z_valid = (ys > y_crotch + 0.01) & (ys < front_lm["chin"][1] - 0.015) & ~bust_band
    fits["trunk", 2] = SliceFit(ys, m_lo, m_hi, zb, zf, z_valid, split=True, sigma=5)
    for leg in ("leg.L", "leg.R"):
        m_lo, m_hi = model_ext(leg, 0)
        p_lo, p_hi = photo_ext(front, (m_lo + m_hi) / 2)
        legs_valid = (ys > 0.03) & (ys < y_crotch - 0.01)  # 발끝은 바닥 그림자와 섞여서 뺀다
        fits[leg, 0] = SliceFit(ys, m_lo, m_hi, p_lo, p_hi, legs_valid)
        m_lo, m_hi = model_ext(leg, 2)
        sides_ext = [photo_ext(s, (m_lo + m_hi) / 2) for s in sides]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p_lo, p_hi = np.nanmean([e[0] for e in sides_ext], 0), np.nanmean([e[1] for e in sides_ext], 0)
        fits[leg, 2] = SliceFit(ys, m_lo, m_hi, p_lo, p_hi, legs_valid, split=True)

    out = v.copy()
    for p in ("trunk", "leg.L", "leg.R"):
        w = body.part_weights[:, PARTS.index(p)]
        for axis in (0, 2):
            out[:, axis] += w * (fits[p, axis].apply(v[:, axis], v[:, 1]) - v[:, axis])
    # 팔: 겨드랑이 아래는 정면 사진의 팔 덩어리에 위치·굵기를 맞추고(앞뒤 굵기도 같은 비율),
    # 팔이 몸통에 붙어 보이는 어깨 쪽은 몸통이 움직인 만큼 따라간다
    for s in ("L", "R"):
        w = body.part_weights[:, PARTS.index(f"arm.{s}")]
        sel = in_use & (part == PARTS.index(f"arm.{s}"))
        top = v[sel][np.argmax(v[sel, 1])]  # 어깨 끝
        shoulder = [fits["trunk", a].apply(np.array([top[a]]), np.array([top[1]]))[0] - top[a] for a in (0, 2)]
        m_lo, m_hi = extents(v[sel, 0], v[sel, 1], ys)
        p_lo, p_hi = photo_ext(front, (m_lo + m_hi) / 2)
        hand_bottom = v[sel, 1].min()
        arm_valid = (ys > hand_bottom + 0.03) & (ys < y_armpit - 0.01)
        fx = SliceFit(ys, m_lo, m_hi, p_lo, p_hi, arm_valid, [(top[1], 1.0)])
        # 주먹: 사진과 모델 손 모양이 달라 굵기는 그대로 두고 위치만 팔을 따라간다
        hand_cols = [i for i, b in enumerate(body.bones) if b.startswith(("wrist", "metacarpal", "finger")) and b.endswith(s)]
        hand = sel & (body.weights[:, hand_cols].sum(1) > 0.5)
        fist_rows = ys < v[hand, 1].max() + 0.01
        fx.s_hi[fist_rows] = fx.s_lo[fist_rows] = 1.0
        fx.shift = np.where(ys >= y_armpit, np.interp(ys, [y_armpit, top[1]], [np.interp(y_armpit, ys, fx.shift), shoulder[0]]), fx.shift)
        z_lo, z_hi = extents(v[sel, 2], v[sel, 1], ys)
        zc = smooth_fill((z_lo + z_hi) / 2, 3, 0.0)
        x_new = fx.apply(v[:, 0], v[:, 1])
        k = np.interp(v[:, 1], ys, (fx.s_hi + fx.s_lo) / 2)
        z_new = np.interp(v[:, 1], ys, zc) + (v[:, 2] - np.interp(v[:, 1], ys, zc)) * k + shoulder[1]
        out[:, 0] += w * (x_new - v[:, 0])
        out[:, 2] += w * (z_new - v[:, 2])

    # 가슴 돌출: 사진 측면의 가장 앞선 점까지 가슴을 통째로 앞으로 민다
    in_band = (ys > y_underbust) & (ys < y_armpit + 0.03)
    want = np.nanmax(zf[in_band])
    band_v = in_use & (out[:, 1] > y_underbust) & (out[:, 1] < y_armpit + 0.03) & (part == PARTS.index("trunk"))
    out[:, 2] += breast_w * np.clip(want - out[band_v, 2].max(), -0.03, 0.04)

    # 4) 머리: 눈·코·턱 기준점
    out = fit_head(body, out, used, front_lm, [s.landmarks for s in sides])
    return out
