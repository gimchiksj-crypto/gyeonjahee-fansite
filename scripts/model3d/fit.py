"""기본 인체를 사진에 맞추기: 자세 → 체형 조절값 최적화 → 얼굴 텍스처 정렬.

메시를 높이별로 자유롭게 늘이면 사진 윤곽엔 맞아도 사람 몸에서 벗어난다
(골반이 승마바지처럼 튀거나 가슴만 떠 있는 식). 그래서 모양은 MakeHuman 체형
조절값(실측 인체 데이터로 만든 둘레·길이·폭 변형)의 조합으로만 바꾸고, 그 값을
사진 윤곽에 가장 가깝게 고른다. 어떤 값이든 해부학 구조 안에 머문다.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage as ndi

from .body import PARTS, Body, load_target, rotation
from .photos import FRONT_ARMPIT_ROW, FRONT_CROTCH_ROW, FRONT_SHOULDER_ROW, FRONT_UNDERBUST_ROW, View, runs, trunk_depth

TIPTOE_DEG = 62  # 까치발: 발목을 발끝 쪽으로 꺾는 각도(측면 사진 기준)
ELBOW_STRAIGHTEN_DEG = 22  # 기본 메시는 팔꿈치가 앞으로 굽어 있어서 사진처럼 편다
FIST_DEG = (70, 95, 70)  # 주먹: 손가락 마디별로 굽히는 각도
THUMB_DEG = (15, 35, 30)
Y_STEP = 0.01  # 실루엣 비교 높이 간격(미터)

_F = "breast/female-young-averagemuscle-averageweight-"
# 체형 조절값: 이름 → (값이 -1일 때 target, +1일 때 target, 최소, 최대)
MODIFIERS = {
    "bust": ("measure/measure-bust-circ-decr", "measure/measure-bust-circ-incr", -1, 1),
    "underbust": ("measure/measure-underbust-circ-decr", "measure/measure-underbust-circ-incr", -1, 1),
    "cup": (_F + "mincup-averagefirmness", _F + "maxcup-averagefirmness", -1, 1),
    "firmness": (_F + "averagecup-minfirmness", _F + "averagecup-maxfirmness", -1, 1),
    "waist": ("measure/measure-waist-circ-decr", "measure/measure-waist-circ-incr", -1, 1),
    "hips": ("measure/measure-hips-circ-decr", "measure/measure-hips-circ-incr", -1, 1),
    "buttocks": ("buttocks/buttocks-volume-decr", "buttocks/buttocks-volume-incr", -1, 1),
    "hip_width": ("hip/hip-scale-horiz-decr", "hip/hip-scale-horiz-incr", -1, 1),
    "hip_depth": ("hip/hip-scale-depth-decr", "hip/hip-scale-depth-incr", -1, 1),
    "torso_width": ("torso/torso-scale-horiz-decr", "torso/torso-scale-horiz-incr", -1, 1),
    "torso_depth": ("torso/torso-scale-depth-decr", "torso/torso-scale-depth-incr", -1, 1),
    "vshape": ("torso/torso-vshape-decr", "torso/torso-vshape-incr", -1, 1),
    "shoulders": ("measure/measure-shoulder-dist-decr", "measure/measure-shoulder-dist-incr", -1, 1),
    "nape_waist": ("measure/measure-napetowaist-dist-decr", "measure/measure-napetowaist-dist-incr", -1, 1),
    "waist_hip": ("measure/measure-waisttohip-dist-decr", "measure/measure-waisttohip-dist-incr", -1, 1),
    "thigh": ("measure/measure-thigh-circ-decr", "measure/measure-thigh-circ-incr", -1, 1),
    "knee": ("measure/measure-knee-circ-decr", "measure/measure-knee-circ-incr", -1, 1),
    "calf": ("measure/measure-calf-circ-decr", "measure/measure-calf-circ-incr", -1, 1),
    "ankle": ("measure/measure-ankle-circ-decr", "measure/measure-ankle-circ-incr", -1, 1),
    "upperleg_len": ("measure/measure-upperleg-height-decr", "measure/measure-upperleg-height-incr", -1, 1),
    "lowerleg_len": ("measure/measure-lowerleg-height-decr", "measure/measure-lowerleg-height-incr", -1, 1),
    "upperarm": ("measure/measure-upperarm-circ-decr", "measure/measure-upperarm-circ-incr", -1, 1),
    "upperarm_len": ("measure/measure-upperarm-length-decr", "measure/measure-upperarm-length-incr", -1, 1),
    "lowerarm_len": ("measure/measure-lowerarm-length-decr", "measure/measure-lowerarm-length-incr", -1, 1),
    "neck": ("measure/measure-neck-circ-decr", "measure/measure-neck-circ-incr", -1, 1),
    # 머리 크기는 사진(그림체라 머리가 작다)을 다 따라가지 않게 좁은 범위만
    "head_width": ("head/head-scale-horiz-decr", "head/head-scale-horiz-incr", -0.8, 0.8),
    "head_height": ("head/head-scale-vert-decr", "head/head-scale-vert-incr", -0.8, 0.8),
}
REGULARIZE = 0.0006  # 조절값을 0(평균 체형)으로 당기는 힘: 사진 오차에 끌려 과하게 비틀리지 않게


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
        # 팔: 위팔과 아래팔 각도를 따로 잰다(팔꿈치 높이는 겨드랑이~손 사이 40% 지점)
        arm = limb_points(front, range(FRONT_ARMPIT_ROW + 10, hand_top), out, skip_center=True)
        elbow_y = arm[:, 1].max() - 0.4 * (arm[:, 1].max() - arm[:, 1].min())
        upper = arm[arm[:, 1] >= elbow_y]
        lower = arm[arm[:, 1] < elbow_y]
        want_upper, want_lower = line_angle(upper) * out, line_angle(lower) * out
        have = limb_angle_model(body.joint(mats, f"upperarm01.{s}"), body.joint(mats, f"lowerarm01.{s}"), out)
        rots[f"upperarm01.{s}"] = rotation([0, 0, 1], (want_upper - have) * out)
        _, mats = body.pose(rots)
        have = limb_angle_model(body.joint(mats, f"lowerarm01.{s}"), body.joint(mats, f"wrist.{s}"), out)
        rots[f"lowerarm01.{s}"] = rotation([0, 0, 1], (want_lower - have) * out) @ rots[f"lowerarm01.{s}"]

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


def keypoint_rows(mask: np.ndarray, center_col: int) -> dict:
    """실루엣에서 가랑이·겨드랑이 행을 찾는다."""
    rows = np.where(mask.any(1))[0]
    top, bottom = rows[0], rows[-1]
    # 가랑이: 몸 가운데 열이 처음으로 비는 행(위에서 아래로)
    crotch = next(r for r in range(top + (bottom - top) // 3, bottom) if not mask[r, center_col])
    # 겨드랑이: 가랑이에서 위로 올라가며 팔이 몸통에서 떨어져 있는 마지막 행
    armpit = crotch
    for r in range(crotch - 10, top, -1):
        if len(runs(mask[r])) >= 3:
            armpit = r
        elif armpit != crotch and r < armpit - 5:
            break
    return {"crotch": crotch, "armpit": armpit}


def head_landmarks(body: Body, v: np.ndarray) -> dict:
    """모델 얼굴 기준점: 눈(눈알 앞면), 코끝, 입, 턱 끝."""
    used = body.used_vertices()
    head_w = body.part_weights[:, PARTS.index("head")]
    eyes = {}
    idx = np.unique(body.faces[body.is_eye])
    for side in ("eye_l", "eye_r"):
        # 눈알 두 개를 x 부호로 나눈다(캐릭터 왼쪽 = +x). 사진 기준점과 맞추려고 앞면(각막) 높이·깊이를 쓴다
        sel = idx[(v[idx, 0] > 0) == (side == "eye_l")]
        c = v[sel].mean(0)
        eyes[side] = np.array([c[0], c[1], v[sel, 2].max()])
    m = np.zeros(len(v), bool)
    m[used] = True
    center = m & (head_w > 0.9) & (np.abs(body.rest[:, 0]) < 0.006)
    nose = v[np.where(center)[0][np.argmax(v[center, 2])]]
    # 입: 윗입술·아랫입술 근육 뼈 끝의 가운데(쉬는 자세 기준)에 가장 가까운 정중선 꼭짓점
    mouth_rest = (body.tail["oris05"] + body.tail["oris01"]) / 2
    mid = np.where(center)[0]
    mouth = v[mid[np.argmin(np.linalg.norm(body.rest[mid] - mouth_rest, axis=1))]]
    return {**eyes, "nose": nose, "mouth": mouth, "chin": model_chin(body, v)}


class ShapeFit:
    """체형 조절값 → 자세 잡고 사진 크기로 놓은 메시. 조절값에 대해 선형이라 빠르게 계산된다."""

    def __init__(self, body: Body, rots: dict, base_dir):
        self.body = body
        posed, mats = body.pose(rots)
        self.posed = posed
        # 조절값의 변형(쉬는 자세 기준)을 자세 잡은 메시로 옮긴다: 뼈 회전만 가중 평균해서 적용
        rot = np.stack([mats[b][:3, :3] for b in body.bones])
        self.rot = np.einsum("nb,bij->nij", body.weights, rot)
        n = len(body.rest)
        self.names = list(MODIFIERS)
        self.lo = np.array([MODIFIERS[k][2] for k in self.names], float)
        self.hi = np.array([MODIFIERS[k][3] for k in self.names], float)
        self.neg, self.pos = [], []
        for k in self.names:
            for target, out in ((MODIFIERS[k][0], self.neg), (MODIFIERS[k][1], self.pos)):
                d = load_target(base_dir, target, n)
                out.append(np.einsum("nij,nj->ni", self.rot, d))
        self.used = body.used_vertices()

    def mesh(self, m: np.ndarray) -> np.ndarray:
        v = self.posed.copy()
        for k, x in enumerate(m):
            if x > 0:
                v += x * self.pos[k]
            elif x < 0:
                v -= x * self.neg[k]
        return v

    def targets(self, m: np.ndarray) -> dict:
        out = {}
        for k, x in zip(self.names, m):
            if abs(x) > 1e-4:
                out[MODIFIERS[k][1] if x > 0 else MODIFIERS[k][0]] = float(abs(x))
        return out


def place_and_scale(body: Body, v: np.ndarray, chin_y: float) -> np.ndarray:
    """발끝을 바닥에, 몸 가운데를 x=0에 놓고 턱 높이가 사진과 같아지게 전체를 같은 비율로 키운다."""
    v = place(v, body.used_vertices())
    return v * (chin_y / model_chin(body, v)[1])


class PhotoTargets:
    """사진에서 잰 높이별 윤곽(정면 폭·측면 깊이)과 기준 높이들."""

    def __init__(self, front: View, sides: list[View], front_lm: dict):
        self.chin = front_lm["chin"][1]
        self.ys = np.arange(0.02, self.chin, Y_STEP)
        kp = keypoint_rows(front.mask, int(round(front.center)))
        self.crotch = float(front.height_of(kp["crotch"]))
        self.armpit = float(front.height_of(kp["armpit"]))
        self.shoulder = float(front.height_of(FRONT_SHOULDER_ROW))
        self.underbust = float(front.height_of(FRONT_UNDERBUST_ROW))
        ys = self.ys
        nan = lambda: np.full(len(ys), np.nan)  # noqa: E731

        self.trunk_x = (nan(), nan())
        self.leg_x = {s: (nan(), nan()) for s in ("L", "R")}
        self.arm_x = {s: (nan(), nan()) for s in ("L", "R")}
        for j, y in enumerate(ys):
            r = int(round(float(front.row_of(y))))
            rs = [(float(front.u_of(a)), float(front.u_of(b))) for a, b in runs(front.mask[r])]
            if not rs:
                continue
            if y > self.crotch + 0.02 and y < self.armpit - 0.01:
                trunk = [q for q in rs if q[0] <= 0 < q[1]]
                if trunk:
                    self.trunk_x[0][j], self.trunk_x[1][j] = trunk[0]
                for side, sign in (("L", 1), ("R", -1)):
                    arm = [q for q in rs if not (q[0] <= 0 < q[1]) and (q[0] + q[1]) * sign > 0]
                    if len(arm) == 1:
                        self.arm_x[side][0][j], self.arm_x[side][1][j] = arm[0]
            if y < self.crotch - 0.02 and len(rs) == 2:
                for side, q in zip(("R", "L"), sorted(rs)):
                    self.leg_x[side][0][j], self.leg_x[side][1][j] = q
        zf_all, zb_all = zip(*(trunk_depth(front, s, ys) for s in sides))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # 사람이 없는 높이는 전부 NaN
            zf, zb = np.nanmean(zf_all, 0), np.nanmean(zb_all, 0)
        keep = (ys > self.crotch + 0.02) & (ys < self.shoulder)
        self.trunk_z = (np.where(keep, zb, np.nan), np.where(keep, zf, np.nan))
        # 다리 앞뒤: 측면에선 두 다리가 겹쳐 보여서 둘을 합친 범위
        legs_z = [nan(), nan()]
        for j, y in enumerate(ys):
            if y >= self.crotch - 0.02:
                continue
            lo, hi = [], []
            for s in sides:
                r = int(round(float(s.row_of(y))))
                rs = runs(s.mask[r])
                if rs:
                    a, b = max(rs, key=lambda q: q[1] - q[0])
                    lo.append(float(s.u_of(a)))
                    hi.append(float(s.u_of(b)))
            if lo:
                legs_z[0][j], legs_z[1][j] = np.mean(lo), np.mean(hi)
        self.legs_z = tuple(legs_z)
        self.eye_dist = front_lm["eye_l"][0] - front_lm["eye_r"][0]
        self.eye_chin = front_lm["eye_l"][1] - front_lm["chin"][1]


def align_depth(v: np.ndarray, photo: PhotoTargets, part: np.ndarray, in_use: np.ndarray) -> np.ndarray:
    """앞뒤(z) 위치: 몸통 앞뒤 가운데가 측면 사진의 몸통 가운데와 겹치게 옮긴다."""
    trunk = in_use & (part == PARTS.index("trunk"))
    lo, hi = extents(v[trunk, 2], v[trunk, 1], photo.ys)
    d = (photo.trunk_z[0] + photo.trunk_z[1]) / 2 - (lo + hi) / 2
    shift = np.nanmedian(d) if np.isfinite(d).any() else 0.0
    return v + [0, 0, shift]


def residuals(body: Body, v: np.ndarray, photo: PhotoTargets, part: np.ndarray, in_use: np.ndarray) -> np.ndarray:
    """모델과 사진의 차이(미터). 사진에 값이 없는 칸은 0."""
    ys = photo.ys
    out = []

    def ext(sel, axis):
        lo, hi = extents(v[sel, axis], v[sel, 1], ys)
        return lo, hi

    def diff(model, target, weight=1.0):
        for m_, t_ in zip(model, target):
            d = m_ - t_
            out.append(np.where(np.isnan(d), 0.0, d) * weight)

    trunk = in_use & (part == PARTS.index("trunk"))
    diff(ext(trunk, 0), photo.trunk_x)
    diff(ext(trunk, 2), photo.trunk_z)
    legs = in_use & np.isin(part, [PARTS.index("leg.L"), PARTS.index("leg.R")])
    diff(ext(legs, 2), photo.legs_z)
    for s in ("L", "R"):
        diff(ext(in_use & (part == PARTS.index(f"leg.{s}")), 0), photo.leg_x[s])
        diff(ext(in_use & (part == PARTS.index(f"arm.{s}")), 0), photo.arm_x[s], 0.5)
    # 기준 높이: 가랑이(다리 길이), 가슴 아랫선
    mid = in_use & (np.abs(v[:, 0]) < 0.01) & (v[:, 1] > 0.3 * photo.chin) & (v[:, 1] < 0.7 * photo.chin)
    out.append(np.array([(v[mid, 1].min() - photo.crotch) * 4]))
    breast_w = body.weights[:, body.bones.index("breast.L")] + body.weights[:, body.bones.index("breast.R")]
    breast = in_use & (breast_w > 0.5)
    out.append(np.array([(v[breast, 1].min() - photo.underbust) * 4]))
    # 얼굴 크기: 눈 사이·눈~턱
    lm = head_landmarks(body, v)
    out.append(np.array([(lm["eye_l"][0] - lm["eye_r"][0] - photo.eye_dist) * 3, (lm["eye_l"][1] - lm["chin"][1] - photo.eye_chin) * 3]))
    return np.concatenate(out)


def fit_shape(body: Body, rots: dict, base_dir, front: View, sides: list[View], front_lm: dict, iterations: int = 15):
    """체형 조절값을 레벤버그-마쿼트 방식으로 고른다."""
    sf = ShapeFit(body, rots, base_dir)
    photo = PhotoTargets(front, sides, front_lm)
    part = np.argmax(body.part_weights, axis=1)
    in_use = np.zeros(len(body.rest), bool)
    in_use[sf.used] = True

    def cost_parts(m):
        v = align_depth(place_and_scale(body, sf.mesh(m), photo.chin), photo, part, in_use)
        r = residuals(body, v, photo, part, in_use)
        return r, float(r @ r + REGULARIZE * m @ m)

    m = np.zeros(len(sf.names))
    r, cost = cost_parts(m)
    mu = 1e-3
    h = 0.05
    for _ in range(iterations):
        jac = np.stack([(cost_parts(np.clip(m + h * e, sf.lo, sf.hi))[0] - r) / h for e in np.eye(len(m))], axis=1)
        a = jac.T @ jac + (REGULARIZE + mu) * np.eye(len(m))
        g = jac.T @ r + REGULARIZE * m
        step = np.linalg.solve(a, -g)
        cand = np.clip(m + step, sf.lo, sf.hi)
        r2, cost2 = cost_parts(cand)
        if cost2 < cost:
            m, r, cost = cand, r2, cost2
            mu *= 0.3
        else:
            mu *= 4
    return sf.targets(m), dict(zip(sf.names, np.round(m, 2))), cost


def fit_to_photos(base_dir, fixed_targets: dict, front: View, sides: list[View], front_lm: dict):
    """반환: (최종 Body, 사진 크기로 놓은 꼭짓점, 고른 조절값)."""
    body = Body(base_dir, fixed_targets)
    rots = pose_from_photos(body, front)
    fitted, values, _ = fit_shape(body, rots, base_dir, front, sides, front_lm)
    # 고른 조절값을 실제 target으로 넣어 다시 만든다(관절 위치도 새 체형에 맞게)
    final = Body(base_dir, {**fixed_targets, **fitted})
    rots = pose_from_photos(final, front)
    v, _ = final.pose(rots)
    v = place_and_scale(final, v, front_lm["chin"][1])
    in_use = np.zeros(len(v), bool)
    in_use[final.used_vertices()] = True
    v = align_depth(v, PhotoTargets(front, sides, front_lm), np.argmax(final.part_weights, axis=1), in_use)
    return final, v, values


def thin_plate(src: np.ndarray, dst: np.ndarray, smooth: float = 1e-6):
    """2D thin-plate spline: 기준점은 정확히 맞추고 사이는 가장 덜 휘게 잇는다. 반환: 점 → 점 함수."""
    def kernel(r):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r > 0, r * r * np.log(r), 0.0)

    n = len(src)
    k = kernel(np.linalg.norm(src[:, None] - src[None], axis=2)) + smooth * np.eye(n)
    p = np.c_[np.ones(n), src]
    a = np.block([[k, p], [p.T, np.zeros((3, 3))]])
    coef = np.linalg.solve(a, np.r_[dst, np.zeros((3, 2))])
    w, aff = coef[:n], coef[n:]

    def apply(q):
        u = kernel(np.linalg.norm(q[:, None] - src[None], axis=2))
        return u @ w + np.c_[np.ones(len(q)), q] @ aff

    return apply


class HeadWarp:
    """얼굴 텍스처 정렬: 모델 얼굴 기준점 → 사진 기준점(thin-plate spline, 머리에만, 목에서 0으로).

    머리 모양은 해부학 비율대로 두고 사진 쪽을 늘이고 줄여 입힌다. 기준점에서 멀어지면
    (정수리·뒤통수) 변환이 크게 휘지 않도록 기준점들의 평균 아핀 변환으로 이어진다."""

    def __init__(self, body: Body, v: np.ndarray, front_lm: dict, side_lms: list[dict]):
        m = head_landmarks(body, v)
        keys = ("eye_l", "eye_r", "nose", "mouth", "chin")
        src = np.array([[m[k][0], m[k][1]] for k in keys])
        dst = np.array([front_lm[k] for k in keys], float)
        self.front = thin_plate(src, dst)
        skeys = ("eye", "nose", "chin")
        side = {k: np.mean([s[k] for s in side_lms], axis=0) for k in skeys}
        m_eye = (m["eye_l"] + m["eye_r"]) / 2
        model_side = {"eye": m_eye, "nose": m["nose"], "chin": m["chin"]}
        src = np.array([[model_side[k][2], model_side[k][1]] for k in skeys])
        dst = np.array([side[k] for k in skeys])
        self.side = np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)[0]
        self.y0, self.y1 = m["chin"][1] - 0.05, m["chin"][1] - 0.01  # 이 높이 사이에서 0 → 1

    def weight(self, y):
        t = np.clip((y - self.y0) / (self.y1 - self.y0), 0, 1)
        return t * t * (3 - 2 * t)

    def front_uv(self, p: np.ndarray):
        """정면·후면 사진에서 찾을 (u, y)."""
        w = self.weight(p[:, 1])[:, None]
        q = self.front(p[:, [0, 1]])
        uv = p[:, [0, 1]] + w * (q - p[:, [0, 1]])
        return uv[:, 0], uv[:, 1]

    def side_uv(self, p: np.ndarray):
        """측면 사진에서 찾을 (u=z, y). p는 오른쪽 옆면 기준(왼쪽은 미리 좌우 대칭)."""
        w = self.weight(p[:, 1])[:, None]
        q = np.c_[p[:, 2], p[:, 1], np.ones(len(p))] @ self.side
        uv = p[:, [2, 1]] + w * (q - p[:, [2, 1]])
        return uv[:, 0], uv[:, 1]
