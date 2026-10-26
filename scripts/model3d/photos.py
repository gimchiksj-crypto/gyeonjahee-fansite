"""사진 처리: 인물 마스크, 사진 ↔ 모델 좌표 변환, 사진 간 정렬."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

EDGE_PX = 14  # 이 픽셀 거리 안쪽의 윤곽 근처 색은 덜 믿는다
HEIGHT_M = 1.65  # 사진 속 인물 키(머리카락 끝~발끝, 미터). 모든 좌표의 기준 축척

# 정면 사진 기준 해부학적 높이(픽셀 행). 이 사진 세트에 맞춘 값이다.
FRONT_SHOULDER_ROW = 300  # 어깨선: 이 아래부터 팔이 몸통 실루엣에 붙어 있다
FRONT_ARMPIT_ROW = 470  # 겨드랑이: 이 아래부터 팔이 따로 떨어진다
FRONT_CROTCH_ROW = 785  # 가랑이: 이 아래 덩어리는 다리
FRONT_UNDERBUST_ROW = 522  # 브라 아랫선(가슴 아래 끝)

# 얼굴 기준점(픽셀). 정면은 (열, 행), 측면은 얼굴이 오른쪽을 보는 사진의 (열, 행)
FRONT_LANDMARKS = {
    "eye_r": (442, 147),
    "eye_l": (490, 147),
    "nose": (470, 182),
    "mouth": (467, 207),
    "chin": (467, 245),
}
SIDE_LANDMARKS = {
    "side.png": {"eye": (545, 135), "nose": (572, 165), "chin": (552, 225), "ear": (467, 170)},
    "side2.png": {"eye": (525, 135), "nose": (556, 165), "chin": (532, 225), "ear": (450, 170)},
}


def background_tinted(rgb: np.ndarray) -> np.ndarray:
    """배경색이 섞인 픽셀(머리카락 가닥 사이, 윤곽 테두리). 빨간 반바지는 제외."""
    a = rgb.astype(float)
    bg = np.median(np.concatenate([a[:, :25], a[:, -25:]], 1), axis=1)
    dist = np.linalg.norm(a - bg[:, None, :], axis=2)
    shorts = a[..., 1] - a[..., 2] < (bg[:, 1] - bg[:, 2])[:, None] - 14
    return (dist < 90) & ~shorts


def segment(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """배경색을 행마다 추정해서 인물 마스크를 만든다."""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    a = rgb.astype(float)
    bg = np.median(np.concatenate([a[:, :25], a[:, -25:]], 1), axis=1)
    dist = np.linalg.norm(a - bg[:, None, :], axis=2)
    # 빨간 반바지는 배경과 색이 가까워서 G-B 차이로 따로 잡는다
    gb = a[..., 1] - a[..., 2]
    bg_gb = (bg[:, 1] - bg[:, 2])[:, None]
    m = (dist > 45) | (gb < bg_gb - 14)
    m = ndi.binary_opening(m, iterations=2)
    lab, n = ndi.label(m)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = lab == (np.argmax(sizes) + 1)
    m = ndi.binary_closing(m, iterations=3)
    m = ndi.binary_fill_holes(m)
    return rgb, m


def runs(row: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, row.astype(np.int8), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


class View:
    """사진 한 장과 모델 좌표 사이의 변환. 가로축은 정면 사진에선 x, 측면 사진에선 z."""

    def __init__(self, rgb: np.ndarray, mask: np.ndarray):
        self.rgb, self.mask = rgb, mask
        # 텍스처로 믿고 쓸 픽셀: 윤곽 테두리와 배경색이 섞인 픽셀은 뺀다
        self.inner = ndi.binary_erosion(mask, iterations=3) & ~ndi.binary_dilation(background_tinted(rgb), iterations=2)
        # 그 밖은 가장 가까운 안쪽 픽셀 색으로 채운다: 표면이 사진 경계를 살짝 넘어도 배경색이 묻지 않게
        _, (ri, ci) = ndi.distance_transform_edt(~self.inner, return_indices=True)
        self.filled = rgb[ri, ci]
        # 윤곽 근처는 주황 배경이 반사돼(림 라이트) 피부가 붉게 찍힌다: 안쪽일수록 믿는다
        self.edge_w = np.clip(ndi.distance_transform_edt(mask) / EDGE_PX, 0, 1) ** 2
        lum = rgb.astype(float).mean(2)
        self.hair = mask & (lum < 70)
        # 피부에 그리면 안 되는 픽셀: 머리카락 + 가닥 사이로 배경이 비쳐 붉게 뜬 머리카락
        self.hairish = ndi.binary_dilation(mask & (lum < 105), iterations=2)
        self.h, self.w = mask.shape
        rows = np.where(mask.any(1))[0]
        self.top, self.bottom = int(rows[0]), int(rows[-1])
        self.scale = HEIGHT_M / (self.bottom - self.top)  # 미터/픽셀
        cols = np.nonzero(mask[FRONT_ARMPIT_ROW:FRONT_CROTCH_ROW])[1]
        self.center = float(np.median(cols))

    def row_of(self, y):
        return self.bottom - np.asarray(y) / self.scale

    def height_of(self, row):
        return (self.bottom - np.asarray(row, float)) * self.scale

    def col_of(self, u):
        return self.center + np.asarray(u) / self.scale

    def u_of(self, col):
        return (np.asarray(col, float) - self.center) * self.scale

    def sample(self, u, y, skin=None):
        """모델 좌표 → (색, 믿을 만한 픽셀인지). skin=True인 점에는 머리카락 픽셀을 쓰지 않는다
        (목·어깨 앞으로 늘어진 머리카락이 피부에 그려지지 않게)."""
        cols, rows = self.col_of(u), self.row_of(y)
        c = np.clip(np.round(cols).astype(int), 0, self.w - 1)
        r = np.clip(np.round(rows).astype(int), 0, self.h - 1)
        ok = self.inner[r, c] & (cols >= 0) & (cols < self.w) & (rows >= 0) & (rows < self.h)
        if skin is not None:
            ok &= ~(skin & self.hairish[r, c])
        return self.filled[r, c].astype(float), ok * np.maximum(self.edge_w[r, c], 0.02)

    def run_at(self, y, u_hint=0.0, mask=None):
        """높이 y에서 u_hint를 포함하는(없으면 가장 가까운) 덩어리의 (u0, u1)."""
        r = int(round(float(self.row_of(y))))
        if not 0 <= r < self.h:
            return None
        rs = runs((self.mask if mask is None else mask)[r])
        if not rs:
            return None
        c = float(self.col_of(u_hint))
        a, b = min(rs, key=lambda q: 0 if q[0] <= c < q[1] else min(abs(q[0] - c), abs(q[1] - c)))
        return float(self.u_of(a)), float(self.u_of(b))


def front_view(path: Path) -> View:
    return View(*segment(path))


def back_view(path: Path, front: View) -> View:
    """후면 사진: 좌우를 뒤집어 정면과 같은 x축으로 만들고 실루엣을 정면에 맞춘다."""
    rgb, mask = segment(path)
    return register_to(front, View(rgb[:, ::-1].copy(), mask[:, ::-1].copy()))


def side_view(path: Path) -> View:
    side = View(*segment(path))
    # 측면 깊이 중심은 몸통 덩어리 기준으로 다시 잡는다
    body_cols = []
    for r in range(side.top + (side.bottom - side.top) // 3, side.top + (side.bottom - side.top) // 2):
        rs = runs(side.mask[r])
        if rs:
            s0, s1 = max(rs, key=lambda q: q[1] - q[0])
            body_cols.append((s0 + s1) / 2)
    side.center = float(np.median(body_cols))
    side.landmarks = {k: (float(side.u_of(c)), float(side.height_of(r))) for k, (c, r) in SIDE_LANDMARKS[path.name].items()}
    return side


def register_to(front: View, other: View) -> View:
    """다른 사진을 행마다 좌우로 늘이고 줄여서 실루엣을 정면 사진에 맞춘다.

    사진끼리 팔다리 위치가 조금씩 달라서, 그대로 투영하면 허벅지 안쪽 같은 곳에
    엉뚱한 픽셀이 묻는다. 같은 높이에서 덩어리(팔·몸통·다리) 개수가 같으면
    덩어리 경계끼리 이어 붙이고, 개수가 다르면 축척만 맞춘다.
    """
    rows = np.arange(front.h)
    src_rows = np.clip(np.round(other.row_of(front.height_of(rows))).astype(int), 0, other.h - 1)
    cols = np.arange(front.w, dtype=float)
    base = other.center + (cols - front.center) * front.scale / other.scale
    offset = np.zeros((front.h, front.w))
    for r in rows:
        fr = runs(front.mask[r])
        orr = runs(other.mask[src_rows[r]])
        if not fr or len(fr) != len(orr):
            continue
        knots_f = np.array([e for run in fr for e in run], float)
        knots_o = np.array([e for run in orr for e in run], float)
        offset[r] = np.interp(cols, knots_f, knots_o - np.interp(knots_f, cols, base))
    offset = ndi.gaussian_filter1d(offset, 4, axis=0)  # 행마다 튀지 않게
    src_cols = np.clip(np.round(base[None, :] + offset).astype(int), 0, other.w - 1)
    rgb = other.rgb[src_rows[:, None], src_cols]
    mask = front.mask & other.mask[src_rows[:, None], src_cols]
    out = View(rgb, mask)
    out.top, out.bottom, out.scale, out.center = front.top, front.bottom, front.scale, front.center
    return out


def trunk_depth(front: View, side: View, ys: np.ndarray):
    """높이별 측면 몸통의 앞·뒤 경계(z). 사람이 없는 높이는 NaN."""
    zf = np.full(len(ys), np.nan)
    zb = np.full(len(ys), np.nan)
    for j, y in enumerate(ys):
        sr = int(round(float(side.row_of(y))))
        if not 0 <= sr < side.h:
            continue
        rs = runs(side.mask[sr] & ~side.hair[sr])
        if not rs:
            continue
        # 가장 긴 덩어리가 몸통(뒤로 돌린 팔·머리 묶음 제외)
        s0, s1 = max(rs, key=lambda r: r[1] - r[0])
        zb[j] = side.u_of(s0)
        zf[j] = side.u_of(s1)
    # 측면 사진은 팔을 등 뒤로 돌리고 있어서 어깨~팔이 떨어지는 지점까지 등이 부풀어 보인다.
    # 그 구간의 등 경계는 어깨선과 팔이 떨어진 첫 높이 사이를 이어서 쓴다
    j_sh = int(np.searchsorted(ys, front.height_of(FRONT_SHOULDER_ROW)))
    j_sep = next(j for j in range(j_sh, 0, -1) if len(runs(side.mask[int(round(float(side.row_of(ys[j]))))])) >= 2)
    line = np.interp(ys[j_sep : j_sh + 1], [ys[j_sep], ys[j_sh]], [zb[j_sep], zb[j_sh]])
    zb[j_sep : j_sh + 1] = np.maximum(zb[j_sep : j_sh + 1], line - 0.01)
    ok = ~np.isnan(zf)
    zf[ok] = ndi.gaussian_filter1d(ndi.median_filter(zf[ok], 5), 2)
    zb[ok] = ndi.gaussian_filter1d(ndi.median_filter(zb[ok], 5), 2)
    return zf, zb


def front_landmarks(front: View) -> dict:
    return {k: (float(front.u_of(c)), float(front.height_of(r))) for k, (c, r) in FRONT_LANDMARKS.items()}
