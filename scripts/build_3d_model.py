"""정면·후면·측면(2장) 사진으로 3D 모델(GLB)을 만든다.

방식: 실루엣 기반 복원.
  1. 단색 배경을 빼서 사진마다 인물 마스크를 만든다.
  2. 키(머리 끝~발끝)를 기준으로 세 사진의 축척을 맞춘다.
  3. 높이마다 정면 실루엣의 폭과 측면 실루엣의 깊이로 타원 단면을 쌓아 복셀 볼륨을 만든다.
     (팔·머리 묶음처럼 몸통과 떨어진 덩어리는 원형 단면으로 처리)
  4. 볼륨을 부드럽게 한 뒤 마칭 큐브로 표면을 뽑는다.
  5. 정면·후면·측면 사진을 표면 방향에 따라 섞어 꼭짓점 색을 입힌다.

출력:
  model/gyeonjahee.glb      꼭짓점 색이 들어간 메시
  model/textures/*.webp     뷰어에서 고해상도 투영에 쓰는 RGBA 사진(알파 = 마스크)
  model/projection.json     사진 ↔ 모델 좌표 정렬값
  (GLB 꼭짓점 색의 알파 = 측면 사진을 써도 되는 정도, 팔은 0)

필요 패키지: numpy, pillow, scipy, scikit-image
"""

from __future__ import annotations

import json
import struct
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "model" / "src"
OUT = ROOT / "model"

HEIGHT_M = 1.65  # 모델 키(미터)
VOXEL = 0.005  # 복셀 크기(미터)

# 정면 사진 기준 해부학적 높이(픽셀 행). 이 사진 세트에 맞춘 값이다.
FRONT_CHIN_ROW = 205  # 턱선: 이 아래~어깨선은 머리카락을 몸통 폭에서 뺀다
FRONT_SHOULDER_ROW = 300  # 어깨선: 이 아래부터 팔이 몸통 실루엣에 붙어 있다
FRONT_ARMPIT_ROW = 470  # 겨드랑이: 이 아래부터 팔이 따로 떨어진다
FRONT_CROTCH_ROW = 785  # 가랑이: 이 아래 덩어리는 다리
FRONT_BUST_TOP_ROW = 330
FRONT_BUST_BOTTOM_ROW = 525
BUST_X = 0.07  # 가슴 중심의 좌우 위치(미터)
BUST_R = 0.075  # 가슴 덩어리 반경(미터)
HEAD_CHIN_DROP = 0.02  # 정면 턱선 아래로 머리 타원체를 얼마나 내릴지(미터)
NECK_MIN_HALF = 0.056  # 목 반지름 최소값(미터)
SHOULDER_SLOPE = 0.06  # 어깨선 위 승모근 경사 높이(미터)


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
    """사진 한 장과 모델 좌표 사이의 변환."""

    def __init__(self, rgb: np.ndarray, mask: np.ndarray, center_rows: tuple[int, int]):
        self.rgb, self.mask = rgb, mask
        self.inner = ndi.binary_erosion(mask, iterations=3) & ~ndi.binary_dilation(background_tinted(rgb), iterations=2)
        # 실루엣 밖은 가장 가까운 안쪽 픽셀 색으로 채운다: 표면이 사진 경계를 살짝 넘어도 배경색이 묻지 않게
        _, (ri, ci) = ndi.distance_transform_edt(~self.inner, return_indices=True)
        self.filled = rgb[ri, ci]
        self.hair = mask & (rgb.astype(float).mean(2) < 70)
        self.h, self.w = mask.shape
        rows = np.where(mask.any(1))[0]
        self.top, self.bottom = int(rows[0]), int(rows[-1])
        self.scale = HEIGHT_M / (self.bottom - self.top)  # 미터/픽셀
        r0, r1 = center_rows
        cols = np.nonzero(mask[r0:r1])[1]
        self.center = float(np.median(cols))

    def row_of(self, y: np.ndarray) -> np.ndarray:
        return self.bottom - y / self.scale

    def row_to_height(self, row: float) -> float:
        return (self.bottom - row) * self.scale

    def col_of(self, u: np.ndarray) -> np.ndarray:
        return self.center + u / self.scale

    def params(self) -> dict:
        return {
            "width": self.w,
            "height": self.h,
            "bottom": self.bottom,
            "scale": self.scale,
            "center": self.center,
        }


def side_profile(front: View, side: View, ys: np.ndarray):
    """높이별 측면 몸통의 앞·뒤 경계(z)와 가슴 돌출량."""
    zf = np.full(len(ys), np.nan)
    zb = np.full(len(ys), np.nan)
    for j, y in enumerate(ys):
        sr = int(round(side.row_of(y)))
        if not 0 <= sr < side.h:
            continue
        row = side.mask[sr]
        if in_neck_band(front, y):
            row = row & ~side.hair[sr]
        rs = runs(row)
        if not rs:
            continue
        # 가장 긴 덩어리가 몸통(뒤로 돌린 팔·머리 묶음 제외)
        s0, s1 = max(rs, key=lambda r: r[1] - r[0])
        zb[j] = (s0 - side.center) * side.scale
        zf[j] = (s1 - side.center) * side.scale
    # 측면 사진은 팔을 등 뒤로 돌리고 있어서 어깨~팔이 떨어지는 지점까지 등이 부풀어 보인다.
    # 그 구간의 등 경계는 어깨선과 팔이 떨어진 첫 높이 사이를 이어서 쓴다
    j_sh = int(np.searchsorted(ys, front.row_to_height(FRONT_SHOULDER_ROW)))
    j_sep = next(
        j for j in range(j_sh, 0, -1)
        if len(runs(side.mask[int(round(side.row_of(ys[j])))])) >= 2
    )
    line = np.interp(ys[j_sep : j_sh + 1], [ys[j_sep], ys[j_sh]], [zb[j_sep], zb[j_sh]])
    zb[j_sep : j_sh + 1] = np.maximum(zb[j_sep : j_sh + 1], line - 0.01)

    ok = ~np.isnan(zf)
    # 두 사진의 미세한 차이로 생기는 가로 주름을 없애려고 세로로 한 번 더 부드럽게
    zf[ok] = ndi.gaussian_filter1d(ndi.median_filter(zf[ok], 5), 3)
    zb[ok] = ndi.gaussian_filter1d(ndi.median_filter(zb[ok], 5), 3)

    # 가슴 구간: 위·아래 경계를 이은 선을 흉곽 앞면으로 보고, 튀어나온 만큼을 가슴으로 뺀다
    bust = np.zeros(len(ys))
    y_top = front.row_to_height(FRONT_BUST_TOP_ROW)
    y_bot = front.row_to_height(FRONT_BUST_BOTTOM_ROW)
    j_top, j_bot = np.searchsorted(ys, y_top), np.searchsorted(ys, y_bot)
    band = slice(j_bot, j_top + 1)
    base = np.interp(ys[band], [ys[j_bot], ys[j_top]], [zf[j_bot], zf[j_top]])
    bust[band] = np.clip(zf[band] - base, 0, None)
    zf[band] = np.minimum(zf[band], base)
    # 가슴 아래쪽이 선반처럼 끊기지 않게 세로로 부드럽게
    bust = ndi.gaussian_filter1d(bust, 3)
    return zf, zb, bust


def in_neck_band(front: View, y: float) -> bool:
    return front.row_to_height(FRONT_SHOULDER_ROW) < y < front.row_to_height(FRONT_CHIN_ROW)


def build_volume(front: View, sides: list[View]):
    xs = np.arange(-0.48, 0.48, VOXEL)
    ys = np.arange(-0.01, HEIGHT_M + 0.02, VOXEL)
    zs = np.arange(-0.22, 0.22, VOXEL)
    vol = np.zeros((len(xs), len(ys), len(zs)), np.float32)
    arm_map = np.zeros((len(xs), len(ys)), np.float32)  # 팔인 곳(측면 사진을 쓰지 않을 곳)
    # 측면 사진이 여러 장이면 깊이 윤곽을 평균 낸다
    profiles = np.array([side_profile(front, side, ys) for side in sides])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 사람이 없는 높이는 전부 NaN
        zf_all, zb_all, bust_all = np.nanmean(profiles, axis=0)

    # 겨드랑이 바로 아래 몸통 폭과 팔 굵기: 팔이 몸통에 붙어 보이는 어깨 구간에 쓴다
    below = runs(front.mask[FRONT_ARMPIT_ROW + 10])
    core = next(r for r in below if r[0] <= front.center < r[1])
    core_half = (core[1] - core[0]) / 2 * front.scale * 1.05
    arm_r = np.median([r[1] - r[0] for r in below if r != core]) / 2 * front.scale

    def body_run(fr, row):
        rs = [r for r in runs(row) if r[0] <= front.center < r[1]]
        return ((rs[0][0] - front.center) * front.scale, (rs[0][1] - front.center) * front.scale) if rs else None

    # 머리: 사진마다 머리카락 윤곽이 달라서 실루엣 그대로 쓰면 버섯 모양이 된다.
    # 머리 위쪽 70%에서 잰 폭·깊이로 타원체를 잡고, 턱 쪽은 타원체를 따라 좁힌다
    y_top = max(y for y in ys if front.row_of(y) >= front.top)
    y_chin = front.row_to_height(FRONT_CHIN_ROW) - HEAD_CHIN_DROP
    head_cy, head_ry = (y_top + y_chin) / 2, (y_top - y_chin) / 2
    upper = [j for j, y in enumerate(ys) if y_chin + 0.6 * (y_top - y_chin) < y <= y_top and not np.isnan(zf_all[j])]
    head_rx = max((lambda r: (r[1] - r[0]) / 2 if r else 0)(body_run(0, front.mask[int(round(front.row_of(ys[j])))])) for j in upper)
    jz = max(upper, key=lambda j: zf_all[j] - zb_all[j])
    head_rz = (zf_all[jz] - zb_all[jz]) / 2
    head_cz = (zf_all[jz] + zb_all[jz]) / 2

    # 목·어깨: 머리카락에 가려진 부분은 해부학적 최소값과 어깨 경사로 보정한다
    y_sh = front.row_to_height(FRONT_SHOULDER_ROW)
    sh_run = body_run(0, front.mask[FRONT_SHOULDER_ROW + 5])
    sh_half = (sh_run[1] - sh_run[0]) / 2
    j_sh = int(np.searchsorted(ys, y_sh))
    sh_zc, sh_zd = (zf_all[j_sh] + zb_all[j_sh]) / 2, (zf_all[j_sh] - zb_all[j_sh]) / 2

    def fill(ix, j, lo, hi):
        inside = (zs[None, :] >= lo[:, None]) & (zs[None, :] <= hi[:, None])
        vol[ix, j, :] = np.maximum(vol[ix, j, :], inside)

    for j, y in enumerate(ys):
        fr = int(round(front.row_of(y)))
        if not 0 <= fr < front.h or np.isnan(zf_all[j]):
            continue
        zb, zf, bust = zb_all[j], zf_all[j], bust_all[j]
        zc, zd = (zb + zf) / 2, (zf - zb) / 2
        head = y > y_chin
        neck = in_neck_band(front, y) and not head
        if head:
            # 아래쪽(턱)은 덜 뾰족하게: 지수가 클수록 각진 턱
            p = 2.0 if y > head_cy else 2.6
            k = max(0.0, 1 - abs((y - head_cy) / head_ry) ** p) ** (1 / p)
            zc, zd = head_cz, head_rz * k
        elif neck:
            # 목 최소 굵기, 어깨선 위 SHOULDER_SLOPE 구간은 승모근 경사로 어깨 깊이까지 키운다
            s = np.clip(1 - (y - y_sh) / SHOULDER_SLOPE, 0, 1) ** 2
            zd = max(zd, NECK_MIN_HALF) + (sh_zd - max(zd, NECK_MIN_HALF)) * s
            zc = zc + (sh_zc - zc) * s

        row = front.mask[fr]
        pieces = []
        if in_neck_band(front, y):
            # 목 주변: 늘어진 머리카락·머리 묶음은 사진마다 모양이 달라 덩어리로 튀어나오므로 뺀다
            row = row & ~front.hair[fr]
        for a, b in runs(row):
            is_body = a <= front.center < b or fr >= FRONT_CROTCH_ROW
            pieces.append((a, b, "body" if is_body else "limb"))

        for a, b, kind in pieces:
            xa = (a - front.center) * front.scale
            xb = (b - front.center) * front.scale
            xc, xw = (xa + xb) / 2, (xb - xa) / 2
            if xw < 2 * VOXEL:
                continue
            if kind == "body" and head:
                hw = min(xw, head_rx * k)
                xa, xb = xc - hw, xc + hw
                xw = hw
            elif kind == "body" and neck:
                hw = max(xw, NECK_MIN_HALF)
                hw = hw + (max(hw, sh_half) - hw) * s
                xa, xb = xc - hw, xc + hw
                xw = hw
            elif head or neck:
                continue  # 머리·목 옆으로 삐져나온 머리카락은 버린다
            ix = (xs >= xa) & (xs <= xb)
            u = xs[ix]
            if kind == "limb":
                # 팔: 원형 단면
                half = np.sqrt(np.clip(xw**2 - (u - xc) ** 2, 0, None))
                fill(ix, j, zc - half, zc + half)
                arm_map[ix, j] = 1
                continue

            attached_arms = FRONT_SHOULDER_ROW < fr < FRONT_ARMPIT_ROW and xw > core_half
            w = core_half if attached_arms else xw
            t = np.clip((u - xc) / w, -1, 1)
            half = zd * np.sqrt(1 - t**2)
            # 가슴: 좌우 두 덩어리로 앞면에만 더한다
            lobe = np.clip(1 - ((np.abs(u - xc) - BUST_X) / BUST_R) ** 2, 0, None)
            front_edge = zc + half + bust * lobe ** 0.7
            fill(ix, j, zc - half, front_edge)
            if attached_arms:
                # 어깨~겨드랑이: 몸통 바깥쪽은 팔 원통으로 잇는다
                d = np.abs(u - xc)
                arm_c = xw - arm_r
                # 몸통과 팔 사이 겨드랑이 살은 아래로 갈수록 얇아진다
                taper = np.clip((FRONT_ARMPIT_ROW - fr) / (FRONT_ARMPIT_ROW - FRONT_SHOULDER_ROW) * 1.6, 0, 1)
                bridge = arm_r * taper * np.clip((d - core_half * 0.7) / (arm_c - core_half * 0.7 + 1e-6), 0, 1) ** 0.5
                tube = np.sqrt(np.clip(arm_r**2 - (d - arm_c) ** 2, 0, None))
                arm = np.where(d < arm_c, np.maximum(bridge, tube), tube)
                fill(ix, j, zc - arm, zc + arm)
                arm_map[ix, j] = np.abs(u - xc) > core_half

    # 세로 방향을 더 흐려서 높이별 단면 사이의 계단을 없앤다
    vol = ndi.gaussian_filter(vol, (1.3, 2.6, 1.3))
    arm_map = ndi.gaussian_filter(ndi.binary_dilation(arm_map > 0, iterations=3).astype(np.float32), 2)
    return vol, arm_map, (xs[0], ys[0], zs[0])


def sample(view: View, cols: np.ndarray, rows: np.ndarray):
    c = np.clip(np.round(cols).astype(int), 0, view.w - 1)
    r = np.clip(np.round(rows).astype(int), 0, view.h - 1)
    ok = view.inner[r, c] & (cols >= 0) & (cols < view.w) & (rows >= 0) & (rows < view.h)
    return view.filled[r, c].astype(float), ok.astype(float)


def colorize(verts, normals, side_ok, front: View, back: View, side: View, side2: View):
    x, y, z = verts.T
    nx, _, nz = normals.T
    cf, of = sample(front, front.col_of(x), front.row_of(y))
    cb, ob = sample(back, back.col_of(x), back.row_of(y))  # 후면은 미리 좌우를 뒤집어 둠
    # 왼쪽 옆면(+x)은 첫 측면 사진, 오른쪽 옆면(-x)은 두 번째 측면 사진(좌우 대칭 가정)
    cs1, os1 = sample(side, side.col_of(z), side.row_of(y))
    cs2, os2 = sample(side2, side2.col_of(z), side2.row_of(y))
    right = (nx < 0)[:, None]
    cs = np.where(right, cs2, cs1)
    os_ = np.where(right[:, 0], os2, os1)
    # 옆을 향한 면(nz≈0)에서는 정면·후면 사진을 반반 섞어 경계가 계단처럼 보이지 않게 한다
    wf = np.clip(nz + 0.25, 0, None) ** 2 * of + 1e-4
    wb = np.clip(-nz + 0.25, 0, None) ** 2 * ob
    # 측면 사진은 바깥쪽 면만 보여준다: 허벅지 안쪽처럼 몸 중심을 향한 면에는 쓰지 않는다
    outward = np.where(np.abs(x) > 0.03, np.clip(np.sign(x) * nx * 5, 0, 1), 1)
    ws = nx**2 * os_ * side_ok * outward
    total = wf + wb + ws
    col = (cf * wf[:, None] + cb * wb[:, None] + cs * ws[:, None]) / total[:, None]
    # 어느 사진에도 안 잡히는 면은 정면/후면 중 가까운 쪽 색(경계 밖은 채워 둔 색)
    fallback = np.where((nz >= 0)[:, None], cf, cb)
    col = np.where((total < 0.01)[:, None], fallback, col)
    return np.clip(col, 0, 255).astype(np.uint8)


def write_glb(path: Path, verts, normals, colors, faces):
    verts = verts.astype(np.float32)
    normals = normals.astype(np.float32)
    rgba = colors
    idx = faces.astype(np.uint32).ravel()
    blobs = [verts.tobytes(), normals.tobytes(), rgba.tobytes(), idx.tobytes()]
    offsets, buf = [], b""
    for blob in blobs:
        offsets.append(len(buf))
        buf += blob + b"\0" * (-len(blob) % 4)
    n = len(verts)
    gltf = {
        "asset": {"version": "2.0", "generator": "gyeonjahee-fansite build_3d_model.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "gyeonjahee"}],
        "meshes": [
            {
                "name": "gyeonjahee",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
                        "indices": 3,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "skin",
                "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.7},
            }
        ],
        "buffers": [{"byteLength": len(buf)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": len(blobs[0]), "target": 34962},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": len(blobs[1]), "target": 34962},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": len(blobs[2]), "target": 34962},
            {"buffer": 0, "byteOffset": offsets[3], "byteLength": len(blobs[3]), "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": n,
                "type": "VEC3",
                "min": verts.min(0).tolist(),
                "max": verts.max(0).tolist(),
            },
            {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5121, "normalized": True, "count": n, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5125, "count": len(idx), "type": "SCALAR"},
        ],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(buf), b"BIN\0") + buf
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


def save_texture(view: View, path: Path, mirror: bool = False):
    # 가장자리의 배경색 번짐을 피하려고 마스크를 살짝 줄여서 알파로 쓴다
    alpha = ndi.binary_erosion(view.mask, iterations=3)
    rgba = np.dstack([view.filled, (alpha * 255).astype(np.uint8)])
    img = Image.fromarray(rgba, "RGBA")
    if mirror:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    img.save(path, "WEBP", quality=88)


def register_to(front: View, other: View) -> View:
    """다른 사진을 행마다 좌우로 늘이고 줄여서 실루엣을 정면 사진에 맞춘다.

    사진끼리 팔다리 위치가 조금씩 달라서, 그대로 투영하면 허벅지 안쪽 같은 곳에
    엉뚱한 픽셀이 묻는다. 같은 높이에서 덩어리(팔·몸통·다리) 개수가 같으면
    덩어리 경계끼리 이어 붙이고, 개수가 다르면 축척만 맞춘다.
    """
    rows = np.arange(front.h)
    src_rows = np.clip(np.round(other.row_of(front.row_to_height(rows))).astype(int), 0, other.h - 1)
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
    out = View(rgb, mask, (FRONT_ARMPIT_ROW, FRONT_CROTCH_ROW))
    out.top, out.bottom, out.scale, out.center = front.top, front.bottom, front.scale, front.center
    return out


def side_view(path: Path) -> View:
    rgb, mask = segment(path)
    side = View(rgb, mask, (FRONT_ARMPIT_ROW, FRONT_CROTCH_ROW))
    # 측면 깊이 중심은 몸통 덩어리 기준으로 다시 잡는다
    body_cols = []
    for r in range(side.top + (side.bottom - side.top) // 3, side.top + (side.bottom - side.top) // 2):
        rs = runs(side.mask[r])
        if rs:
            s0, s1 = max(rs, key=lambda q: q[1] - q[0])
            body_cols.append((s0 + s1) / 2)
    side.center = float(np.median(body_cols))
    return side


def main():
    f_rgb, f_mask = segment(SRC / "front.png")
    b_rgb, b_mask = segment(SRC / "back.png")
    # 후면은 좌우를 뒤집어서 정면과 같은 x축을 쓰게 한다
    b_rgb, b_mask = b_rgb[:, ::-1].copy(), b_mask[:, ::-1].copy()

    front = View(f_rgb, f_mask, (FRONT_ARMPIT_ROW, FRONT_CROTCH_ROW))
    back = register_to(front, View(b_rgb, b_mask, (FRONT_ARMPIT_ROW, FRONT_CROTCH_ROW)))
    side = side_view(SRC / "side.png")
    side2 = side_view(SRC / "side2.png")

    vol, arm_map, origin = build_volume(front, [side, side2])
    verts, faces, normals, _ = marching_cubes(vol, 0.5, spacing=(VOXEL,) * 3)
    verts += np.array(origin)
    # 표면 법선이 바깥을 향하게 맞춘다
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    if np.mean(np.sum(fn * normals[faces[:, 0]], 1)) < 0:
        faces = faces[:, ::-1]
    if np.mean(np.sum(normals * (verts - verts.mean(0)) * [1, 0, 1], 1)) < 0:
        normals = -normals
        faces = faces[:, ::-1]

    # 측면 사진은 팔 자세가 달라서 팔에는 쓰지 않는다. 이 값은 꼭짓점 색의 알파로도 넘긴다
    ax = np.clip(np.round((verts[:, 0] - origin[0]) / VOXEL).astype(int), 0, arm_map.shape[0] - 1)
    ay = np.clip(np.round((verts[:, 1] - origin[1]) / VOXEL).astype(int), 0, arm_map.shape[1] - 1)
    side_ok = 1 - np.clip(arm_map[ax, ay] * 1.5, 0, 1)
    rgb = colorize(verts, normals, side_ok, front, back, side, side2)
    colors = np.concatenate([rgb, np.round(side_ok * 255).astype(np.uint8)[:, None]], 1)
    # 바닥이 y=0, 키 방향이 +y, 정면이 +z
    write_glb(OUT / "gyeonjahee.glb", verts, normals, colors, faces)

    tex = OUT / "textures"
    tex.mkdir(exist_ok=True)
    save_texture(front, tex / "front.webp")
    save_texture(back, tex / "back.webp", mirror=True)  # 원본 방향으로 저장
    save_texture(side, tex / "side.webp")
    save_texture(side2, tex / "side2.webp")
    (OUT / "projection.json").write_text(
        json.dumps(
            {
                "front": front.params(),
                "back": {**back.params(), "mirrored": True},
                "side": side.params(),
                "side2": side2.params(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"vertices={len(verts)} faces={len(faces)} glb={(OUT / 'gyeonjahee.glb').stat().st_size / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
