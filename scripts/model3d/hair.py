"""머리카락: 두피에서 사진 속 머리카락 윤곽까지 부풀린 껍데기 + 묶은 머리(번)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse import csgraph

from .bake import Projector, vertex_normals
from .photos import View

MAX_THICKNESS = 0.07  # 두피에서 머리카락 바깥면까지 최대 거리(미터)
STEP = 0.0015
BANGS_THICKNESS = 0.008


def adjacency(n: int, tris: np.ndarray) -> sparse.csr_matrix:
    i = np.r_[tris[:, 0], tris[:, 1], tris[:, 2], tris[:, 1], tris[:, 2], tris[:, 0]]
    j = np.r_[tris[:, 1], tris[:, 2], tris[:, 0], tris[:, 0], tris[:, 1], tris[:, 2]]
    a = sparse.csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    a.data[:] = 1
    return a


def smooth_values(a: sparse.csr_matrix, x: np.ndarray, iterations: int) -> np.ndarray:
    deg = np.maximum(np.asarray(a.sum(1)).ravel(), 1)
    for _ in range(iterations):
        x = 0.5 * x + 0.5 * (a @ x) / (deg if x.ndim == 1 else deg[:, None])
    return x


def taubin(v: np.ndarray, tris: np.ndarray, weight: np.ndarray, iterations: int = 20) -> np.ndarray:
    """부피가 줄지 않는 매끈하게 하기(Taubin λ/μ). weight=0인 곳은 그대로."""
    a = adjacency(len(v), tris)
    deg = np.maximum(np.asarray(a.sum(1)).ravel(), 1)[:, None]
    w = weight[:, None]
    for _ in range(iterations):
        for f in (0.5, -0.53):
            v = v + f * w * ((a @ v) / deg - v)
    return v


def subdivide(v: np.ndarray, tris: np.ndarray):
    """삼각형마다 변의 가운데에 점을 넣어 넷으로 나눈다."""
    e = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    uniq, inv = np.unique(e, axis=0, return_inverse=True)
    mid = len(v) + inv.reshape(3, -1).T  # (삼각형, 변 01/12/20)
    v2 = np.concatenate([v, (v[uniq[:, 0]] + v[uniq[:, 1]]) / 2])
    a, b, c = tris.T
    ab, bc, ca = mid.T
    t2 = np.concatenate([np.c_[a, ab, ca], np.c_[ab, b, bc], np.c_[ca, bc, c], np.c_[ab, bc, ca]])
    return v2, t2


def clean_hair(view: View) -> np.ndarray:
    """머리카락 마스크: 하이라이트로 뚫린 구멍을 메운다."""
    hair = ndi.binary_closing(view.hair, iterations=4) & view.mask
    return ndi.binary_opening(hair, iterations=1)


def pix(view: View, u, y, img) -> np.ndarray:
    c = np.round(view.col_of(u)).astype(int)
    r = np.round(view.row_of(y)).astype(int)
    ok = (c >= 0) & (c < view.w) & (r >= 0) & (r < view.h)
    out = np.zeros(len(u), bool)
    out[ok] = img[r[ok], c[ok]]
    return out


def in_hull(p: np.ndarray, front: View, back: View, sides: list[View], warp) -> np.ndarray:
    """사진 윤곽(정면·후면·측면) 안쪽인지. 얼굴 정렬 변환을 거친 사진 좌표로 본다."""
    fu, fy = warp.front_uv(p)
    su, sy = warp.side_uv(p)
    inside = pix(front, fu, fy, front.mask) & pix(back, fu, fy, back.mask)
    side = np.zeros(len(p), bool)
    for s in sides:
        side |= pix(s, su, sy, s.mask)
    return inside & side


def hair_cap(v, tris, head_w, chin_y, brow_y, neck_y, front: View, back: View, sides: list[View], proj: Projector):
    """머리 표면 중 사진에서 머리카락으로 보이는 부분을 바깥쪽 윤곽까지 부풀린다."""
    hair_f, hair_b = clean_hair(front), clean_hair(back)
    hair_s = [clean_hair(s) for s in sides]
    # 머리 + 목 뒤·옆(단발 끝과 묶은 머리가 목까지 내려온다)
    head = ((head_w > 0.5) & (v[:, 1] > chin_y - 0.01)) | ((v[:, 1] > neck_y) & (v[:, 1] <= chin_y))
    t_head = tris[head[tris].all(1)]
    keep, inv = np.unique(t_head, return_inverse=True)
    p, t = subdivide(*subdivide(v[keep], inv.reshape(-1, 3)))  # 머리 메시는 성겨서 가장자리가 들쭉날쭉하다
    n = vertex_normals(p, t)
    a = adjacency(len(p), t)

    # 머리카락인 정도: 표면이 향한 쪽 사진에서 머리카락 픽셀인지, 보이는 곳인지로 가중 평균
    x, y, z = p.T
    w_f = np.clip(n[:, 2], 0, None) ** 2 * proj.d_front.visible(p)
    w_b = np.clip(-n[:, 2], 0, None) ** 2 * proj.d_back.visible(p)
    fu, fy = proj.warp.front_uv(p)
    su, sy = proj.warp.side_uv(p)
    score = w_f * pix(front, fu, fy, hair_f) + w_b * pix(back, fu, fy, hair_b)
    total = w_f + w_b
    for s, hs in zip(sides, hair_s):
        w_s = n[:, 0] ** 2 / len(sides)
        score += w_s * pix(s, su, sy, hs)
        total += w_s
    hairness = score / np.maximum(total, 1e-6)
    # 얼굴(눈썹 아래 정면)은 속눈썹·눈동자가 어두워도 머리카락이 아니다
    face = (y < brow_y) & (n[:, 2] > 0.3) & (np.abs(x) < 0.05)
    hairness[face] = 0
    hairness = smooth_values(a, hairness, 20)
    # 정수리·뒤통수는 사진 각도상 판정이 애매해도 머리카락이다
    hairness = np.maximum(hairness, np.clip((y - (y.max() - 0.06)) / 0.03, 0, 1))

    # 법선 방향으로 사진 윤곽(시각 껍질)을 벗어나기 직전까지 부풀린다
    thick = np.zeros(len(p))
    alive = np.ones(len(p), bool)
    for k in range(1, int(MAX_THICKNESS / STEP) + 1):
        q = p + n * (k * STEP)
        alive &= in_hull(q, front, back, sides, proj.warp)
        thick[alive] = k * STEP
    # 가장자리로 갈수록 얇게: 머리카락 끝이 피부에 붙어 들뜬 조각이 생기지 않게
    edge = np.clip((hairness - 0.4) / 0.45, 0, 1)
    thick = thick * edge * edge * (3 - 2 * edge)
    # 앞머리는 이마에 얹힌 얇은 층이다(정면 윤곽이 옆머리 때문에 넓어서 그대로 두면 앞으로 튀어나온다)
    bangs = (n[:, 2] > 0.4) & (y < brow_y + 0.05)
    thick[bangs] = np.minimum(thick[bangs], BANGS_THICKNESS)
    thick = smooth_values(a, thick, 25)
    cap = p + n * (thick[:, None] + 0.0015)
    face_keep = hairness[t].mean(1) > 0.4
    t_cap = t[face_keep]
    cap = taubin(cap, t_cap, np.ones(len(cap)), 10)
    # 큰 덩어리만 남긴다(떨어져 나온 작은 조각 제거)
    lab = csgraph.connected_components(adjacency(len(cap), t_cap), directed=False)[1]
    sizes = np.bincount(lab[t_cap[:, 0]], minlength=lab.max() + 1)
    t_cap = t_cap[sizes[lab[t_cap[:, 0]]] > 0.05 * sizes.max()]
    used, inv = np.unique(t_cap, return_inverse=True)
    cap, t_cap = cap[used], inv.reshape(-1, 3)
    n_cap = vertex_normals(cap, t_cap)
    return cap, t_cap, n_cap


def buns(front: View, sides: list[View], chin_y: float, shoulder_y: float):
    """턱 아래로 늘어진 머리 묶음(번): 정면 사진의 머리카락 덩어리마다 타원체 하나."""
    hair = clean_hair(front)
    r0, r1 = int(front.row_of(chin_y)), int(front.row_of(shoulder_y))
    band = np.zeros_like(hair)
    band[r0:r1] = hair[r0:r1]
    lab, k = ndi.label(band)
    # 측면 사진의 같은 높이 머리카락 덩어리에서 앞뒤 위치를 잡는다
    z_centers, z_radii = [], []
    for s in sides:
        hs = clean_hair(s)
        sr0, sr1 = int(s.row_of(chin_y)), int(s.row_of(shoulder_y))
        cols = np.nonzero(hs[sr0:sr1])[1]
        if len(cols):
            z_centers.append(float(s.u_of(np.median(cols))))
            z_radii.append(float((np.percentile(cols, 90) - np.percentile(cols, 10)) / 2 * s.scale))
    zc = float(np.mean(z_centers)) if z_centers else -0.06
    rz = float(np.mean(z_radii)) if z_radii else 0.03
    out = []
    for i in range(1, k + 1):
        rr, cc = np.nonzero(lab == i)
        if len(rr) < 300:
            continue
        cu, cy = float(front.u_of(cc.mean())), float(front.height_of(rr.mean()))
        rx = (cc.max() - cc.min()) / 2 * front.scale
        ry = (rr.max() - rr.min()) / 2 * front.scale
        out.append((np.array([cu, cy, zc]), np.array([rx, ry, min(rz, 1.2 * max(rx, ry))])))
    return out


def hair_colors(rgb: np.ndarray) -> np.ndarray:
    """사진 머리카락 색은 가닥 사이로 비친 배경(주황)이 섞여 붉게 뜬다: 밝기만 살리고 색은 검은 머리로."""
    lum = rgb.astype(float) @ [0.299, 0.587, 0.114]
    lum = np.clip(lum / max(np.median(lum), 1), 0.4, 2.5)
    return np.clip(np.array([30, 25, 27]) * lum[:, None], 0, 255).astype(np.uint8)


def ellipsoid(center, radii, n_u: int = 24, n_v: int = 16):
    u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    w = np.linspace(0, np.pi, n_v)
    uu, ww = np.meshgrid(u, w)
    unit = np.c_[(np.sin(ww) * np.cos(uu)).ravel(), np.cos(ww).ravel(), (np.sin(ww) * np.sin(uu)).ravel()]
    verts = center + unit * radii
    tris = []
    for i in range(n_v - 1):
        for j in range(n_u):
            a, b = i * n_u + j, i * n_u + (j + 1) % n_u
            c, d = a + n_u, b + n_u
            tris += [(a, c, b), (b, c, d)]
    normals = unit / radii
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return verts, np.array(tris), normals
