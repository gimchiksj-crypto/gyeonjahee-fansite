"""머리카락: 두피에서 사진 속 머리카락 윤곽까지 부풀린 껍데기 + 묶은 머리(번)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from .bake import Projector, vertex_normals
from .photos import View

MAX_THICKNESS = 0.07  # 두피에서 머리카락 바깥면까지 최대 거리(미터)
STEP = 0.0015
BANGS_THICKNESS = 0.008
WRAP = ("nearest", "wrap")  # 구면 격자: 극각 방향은 끝값 유지, 방위각 방향은 한 바퀴 이어짐


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


def smooth_boundary(v: np.ndarray, tris: np.ndarray, iterations: int) -> np.ndarray:
    """열린 가장자리(머리선)를 가장자리를 따라 매끈하게: 격자 칸 모양 계단을 없앤다."""
    e = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    uniq, count = np.unique(e, axis=0, return_counts=True)
    edges = uniq[count == 1]  # 삼각형 하나에만 속한 변 = 가장자리
    if not len(edges):
        return v
    n = len(v)
    a = sparse.csr_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n))
    deg = np.asarray(a.sum(1)).ravel()
    on = deg > 0
    v = v.copy()
    for _ in range(iterations):
        avg = (a @ v)[on] / deg[on, None]
        v[on] = 0.5 * v[on] + 0.5 * avg
    # 가장자리 바로 안쪽 줄도 조금 따라오게(가장자리만 움직여 생기는 주름 방지)
    return taubin(v, tris, np.where(on, 0.0, 1.0), 5)


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
    """머리카락 껍데기: 머리 중심에서 사방으로 쏜 방향마다 두피 높이와 머리카락 두께를 정한다.

    머리 메시 표면을 법선으로 부풀리면 귀·이목구비 굴곡 때문에 껍데기가 구겨진다.
    구면 격자 위에서 만들면 위상이 단순해서 항상 매끈한 한 장이 된다."""
    hair_f, hair_b = clean_hair(front), clean_hair(back)
    hair_s = [clean_hair(s) for s in sides]
    used = np.unique(tris)
    head = used[((head_w[used] > 0.5) & (v[used, 1] > chin_y - 0.01)) | ((v[used, 1] > neck_y) & (v[used, 1] <= chin_y))]
    pts = v[head]
    skull = pts[pts[:, 1] > chin_y + 0.04]
    c = (skull.min(0) + skull.max(0)) / 2

    # 구면 격자: 극각 0(정수리) ~ 155°(목 뒤), 방위각은 한 바퀴
    n_t, n_p = 144, 256
    theta = np.radians(np.linspace(0, 155, n_t))
    phi = np.linspace(0, 2 * np.pi, n_p, endpoint=False)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    dirs = np.stack([np.sin(tt) * np.sin(pp), np.cos(tt), np.sin(tt) * np.cos(pp)], -1).reshape(-1, 3)

    # 방향마다 두피까지 거리: 그 방향 근처(각도 6° 안) 머리 꼭짓점의 최대 투영
    rel = pts - c
    unit = rel / np.linalg.norm(rel, axis=1, keepdims=True)
    tree = cKDTree(unit)
    r_skin = np.zeros(len(dirs))
    for i, nb in enumerate(tree.query_ball_point(dirs, 2 * np.sin(np.radians(4)))):
        if nb:
            r_skin[i] = (rel[nb] @ dirs[i]).max()
    grid = lambda a: a.reshape(n_t, n_p)  # noqa: E731
    missing = grid(r_skin) == 0
    r = grid(r_skin)
    _, (ri, ci) = ndi.distance_transform_edt(missing, return_indices=True)  # 빈 방향은 가장 가까운 방향 값
    r = r[ri, ci]
    r = ndi.gaussian_filter(r, 2.0, mode=WRAP)
    p = c + dirs * r.reshape(-1, 1)
    n = dirs
    x, y, z = p.T

    # 머리카락인 정도: 표면이 향한 쪽 사진에서 머리카락 픽셀인지, 보이는 곳인지로 가중 평균
    w_f = np.clip(n[:, 2], 0, None) ** 2
    w_b = np.clip(-n[:, 2], 0, None) ** 2
    fu, fy = proj.warp.front_uv(p)
    su, sy = proj.warp.side_uv(p)
    score = w_f * pix(front, fu, fy, hair_f) + w_b * pix(back, fu, fy, hair_b)
    total = w_f + w_b
    for s, hs in zip(sides, hair_s):
        w_s = n[:, 0] ** 2 / len(sides)
        score += w_s * pix(s, su, sy, hs)
        total += w_s
    hairness = grid(score / np.maximum(total, 1e-6))
    # 얼굴(눈썹 아래 정면)은 속눈썹·눈동자가 어두워도 머리카락이 아니다
    face = grid((y < brow_y) & (n[:, 2] > 0.3) & (np.abs(x) < 0.055))
    hairness[face] = 0
    hairness = ndi.gaussian_filter(hairness, 3.0, mode=WRAP)
    # 정수리·뒤통수는 사진 각도상 판정이 애매해도 머리카락이다
    hairness = np.maximum(hairness, grid(np.clip((y - (y.max() - 0.06)) / 0.03, 0, 1)))

    # 바깥쪽으로 사진 윤곽(시각 껍질)을 벗어나기 직전까지
    thick = np.zeros(len(p))
    alive = np.ones(len(p), bool)
    for k in range(1, int(MAX_THICKNESS / STEP) + 1):
        alive &= in_hull(p + n * (k * STEP), front, back, sides, proj.warp)
        thick[alive] = k * STEP
    thick = grid(thick)
    # 가장자리로 갈수록 얇게: 머리카락 끝이 피부에 붙어 들뜬 조각이 생기지 않게
    edge = np.clip((hairness - 0.4) / 0.45, 0, 1)
    thick = thick * edge * edge * (3 - 2 * edge)
    # 앞머리는 이마에 얹힌 얇은 층이다(정면 윤곽이 옆머리 때문에 넓어서 그대로 두면 앞으로 튀어나온다)
    bangs = grid((n[:, 2] > 0.4) & (y < brow_y + 0.05))
    thick[bangs] = np.minimum(thick[bangs], BANGS_THICKNESS)
    thick = ndi.gaussian_filter(thick, 4.0, mode=WRAP)
    cap = p + n * (thick.reshape(-1, 1) + 0.002)

    # 격자 → 삼각형(방위각은 이어 붙인다), 머리카락인 칸만
    i, j = np.meshgrid(np.arange(n_t - 1), np.arange(n_p), indexing="ij")
    a_, b_ = i * n_p + j, i * n_p + (j + 1) % n_p
    c_, d_ = a_ + n_p, b_ + n_p
    quads = np.stack([a_, c_, d_, b_], -1).reshape(-1, 4)
    keep = hairness.ravel()[quads].mean(1) > 0.4
    quads = quads[keep]
    t_cap = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    # 큰 덩어리만 남긴다(떨어져 나온 작은 조각 제거)
    lab = csgraph.connected_components(adjacency(len(cap), t_cap), directed=False)[1]
    sizes = np.bincount(lab[t_cap[:, 0]], minlength=lab.max() + 1)
    t_cap = t_cap[sizes[lab[t_cap[:, 0]]] > 0.05 * sizes.max()]
    used, inv = np.unique(t_cap, return_inverse=True)
    cap, t_cap = cap[used], inv.reshape(-1, 3)
    cap = smooth_boundary(cap, t_cap, 30)
    n_cap = vertex_normals(cap, t_cap)
    if np.mean(np.sum(n_cap * (cap - c), 1)) < 0:  # 바깥을 보게
        t_cap = t_cap[:, ::-1]
        n_cap = -n_cap
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
