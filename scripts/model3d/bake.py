"""사진 → 모델 색: 가림 판정용 깊이 지도, UV 텍스처 굽기."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .photos import View

# 사진마다 카메라 방향. 측면 사진은 얼굴이 화면 오른쪽(+z)을 보므로 카메라는 -x 쪽에서
# 캐릭터의 오른쪽 옆면을 찍은 것이다. 왼쪽 옆면은 좌우 대칭으로 같은 사진을 쓴다.
TEXTURE_SIZE = 2048
VISIBLE_TOL = 0.006  # 깊이 지도보다 이만큼(미터) 안쪽까지는 보인다고 친다


def triangles(quads: np.ndarray) -> np.ndarray:
    return np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])


def vertex_normals(v: np.ndarray, tris: np.ndarray) -> np.ndarray:
    fn = np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def surface_samples(v: np.ndarray, tris: np.ndarray, n: int = 6) -> np.ndarray:
    """삼각형마다 고르게 점을 뿌린다(깊이 지도용)."""
    a, b = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    keep = a + b <= 1
    a, b = a[keep], b[keep]
    p0, p1, p2 = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
    return (p0[:, None] + a[None, :, None] * (p1 - p0)[:, None] + b[None, :, None] * (p2 - p0)[:, None]).reshape(-1, 3)


class DepthMap:
    """사진 화면 기준 깊이 지도. axis=사진 가로축(0=x, 2=z), depth=깊이 축, sign=카메라 쪽 부호."""

    def __init__(self, view: View, points: np.ndarray, axis: int, depth: int, sign: int):
        self.view, self.axis, self.depth, self.sign = view, axis, depth, sign
        cols = np.round(view.col_of(points[:, axis])).astype(int)
        rows = np.round(view.row_of(points[:, 1])).astype(int)
        ok = (cols >= 0) & (cols < view.w) & (rows >= 0) & (rows < view.h)
        d = np.full((view.h, view.w), -np.inf)
        np.maximum.at(d, (rows[ok], cols[ok]), sign * points[ok, depth])
        self.d = ndi.maximum_filter(d, 3)  # 점 사이 틈 메우기

    def nearest(self, u, y):
        c = np.clip(np.round(self.view.col_of(u)).astype(int), 0, self.view.w - 1)
        r = np.clip(np.round(self.view.row_of(y)).astype(int), 0, self.view.h - 1)
        return self.d[r, c]

    def visible(self, p: np.ndarray) -> np.ndarray:
        return self.sign * p[:, self.depth] >= self.nearest(p[:, self.axis], p[:, 1]) - VISIBLE_TOL


class Projector:
    """위치·법선 → 색. 정면·후면·측면 사진을 표면 방향과 가림 여부에 따라 섞는다."""

    def __init__(self, front: View, back: View, sides: list[View], v: np.ndarray, tris: np.ndarray):
        pts = surface_samples(v, tris)
        self.front, self.back, self.sides = front, back, sides
        self.d_front = DepthMap(front, pts, 0, 2, +1)
        self.d_back = DepthMap(back, pts, 0, 2, -1)
        self.d_right = [DepthMap(s, pts, 2, 0, -1) for s in sides]
        mirrored = pts * [-1, 1, 1]
        self.d_left = [DepthMap(s, mirrored, 2, 0, -1) for s in sides]

    def colors(self, p: np.ndarray, n: np.ndarray, side_ok: np.ndarray, skin=None, with_known=False):
        """skin: 머리카락 픽셀을 쓰면 안 되는 점(목·몸). with_known: 믿을 만한 사진 픽셀이 있었는지도 돌려준다."""
        x, y, z = p.T
        nx, nz = n[:, 0], n[:, 2]
        cf, of = self.front.sample(x, y, skin)
        cb, ob = self.back.sample(x, y, skin)
        of *= self.d_front.visible(p)
        ob *= self.d_back.visible(p)
        # 옆을 향한 면(nz≈0)에서는 정면·후면 사진을 반반 섞어 경계가 계단처럼 보이지 않게 한다
        wf = np.clip(nz + 0.25, 0, None) ** 2 * of
        wb = np.clip(-nz + 0.25, 0, None) ** 2 * ob
        acc = cf * wf[:, None] + cb * wb[:, None]
        total = wf + wb
        # 측면 사진(오른쪽 옆면 원본, 왼쪽은 좌우 대칭)
        mirrored = p * [-1, 1, 1]
        for s, d_r, d_l in zip(self.sides, self.d_right, self.d_left):
            right = nx < 0
            q = np.where(right[:, None], p, mirrored)
            c, o = s.sample(q[:, 2], q[:, 1], skin)
            vis = np.where(right, d_r.visible(p), d_l.visible(mirrored))
            w = nx**2 * o * vis * side_ok / len(self.sides)
            acc += c * w[:, None]
            total += w
        col = acc / np.maximum(total, 1e-9)[:, None]
        # 어느 사진에도 안 잡히는 면은 정면/후면 중 가까운 쪽 색(경계 밖은 채워 둔 색)
        fallback = np.where((nz >= 0)[:, None], cf, cb)
        out = np.clip(np.where((total < 0.01)[:, None], fallback, col), 0, 255).astype(np.uint8)
        return (out, total >= 0.01) if with_known else out


def bake_texture(proj: Projector, v, n, side_ok, skin, tris, uv_tris, uvs) -> np.ndarray:
    """UV 공간에 삼각형 번호를 그려서, 텍셀마다 3D 위치·법선을 구해 색을 굽는다."""
    size = TEXTURE_SIZE
    ids = Image.new("I", (size, size), -1)
    draw = ImageDraw.Draw(ids)
    tex_xy = np.c_[uvs[:, 0] * size, (1 - uvs[:, 1]) * size]
    for t, corners in enumerate(uv_tris):
        draw.polygon([tuple(tex_xy[c]) for c in corners], fill=t)
    ids = np.array(ids)
    rows, cols = np.nonzero(ids >= 0)
    t = ids[rows, cols]
    a, b, c = (tex_xy[uv_tris[t, k]] for k in range(3))
    pt = np.c_[cols + 0.5, rows + 0.5]
    # 무게중심 좌표
    v0, v1, v2 = b - a, c - a, pt - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    l0 = 1 - l1 - l2
    lam = np.clip(np.c_[l0, l1, l2], 0, 1)
    lam /= lam.sum(1, keepdims=True)
    tri = tris[t]
    P = sum(lam[:, k, None] * v[tri[:, k]] for k in range(3))
    N = sum(lam[:, k, None] * n[tri[:, k]] for k in range(3))
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    S = sum(lam[:, k] * side_ok[tri[:, k]] for k in range(3))
    K = sum(lam[:, k] * skin[tri[:, k]] for k in range(3)) > 0.5
    img = np.zeros((size, size, 3), np.uint8)
    col, known = proj.colors(P, N, S, K, with_known=True)
    img[rows, cols] = col
    have = np.zeros((size, size), bool)
    have[rows, cols] = known
    # 어느 사진에서도 못 본 텍셀과 UV 섬 바깥(틈)은 가장 가까운 확실한 텍셀 색으로 메운다
    _, (ri, ci) = ndi.distance_transform_edt(~have, return_indices=True)
    return img[ri, ci]
