"""정면·후면·측면 사진으로 여성 3D 모델(GLB)을 만든다.

방식: 사람 몸 기본 메시를 사진에 맞춰 변형한다.
  1. 출발점은 MakeHuman 기본 인체(얼굴·손가락·발가락·UV·골격 포함, CC0)에
     동아시아 20대 여성 몸형을 적용한 메시다. (model/base/)
  2. 정면 사진에서 팔·다리 각도를 재서 골격으로 자세를 잡는다(까치발, 주먹 포함).
  3. 턱 높이로 전체 크기를, 가랑이·겨드랑이 높이로 몸 비율을 맞춘다.
  4. 높이별로 몸통·다리·팔의 좌우 폭(정면 사진)과 앞뒤 깊이(측면 사진)를 실루엣에 맞춘다.
     부위 경계는 스키닝 가중치로 섞어서 이음매 없이 이어진다.
  5. 눈·코·턱 기준점이 사진과 겹치게 머리를 옮기고 늘인다.
  6. 사진을 표면 방향과 가림 여부에 따라 섞어 UV 텍스처로 굽는다.
  7. 사진에서 머리카락으로 보이는 곳 중 몸 바깥을 채워 머리카락 메시를 만든다.

사진끼리 자세·비율이 조금씩 다른 부분(측면 사진만 팔을 뒤로 모음, 머리카락 윤곽 등)은
정면 사진을 기준으로 삼고 나머지는 보조로만 쓴다.

출력: model/gyeonjahee.glb (몸: UV 텍스처, 눈·머리카락: 꼭짓점 색)
필요 패키지: numpy, pillow, scipy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from model3d import bake, fit, hair, photos
from model3d.body import PARTS, Body
from model3d.gltf import GLB

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "model" / "src"
BASE = ROOT / "model" / "base"
OUT = ROOT / "model" / "gyeonjahee.glb"

# 기본 메시에 적용할 몸형(MakeHuman target, 가중치)
BODY_TARGETS = {
    "macrodetails/asian-female-young": 1.0,
    "macrodetails/universal-female-young-averagemuscle-minweight": 0.3,
    "breast/female-young-averagemuscle-averageweight-maxcup-averagefirmness": 1.0,
    "breast/female-young-averagemuscle-averageweight-averagecup-maxfirmness": 0.6,
    # 표정: 정면 사진처럼 왼쪽 눈 윙크, 오른쪽 눈웃음, 미소(입은 다물어 이는 텍스처로만)
    "expression/units/asian/eye-left-closure": 1.0,
    "expression/units/asian/eye-right-slit": 0.7,
    "expression/units/asian/mouth-corner-puller": 0.6,
}


EYE_SCALE = 0.82
EYE_SETBACK = 0.005  # 미터


def split_by_uv(v, normals, tris, uv_tris, uvs):
    """glTF는 꼭짓점마다 UV가 하나라서, UV 이음선에서 꼭짓점을 나눈다."""
    pairs = np.c_[tris.ravel(), uv_tris.ravel()]
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    uv = uvs[uniq[:, 1]] * [1, -1] + [0, 1]  # glTF UV는 위가 0
    return v[uniq[:, 0]], normals[uniq[:, 0]], uv, inv.reshape(-1, 3)


def compact(v, normals, tris):
    used, inv = np.unique(tris, return_inverse=True)
    return v[used], normals[used], inv.reshape(-1, 3)


def smooth_clothed(body: Body, v: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """브라·반바지로 덮인 곳은 기본 메시의 해부학적 굴곡(유두 등)을 매끈하게 편다."""
    used = body.used_vertices()
    breast = [i for i, b in enumerate(body.bones) if b.startswith("breast")]
    # 가슴을 들어 올리며 생긴 아랫선 주름까지 같이 펴도록 가슴 덩어리 전체를 넉넉히
    w = np.clip(np.maximum(body.weights[:, breast].sum(1) * 4, fit.breast_mass_weight(body, v, used) * 1.5), 0, 1)
    # 사타구니 앞쪽(반바지)
    crotch_y = v[used][np.abs(v[used, 0]) < 0.004][:, 1]
    crotch_y = crotch_y[(crotch_y > 0.6) & (crotch_y < 1.0)].min()
    near = np.exp(-((v[:, 0] / 0.05) ** 2) - (((v[:, 1] - crotch_y - 0.03) / 0.05) ** 2))
    w = np.maximum(w, np.clip(near * 1.5, 0, 1))
    return hair.taubin(v, tris, w, iterations=60)


def seat_eyes(body: Body, v: np.ndarray) -> np.ndarray:
    """얼굴을 사진에 맞춰 줄이고 납작하게 하면서 눈알이 광대 앞으로 튀어나온다: 줄여서 안으로 넣는다."""
    v = v.copy()
    idx = np.unique(body.faces[body.is_eye])
    for side in (v[idx, 0] > 0, v[idx, 0] <= 0):
        e = idx[side]
        c = v[e].mean(0)
        v[e] = c + (v[e] - c) * EYE_SCALE + [0, 0, -EYE_SETBACK]
    return v


def main():
    front = photos.front_view(SRC / "front.png")
    back = photos.back_view(SRC / "back.png", front)
    sides = [photos.side_view(SRC / name) for name in ("side.png", "side2.png")]
    landmarks = photos.front_landmarks(front)

    body = Body(BASE, BODY_TARGETS)
    v = fit.fit_to_photos(body, front, sides, landmarks)

    tris = bake.triangles(body.faces)
    uv_tris = bake.triangles(body.face_uvs)
    eye = np.r_[body.is_eye, body.is_eye]
    v = smooth_clothed(body, v, tris[~eye])
    v = seat_eyes(body, v)
    normals = np.zeros_like(v)
    normals_body = bake.vertex_normals(v, tris[~eye])
    normals_eye = bake.vertex_normals(v, tris[eye])
    eye_verts = np.unique(tris[eye])
    normals[:] = normals_body
    normals[eye_verts] = normals_eye[eye_verts]
    # 측면 사진은 팔 자세가 달라서 팔에는 쓰지 않는다
    side_ok = 1 - np.clip(1.5 * (body.part_weights[:, PARTS.index("arm.L")] + body.part_weights[:, PARTS.index("arm.R")]), 0, 1)

    proj = bake.Projector(front, back, sides, v, tris[~eye])
    # 목·몸·턱 아래쪽: 앞으로 늘어진 머리카락 픽셀은 피부에 그리지 않는다(머리카락은 따로 만든다)
    not_head = (body.part_weights[:, PARTS.index("head")] < 0.5) | (v[:, 1] < landmarks["eye_l"][1] - 0.03)
    texture = bake.bake_texture(proj, v, normals, side_ok, not_head, tris[~eye], uv_tris[~eye], body.uvs)

    glb = GLB()
    skin = glb.material("skin", glb.texture(texture), roughness=0.6)
    bv, bn, buv, bt = split_by_uv(v, normals, tris[~eye], uv_tris[~eye], body.uvs)
    glb.primitive(bv, bn, bt, skin, uvs=buv)

    ev, en, et = compact(v, normals, tris[eye])
    glb.primitive(ev, en, et, glb.material("eyes", roughness=0.15), colors=proj.colors(ev, en, np.ones(len(ev))))

    hair_mat = glb.material("hair", roughness=0.45, double_sided=True)
    head_w = body.part_weights[:, PARTS.index("head")]
    chin_y = landmarks["chin"][1]
    brow_y = landmarks["eye_l"][1] + 0.012
    neck_y = float(front.height_of(photos.FRONT_SHOULDER_ROW)) + 0.02
    cv, ct, cn = hair.hair_cap(v, tris[~eye], head_w, chin_y, brow_y, neck_y, front, back, sides, proj)
    glb.primitive(cv, cn, ct, hair_mat, colors=hair.hair_colors(proj.colors(cv, cn, np.ones(len(cv)))))
    for center, radii in hair.buns(front, sides, chin_y, float(front.height_of(photos.FRONT_SHOULDER_ROW))):
        bv_, bt_, bn_ = hair.ellipsoid(center, radii)
        glb.primitive(bv_, bn_, bt_, hair_mat, colors=hair.hair_colors(proj.colors(bv_, bn_, np.ones(len(bv_)))))

    glb.save(OUT, "gyeonjahee")
    print(f"body={len(bv)} eyes={len(ev)} hair={len(cv)} vertices, glb={OUT.stat().st_size / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
