"""MakeHuman 기본 인체 메시: 몸형 변형(target), 골격, 스키닝으로 자세 잡기."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 쓸 메시 그룹: 몸 + 눈알(눈알이 없으면 눈 자리가 구멍으로 보인다)
BODY_GROUP = "body"
EYE_GROUPS = ("helper-l-eye", "helper-r-eye")

# 뼈 → 부위. 실루엣 맞추기를 부위별로 따로 한다
ARM_BONES = ("shoulder01", "upperarm", "lowerarm", "wrist", "metacarpal", "finger")
LEG_BONES = ("upperleg", "lowerleg", "foot", "toe")
NECK_BONES = ("neck01", "neck02", "neck03")
TRUNK_BONES = ("root", "spine", "pelvis", "breast", "clavicle")
PARTS = ("trunk", "head", "arm.L", "arm.R", "leg.L", "leg.R")


def part_of(bone: str) -> str:
    side = bone[-2:] if bone[-2:] in (".L", ".R") else ""
    if bone.startswith(ARM_BONES):
        return "arm" + side
    if bone.startswith(LEG_BONES):
        return "leg" + side
    if bone.startswith(TRUNK_BONES) or bone in NECK_BONES:
        return "trunk"
    return "head"  # 머리·턱·눈·얼굴 근육·혀


def rotation(axis, degrees: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = np.radians(degrees)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def load_obj(path: Path):
    verts, uvs, faces, face_uvs, groups = [], [], [], [], []
    group = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                uvs.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("g "):
                group = line.split()[1]
            elif line.startswith("f "):
                corners = [c.split("/") for c in line.split()[1:]]
                faces.append([int(c[0]) - 1 for c in corners])
                face_uvs.append([int(c[1]) - 1 for c in corners])
                groups.append(group)
    return np.array(verts), np.array(uvs), np.array(faces), np.array(face_uvs), np.array(groups)


def load_target(base_dir: Path, name: str, n: int) -> np.ndarray:
    """target 하나의 꼭짓점 변위(미터, 쉬는 자세 기준)."""
    npz = np.load(base_dir / "targets.npz")
    key = name.replace("/", "__")
    d = np.zeros((n, 3))
    d[npz[key + "__index"].astype(int)] = npz[key + "__vector"].astype(float) * 1e-4  # 1/1000 dm → m
    return d


class Body:
    def __init__(self, base_dir: Path, targets: dict[str, float]):
        verts, self.uvs, faces, face_uvs, groups = load_obj(base_dir / "base.obj")
        npz = np.load(base_dir / "targets.npz")
        for name, weight in targets.items():
            key = name.replace("/", "__")
            idx = npz[key + "__index"].astype(int)
            verts[idx] += weight * npz[key + "__vector"].astype(float) * 1e-3  # MakeHuman은 1/1000 단위로 저장
        verts *= 0.1  # 데시미터 → 미터

        keep = (groups == BODY_GROUP) | np.isin(groups, EYE_GROUPS)
        self.faces, self.face_uvs = faces[keep], face_uvs[keep]
        self.is_eye = np.isin(groups[keep], EYE_GROUPS)
        self.rest = verts

        skel = json.loads((base_dir / "default.mhskel").read_text())
        joints = {name: verts[idx].mean(0) for name, idx in skel["joints"].items()}
        bones = skel["bones"]
        order, seen = [], set()

        def visit(b):
            if b in seen:
                return
            parent = bones[b]["parent"]
            if parent:
                visit(parent)
            seen.add(b)
            order.append(b)

        for b in bones:
            visit(b)
        self.bones = order
        self.parent = {b: bones[b]["parent"] for b in order}
        self.head = {b: joints[bones[b]["head"]] for b in order}
        self.tail = {b: joints[bones[b]["tail"]] for b in order}

        weights = json.loads((base_dir / "default_weights.mhw").read_text())["weights"]
        self.weights = np.zeros((len(verts), len(order)), np.float32)
        col = {b: i for i, b in enumerate(order)}
        for b, pairs in weights.items():
            pairs = np.array(pairs)
            self.weights[pairs[:, 0].astype(int), col[b]] = pairs[:, 1]
        # 눈알(보조 메시)은 가중치가 없어서 머리뼈에 붙인다
        eye_verts = np.unique(self.faces[self.is_eye])
        self.weights[eye_verts] = 0
        self.weights[eye_verts, col["head"]] = 1
        total = self.weights.sum(1, keepdims=True)
        self.weights[total[:, 0] == 0, col["root"]] = 1
        self.weights /= np.maximum(self.weights.sum(1, keepdims=True), 1e-9)

        # 부위별 가중치(스키닝 가중치를 부위로 묶은 것): 부위 경계에서 변형이 부드럽게 섞인다
        self.part_weights = np.zeros((len(verts), len(PARTS)), np.float32)
        for i, b in enumerate(order):
            self.part_weights[:, PARTS.index(part_of(b))] += self.weights[:, i]

    def used_vertices(self) -> np.ndarray:
        return np.unique(self.faces)

    def bone_transforms(self, rotations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """뼈마다 4x4 변환. 회전은 쉬는 자세(rest) 좌표계에서 뼈 머리를 축으로 준다."""
        out = {}
        for b in self.bones:
            m = np.eye(4)
            if b in rotations:
                r, h = rotations[b], self.head[b]
                m[:3, :3] = r
                m[:3, 3] = h - r @ h
            p = self.parent[b]
            out[b] = out[p] @ m if p else m
        return out

    def pose(self, rotations: dict[str, np.ndarray]):
        """선형 블렌드 스키닝. 반환: (자세 잡은 꼭짓점, 뼈별 4x4 변환)."""
        mats = self.bone_transforms(rotations)
        out = np.zeros_like(self.rest)
        for i, b in enumerate(self.bones):
            w = self.weights[:, i]
            nz = w > 0
            if nz.any():
                m = mats[b]
                out[nz] += w[nz, None] * (self.rest[nz] @ m[:3, :3].T + m[:3, 3])
        return out, mats

    def joint(self, mats, bone: str, end: str = "head") -> np.ndarray:
        p = self.head[bone] if end == "head" else self.tail[bone]
        # 뼈 머리는 부모의 변환을 따른다(자기 회전의 축이므로)
        m = mats[self.parent[bone]] if end == "head" and self.parent[bone] else mats[bone]
        return m[:3, :3] @ p + m[:3, 3]
