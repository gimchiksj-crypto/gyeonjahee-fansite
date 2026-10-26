"""최소한의 glTF 2.0 바이너리(GLB) 작성기: 여러 프리미티브, 텍스처, 꼭짓점 색."""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


class GLB:
    def __init__(self):
        self.buf = bytearray()
        self.views, self.accessors, self.materials, self.images, self.textures = [], [], [], [], []
        self.primitives = []

    def _blob(self, data: bytes, target: int | None = None) -> int:
        self.buf += b"\0" * (-len(self.buf) % 4)
        view = {"buffer": 0, "byteOffset": len(self.buf), "byteLength": len(data)}
        if target:
            view["target"] = target
        self.buf += data
        self.views.append(view)
        return len(self.views) - 1

    def _accessor(self, arr: np.ndarray, kind: str, comp: int, target: int, normalized=False, bounds=False) -> int:
        acc = {"bufferView": self._blob(arr.tobytes(), target), "componentType": comp, "count": len(arr), "type": kind}
        if normalized:
            acc["normalized"] = True
        if bounds:
            acc["min"], acc["max"] = arr.min(0).tolist(), arr.max(0).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def texture(self, img: np.ndarray, quality: int = 90) -> int:
        out = io.BytesIO()
        Image.fromarray(img).save(out, "JPEG", quality=quality)
        self.images.append({"bufferView": self._blob(out.getvalue()), "mimeType": "image/jpeg"})
        self.textures.append({"source": len(self.images) - 1})
        return len(self.textures) - 1

    def material(self, name: str, texture: int | None = None, roughness: float = 0.7, double_sided=False) -> int:
        pbr = {"metallicFactor": 0.0, "roughnessFactor": roughness}
        if texture is not None:
            pbr["baseColorTexture"] = {"index": texture}
        mat = {"name": name, "pbrMetallicRoughness": pbr}
        if double_sided:
            mat["doubleSided"] = True
        self.materials.append(mat)
        return len(self.materials) - 1

    def primitive(self, verts, normals, tris, material: int, uvs=None, colors=None):
        attrs = {
            "POSITION": self._accessor(verts.astype(np.float32), "VEC3", 5126, 34962, bounds=True),
            "NORMAL": self._accessor(normals.astype(np.float32), "VEC3", 5126, 34962),
        }
        if uvs is not None:
            attrs["TEXCOORD_0"] = self._accessor(uvs.astype(np.float32), "VEC2", 5126, 34962)
        if colors is not None:
            rgba = np.c_[colors, np.full(len(colors), 255)].astype(np.uint8)
            attrs["COLOR_0"] = self._accessor(rgba, "VEC4", 5121, 34962, normalized=True)
        idx = self._accessor(tris.astype(np.uint32).ravel(), "SCALAR", 5125, 34963)
        self.primitives.append({"attributes": attrs, "indices": idx, "material": material})

    def save(self, path: Path, name: str):
        gltf = {
            "asset": {"version": "2.0", "generator": "gyeonjahee-fansite build_3d_model.py"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": name}],
            "meshes": [{"name": name, "primitives": self.primitives}],
            "materials": self.materials,
            "buffers": [{"byteLength": len(self.buf)}],
            "bufferViews": self.views,
            "accessors": self.accessors,
        }
        if self.images:
            gltf["images"], gltf["textures"] = self.images, self.textures
            gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987}]
            for t in self.textures:
                t["sampler"] = 0
        js = json.dumps(gltf, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 4)
        self.buf += b"\0" * (-len(self.buf) % 4)
        body = struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(self.buf), b"BIN\0") + bytes(self.buf)
        path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
