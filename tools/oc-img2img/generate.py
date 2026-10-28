#!/usr/bin/env python3
"""OC 외형을 IP-Adapter로 고정한 채 복장별 이미지를 일괄 생성한다.

ComfyUI가 http://127.0.0.1:8188 에서 실행 중이어야 한다. 표준 라이브러리만 사용.

예)
  # img2img: 구도/포즈용 원본 + OC 레퍼런스
  python generate.py --ref input/oc_ref.png --source input/pose.png
  # txt2img: 레퍼런스만으로 새 구도
  python generate.py --ref input/oc_ref.png --seeds 3
  # 워크플로 JSON만 뽑기 (ComfyUI에서 직접 열어 수정할 때)
  python generate.py --ref input/oc_ref.png --source input/pose.png --dump workflows/oc_img2img_api.json
"""
import argparse
import json
import random
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent

PRESETS = {
    "plus": "PLUS (high strength)",       # 얼굴 + 머리 + 전체 분위기
    "face": "PLUS FACE (portraits)",      # 얼굴 위주, 복장 간섭이 적음
}

NEGATIVE = (
    "lowres, blurry, bad anatomy, bad hands, extra fingers, deformed face, "
    "different person, inconsistent face, watermark, text, signature"
)


def read_lines(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.lstrip().startswith("#")]


def build_graph(a, ref_name, source_name, positive, seed):
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": a.ckpt}},
        "2": {"class_type": "IPAdapterUnifiedLoader",
              "inputs": {"model": ["1", 0], "preset": PRESETS[a.preset]}},
        "3": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
        "4": {"class_type": "IPAdapterAdvanced",
              "inputs": {"model": ["2", 0], "ipadapter": ["2", 1], "image": ["3", 0],
                         "weight": a.ip_weight, "weight_type": "linear",
                         "combine_embeds": "concat", "start_at": 0.0, "end_at": 1.0,
                         "embeds_scaling": "V only"}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["1", 1]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "10": {"class_type": "KSampler",
               "inputs": {"model": ["4", 0], "seed": seed, "steps": a.steps, "cfg": a.cfg,
                          "sampler_name": "dpmpp_2m", "scheduler": "karras",
                          "positive": ["8", 0], "negative": ["9", 0],
                          "latent_image": ["7", 0], "denoise": 1.0}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "oc"}},
    }
    if source_name:
        g["5"] = {"class_type": "LoadImage", "inputs": {"image": source_name}}
        g["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["1", 2]}}
        g["10"]["inputs"]["denoise"] = a.denoise
    else:
        g["7"] = {"class_type": "EmptyLatentImage",
                  "inputs": {"width": a.width, "height": a.height, "batch_size": 1}}
    return g


def api(server, path, data=None, headers=None):
    req = urllib.request.Request(f"http://{server}{path}", data=data, headers=headers or {})
    with urllib.request.urlopen(req) as r:
        return r.read()


def upload(server, path):
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
        f"filename=\"{Path(path).name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
    ).encode() + Path(path).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    res = api(server, "/upload/image", body,
              {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.loads(res)["name"]


def run(server, graph, out_dir, tag):
    res = json.loads(api(server, "/prompt", json.dumps({"prompt": graph}).encode(),
                         {"Content-Type": "application/json"}))
    pid = res["prompt_id"]
    while True:
        hist = json.loads(api(server, f"/history/{pid}"))
        if pid in hist:
            break
        time.sleep(1)
    status = hist[pid].get("status", {})
    if status.get("status_str") == "error":
        raise RuntimeError(json.dumps(status.get("messages", []), ensure_ascii=False))
    saved = []
    for node in hist[pid]["outputs"].values():
        for img in node.get("images", []):
            q = urllib.parse.urlencode({k: img[k] for k in ("filename", "subfolder", "type")})
            dest = out_dir / f"{tag}_{img['filename']}"
            dest.write_bytes(api(server, f"/view?{q}"))
            saved.append(dest)
    return saved


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="OC 외형 레퍼런스 이미지 (얼굴이 잘 보이는 정면/반측면)")
    p.add_argument("--source", help="img2img 원본 (구도·포즈). 없으면 txt2img")
    p.add_argument("--ckpt", default="sd_xl_base_1.0.safetensors", help="models/checkpoints 안의 SDXL 체크포인트")
    p.add_argument("--preset", choices=PRESETS, default="plus")
    p.add_argument("--ip-weight", type=float, default=0.8, help="외형 고정 강도 (0.6~1.0)")
    p.add_argument("--denoise", type=float, default=0.65, help="img2img 변형 강도 (0.5~0.8)")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=6.0)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=1216)
    p.add_argument("--seeds", type=int, default=2, help="복장당 생성 장수")
    p.add_argument("--character", default=str(HERE / "character.txt"))
    p.add_argument("--outfits", default=str(HERE / "outfits.txt"))
    p.add_argument("--server", default="127.0.0.1:8188")
    p.add_argument("--out", default=str(HERE / "outputs"))
    p.add_argument("--dump", help="첫 번째 복장의 워크플로(API JSON)만 저장하고 종료")
    a = p.parse_args()

    character = ", ".join(read_lines(a.character))
    outfits = read_lines(a.outfits)

    if a.dump:
        g = build_graph(a, Path(a.ref).name, a.source and Path(a.source).name,
                        f"{character}, {outfits[0]}", 0)
        Path(a.dump).write_text(json.dumps(g, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"저장: {a.dump}")
        return

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_name = upload(a.server, a.ref)
    source_name = upload(a.server, a.source) if a.source else None

    for i, outfit in enumerate(outfits, 1):
        for _ in range(a.seeds):
            seed = random.randint(0, 2**32 - 1)
            graph = build_graph(a, ref_name, source_name, f"{character}, {outfit}", seed)
            try:
                files = run(a.server, graph, out_dir, f"{i:02d}")
            except Exception as e:  # 한 장 실패해도 나머지는 계속
                print(f"[{i:02d}] 실패: {e}", file=sys.stderr)
                continue
            for f in files:
                print(f"[{i:02d}] seed={seed} {outfit[:40]} -> {f}")


if __name__ == "__main__":
    main()
