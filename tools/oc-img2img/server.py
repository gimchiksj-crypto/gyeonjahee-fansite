#!/usr/bin/env python3
"""폰에서 쓰는 OC 생성 웹앱 서버.

ComfyUI(기본 127.0.0.1:8188) 앞에 붙어서 모바일 UI(web/)와 생성 API를 제공한다.
원격 접속은 Tailscale `tailscale serve`로 HTTPS를 씌우는 걸 전제로 기본 127.0.0.1에만 열린다.

  APP_PASSWORD=비밀번호 python server.py              # http://127.0.0.1:8787
  APP_PASSWORD=비밀번호 python server.py --host 0.0.0.0   # 같은 와이파이에서 직접 접속
"""
import argparse
import base64
import hashlib
import hmac
import json
import mimetypes
import os
import queue
import random
import re
import secrets
import sys
import threading
import time
import urllib.parse
import uuid
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import generate as gen

HERE = Path(__file__).resolve().parent
WEB = HERE / "web"
OUT = HERE / "outputs" / "web"
INPUT = HERE / "input"
SESSION_SECRET = secrets.token_bytes(32)
MAX_BODY = 40 * 1024 * 1024

jobs = {}           # id -> job dict
jobs_lock = threading.Lock()
work = queue.Queue()
state = {"ref": None, "source": None}   # 마지막으로 올린 이미지(ComfyUI 쪽 파일명)


def session_token():
    return hmac.new(SESSION_SECRET, b"ok", hashlib.sha256).hexdigest()


def worker(cfg):
    while True:
        job_id = work.get()
        with jobs_lock:
            job = jobs[job_id]
            job["status"] = "running"
        try:
            a = SimpleNamespace(**job["params"])
            graph = gen.build_graph(a, job["ref"], job["source"], job["prompt"], job["seed"])
            files = gen.run(cfg.comfy, graph, OUT, job_id)
            with jobs_lock:
                job["status"] = "done"
                job["images"] = [f.name for f in files]
        except Exception as e:
            with jobs_lock:
                job["status"] = "error"
                job["error"] = str(e)[:500]


def save_data_url(data_url, stem):
    m = re.match(r"data:image/(png|jpeg|webp);base64,(.+)", data_url or "", re.S)
    if not m:
        raise ValueError("이미지 형식이 올바르지 않아요 (png/jpeg/webp)")
    ext = "jpg" if m.group(1) == "jpeg" else m.group(1)
    path = INPUT / f"{stem}.{ext}"
    path.write_bytes(base64.b64decode(m.group(2)))
    return path


class Handler(BaseHTTPRequestHandler):
    cfg = None
    server_version = "OCStudio"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    # ---- helpers ----
    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, cache="no-cache"):
        data = path.read_bytes()
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".webmanifest":
            ctype = "application/manifest+json"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authed(self):
        c = SimpleCookie(self.headers.get("Cookie", ""))
        tok = c["oc_session"].value if "oc_session" in c else ""
        return hmac.compare_digest(tok, session_token())

    def read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BODY:
            raise ValueError("요청이 너무 커요")
        return json.loads(self.rfile.read(n) or b"{}")

    # ---- routes ----
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            return self.send_file(WEB / "index.html")
        if path.startswith("/api/"):
            if not self.authed():
                return self.send_json({"error": "login"}, 401)
            if path == "/api/config":
                return self.send_json(self.config())
            if path == "/api/jobs":
                with jobs_lock:
                    items = sorted(jobs.values(), key=lambda j: j["created"], reverse=True)[:200]
                    return self.send_json([{k: j[k] for k in ("id", "status", "outfit", "seed",
                                                              "images", "error")} for j in items])
            if path.startswith("/api/image/"):
                name = Path(urllib.parse.unquote(path[len("/api/image/"):])).name
                f = OUT / name
                if f.is_file():
                    return self.send_file(f, "private, max-age=31536000")
            return self.send_json({"error": "not found"}, 404)
        f = (WEB / path.lstrip("/")).resolve()
        if f.is_file() and WEB in f.parents:
            return self.send_file(f)
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            body = self.read_json()
        except Exception as e:
            return self.send_json({"error": str(e)}, 400)

        if path == "/api/login":
            if hmac.compare_digest(str(body.get("password", "")), self.cfg.password):
                self.send_response(204)
                self.send_header("Set-Cookie", f"oc_session={session_token()}; Path=/; HttpOnly; "
                                 "SameSite=Strict; Max-Age=2592000")
                self.end_headers()
                return
            time.sleep(1.5)
            return self.send_json({"error": "비밀번호가 틀렸어요"}, 403)

        if not self.authed():
            return self.send_json({"error": "login"}, 401)
        if path == "/api/generate":
            try:
                return self.send_json(self.enqueue(body))
            except Exception as e:
                return self.send_json({"error": str(e)}, 400)
        if path == "/api/clear":
            with jobs_lock:
                for k in [k for k, j in jobs.items() if j["status"] in ("done", "error")]:
                    del jobs[k]
            return self.send_json({"ok": True})
        self.send_json({"error": "not found"}, 404)

    # ---- logic ----
    def config(self):
        ckpts = []
        try:
            info = json.loads(gen.api(self.cfg.comfy, "/object_info/CheckpointLoaderSimple"))
            ckpts = info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        except Exception:
            pass
        return {
            "character": ", ".join(gen.read_lines(HERE / "character.txt")),
            "outfits": gen.read_lines(HERE / "outfits.txt"),
            "checkpoints": ckpts,
            "comfy_ok": bool(ckpts),
            "has_ref": state["ref"] is not None,
            "has_source": state["source"] is not None,
        }

    def enqueue(self, b):
        if b.get("ref"):
            state["ref"] = gen.upload(self.cfg.comfy, save_data_url(b["ref"], "web_ref"))
        if b.get("source"):
            state["source"] = gen.upload(self.cfg.comfy, save_data_url(b["source"], "web_source"))
        if b.get("clear_source"):
            state["source"] = None
        if not state["ref"]:
            raise ValueError("외형 레퍼런스 이미지를 먼저 올려주세요")

        outfits = [o.strip() for o in b.get("outfits", []) if o.strip()]
        if not outfits:
            raise ValueError("복장을 하나 이상 골라주세요")
        count = max(1, min(int(b.get("count", 1)), 8))
        use_source = bool(b.get("use_source")) and state["source"] is not None
        character = str(b.get("character", "")).strip()
        params = dict(
            ckpt=str(b.get("ckpt") or "sd_xl_base_1.0.safetensors"),
            preset=b.get("preset") if b.get("preset") in gen.PRESETS else "plus",
            ip_weight=float(b.get("ip_weight", 0.8)),
            denoise=float(b.get("denoise", 0.65)),
            steps=max(10, min(int(b.get("steps", 30)), 60)),
            cfg=float(b.get("cfg", 6.0)),
            width=int(b.get("width", 832)) // 8 * 8,
            height=int(b.get("height", 1216)) // 8 * 8,
        )
        created = []
        with jobs_lock:
            for outfit in outfits[:20]:
                for _ in range(count):
                    jid = uuid.uuid4().hex[:10]
                    jobs[jid] = dict(
                        id=jid, status="queued", outfit=outfit, seed=random.randint(0, 2**32 - 1),
                        prompt=f"{character}, {outfit}" if character else outfit,
                        ref=state["ref"], source=state["source"] if use_source else None,
                        params=params, images=[], error=None, created=time.time(),
                    )
                    created.append(jid)
                    work.put(jid)
        return {"queued": created}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--comfy", default="127.0.0.1:8188", help="ComfyUI 주소")
    p.add_argument("--password", default=os.environ.get("APP_PASSWORD"))
    cfg = p.parse_args()
    if not cfg.password or len(cfg.password) < 8:
        sys.exit("APP_PASSWORD(또는 --password)를 8자 이상으로 지정해주세요.")

    OUT.mkdir(parents=True, exist_ok=True)
    INPUT.mkdir(exist_ok=True)
    Handler.cfg = cfg
    threading.Thread(target=worker, args=(cfg,), daemon=True).start()
    print(f"OC Studio: http://{cfg.host}:{cfg.port}  (ComfyUI: {cfg.comfy})")
    ThreadingHTTPServer((cfg.host, cfg.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
