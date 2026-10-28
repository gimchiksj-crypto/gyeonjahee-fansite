# OC img2img — 외형 고정 + 복장 바리에이션

ComfyUI + SDXL + IP-Adapter(Plus / Plus Face)로 **오리지널 캐릭터의 얼굴·헤어 외형을 유지**하면서
복장·배경·구도를 바꿔 여러 장을 뽑는 로컬 세팅입니다.

> 레퍼런스 이미지는 직접 만든 OC 이미지만 사용하세요. 실존 인물 사진은 넣지 않습니다.

폰에서 원격으로 쓰는 앱은 [7번](#7-폰-앱-oc-studio--밖에서-원격으로-쓰기)을 보세요.

## 1. 요구 사양

| 항목 | 권장 |
|---|---|
| GPU | NVIDIA VRAM 8GB 이상 (12GB+ 쾌적) |
| Python | 3.10 ~ 3.12 |
| 디스크 | 약 15GB (SDXL 체크포인트 7GB 포함) |
| 기타 | git, curl |

## 2. 설치

```bash
# Linux / macOS
./setup.sh                 # 체크포인트까지 받기
SKIP_CKPT=1 ./setup.sh     # 체크포인트는 직접 넣을 때
```

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File setup.ps1          # 또는 -SkipCkpt
```

설치되는 것:

- `ComfyUI/` 본체 + `ComfyUI_IPAdapter_plus`, `ComfyUI-Manager`
- `models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`
- `models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors`, `ip-adapter-plus-face_sdxl_vit-h.safetensors`
- `models/checkpoints/sd_xl_base_1.0.safetensors` (기본값)

**체크포인트 선택**: OC가 애니/일러스트 그림체면 SDXL 기반 애니 체크포인트(Illustrious·NoobAI·Pony 계열 등),
실사 그림체면 SDXL 실사 체크포인트를 `ComfyUI/models/checkpoints/`에 넣고 `--ckpt 파일명`으로 지정하세요.
SDXL base는 동작 확인용이에요.

## 3. 실행

```bash
cd ComfyUI && source venv/bin/activate && python main.py   # 서버 켜두기 (http://127.0.0.1:8188)
```

다른 터미널에서:

```bash
# img2img — pose.png의 구도·포즈를 살리고 외형은 oc_ref.png, 복장은 outfits.txt
python generate.py --ref input/oc_ref.png --source input/pose.png

# txt2img — 레퍼런스만으로 새 구도, 복장당 3장
python generate.py --ref input/oc_ref.png --seeds 3 --ckpt mymodel.safetensors
```

결과는 `outputs/NN_oc_xxxxx_.png` (NN = outfits.txt 줄 번호)로 저장됩니다.

## 4. 외형/복장 편집

- `character.txt` — 매번 고정할 외형 태그 (머리색, 눈색, 피부, 특징적인 액세서리 등)
- `outfits.txt` — 한 줄에 복장+배경 1종. 줄을 늘리면 그만큼 바리에이션이 늘어나요.

## 5. 튜닝 가이드

| 증상 | 조정 |
|---|---|
| 얼굴이 레퍼런스와 달라짐 | `--ip-weight` 0.9~1.0, `character.txt` 태그 보강 |
| 복장이 레퍼런스 옷으로 끌려감 | `--preset face` (얼굴 위주) + `--ip-weight` 0.7 |
| img2img 결과가 원본과 너무 같음 | `--denoise` 0.7~0.8 |
| 원본 구도가 무너짐 | `--denoise` 0.5~0.6 |
| 그림이 타거나 과포화 | `--cfg` 4.5~5.5 |

레퍼런스는 **얼굴이 크게 보이는 정면~반측면, 단색 배경, 1024px 내외**가 가장 잘 먹혀요.

## 6. 일관성을 더 올리고 싶을 때 — 캐릭터 LoRA

IP-Adapter만으로는 외형이 80~90% 정도 유지돼요. 수십 장 단위로 같은 캐릭터를 꾸준히 뽑을 거면:

1. 이 세팅으로 뽑은 결과 중 외형이 잘 맞은 것 15~30장을 고릅니다 (복장·각도가 다양하게).
2. [kohya_ss](https://github.com/bmaltais/kohya_ss)로 SDXL 캐릭터 LoRA를 학습합니다 (트리거 단어 예: `myoc`).
3. ComfyUI에서 LoRA + IP-Adapter(weight 0.4~0.6)를 같이 쓰면 외형이 거의 고정돼요.

## 7. 폰 앱 (OC Studio) — 밖에서 원격으로 쓰기

PC에서 ComfyUI랑 웹앱 서버를 켜두고, 폰(갤럭시 등)에서 앱처럼 접속해서 쓰는 구성이에요.

```
[갤럭시] ──HTTPS(Tailscale)──▶ [PC] server.py :8787 ──▶ ComfyUI :8188 (GPU)
```

### 7-1. PC에서 실행

```bash
APP_PASSWORD='8자이상비밀번호' ./start.sh                                   # Linux / macOS
```
```powershell
$env:APP_PASSWORD='8자이상비밀번호'; powershell -ExecutionPolicy Bypass -File start.ps1   # Windows
```

서버는 기본적으로 PC 내부(127.0.0.1)에만 열려요. 외부 접속은 아래 Tailscale로만 받아요.

### 7-2. Tailscale로 원격 연결 (포트포워딩 불필요)

1. PC와 폰 모두 [Tailscale](https://tailscale.com/download)을 설치하고 같은 계정으로 로그인합니다.
   (갤럭시는 Play 스토어에서 "Tailscale")
2. 관리 콘솔(login.tailscale.com) → DNS에서 **MagicDNS**와 **HTTPS Certificates**를 켭니다.
3. PC에서 한 번만 실행하면 재부팅 뒤에도 유지돼요:
   ```bash
   tailscale serve --bg 8787
   ```
   출력에 나오는 `https://<PC이름>.<tailnet>.ts.net` 이 접속 주소예요.
4. 폰에서 Tailscale을 켜고 Chrome으로 그 주소에 접속한 뒤 비밀번호를 입력합니다.

`tailscale serve`는 내 Tailscale 계정에 연결된 기기에서만 열려요. 인터넷 전체에 공개하는 `tailscale funnel`은 쓰지 마세요.

### 7-3. 홈 화면에 앱으로 설치 (갤럭시 S25 Ultra)

- **Chrome**: 주소창 오른쪽 ⋮ → **홈 화면에 추가** → **설치**
- **삼성 인터넷**: 하단 ≡ → **현재 페이지 추가** → **홈 화면**

설치하면 주소창 없는 전체화면 앱으로 열려요. HTTPS 주소(ts.net)로 접속해야 앱 설치가 돼요.

### 7-4. 앱 사용법

- **레퍼런스**: OC 외형 이미지예요. 한 번 올리면 서버가 기억해서 다음부터는 안 올려도 돼요.
- **원본**: 선택 사항이에요. 넣으면 img2img(그 구도·포즈 유지), 비우면(×) 새 구도로 그려요.
- **복장 칩**: 여러 개를 선택할 수 있어요. 직접 추가한 칩은 길게 누르면 삭제돼요.
- **결과**: 탭하면 크게 보여요. **저장**은 폰에 다운로드, **원본으로**는 그 결과를 다음 생성의 구도로 써요.
- 상단 점이 초록색이면 ComfyUI가 연결된 상태예요.

같은 와이파이에서만 쓸 거면 `python server.py --host 0.0.0.0` 후 `http://PC내부IP:8787`로도 접속돼요.
다만 이때는 HTTP라 홈 화면 "설치" 대신 바로가기로만 추가돼요.

## 8. 워크플로 파일

`workflows/oc_img2img_api.json`, `workflows/oc_txt2img_api.json`은 ComfyUI API 포맷이에요.
설정을 바꾼 버전을 다시 뽑으려면 `--dump`를 쓰세요:

```bash
python generate.py --ref input/oc_ref.png --source input/pose.png --ip-weight 0.9 --dump workflows/custom.json
```
