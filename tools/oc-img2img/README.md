# OC img2img — 외형 고정 + 복장 바리에이션

ComfyUI + SDXL + IP-Adapter(Plus / Plus Face)로 **오리지널 캐릭터의 얼굴·헤어 외형을 유지**하면서
복장·배경·구도를 바꿔 여러 장을 뽑는 로컬 세팅입니다.

> 레퍼런스 이미지는 직접 만든 OC 이미지만 사용하세요. 실존 인물 사진은 넣지 않습니다.

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

## 7. 워크플로 파일

`workflows/oc_img2img_api.json`, `workflows/oc_txt2img_api.json`은 ComfyUI API 포맷이에요.
설정을 바꾼 버전을 다시 뽑으려면 `--dump`를 쓰세요:

```bash
python generate.py --ref input/oc_ref.png --source input/pose.png --ip-weight 0.9 --dump workflows/custom.json
```
