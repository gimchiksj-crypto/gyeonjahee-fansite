# 기본 인체 메시

`scripts/build_3d_model.py`가 사진에 맞춰 변형하는 출발점이 되는 인체 데이터.
모두 [MakeHuman](http://www.makehumancommunity.org) 커뮤니티 에셋이며 **CC0**(퍼블릭 도메인)이다.

| 파일 | 내용 |
| --- | --- |
| `base.obj` | MakeHuman 기본 메시(hm08). 얼굴·손가락·발가락·UV 포함 |
| `targets.npz` | 몸형·표정 변형값: 동아시아 여성 20대, 마른 체형, 가슴 크기·탄력, 윙크·눈웃음·미소 (MakeHuman `targets.npz`에서 필요한 것만 추출) |
| `default.mhskel` | 기본 골격(163개 뼈)과 관절 위치 정의 |
| `default_weights.mhw` | 골격 스키닝 가중치 |
