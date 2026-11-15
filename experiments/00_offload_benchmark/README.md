# 00. fp16 offload vs NF4 벤치마크

LLaVA-1.5-7B를 Titan V(12GB)에서 **fp16 원본 + CPU offload**로 돌릴 수 있는지, 그리고 **4-bit NF4**와 속도·출력이 얼마나 다른지 잰다. C3 프로젝트의 첫 실험이고, 결과에 따라 본 실험을 fp16으로 할지 NF4로 할지 정한다. 배경은 [HANDOFF.md](../../HANDOFF.md) 3절과 6절에 있다.

## 재는 것

| 항목 | 이유 |
|---|---|
| POPE 질문 1개당 시간 (fp16 offload, NF4) | POPE 전체(9,000문항)를 fp16으로 돌릴 수 있는지 판단 |
| 캡션 생성 토큰당 시간 | CHAIR(500장)를 fp16으로 돌릴 수 있는지 판단 |
| peak VRAM, CPU RAM | 12GB 안에 들어가는지 확인 |
| logit의 NaN/inf 여부 | Volta의 fp16 연산이 안정적인지 확인 |
| 같은 문항에서 fp16과 NF4의 `s = logit(Yes) − logit(No)` 차이 | 양자화가 yes/no 판단을 얼마나 바꾸는지 측정 |

## 비용

| 항목 | 크기·시간 | 근거 |
|---|---|---|
| 모델 `llava-hf/llava-1.5-7b-hf` | **14.13 GB** 다운로드 | HF API로 확인 |
| Python 패키지 (PyTorch 포함) | 다운로드 약 3 GB, 설치 후 약 6 GB | 추정 |
| POPE 질문 파일 + COCO 이미지 20장 | 약 4 MB | 질문 파일 0.37 MB는 확인, 이미지는 추정 |
| 실행 시간 (다운로드 제외) | 약 20–30분 | 추정 |

디스크는 25 GB 이상 비어 있어야 안전하다. GPU 학습은 없고 추론만 한다.

## 실행 방법

저장소를 GPU 컴퓨터로 가져온 뒤, 이 폴더에서 차례로 실행한다.

```bash
cd experiments/00_offload_benchmark

# 1. 환경 만들기 (conda가 없으면: python3.10 -m venv .venv && source .venv/bin/activate)
conda create -n c3bench python=3.10 -y
conda activate c3bench
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 2. 환경 점검 (다운로드 없음, 1분 이내). 모두 OK가 나와야 다음으로 간다
python check_env.py

# 3. 모델 다운로드 (14.13 GB)
huggingface-cli download llava-hf/llava-1.5-7b-hf

# 4. 측정. 한 번에 한 mode씩 돌린다
python bench.py --mode nf4
python bench.py --mode fp16_offload

# 5. 비교와 판정. results/summary.md가 만들어진다
python summarize.py
```

참고
- GPU가 여러 장이면 Titan V가 0번이 되도록 `CUDA_VISIBLE_DEVICES=<번호>`를 명령 앞에 붙인다.
- 홈 디스크가 작으면 3단계 전에 `export HF_HOME=/큰디스크/hf_cache`로 모델 저장 위치를 바꾼다. 4단계에서도 같은 값이 설정돼 있어야 한다.
- 인터넷이 막힌 서버라면 `--synthetic`을 붙이면 무작위 이미지로 속도와 메모리만 잰다. 이때 정확도와 양자화 비교는 의미가 없다.

## 결과 읽는 법

`python summarize.py`가 출력하는 표를 이렇게 본다.

1. **속도와 메모리.** `s/문항`과 `POPE 전체 환산 (h)`이 핵심이다. 환산 시간은 방법 1개·설정 1개당 시간이다.
2. **출력 품질.** 문항이 120개라서 accuracy와 AUROC는 참고용이다.
3. **fp16과 NF4의 출력 차이.** 답 일치율, 평균 |Δs|, yes 비율 차이를 본다. NF4가 yes 비율을 한쪽으로 미는지가 C3에서 가장 중요한 정보다.
4. **판정.** fp16 offload 속도에 따라 자동으로 나온다.

| fp16 offload 속도 | 판정 |
|---|---|
| 1 s/문항 이하 | POPE 전체를 fp16으로 돌린다 |
| 1–3 s/문항 | fp16은 subset에서 기준점·양자화 비교용으로만 쓰고, 본 실험은 NF4로 한다 |
| 3 s/문항 초과 | NF4 위주로 하고, fp16은 500문항에서 양자화 차이만 잰다 |

기준값은 제안값이다. 전체 GPU 예산(40–80시간)에 맞춰 조정할 수 있다.

측정이 끝나면 `results/*.json`과 `results/summary.md`를 커밋하거나 복사해서 가져온다. 그 값으로 제안서의 Resources 항목을 채운다.

## 추가로 해 볼 설정

기본 측정이 끝난 뒤, 시간이 되면 아래를 돌려 본다. `--tag`를 붙이면 결과 파일이 따로 저장된다.

```bash
# GPU에 가중치를 더 많이 올린다. offload되는 layer가 줄어 빨라지지만 메모리 부족 위험이 있다
python bench.py --mode fp16_offload --gpu-mem 10.5GiB --tag 10.5g

# 본 실험(PAI, VAR)은 attention 값을 직접 고쳐야 해서 eager 구현이 필요하다. 그때의 속도
python bench.py --mode fp16_offload --attn eager --tag eager
python bench.py --mode nf4 --attn eager --tag eager

# 양자화 차이를 더 정확히 잰다 (이미지 100장, 두 mode 모두 같은 문항 수로)
python bench.py --mode nf4 --num-questions 600 --tag 600
python bench.py --mode fp16_offload --num-questions 600 --tag 600
```

## 문제가 생기면

| 증상 | 대응 |
|---|---|
| `check_env.py`에서 `sm_70 지원`이 FAIL | 설치된 PyTorch가 Volta를 지원하지 않는 빌드다. 1단계의 `torch==2.5.1` cu118 설치 명령을 다시 실행한다 |
| `check_env.py`에서 CUDA GPU가 FAIL | `nvidia-smi`가 되는지 확인한다. 드라이버가 520보다 낮으면 cu118 빌드가 동작하지 않는다 |
| `bitsandbytes NF4 연산`이 FAIL | 오류 메시지를 그대로 기록해 둔다. NF4를 못 쓰면 계획을 바꿔야 하므로 중요한 정보다 |
| 로딩이나 추론 중 GPU 메모리 부족 | `--gpu-mem 9GiB`처럼 낮춘다. `nvidia-smi`로 다른 프로세스가 GPU를 쓰는지도 확인한다 |
| "disk로 밀려났습니다" 오류 | CPU RAM 예산이 부족하다. `--cpu-mem`을 늘린다. RAM 자체가 16 GB보다 작으면 fp16 offload는 어렵다 |
| NaN/inf 문항이 1개 이상 | 숫자를 기록하고 멈춘다. fp16 overflow일 수 있어서 원인을 먼저 확인해야 한다 |
| COCO 이미지 다운로드 실패 | 잠시 뒤 다시 실행한다. 받은 파일은 `data/`에 남아 있어서 이어서 받는다 |

## 동작 원리

- **fp16 offload.** `device_map="auto"`와 `max_memory`를 주면 accelerate가 GPU 예산(기본 10 GiB) 안에 들어가는 layer만 GPU에 올린다. 나머지 layer의 가중치는 CPU RAM에 둔다. forward가 그 layer에 도달하면 가중치를 GPU로 복사해 계산하고, 끝나면 GPU에서 내린다. 그래서 **계산은 전부 GPU에서 fp16으로** 하고, 느려지는 원인은 매번 가중치를 복사하는 시간이다.
- **NF4.** bitsandbytes가 language model의 Linear layer 가중치를 4-bit로 저장하고, 계산할 때 fp16으로 풀어 쓴다. vision tower와 projector, lm_head는 fp16으로 남긴다(`config.yaml`의 `skip_modules`). 실행 로그의 "Linear 양자화 현황"에서 실제로 그렇게 됐는지 확인할 수 있다.
- **POPE 질문 1개 = forward 1회.** 프롬프트 끝(`ASSISTANT:`) 위치의 logit에서 `Yes`와 `No` 토큰 값을 읽는다. 답을 생성하지 않아도 되므로 offload가 느려도 감당할 만하다. 반대로 캡션 생성은 토큰마다 모든 layer를 다시 지나가서 offload 비용이 토큰 수만큼 곱해진다.

## 검증 상태

이 스크립트는 GPU가 없는 Mac에서 작성했다.

- **확인한 것**
  - 소형 무작위 모델로 CPU에서 전체 흐름을 실행했다 (`--dry-run`): 데이터 다운로드, 프롬프트 구성, forward, 생성, 결과 저장, 요약.
  - 실제 processor로 입력이 627토큰(이미지 576 + 텍스트 51)이 되고, `Yes`/`No` 토큰 id가 3869/1939로 잡히는 것을 확인했다.
  - accelerate 1.3.0 소스에서, GPU가 있을 때 `cpu`로 배치된 모듈의 실행 장치가 GPU로 설정되는 것을 확인했다.
  - transformers 4.49.0 소스에서 `skip_modules` 이름 매칭 규칙을 확인했다.
  - 고정한 패키지 버전이 PyPI와 PyTorch 저장소에 실제로 있는지 확인했다.
- **확인하지 못한 것** (GPU에서 처음 실행됨)
  - 실제 7B 모델의 offload 로딩과 속도
  - Volta에서의 bitsandbytes 동작
  - `check_env.py`의 GPU 검사 부분

처음 실행에서 오류가 나면 메시지를 그대로 남겨 두고 고친다.

## 외부 자원과 AI 사용 기록

- 모델: `llava-hf/llava-1.5-7b-hf` (LLaVA-1.5-7B의 HF transformers 형식)
- 데이터: POPE 질문 파일 ([RUCAIBox/POPE](https://github.com/RUCAIBox/POPE)), COCO val2014 이미지
- 라이브러리: PyTorch 2.5.1, transformers 4.49.0, accelerate 1.3.0, bitsandbytes 0.45.0
- 코드: 이 폴더의 스크립트는 Claude Code로 작성했다.
