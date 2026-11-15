# 작업 인수인계 (2026-11-15 작성 → 2026-11-16 작업 컴퓨터에서 이어서)

> **다음 세션의 Claude에게.** 이 파일을 [CLAUDE.md](CLAUDE.md) 다음으로 먼저 읽으세요. CLAUDE.md 6절("Topic Discovery")은 **이미 끝난 단계**예요. 지금은 **C3 주제로 제안서를 쓰는 단계**입니다.
>
> 읽는 순서
> 1. 이 파일
> 2. [proposal/C3_proposal_draft_v0.md](proposal/C3_proposal_draft_v0.md): 제안서 초안
> 3. [reports/CV 텀프로젝트 주제 탐색.md](reports/CV%20텀프로젝트%20주제%20탐색.md)의 §1 C3, §3.2: C3 심층 분석
> 4. [research_notes/CV 텀프로젝트 주제 탐색/novelty_hallucination.md](research_notes/CV%20텀프로젝트%20주제%20탐색/novelty_hallucination.md): novelty 조사 근거
>
> 표기 규칙: **[검증]** 1차 출처에서 확인 / **[추정]** 계산·경험에 근거한 추정 / **[가설]** 실험으로 검증할 주장 / **[미검증]** 확인 필요

---

## 1. 현재 상태와 결정 사항

- **주제 탐색 완료** (2026-11-15). 보고서는 C1(DINOv3 positional bias)을 추천했지만, **사용자가 C3를 선택했다.**
- **C3**: LVLM attention-intervention hallucination 완화 방법의 성능 향상이 실제 grounding 개선인지, operating point 이동인지를 감사(audit)한다. 프로젝트 유형은 D(Analysis)와 C(Reproduce & extend)다.
- 사용자는 **(a) "바로 제안서 준비 시작"** 경로를 골랐다. fp16 offload 속도 측정은 제안서와 병행하며, 내일 작업 컴퓨터에서 처음 할 일이다.
- 팀 인원은 **1인으로 가정**했다 [미검증, 사용자 확인 필요].
- 일정 [검증, 강의 PDF]:
  - 제안서: **11/18(수) 23:59 KLMS**, 1페이지 PDF, 배점 5%
  - 구현·실험: 11/18–12/6
  - 발표: 12/7(월) 또는 12/9(수), 1인 기준 10분 발표 + 2분 Q&A
  - 슬라이드 제출: 12/9 23:59, 배점 25%
- 반드시 지킬 규칙 (CLAUDE.md):
  - 사용자 승인 없이 대규모 다운로드·학습·구현을 시작하지 않는다.
  - 다운로드·장시간 실행 전에는 목적과 비용을 먼저 설명한다.
  - 검증된 사실, 추정, 가설을 구분해서 쓴다.
  - 커밋은 사용자가 요청할 때만 한다.

---

## 2. C3 한 장 요약

- **연구 질문.** PAI·VAR 같은 attention 개입 방법의 POPE/CHAIR 개선은 정말 시각적 grounding이 좋아진 결과인가? 아니면 yes 비율이나 캡션 길이 같은 operating point가 이동한 결과인가?
- **선례.** Mirage (Yin, Si, Wang, NeurIPS 2025, arXiv 2504.10020, 코드 `ustc-hyin/cd_rethink`)는 같은 질문을 **contrastive decoding(VCD, ICD, SID)에 대해서만** 다뤘다 [검증].
- **가설 [가설]:**
  - **H1 (POPE).** baseline에 yes 비율만 맞추는 scalar logit bias를 걸면 accuracy 이득의 70% 이상이 복원되고, P(Yes) 기반 AUROC는 거의 변하지 않는다. 복원율이 30% 미만이면 기각한다.
  - **H2 (CHAIR).** 각 방법의 (CHAIR_i, recall) 점은 baseline의 EOS-bias sweep frontier 위나 그 근처에 놓인다. frontier를 엄격히 지배하면 기각한다.
  - **H3 (메커니즘).** 개입 강도를 맞춘 random-head/token 개입이 이득의 상당 부분을 재현한다. 재현하지 못하면, 그리고 GT mask 안 attention의 증가가 정답 여부를 예측하면, "이미지에 attention을 더 준다"는 메커니즘 주장을 지지한다.
- **Positive control.** VCD에서는 Mirage의 결론이 재현되어야 한다. 재현되지 않으면 우리 프로토콜을 의심해야 한다.
- **Novelty 상태 (11-15 조사 기준).** 특정 아이디어 수준에서는 OPEN이고, 방법론 수준에서는 PARTIALLY COVERED다. 가장 가까운 연구는 세 가지다.
  - Mirage
  - Rethinking-CD (arXiv 2607.25196, NeurIPS 2026 workshop용, CD만 다룸, scalar-bias sweep 사용)
  - attention 쪽 통제 연구 두 편:
    - arXiv 2608.24966 (COLM 2026 workshop): CHAIRs 0.370→0.230 사이에 recall 0.78→0.70
    - arXiv 2610.13111: discovered visual head가 random head보다 steering을 더 잘하지 못함

  모두 [검증].
- **어느 결과든 보고 가치가 있다.** 이득이 통제 후에도 살아남으면 "attention 개입은 CD와 다르다"는 증거가 된다. 사라지면 Mirage를 attention 계열로 확장한 결과가 된다.

---

## 3. 핵심 이슈: "7B 모델은 4-bit로만 돌아서 논문 수치를 재현할 수 없다"는 말의 의미

> 2026-11-15 대화에서 사용자에게 설명한 내용을 정리했다.

**한 줄 요약:** Titan V(12GB)에는 7B 모델 원본이 들어가지 않는다. 그래서 4-bit로 압축한 버전을 써야 하고, 그러면 논문 저자들과 "같은 모델"로 실험하는 게 아니게 된다.

### 3.1 왜 4-bit로만 돌아가나

감사 대상 방법(PAI, VAR, OPERA, VCD)은 거의 모두 **LLaVA-1.5-7B**(파라미터 약 70억 개)로 실험했다. 메모리는 파라미터 수 × 파라미터당 바이트 수다.

| 정밀도 | 파라미터당 | 7B 가중치 | Titan V 12GB |
|---|---|---|---|
| fp32 | 4 B | 약 28 GB | ✗ |
| **fp16 (논문 설정)** | 2 B | **약 14 GB** | **✗** (가중치만으로 초과) |
| 8-bit | 1 B | 약 7 GB | 크기는 들어가지만 bitsandbytes LLM.int8이 Volta(CC 7.0)를 지원하지 않음 (CC 7.5+ 필요) [검증] |
| **4-bit (NF4)** | 약 0.5 B | **약 4 GB** | **✓** |

### 3.2 4-bit(NF4) 양자화란

- fp16은 숫자 하나를 65,536가지 값으로 표현하고, 4-bit는 **16가지 값으로 반올림**한다.
- 가중치 64개씩 블록으로 묶고 블록마다 스케일을 따로 둬서 손실을 줄인다.
- 그래도 모든 가중치에 작은 오차가 생기고, 이 오차가 32개 layer를 지나며 누적된다.
- 그래서 **같은 입력에 대한 출력 확률이 조금씩 달라진다.**

### 3.3 "논문 수치를 재현할 수 없다"가 왜 문제인가

- baseline reproduction의 목적은 **내 구현이 맞는지 확인하는 기준점(anchor)**을 얻는 것이다.
  - 예: 논문의 "LLaVA-1.5-7B POPE accuracy X"를 같은 설정에서 X ± 0.5 정도로 재현하면 코드가 맞다는 근거가 된다.
- 4-bit에서 숫자가 다르게 나오면 원인이 **내 버그인지 양자화인지 구분할 수 없다.** 즉 기준점을 잃는다.

### 3.4 C3에서 특히 더 아픈 이유

1. C3가 측정하려는 것은 "개선이 yes/no 확률 분포의 작은 이동인가"인데, **양자화도 바로 그 분포를 조금씩 움직인다.** 측정 대상과 교란 변수가 같은 종류다.
2. 방법들의 하이퍼파라미터(예: PAI α=0.5, γ=1.1)는 fp16 모델 기준으로 튜닝됐다. 4-bit에서는 방법이 원래보다 약하거나 강해 보일 수 있다.
3. 리뷰어가 "4-bit 결론이 fp16에서도 성립하나?"라고 물으면 답할 수 없다.

### 3.5 치명적이지는 않다: 대응책

같은 4-bit 모델 안에서 방법과 대조군을 비교하는 것은 여전히 공정하다. 문제는 검증 기준점과 일반화다.

1. **fp16 + CPU offload (주 대응책, 측정 필요).** 일부 layer를 CPU RAM에 두고 필요할 때 GPU로 옮겨 fp16 원본을 돌린다.
   - POPE는 질문당 forward가 한 번이라 느려도 감당할 만하다. POPE 1개 설정에 약 1–3시간으로 추정한다 [추정, 6절에서 측정].
   - CHAIR는 토큰마다 forward가 필요해 설정당 약 5시간 이상이 걸린다 [추정]. 그래서 CHAIR는 NF4로 돌리거나 fp16은 subset만 쓴다.
2. **양자화를 통제 변수로 바꾸기.** POPE에서 fp16(offload)과 NF4를 모두 돌려 "양자화가 yes 비율과 방법 간 순위를 얼마나 바꾸는가"를 측정·보고한다. 약점을 분석 항목으로 바꾸는 셈이다.
3. **보조 모델로 Qwen2.5-VL-3B (fp16, 약 7.5GB).** 논문 비교 수치는 없다. Qwen2 계열은 fp16 attention overflow로 NaN이 생긴 사례가 있다 (transformers PR #33312) [검증].
4. **12월 선배 서버 (RTX 3090 24GB × 4로 추정) [미검증].** fp16 7B가 들어간다. 확보되면 검증용으로 쓰고, 핵심 계획에는 넣지 않는다.

**판단:** 1번이 실제로 돌아가면 C3의 가장 큰 약점이 많이 해소되고, 실현 가능성 점수가 3점에서 4점으로 오른다. 그래서 **6절의 측정이 가장 중요한 첫 실험이다.**

---

## 4. 11-15에 새로 확인한 사실 (1차 출처 fetch)

### PAI
"Paying More Attention to Image: A Training-Free Method for Alleviating Hallucination in LVLMs", Liu et al., ECCV 2024, arXiv 2407.21771, repo `LALBJ/PAI`.

- **구성 요소가 두 개다** [검증, arXiv HTML]:
  - §4.1 "Pay More Attention to Image": softmax 전 image-token attention을 α로 증폭한다.
  - §4.2 "Image-Centric Logit Refine": `γ·p(y|image,text) − (γ−1)·p(y|text)`로 **text-only 입력과 대비**한다.
- **함의 [가설]:** PAI의 γ 항은 사실상 contrastive decoding이고, Mirage가 비판한 영역에 속한다. 그래서 **α-only와 γ-only로 분해한 ablation**이 C3의 핵심 실험 중 하나가 된다. 제안서에도 반영했다.
- **기본값:** 논문 §5.1은 LLaVA에 α=0.5, γ=1.1을 쓴다. **repo README 예시는 α=0.2라서 논문과 repo가 다르다** [검증]. README의 layer 범위는 `--start-layer 2 --end-layer 32`. 재현할 때 어느 값을 쓸지 기록해야 한다.
- **평가 설정** [검증]:
  - CHAIR: COCO 2014 val에서 500장을 샘플링, `max_new_tokens 512`
  - POPE: COCO 500장, split당 질문 6개(= split당 3,000문항), accuracy/F1
  - 디코딩: greedy, beam(5), nucleus sampling
- **환경 (`environment.yml`)** [검증]: python 3.10.14, **torch 2.0.1**, torchvision 0.15.2, **transformers 4.37.0**, accelerate 0.31.0.
  - POPE 스크립트(`pope_eval.py`, `pope_ans.py`)와 CHAIR 스크립트(`chair_eval.py`, `chair.py`)가 포함돼 있다. 평가 코드는 OPERA 기반이다.
  - transformers의 attention forward를 교체하는 방식으로 개입한다.
- **Titan V 관련 [추정]:** torch 2.0.1은 sm_70(Volta)를 확실히 지원한다. 그래서 PyTorch 2.15의 Volta 지원 중단 문제는 **PAI 환경을 쓰면 피할 수 있다.**

### VAR
"See What You Are Told: Visual Attention Sink in Large Multimodal Models", Kang et al., ICLR 2025, arXiv 2503.03321, repo `seilk/VisAttnSink`.

- **Sink token** [검증]: LLaVA-1.5-7B에서는 sink 차원 {1415, 2533}의 RMS-normalized hidden state 최댓값 ϕ(x)가 τ=20 이상인 visual token이다.
- **Image-centric head** [검증]: visual attention 합이 0.2 이상인 head 중에서 visual non-sink ratio가 ρ 이상인 head다. task별 ρ 값은 [미검증].
- **평가 모델·벤치마크** [미검증, 요약에서 얻음]: LLaVA-1.5-7B/13B, Qwen2-VL-7B, InternVL2-8B 등을 썼고 POPE와 CHAIR가 포함된다.
- **환경** [검증]: torch 2.4.0, transformers는 git commit 고정(`33868a05…`), accelerate 0.34.0. 평가는 LLaVA eval 방식을 따른다.
- **주의:** PAI와 transformers 버전이 달라서 **한 환경에서 두 공식 코드를 같이 돌리기 어렵다** [추정]. 그래서 제안서에서는 "통합 harness 재구현 + 공식 코드와 subset에서 출력 일치 검증"을 own work로 잡았다.

### POPE
- COCO split당 500장 × 질문 6개 = 3,000문항이고, 3개 split을 합치면 9,000문항이다 [검증, PAI 논문 기준]. POPE 원 논문(arXiv 2305.10355)으로 한 번 더 확인하면 좋다.

---

## 5. 제안서 완성까지 남은 작업 (11/16–11/18)

초안은 [proposal/C3_proposal_draft_v0.md](proposal/C3_proposal_draft_v0.md)에 있다. 영어로 썼고 강의 PDF p.9의 필수 항목을 모두 포함한다. 본문이 약 880단어라서 1페이지에 맞추려면 약 750단어로 줄여야 한다 [추정].

### 11/16 (월): 작업 컴퓨터
- [ ] **환경 확인:**
  - `nvidia-smi`: 드라이버 버전, Titan V 인식 여부
  - CUDA, conda/python
  - **CPU RAM 용량** (offload에 32GB 이상 필요 [추정])
  - 디스크 여유 공간
- [ ] **환경 전략 결정.** PAI의 `environment.yml`(torch 2.0.1 + transformers 4.37.0)로 시작하는 것을 권장한다. torch 2.15 이상은 Volta 지원이 중단됐다는 보고가 있다 [미검증].
- [ ] **(사용자 승인 필요) 다운로드:**
  - LLaVA-1.5-7B 가중치 약 14GB [추정]
  - COCO val2014 약 6GB [미검증]
  - POPE 질문 파일 (작음)

  PAI 코드는 원본 LLaVA 형식(`liuhaotian/llava-v1.5-7b`)을 쓰는지, HF 형식(`llava-hf/llava-1.5-7b-hf`)을 쓰는지 repo에서 확인한 뒤 받는다.
- [ ] **fp16 offload 속도 측정** (6절 사양, 약 30분).
- [ ] 측정 결과로 제안서 Resources의 `[TODO: insert the measured fp16-offload speed in s/probe]`를 채운다.

### 11/17 (화)
- [ ] **Novelty 재확인** (약 30분):
  - OpenReview ICLR 2027 submission에서 "hallucination", "PAI", "attention sink", "POPE", "CHAIR", "spurious", "mirage" 검색
  - Google Scholar에서 Mirage의 "Cited by"
  - arXiv 2026년 11월 신규 논문

  attention 개입 계열을 감사한 논문이 새로 나왔으면 차별화 문장을 수정한다.
- [ ] **인용 검증.** 저자·학회를 1차 출처로 확인한다. 아래는 일부가 기억에 의존한 항목이다.
  - VCD: Leng et al., CVPR 2024
  - OPERA: Huang et al., CVPR 2024
  - PAI: Liu et al., ECCV 2024. 학회명은 repo 기준이고, 한 검색 요약은 ICLR 2025로 잘못 표기했다.
  - VAR: Kang et al., ICLR 2025 [검증]
  - ICT: Chen et al., CVPR 2025
  - Mirage: Yin et al., NeurIPS 2025 [검증]
  - POPE: Li et al., EMNLP 2023
  - CHAIR: Rohrbach et al., EMNLP 2018
  - RePOPE: Neuhaus & Hein, arXiv 2504.15707
  - Sandhu et al., arXiv 2608.24966 [검증]
  - Kadir & Sonntag, arXiv 2610.13111 [검증]
- [ ] 이름과 학번을 채운다. 팀 인원도 확인한다.

### 11/18 (수)
- [ ] **1페이지 PDF로 변환.** 이 Mac에는 LaTeX 엔진이 없고 pandoc만 있다(`/opt/anaconda3/bin/pandoc`). Overleaf나 작업 컴퓨터의 LaTeX를 쓴다. 10pt, 여백 약 2cm. 1페이지를 넘으면 Related work의 OPERA/VCD 문장과 Ablations 줄부터 줄인다.
- [ ] 교정 후 **23:59 전 KLMS에 제출.**

---

## 6. fp16 offload 속도 측정 스크립트 사양 (내일 구현)

- **목적.** fp16 원본 LLaVA-1.5-7B로 POPE를 돌릴 수 있는지(재현 기준점 확보), NF4와 얼마나 다른지 판단한다.
- **위치(제안).** `experiments/00_offload_benchmark/`에 config(yaml)와 script, 결과 `results.json`을 둔다.
- **설정:**
  - A. fp16 + `device_map="auto"`, `max_memory={0: "10GiB", "cpu": "<RAM-여유>GiB"}` (accelerate offload)
  - B. NF4: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)`
  - attention 구현: eager 또는 sdpa (fp16, FlashAttention 없음)
- **입력.** POPE-COCO adversarial split의 처음 50문항. 프롬프트는 PAI/OPERA 템플릿과 동일하게 한다.
- **측정 항목:**
  - 질문당 시간(s/probe; warm-up 3개 제외, 중앙값과 p90)
  - peak VRAM (`torch.cuda.max_memory_allocated`)
  - CPU RAM 사용량
  - logit의 NaN/inf 여부
- **비교.** 같은 50문항에서 A와 B의 s = logit(Yes) − logit(No)를 비교한다. 답 일치율, s의 평균 절대 차이, yes 비율 차이를 본다.
- **판정 기준 [제안]** (POPE 전체 9,000문항 기준):

  | fp16-offload 속도 | 판정 |
  |---|---|
  | ≤ 1 s/probe | fp16로 전체 POPE 가능 (설정당 약 2.5시간). 주요 설정을 fp16로 돌린다 |
  | 1–3 s/probe | fp16은 split 1개(3,000문항) 또는 1,000문항 subset에서 anchor·양자화 비교용으로만 쓰고, 본 실험은 NF4로 한다 |
  | > 3 s/probe | NF4 위주로 하고, fp16은 500문항 subset에서 양자화 gap만 측정한다 |

- **호환성 확인.** PAI처럼 attention forward를 교체하는 방식이 accelerate offload hook과 충돌하지 않는지, 1문항으로 확인한다 [미검증 리스크].
- **기록.** seed, torch/transformers/accelerate/bitsandbytes 버전, 드라이버, GPU 이름을 `results.json`에 같이 저장한다.

---

## 7. 주차별 실행 계획 (C3 버전, 11/18–12/6)

| 기간 | 할 일 | 결정 게이트 |
|---|---|---|
| **W0** 11/16–11/18 | 환경 구축, offload 측정, 제안서 완성·제출 | 11/18 제출 |
| **W1** 11/19–11/25 | ① PAI 공식 코드로 vanilla와 PAI의 POPE를 subset에서 재현 ② 통합 harness(HF transformers + hook) 작성 ③ harness의 vanilla/PAI 출력이 공식 코드와 일치하는지 검증 (300문항 답 일치 ≥ 99% [제안]) ④ VCD 추가 | **G1 (11/24):** vanilla POPE acc가 공개 LLaVA-1.5-7B 수치와 ±1.5pt 이내(fp16 기준; NF4만 가능하면 방향만 확인)이고, PAI가 vanilla보다 좋아지는 방향이 재현되면 통과 |
| **W2** 11/26–12/2 | ① VAR 추가 ② T1: 전 방법의 AUROC/AP, matched yes-ratio, scalar-bias fake fix ③ T2: CHAIR 500장(NF4), baseline EOS-bias sweep + PAI + VAR ④ PAI α-only / γ-only 분해 | **G2 (11/29):** 3개 이상 방법의 T1 수치 확보 |
| **W3** 12/3–12/6 | ① T3: magnitude-matched random-head 개입, COCO mask 안 attention 비율 ② ablation(α, layer 범위) ③ paired bootstrap CI ④ 실패 사례 갤러리 ⑤ 슬라이드 초안 | 12/6 결과 고정 |
| 발표 | 12/7 또는 12/9 발표, 슬라이드 12/9 23:59 제출 | |

- **GPU 예산.** 40–80 GPU-시간 [추정, ±2배]. OPERA는 beam search가 무거워서 stretch로 두고, 100장 subset으로만 돌린다.
- **우선순위.** PAI, VAR, VCD가 핵심이고 OPERA, ICT, Qwen2.5-VL-3B는 stretch다.

---

## 8. 성공/실패 기준과 fallback

- **가설 결과는 어느 방향이든 성공이다.** H1–H3가 지지되든 기각되든 보고할 가치가 있다.
- **프로젝트 실패 신호와 대응:**
  1. **공식 코드조차 우리 환경에서 PAI 이득을 재현하지 못하는 경우.** 재현 실패 자체를 분석한다(양자화, 하이퍼파라미터 α=0.5 vs 0.2, 프롬프트 차이). T1/T2는 VCD와 baseline으로 진행해서 Mirage 재현 + 새 지표 적용 형태로 축소한다.
  2. **fp16 offload가 불가능하거나 너무 느린 경우.** NF4 주력으로 가고, 양자화 gap은 subset으로만 측정한다. 리스크는 제안서와 발표에서 명시한다.
  3. **11/24(G1)까지 환경이나 재현이 막힌 경우.** 주제 전환 마감은 11/24다. 전환한다면 C1(DINOv3 positional bias)로 간다. 보고서 §4에 계획이 준비돼 있고, 연산 예산은 20–40 GPU-시간으로 더 작다.

---

## 9. ICCV 2027로 확장하려면 (Stretch)

- **일정.** ICCV 2027은 홍콩에서 열리고, 본회의는 2027-10-05~07이다 [검증]. 논문 마감은 아직 발표되지 않았고, ICCV 2025 패턴으로 보면 2027년 3월 초로 추정된다 [추정].
- **필요한 것 [추정]:**
  - 비양자화 7B 이상 모델 (3090 서버 등 GPU 확보가 전제)
  - CVPR 2026 방법(ACG, EnAR 등)을 포함한 약 10개 방법
  - Qwen3-VL급 최신 모델
  - 프로토콜·코드 공개
  - 메커니즘 설명
- **리스크.** Rethinking-CD 저자들이 "newer methods"를 future work로 명시했다. 그래서 3–4개월 안에 선점될 위험은 중간 수준이다 [추정].
- **단기 vs 장기 contribution:**
  - 수업 프로젝트: 프로토콜 + 2–3개 방법에 대한 감사
  - 논문: 일반화 가능한 발견(예: "attention 개입 이득의 대부분이 operating point 이동이다") + 다수 방법·모델 + 메커니즘

---

## 10. 사용자에게 확인할 것

- [ ] 팀 인원 (1인 가정)
- [ ] 작업 컴퓨터 사양: CPU RAM, 디스크, 드라이버 버전, 원격 접속 방식
- [ ] 다운로드 승인 (LLaVA-1.5-7B 약 14GB, COCO val2014 약 6GB)
- [ ] PDF 작성 도구 (Overleaf/LaTeX 또는 다른 도구)
- [ ] 선배 3090 서버 확보 여부 (12월)

---

## 11. 파일 지도

| 경로 | 내용 |
|---|---|
| [CLAUDE.md](CLAUDE.md) | 연구 파트너 페르소나와 지침. 6절은 완료된 단계 |
| HANDOFF.md | 이 파일. 현재 상태와 다음 할 일 |
| [proposal/C3_proposal_draft_v0.md](proposal/C3_proposal_draft_v0.md) | 제안서 초안 v0 (영어) |
| [reports/CV 텀프로젝트 주제 탐색.md](reports/CV%20텀프로젝트%20주제%20탐색.md) | 주제 탐색 최종 보고서 (C1–C7, 비교표, Top-3 심층 분석) |
| `research_notes/CV 텀프로젝트 주제 탐색/` | 조사 노트 7개 (novelty_hallucination.md, vlm_generation_ct.md, iccv2027_titanv_constraints.md 등) |
| [TermProject_2026Fall.pdf](TermProject_2026Fall.pdf) | 강의 프로젝트 안내 |

**AI 사용 기록.**
- 문헌 조사: Claude Code 서브에이전트가 웹 검색, arXiv, S2, GitHub를 활용했다.
- 작성: 보고서, 이 파일, 제안서 초안을 Claude Code(Claude Opus 5.5)로 작성했다.
- 제출물에는 AI 도구 사용 사실을 반드시 밝힌다 (강의 규정).
