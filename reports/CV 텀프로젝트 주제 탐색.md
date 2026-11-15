# DINOv3 위치 편향 분석이 세 조건을 가장 잘 통과한다

> 작성일 2026-11-15(일) · 대상: CTP.49900 Topics in Cultural Technology <Computer Vision> (KAIST GSCT, Prof. Sunghoon Im), 1인 텀프로젝트 · 근거: `research_notes/CV 텀프로젝트 주제 탐색/`의 노트 7개
>
> 표기: **[검증]** 노트가 1차·공식 출처에서 확인한 사실 · **[추정]** 수치 추정(측정값 아님) · **[가설]** 우리가 세운 검증 대상 주장, 또는 저자가 말하지 않은 우리의 추론 · **[미검증]** 노트가 확인하지 못했다고 표시한 항목(그대로 옮김)

---

## 0. Executive summary: C1을 추천하고, 학습 모듈을 하나 덧붙인다

**추천 주제는 C1이다. DINOv3 dense feature의 절대 위치 성분이 semantic correspondence(SPair-71k)에서는 해롭고 temporal correspondence(TAP-Vid point tracking)에서는 이로운지, 그리고 이 성분이 좌우(left/right) 대응 오류를 설명하는지를 검증하는 분석+경량 방법 프로젝트다.** 한 줄 이유: Titan V 한 장으로 3주 안에 확실히 끝낼 수 있고, 아직 아무도 시험하지 않은 반증 가능한 가설이 두 개 있다. INSID3(CVPR 2026 Oral)는 이 위치 부분공간을 제거하면 DINOv3-B의 SPair PCK@0.10이 **45.0에서 50.0으로** 오른다고 보고했다. 그러면서도 위치 정보를 "trade-off"라고 부르고, 위치가 유리할 비디오 쪽은 시험하지 않았다 ([INSID3](https://arxiv.org/html/2603.28480v1)). 2026-11-15 기준 novelty check에서 tracking 쪽 효과(H1)와 좌우 오류 귀인(H2)을 다룬 논문은 찾지 못했다. 적응형 debias(H3)만 FoRIS가 개념적으로 먼저 제안했다 [검증. 단 Google Scholar와 ECCV/NeurIPS 2026 목록은 전수 확인하지 못함]. 필요한 데이터 3종(합계 약 2.7 GB)과 DINOv3 가중치(게이트 없는 timm 버전)는 모두 바로 내려받을 수 있음을 확인했다 [검증]. 연산은 frozen inference 20–40 GPU-시간 수준이다 [추정]. 약점도 분명하다. 상위 세 후보가 모두 inference-only라 학습 루프 경험이 빠진다. 그래서 3주차에 직접 구현해 학습하는 소형 모듈(learned soft projection 또는 regime gate)을 stretch로 붙이기를 권한다. ICCV 2027은 홍콩, 본회의 **2027-10-05~07**로 확정됐다. 논문 마감은 아직 공지되지 않았고, 2025년 패턴으로 보면 2027년 3월 초로 추정된다 ([ICCV 2027 Dates](https://iccv.thecvf.com/Conferences/2027/Dates)).

| 구분 | 선택 | 한 줄 근거 |
|---|---|---|
| **최종 추천** | **C1** DINOv3 위치 성분 × 대응 regime | 실현가능성 최고, 가설 2개 미개척, 부정적 결과여도 발표 가능 |
| (a) 수업용으로 가장 안전 | **C1** (C2가 근소한 차이로 2위) | 등록·승인·양자화 없이 데이터와 가중치 확보, VRAM 6 GB 미만 [추정] |
| (b) CV 연구 경험이 가장 큼 | **C3** LVLM attention-intervention hallucination 감사 | 재구현·통제 설계·통계 분석 밀도가 가장 높음. 단 7B 모델은 NF4로만 돌아 실현가능성이 낮음 |
| (c) ICCV 2027 잠재력 최대 (현재 자원 기준) | **C1** (GPU 제약이 풀리면 C3의 상한이 더 높을 수 있음) | 논문 규모 실험도 Titan V로 가능. C3는 비양자화 7B 이상 모델이 필요 |

---

## 1. Step 1, 후보 발굴: 새 방법 경쟁은 포화됐고, 남은 틈은 측정과 메커니즘이다

사용자 결정에 따라 문화기술(culture-technology) 응용과 3D vision 주제는 후보에서 뺐다. 나머지 영역(feature correspondence/tracking, open-vocabulary segmentation과 ViT artifact, VLM hallucination·효율화, diffusion/flow 편집)의 노트 7개를 종합하면 같은 패턴이 반복된다. 첫째, **"새 방법" 경쟁은 거의 모든 영역에서 포화됐다.** training-free OVSS의 8-benchmark 평균은 2024~2026년에 약 38 → 44.4 → 48.6 → 50.7로 올랐다. 최근 개선폭은 1 mIoU 안팎이라 프로토콜 차이만으로도 뒤집힌다 ([GLA-CLIP](https://arxiv.org/html/2603.23030), [OV-Stitcher, snippet 수준](https://openaccess.thecvf.com/content/CVPR2026F/html/Moon_OV-Stitcher_A_Global_Context-Aware_Framework_for_Training-Free_Open_Vocabulary_Semantic_CVPRF_2026_paper.html)). 비공식 집계로 CVPR 2026에만 hallucination 논문이 33편(그중 약 21편이 training-free 개입)이고 ([papernotes, 비공식](https://en.papernotes.org/CVPR2026/hallucination/)), visual token pruning 논문은 약 30편이다 ([papernotes, 비공식](https://en.papernotes.org/CVPR2026/vlm_efficiency/)). 둘째, 그 반대편에는 **"인기 있는 개입이 정말 주장된 이유로 작동하는가"를 통제된 프로토콜로 재는 틈**이 여러 곳에 남아 있다. 이런 측정은 frozen inference로 할 수 있어서 12 GB Titan V와 3주 일정에 맞는다. 아래 7개 후보는 이 기준으로 골랐다.

### C1. DINOv3의 절대 위치 성분: 같은 개입이 semantic과 temporal correspondence에서 반대로 작용하는가 (유형 D + 경량 A, INSID3 재현이라는 C 요소 포함)

**문제 (Input → Output).** 입력은 source 이미지, target(다른 인스턴스의 이미지 또는 같은 비디오의 다른 프레임), query 점이다. 출력은 target 위의 대응점이며, tracking에서는 프레임별 위치와 가시성이다. 분석 결과물은 위치 부분공간 제거 강도(rank s), 과제 regime, 변위 크기에 따른 정확도 곡선이다.

**핵심 선행연구.**
- **INSID3** ("INSID3: Training-Free In-Context Segmentation with DINOv3", Cuttano et al., **CVPR 2026 Oral**, Apache-2.0 코드). Gaussian noise 이미지를 DINOv3에 넣고 SVD로 상위 s개 방향을 구해 이를 위치 부분공간으로 본 뒤, 특징을 그 직교여공간으로 투영한다. SPair PCK@0.10은 S 43.8→45.7, **B 45.0→50.0**, L 50.6→52.0으로 오르고, DINOv2는 +0.3에 그친다 [검증] ([arXiv HTML](https://arxiv.org/html/2603.28480v1), [code](https://github.com/visinf/insid3)).
- **"Telling Left from Right"** (GeoAware-SC, Zhang et al., **CVPR 2024**). 좌우 대칭 부위가 걸린 geo-aware subset이 SPair 쌍의 82.4%, keypoint의 59.6%를 차지한다. DINOv2+NN의 PCK@0.10은 전체 53.9에서 geo-aware 42.0으로 떨어진다 [검증] ([arXiv HTML](https://arxiv.org/html/2311.17034), [code](https://github.com/Junyi42/GeoAware-SC)).
- **Tracking 기준점.** Chrono ("Exploring Temporally-Aware Features for Point Tracking", Kim et al., **CVPR 2025**) ([arXiv](https://arxiv.org/abs/2501.12218)). frozen DINOv3-B의 zero-shot TAP-Vid-DAVIS δavg는 38.9 ([M2P](https://arxiv.org/html/2603.17813)) 또는 41.3 ([DiTracker](https://arxiv.org/html/2512.20606), 표 파싱 불확실 [미검증])로 보고됐다.

**한계: 저자가 명시한 것 vs 우리가 추론한 것.**
- *저자 명시* [검증]
  - INSID3는 위치 정보를 "trade-off"라고 쓴다(하늘 분할에는 유리). 모든 위치 편향을 제거했다고 주장하지 않는다. 하이퍼파라미터는 COCO-20i에 맞췄다. Gram anchoring이 원인이라는 설명은 검증하지 않은 가설이라고 밝힌다 ([arXiv HTML](https://arxiv.org/html/2603.28480v1)).
  - Unified benchmark 논문은 좌우 대칭 부위 혼동을 미해결 실패 유형으로 적는다 ([arXiv](https://arxiv.org/html/2505.18060)).
  - GeoAware-SC는 좌우 오류를 기하·자세 인식 부족 탓으로 보며, 위치 임베딩은 언급하지 않는다.
- *우리 추론* [가설]
  1. INSID3는 debias가 *도움이 되는* 쪽만 시험했다. 프레임 간 변위가 작아 절대 위치가 유효한 prior인 비디오 쪽은 비어 있다.
  2. 서로 반대 방향을 보는 쌍에서는 위치 누출이 매칭을 거울 쪽 부위로 끌어당길 수 있다.
  3. GeoAware-SC의 test-time flip 정렬은 이미지를 뒤집으면서 절대 위치까지 함께 반전한다. 그 이득의 일부는 위치 재정렬 효과일 수 있다.

**구체 아이디어.**
- **H1**: debias는 SPair를 돕지만 TAP-Vid와 DAVIS propagation에는 중립이거나 해롭다. 해는 작은 변위 track에 몰린다.
- **H2**: geo-aware subset에서 flip 오류 감소는 반대 방향 쌍에 집중된다.
- **H3**: 변위·신뢰도 기반 regime-aware gate가 항상 켜기, 항상 끄기, FoRIS gate를 모두 이긴다.
- 부가 과제: rank s가 정의된 공간을 확인한다(3.1절). 3주차 stretch로 직접 학습하는 소형 모듈을 붙인다(5.1절).

**데이터·지표.**
- SPair-71k: 70,958쌍, 18 카테고리, 약 227 MB, 바로 다운로드 가능 [검증]. 지표는 PCK@0.05/0.10(per-image와 per-keypoint), geo-aware PCK, flip-error rate.
- TAP-Vid-DAVIS: DAVIS val 비디오 30개, 약 1.67 GB [검증]. 지표는 δavg, OA, AJ (query-first).
- DAVIS-2017 trainval 480p: 약 833 MB [검증]. 지표는 label propagation J&F.
- 출처: [TAP-Vid README](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid/README.md), [SPair-71k](http://cvlab.postech.ac.kr/research/SPair-71k/), [DAVIS zip](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip).

**난이도·compute.** 난이도는 낮음~중간이다. ViT-L을 fp16, 960 px로 돌려도 VRAM 6 GB 미만이다 [추정, ±2×]. SPair 특징 추출은 모델·해상도 조합마다 수 GPU-시간, TAP-Vid NN tracking은 설정당 1 GPU-시간 미만이다 [추정]. 전체 20–40 GPU-시간 [추정]. 학습은 없다(선택 모듈 제외).

**12-06까지 최소 결과물.** 세 과제를 하나로 묶은 평가 harness, INSID3 재현표, H1의 rank×변위 곡선, H2 flip-error 표, ablation(rank·layer·해상도·모델 크기), 실패 갤러리.

**학습 가치·논문화.** 특징 분석, 평가 프로토콜의 엄밀성, 선형대수적 개입을 익히는 학습 가치는 높다. 단 학습 루프가 없다. 수업 수준의 기여로는 탄탄하다. 논문 수준이 되려면 "절대 위치 누출이 좌우 flip의 측정 가능한 몫을 설명하고, 변위 prior가 성립할 때 정확히 이롭다"는 일반 원리와, 두 regime을 모두 개선하는 규칙이 함께 나와야 한다.

**주요 리스크.**
- INSID3·FoRIS와 겹친다. INSID3 그룹은 MARCO와 INSID3로 활발하게 연구 중이다.
- 효과가 작을 수 있다(L에서 +1.4).
- "예상된 결과"라는 비판을 받을 수 있다.
- 논문마다 프로토콜이 다르다. 같은 DINOv3-L의 SPair zero-shot이 58.4 / 50.6 / 61.2로 서로 다르게 보고된다 ([Zhang et al.](https://arxiv.org/html/2505.18060), [INSID3](https://arxiv.org/html/2603.28480v1), [DINOv3](https://arxiv.org/html/2508.10104v1)).

**가장 가까운 연구 (novelty check: PARTIALLY COVERED, 대부분 OPEN).**
- **FoRIS** (arXiv 2609.03384, venue 없음). 의미 정렬 점수 s_sem이 0.80 미만일 때만 debias하는 per-pair 적응 방식을 이미 제안했다. 다만 분할 과제가 중심이고, SPair에서는 항상 켠 경우와 똑같이 46.80→52.60이다 [검증] ([arXiv HTML](https://arxiv.org/html/2609.03384)).
- **Preserve-Then-Resolve** (arXiv 2604.23670). layer 19, rank 200 debias가 wide-baseline 매칭에 도움이 된다고 보고한다. 작은 움직임이나 비디오 regime은 다루지 않는다 [검증] ([arXiv HTML](https://arxiv.org/html/2604.23670v2)).
- H1·H2를 시험한 논문은 찾지 못했다 ([novelty 노트의 Semantic Scholar 인용 목록](https://api.semanticscholar.org/graph/v1/paper/arXiv:2603.28480/citations?fields=title,year,venue,externalIds,publicationDate&limit=100)).

### C2. Training-free OVSS 개선은 무엇을 고치는가: test-time registers × 5개 host 요인 실험 (유형 D + 소형 A)

**문제.** 입력은 이미지와 클래스 이름 집합, 출력은 frozen CLIP ViT-B/16 기반의 픽셀별 클래스 맵이다. 진단값(token norm, 이상치 토큰에 몰린 attention 질량, patch-text 유사도 맵)도 함께 산출한다.

**핵심 선행연구.**
- **Test-time registers (TTR)** ("Vision Transformers Don't Need Trained Registers", Jiang et al., **NeurIPS 2025 spotlight**). 소수의 "register neuron"을 추가 토큰 하나로 옮기면 학습된 register를 흉내 낼 수 있다. OpenCLIP의 ImageNet-segmentation mIoU는 34.7→40.0 [검증] ([arXiv HTML](https://arxiv.org/html/2506.08010), [code](https://github.com/nickjiang2378/test-time-registers)).
- **PH-Reg** ("Vision Transformers with Self-Distilled Registers", **NeurIPS 2025 spotlight**). OpenAI CLIP ViT-B/16, 8-benchmark 평균: MaskCLIP 32.21, **MaskCLIP+DVT 30.96**, SCLIP 40.08, ClearCLIP 39.52, NACLIP 39.41, NACLIP+DVT 39.91, PH-Reg 41.85 [검증] ([arXiv HTML](https://arxiv.org/html/2505.21501v3)).
- **Shi et al.** ("Vision Transformers Need More Than Registers", **CVPR 2026**). high-norm 토큰은 "lazy aggregation"의 증상이지 근본 원인이 아니라고 주장한다 [검증] ([arXiv](https://arxiv.org/abs/2602.22394)).

**한계.**
- *저자 명시* [검증]
  - PH-Reg는 TTA에 의존하고, DINOv2에서는 DVT보다 못할 때가 있다. DVT의 "정적 artifact" 가정은 DINOv2에는 맞지만 CLIP에는 맞지 않는다고 적었다.
  - Shi et al.의 해법(LaSt-ViT)은 사전학습 단계에서 적용된다.
  - TTR README는 hook 때문에 forward가 느려진다고 경고한다.
- *우리 추론* [가설]
  - PH-Reg는 OpenAI CLIP의 artifact 토큰이 오히려 *low-norm*이라고 보고했다. TTR은 OpenAI CLIP ViT-B/16을 시험한 적이 없다. 따라서 high-norm 기준으로 neuron을 찾는 TTR이 표준 OVSS 백본에서는 빗나갈 수 있다 ([novelty_ovss 근거: PH-Reg](https://arxiv.org/html/2505.21501v3)).
  - 경쟁하는 설명들(residual/FFN 경로, [CLS] 기반 global patch, anomaly token, lazy aggregation)을 한 프로토콜에서 맞붙인 연구가 없다.

**아이디어.** host {MaskCLIP, SCLIP, ClearCLIP, NACLIP, ProxyCLIP} × fix {없음, TTR, register-neuron 제거, top-k high-norm 마스킹, [CLS]-attention 제거}의 요인 실험. 가설은 3.3절에서 다룬다.

**데이터·지표.** VOC21/20, PC60/59, COCO-Object/Stuff, Cityscapes, ADE20K의 mIoU. 크기는 [미검증]: VOC 약 2 GB, ADE val 약 1 GB, Cityscapes 약 11 GB(등록 필요).

**Compute.** ViT-B/16, 448 px에서 4 GB 미만이고, 8-benchmark 한 번에 0.5–2시간, 약 20회 실행에 1–2 GPU-일이다 [추정]. GLA-CLIP은 모든 실험을 Titan RTX 한 장으로 돌렸다 [검증] ([GLA-CLIP](https://arxiv.org/html/2603.23030)).

**최소 결과물.** 한 프로토콜로 재현한 baseline 4개와 공개 수치 대비 편차, 요인 실험 표, 진단 플롯, 실패 사례, ablation(register 수·개입 layer·임계값).

**학습·논문화.** ViT 내부를 깊이 이해하게 되어 학습 가치가 높다. 논문화 창은 좁다. "TTR이 별로 효과가 없다"는 결론만으로는 Shi et al.의 확인에 그친다.

**리스크.** 프로토콜 편차(SCLIP 평균이 논문에 따라 38.2와 40.08)가 효과 크기보다 클 수 있다. Volta에서 mmcv/mmseg 호환이 확인되지 않았다 [미검증]. Cityscapes는 등록이 필요하다.

**가장 가까운 연구 (novelty: PARTIALLY COVERED, 특정 실험 칸은 OPEN).** PH-Reg, Shi et al., RF-CLIP(AAAI 2026, [arXiv](https://arxiv.org/html/2511.16170v1)). 시간 민감도가 높다. DisenCLIP(ACM MM 2026)은 내용을 확인하지 못했다 [미검증].

### C3. LVLM attention-intervention hallucination 완화는 grounding을 개선하는가, operating point만 옮기는가 (유형 D + C)

**문제.** 입력은 이미지와 yes/no probe("Is there a {object}?") 또는 캡션 프롬프트, 출력은 답이나 캡션이다. 연구 대상은 inference-time 개입 방법군이고, 측정량은 통제된 프로토콜에서의 hallucination이다.

**핵심 선행연구.**
- **Mirage** ("The Mirage of Performance Gains: Why Contrastive Decoding Fails to Mitigate Object Hallucinations in MLLMs?", Yin et al., **NeurIPS 2025**). contrastive decoding의 POPE 이득이 출력 분포의 조잡한 단방향 조정과, sampling을 greedy로 퇴화시키는 효과에서 나온다고 보였다. 감사 대상은 VCD·ICD·SID뿐이다 [검증] ([arXiv HTML v3](https://arxiv.org/html/2504.10020v3)).
- **PAI** (**ECCV 2024**, 공식 repo 기준) ([repo](https://github.com/LALBJ/PAI)).
- **VAR** ("See What You Are Told: Visual Attention Sink in Large Multimodal Models", **ICLR 2025**) ([arXiv](https://arxiv.org/abs/2503.03321)).
- 감사 대상에 ICT (CVPR 2025)와 OPERA (CVPR 2024)도 포함된다.

**한계.**
- *저자 명시* [검증]: Mirage의 결론은 contrastive decoding에만 해당한다. VAR은 image-centric head 안에서만 attention을 재분배한다 ([ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/da8a39bc39ae1c89dd6ebb1e3bcbb3f3-Paper-Conference.pdf)).
- *우리 추론* [가설]: attention 증폭 방법은 캡션을 짧게 만들어 CHAIR를 낮추고, yes 비율을 옮겨 POPE를 올릴 수 있다. 간접 증거는 이렇다.
  - COLM 2026 workshop 논문: CHAIRs 0.370→0.230이지만 object recall 0.78→0.70 [검증] ([arXiv](https://arxiv.org/abs/2608.24966)).
  - 발견된 visual head가 무작위 head보다 steering을 잘하지 못한다 [검증] ([arXiv](https://arxiv.org/abs/2610.13111)).

**아이디어.**
- P(Yes)에서 AUROC를 계산하고, yes 비율을 맞춘 조건에서 POPE를 비교한다.
- EOS-bias 또는 max_new_tokens sweep으로 CHAIR–recall 곡선을 그린다.
- 가짜 수정(fake fix) 통제를 둔다: 양방향 logit bias, EOS boost, 크기를 맞춘 random-head boost.
- COCO mask로 grounding을 확인한다.

**데이터·지표.**
- POPE (COCO 3개 split, split당 약 3,000문항으로 기억 [미검증]), CHAIR (COCO val2014 약 500장, 약 6 GB [미검증]), AMBER, RePOPE 레이블.
- 공식 repo: [POPE](https://github.com/RUCAIBox/POPE), [AMBER](https://github.com/junyangwang0410/AMBER).

**Compute.** LLaVA-1.5-7B는 NF4로 4–5 GB에 activation 1–2 GB, Qwen2.5-VL-3B는 fp16으로 약 7.5 GB다. 전체 40–80 GPU-시간 [추정, ±2×]. OPERA는 beam search 때문에 약 100장 subset으로 제한한다.

**최소 결과물.** baseline과 VCD·PAI·VAR(OPERA는 subset)을 두 모델에서 실행. POPE acc/F1/yes 비율/AUROC, 길이를 통제한 CHAIR 곡선, spurious baseline 2개, per-head attention 그림, 실패 갤러리.

**학습·논문화.** LVLM 내부(attention hook, logit, KV cache)와 통계적 평가를 익히는 학습 가치가 매우 높다. 논문 수준으로는 현 상태에서 중하, 확장하면 중이다.

**리스크.**
- NF4 양자화가 hallucination 비율 자체를 바꿀 수 있다(교란 변수).
- Qwen 계열은 fp16에서 attention overflow가 난 사례가 있다 ([transformers PR #33312](https://github.com/huggingface/transformers/pull/33312)).
- 재구현 결과가 원 논문과 어긋날 수 있다.
- ICLR 2027 투고 풀에 숨은 경쟁작이 있을 수 있다.

**가장 가까운 연구 (novelty: 특정 아이디어는 OPEN, 방법론 수준에서는 PARTIALLY COVERED).**
- Mirage.
- Rethinking-CD (arXiv 2607.25196, NeurIPS 2026 VLM4RWD workshop용). scalar-bias sweep을 쓰지만 여전히 CD만 다루며, "newer methods"를 future work로 남겼다 [검증] ([arXiv](https://arxiv.org/abs/2607.25196)).
- 위의 COLM workshop 논문과 2610.13111.

### C4. Inversion-free flow 편집은 언제 실패하는가: PIE-Bench 위의 FlowEdit 분석과 적응형 n_max (유형 C + D, 경량 A)

**문제.** 입력은 실사 이미지, source 프롬프트, target 프롬프트다. 출력은 target 텍스트를 따르면서 편집하지 않은 구조와 배경은 보존한 이미지다.

**핵심 선행연구.**
- **FlowEdit** ("FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models", Kulikov et al., **ICCV 2025**, oral 목록 등재). 자체 수집한 70+장으로 평가했고 **PIE-Bench는 쓰지 않았다** [검증] ([CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Kulikov_FlowEdit_Inversion-Free_Text-Based_Editing_Using_Pre-Trained_Flow_Models_ICCV_2025_paper.html), [arXiv HTML](https://arxiv.org/html/2412.08629v2), [code](https://github.com/fallenshock/FlowEdit)).
- **PnP Inversion / PIE-Bench** (**ICLR 2024**). 이미지 700장, 편집 유형 10종, 편집 마스크와 평가 스크립트를 제공한다 [검증] ([repo](https://github.com/cure-lab/PnPInversion)).

**한계.**
- *저자 명시* [검증]: FlowEdit은 "큰 영역을 상당히 바꾸는 데 종종 한계"가 있고, "target 분포로의 정확한 사상이 보장되지 않는 heuristic"이다.
- *우리 추론* [가설]: 최적 n_max는 편집 유형마다 다르다. noise 평균(n_avg)은 텍스트 충실도보다 artifact를 줄이는 쪽에 기여한다.

**데이터·지표.** PIE-Bench(구조 거리, 편집 외 영역의 PSNR/LPIPS/MSE/SSIM, CLIP 유사도).

**Compute.** T5를 뺀 SD3-medium은 약 6 GB다. 1024² 이미지 한 장에 1–3분 [추정]. 200장 × 설정 1개에 3–10 GPU-시간, sweep 10개면 30–100 GPU-시간 [추정]이라 빠듯하다.

**최소 결과물.** FlowEdit·ODE inversion·SDEdit을 층화한 200장 subset에서 비교. 유형별·마스크 면적별 분해, n_max/n_avg sweep, 적응형 변형 하나, 실패 갤러리.

**학습·논문화.** rectified flow, CFG, 편집 이론을 배우는 학습 가치는 높다. 논문화는 중간이다(2026년 후속작이 많음).

**리스크.**
- 1024² 해상도의 연산 시간.
- T5를 빼서 생기는 교란.
- SD3 가중치 게이트와 PIE-Bench Google Form 승인에 걸리는 시간을 모른다.
- bf16 기본값을 fp16으로 바꿔야 한다.
- PIE-Bench 해상도(512로 추정)와 SD3(1024)가 다를 수 있다 [미검증].

**가장 가까운 연구 (전용 novelty check는 하지 않음).** SNR-Edit (arXiv 2601.19180, [HTML](https://arxiv.org/html/2601.19180)), EKFEdit (IJCAI 2026), UniEdit-Flow, FlowBypass.

### C5. Feature upsampler는 대응에 도움이 되는가 해가 되는가: semantic vs temporal regime (유형 D)

**문제.** 이미지와 저해상도 DINO 특징을 받아 고해상도 특징을 만들고, 이를 SPair PCK와 TAP-Vid δavg로 평가한다.

**핵심 선행연구.**
- **ViT-Up** (arXiv 2606.14024, venue 없음). DINOv3-S+의 SPair PCK@0.10: bilinear 51.27, **AnyUp 37.63**, JAFAR 36.82, ViT-Up 55.44 [검증] ([arXiv HTML](https://arxiv.org/html/2606.14024v1)).
- **WRC** (arXiv 2605.17472). DAVIS VOS J&F에서는 반대로 AnyUp이 bilinear보다 낫다(71.21 vs 62.52) [검증] ([arXiv HTML](https://arxiv.org/html/2605.17472)).
- **AnyUp** (arXiv 2510.12764). ICLR 2026이라는 2차 출처가 있다 [미검증] ([arXiv HTML](https://arxiv.org/html/2510.12764v2)). LoftUp (ICCV 2025)도 비교 대상이다.

**한계.**
- *저자 명시*: AnyUp은 출력이 저해상도 특징의 선형 결합이라고 가정하고, PE artifact를 제거하지 않는다.
- *우리 추론* [가설]: upsampler들이 매끄러움을 목표로 학습되어 descriptor의 변별력이 떨어진다.

**아이디어.** SPair와 DAVIS에서 결과가 엇갈리는 현상을 매끄러움 대 변별력의 trade-off로 설명하고, point tracking이 VOS 쪽(도움)으로 가는지 SPair 쪽(해)으로 가는지 본다.

**Compute.** inference만 한다. AnyUp backward는 448²에서 약 12.9 GB(H100 기준)라 재학습은 불가능하다.

**최소 결과물.** 공개 upsampler 5–7개를 한 harness에서 비교하고, 변별력 진단을 붙인다.

**학습·논문화.** 학습 가치는 중간, 논문화는 낮다.

**리스크.** 백본별 checkpoint가 없을 수 있다. ViT-Up 결과를 되풀이하는 데 그칠 수 있다. "공개 코드를 그대로 돌린 것"처럼 보일 위험이 있다.

**가장 가까운 연구.** ViT-Up, WRC. TAP-Vid로 평가한 upsampler 논문은 찾지 못했다(전용 novelty check는 아님).

### C6. 12 GB로 DINOv3를 point tracking에 약지도 적응시키기: M2P-lite (유형 C)

**문제.** 입력은 비디오와 query 점, 출력은 프레임별 track과 가시성이다.

**핵심 선행연구.**
- **M2P** (arXiv 2603.17813, venue 없음, **코드 미공개**). LoRA(rank 64)와 VOS 마스크로 학습해 DINOv3-B의 δavg를 38.9에서 53.5로 올렸다. RTX 4090 2장으로 약 하루 걸렸다 [검증] ([arXiv HTML](https://arxiv.org/html/2603.17813)).
- **Chrono** (CVPR 2025)와 **DINO-Tracker** (ECCV 2024, [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3799_ECCV_2024_paper.php)).

**한계.**
- *저자 명시* (M2P): LoRA와 작은 데이터셋이 용량을 제한한다. raw feature는 방해물과 심한 변형에서 실패한다. patch-16이 Kinetics의 δ0를 해친다.
- *우리 추론*: 축소 재현은 원 논문 수치와 공정하게 비교할 수 없다.

**아이디어.** DINOv3-S에 mask-consistency와 Procrustes loss로 LoRA를 학습한다. 데이터는 DAVIS-2017 train(60개 비디오) 또는 YouTube-VOS subset.

**Compute.** 256–384 px, batch 4–8, fp16 LoRA는 12 GB에 들어간다 [추정]. 전체 recipe는 Titan V로 2–4일 [추정]. Chrono를 학습시키려면 273 GB 데이터와 xformers가 필요해 불가능하다.

**최소 결과물.** 1주차에 zero-shot과 Chrono checkpoint 평가, 2주차에 학습 곡선, 3주차에 데이터 규모·loss ablation.

**학습·논문화.** 실제 학습 루프를 돌리므로 학습 가치는 최고다. 논문화는 낮다(M2P, Chrono, DiTracker가 있고, 상당수가 KAIST에서 나옴).

**리스크.** 코드가 없다. xformers가 Volta에서 동작하는지 미확인이다 [미검증]. 학습이 불안정할 수 있다.

### C7. Training-free OVSS의 강건성과 TTA 가산성: 분포 이동에서도 순위가 유지되는가 (유형 D + C)

**문제.** 입력은 corruption·악천후·합성 이미지와 클래스 이름, 출력은 분할 맵이다. 선택적으로 이미지별 LayerNorm TTA를 붙인다.

**핵심 선행연구.**
- **MLMP** (NeurIPS 2025). corruption 15종, ACDC, GTA-V를 포함한 OVSS-TTA 벤치마크다(v2 기준 시나리오 87개, v1은 82개로 서로 다름). NACLIP ViT-L/14의 V20 clean은 75.91, V20-C는 69.01이다. MLMP는 V100에서 peak 약 2.1 GB를 쓰고 파라미터 102,400개만 갱신한다 [검증] ([arXiv HTML](https://arxiv.org/html/2505.21844v2), [code](https://github.com/dosowiechi/MLMP)).
- DAF (arXiv 2608.29923)와 TPA (arXiv 2608.08290).

**한계.**
- *저자 명시*: MLMP의 layer 가중치는 공유 projection head의 entropy에 의존한다. DAF는 entropy minimization이 patch 수준 클래스 붕괴를 낳는다고 보고한다.
- *우리 추론* [가설]: VFM-proxy 계열은 덜 열화된다(H-R1). 분포 이동에서 순위가 보존되지 않는다(Kendall τ < 0.6). host가 강할수록 TTA 이득이 줄어든다(H-T1).

**Compute.** 6 GB 이하. subset만 쓰면 10–20 GPU-시간 [추정].

**학습·논문화.** TTA는 실제 gradient 갱신이 들어가 최적화 루프를 경험할 수 있다. 논문화는 낮다.

**리스크.** TTA-for-OVSS 논문이 5편 이상 있어 붐빈다. ACDC·Cityscapes는 등록이 필요하다. "모든 방법이 비슷하게 열화된다"는 밋밋한 결과가 나올 수 있다.

**가장 가까운 연구 (novelty_ovss Q4).** MLMP, DAF, TPA, TMLR "Multifaceted Evaluation Protocol" ([OpenReview](https://openreview.net/forum?id=6LG7oFBbLH)). "적응 없이 설계 계열을 분포 이동에서 비교"하는 칸은 열려 있지만, 단독 기여로는 작다.

### 검토 후 제외한 방향

다음 방향은 근거와 함께 뺐다.
- **새 supervised/pseudo-label SC adapter**: last-4 block fine-tuning만으로 SPair 90.6이고, annotation-free 방법도 DIY-SC 71.6, 3D-SC 73.0에 이르렀다 ([Zhang et al.](https://arxiv.org/html/2505.18060), [3D-SC](https://arxiv.org/html/2605.30093)).
- **diffusion-feature SC**: SD2.1+DINOv3(58.1)가 DINOv3 단독(58.4)보다 낮다는 보고가 있다.
- **새 feature upsampler 학습**: 경쟁작이 8개 이상이고 AnyUp backward가 약 12.9 GB다.
- **video-DiT 기반 tracking**: CogVideoX-5B가 12 GB에 들어가지 않고, 모델이 bf16 기본이다.
- **dense matcher 학습**: 다중 GPU가 필요하고, 핵심 분석 질문은 RoMa-Ω가 이미 답했다.
- **새 training-free OVSS 기법**: 포화됐다.
- **PH-Reg/DVT 학습**: 150–365 GPU-시간.
- **새 hallucination 디코딩이나 token pruning 기법**: 포화됐고, pruning에서는 random 선택이 자주 경쟁력이 있다 ([ACL Findings 2025](https://aclanthology.org/2025.findings-acl.802/)).
- **style injection**: Scheduled Style Injection(NTIRE 2026)이 가장 뻔한 분석을 먼저 했다 ([arXiv](https://arxiv.org/abs/2605.26538)).
- **FLUX·비디오 diffusion**: fp16으로 12 GB에 들어가지 않는다.

---

## 2. Step 2, 비교 평가: 단순 합계는 C1과 C3가 동점이고 Titan V 실현가능성이 순위를 가른다

점수(1–5)는 노트의 근거를 바탕으로 한 **주관적 판단 [추정]**이다. 가중합은 사용자 우선순위를 반영해 학습가치(1순위)와 실현가능성(2순위, 필수 조건)에 ×2, 나머지에 ×1을 주었다(최대 45점).

| 후보 | 연구가치 | 학습가치 | 실현가능성 | 신규성 | 평가명확성 | 커리어 | 확장성 | 단순합(35) | 가중합(45) | 순위 |
|---|---|---|---|---|---|---|---|---|---|---|
| **C1** DINOv3 위치 × regime | 4 | 4 | **5** | 4 | 4 | 4 | 4 | 29 | **38** | **1** |
| **C3** Hallucination 개입 감사 | 4 | **5** | 3 | 4 | 4 | **5** | 4 | 29 | 37 | 2 |
| **C2** OVSS artifact × host | 4 | 4 | 4 | 3 | 4 | 4 | 3 | 26 | 34 | 3 |
| C7 OVSS 강건성 × TTA | 3 | 4 | 4 | 2 | 4 | 3 | 3 | 23 | 31 | 4 |
| C6 M2P-lite tracking 적응 | 3 | **5** | 2 | 2 | 4 | 4 | 3 | 23 | 30 | 5 |
| C5 Upsampler × regime | 3 | 3 | 4 | 2 | 4 | 3 | 2 | 21 | 28 | 6 |
| C4 FlowEdit PIE-Bench | 3 | 4 | 2 | 3 | 3 | 4 | 3 | 22 | 28 | 7 |

**C1**
- 실현가능성 5점의 근거가 가장 단단하다. 데이터와 가중치 링크가 모두 HTTP 200으로 확인됐고, 등록·승인·양자화가 필요 없다 ([novelty_dinov3 노트의 링크 확인](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m)).
- 학습가치는 4점에 그친다. 핵심 개입이 약 30줄짜리 투영이고 학습 루프가 없다.
- 평가명확성도 4점이다. 같은 모델의 SPair zero-shot이 논문마다 58.4/50.6/61.2로 다를 만큼 프로토콜에 민감하다.

**C3**
- 학습가치 5점: 공개 방법을 다른 모델에 재구현하고, fake-fix 통제를 설계하고, AUROC와 bootstrap 통계, per-token grounding 분석까지 한다. 커리어 5점: VLM 수요가 크다.
- 실현가능성은 3점이다. LLaVA-1.5-7B는 fp16으로 들어가지 않아 NF4만 가능하다(LLM.int8은 sm_70 미지원) ([bitsandbytes docs](https://huggingface.co/docs/bitsandbytes/main/en/installation)). 그래서 **공개 수치를 그대로 재현했는지 확인할 수 없다.**

**C2**
- 실험 칸 자체는 비어 있다.
- 그러나 "TTR이 거의 효과가 없다"는 결과가 Shi et al.의 확인처럼 읽힐 위험 때문에 신규성은 3점이다.
- 8개 데이터셋 확보(Cityscapes 등록)와 mmseg 환경 문제로 실현가능성은 4점이다.

**C7·C6·C5·C4**
- C7과 C6는 각각 TTA 붐비는 영역과 코드 없는 재구현이 약점이다.
- C5와 C4는 가중합이 같다. 필수 조건(2순위)을 기준으로 실현가능성이 높은 C5를 앞에 두었다.
- C4는 연산 시간과 승인 대기로 기한 위험이 가장 크다.

**세 가지 지정.**
- **(a) 수업용으로 가장 안전: C1.** C2와의 차이는 데이터 접근(등록 필요 여부)과 환경 위험이다. C5는 C1만큼 안전하지만, 공개 코드를 그대로 돌린 것처럼 보일 위험 때문에 채점상 얇다.
- **(b) CV 연구 경험이 가장 큼: C3.** 연구의 핵심인 "통제 설계와 반증"이 가장 촘촘하다. 대가로 NF4 교란과 fp16 overflow라는 기한 위험을 진다. 학습 루프 경험 자체만 보면 C6가 가장 크지만, 필수 조건을 위협해 추천하지 않는다.
- **(c) ICCV 2027 잠재력 최대: C1 (현재 자원 기준).** 가설 두 개가 미개척이고, 논문 규모 확장(백본·regime 추가)도 frozen feature로 Titan V에서 가능하다. C3는 선례(Mirage, NeurIPS 2025 본트랙)가 있고 독자층이 넓다. 하지만 리뷰어는 비양자화 7B 이상 모델을 기대할 것이므로 GPU가 확보될 때만 상한이 열린다.

---

## 3. Step 3, Top-3 심층 분석: 세 후보 모두 "인기 개입이 정말 그 이유로 작동하는가"를 묻는다

### 3.1 C1: DINOv3 위치 성분 × 대응 regime

**핵심 가설 (반증 가능).**
- **H1**: 같은 rank s에서 debias는 SPair PCK@0.10을 올리지만 TAP-Vid-DAVIS δavg는 낮춘다. 하락은 GT 변위 하위 1/3 구간에 몰린다. 작은 변위 구간에서 debias 효과의 95% CI가 0 이상이면 기각한다.
- **H2**: geo-aware keypoint의 flip-error 감소량을 "반대 방향 쌍 − 같은 방향 쌍"으로 계산한 difference-in-differences가 0보다 크다. 차이가 없으면 "좌우 오류는 위치 탓이 아니다"로 결론 낸다.
- **H2b**: GeoAware-SC flip 정렬의 이득은 debias 후에 줄어든다. 그렇다면 그 이득의 일부가 위치 재정렬이라는 뜻이다.
- **H3**: 변위나 신뢰도로 debias 강도를 조절하는 gate가 두 regime에서 동시에 최선의 고정 설정에 근접하면서 FoRIS의 s_sem gate를 이긴다.

모두 [가설]이다. 특히 H1은 부호가 뒤집힌다는 *양방향* 예측이라 "이득이 줄어든다"는 단방향 감사보다 반증력이 강하다. 이 점이 C1의 방법론적 장점이다.

**Baseline 재현 가능성.**

| 항목 | 공개 자원 | 재현할 공개 수치 | 주의 |
|---|---|---|---|
| SPair raw NN + INSID3 debias | [visinf/insid3](https://github.com/visinf/insid3) (Apache-2.0, keypoint matching 스크립트) | DINOv3-B PCK@0.10 **45.0→50.0** (논문, s=500) / **46.8→52.6** (README, s=20, 768 px) [검증] | 공식 `.pth`를 기대하므로 timm 가중치를 쓰려면 loader를 고쳐야 함 |
| SPair 고해상도 zero-shot | [Visual-AI/Semantic-Correspondence](https://github.com/Visual-AI/Semantic-Correspondence) | DINOv3-L 58.4 @960, per-image [검증] | RTX 3090 한 장으로 실행된 코드 |
| Geo-aware subset, flip test | [GeoAware-SC](https://github.com/Junyi42/GeoAware-SC) (`--COMPUTE_GEOAWARE_METRICS`, `pose_awareness_test.py`) | DINOv2+NN 53.9 / geo-aware 42.0 [검증] | torch 1.13.1 환경이라 지표 코드를 자체 harness로 이식하는 편이 낫다 |
| TAP-Vid-DAVIS zero-shot | tapnet 평가 코드 | DINOv3-B δavg 38.9 (M2P) / 41.3 (DiTracker) | 두 논문의 프로토콜이 다름. DiTracker는 soft-argmax, 30×52 특징 해상도 |
| DAVIS-2017 label propagation | 자체 구현 필요 (DINOv3 평가 코드 공개 여부 미확인) | DINOv3 distilled B 77.2 / L 79.9 J&F @960 ([DINOv3](https://arxiv.org/html/2508.10104v1)) | 다른 노트에는 83.3이라는 수치도 있어 충돌 [미검증] |

**반드시 1일차에 확인할 의문점.** 노트에 따르면 INSID3 논문은 모든 데이터셋에 s=500을 고정했다. 그런데 ViT-S의 채널 차원은 384라서, 채널 공간에서 rank 500 투영은 정의상 불가능하다. 투영이 채널 공간이 아니라 토큰(공간) 축에서 정의됐을 가능성이 있다. 그렇다면 features 노트의 "ViT-S에서 의미 신호를 과다하게 제거한다"는 추론은 성립하지 않는다. 또 부분공간이 토큰 수, 즉 해상도에 묶이므로 해상도마다 다시 계산해야 한다 [미검증]. 코드로 확인한 뒤 H4(width로 정규화한 rank sweep)의 형태를 정한다.

**최소 실험 설계.**
- **독립변수**
  - rank s ∈ {0, 20, 200, 500} (정의 공간 확인 후 비율로도 표기)
  - 모델 ∈ {DINOv3 S/B/L, DINOv2-B 대조군}
  - layer ∈ {마지막, 중간(예: Preserve-Then-Resolve의 layer 19)}
  - 해상도 ∈ {512, 768, 960}
  - regime ∈ {SPair 교차 인스턴스, TAP-Vid temporal, DAVIS propagation}
  - 프레임 간격 Δt와 GT 변위 구간
- **종속변수**: PCK@0.05/0.10(per-image·per-keypoint), geo-aware PCK, flip-error rate, δavg, OA(cycle-check 가시성), AJ, J&F.
- **통제**
  - **같은 rank의 무작위 부분공간 투영.** "아무 s개 차원을 지워도 생기는 효과"를 걸러낸다. 가장 중요한 통제다.
  - DINOv2. 위치 성분이 약하다는 음성 대조군이다(+0.3).
  - 모든 baseline을 같은 harness에서 다시 실행한다.
  - 비디오 단위 bootstrap CI. TAP-Vid는 비디오가 30개뿐이라 track 단위 CI는 과신을 낳는다.
- **Ablation**: noise 이미지 수(1 vs 5; INSID3는 1장 57.6, 5장 57.7 mIoU), noise 종류(Gaussian vs FoRIS식 zero-input), rank, layer, 해상도, 모델 크기.

**예상 실패 원인.**
1. 효과 크기가 작다(INSID3의 L은 +1.4). TAP-Vid의 오차는 patch-16의 거친 해상도가 지배할 수 있다.
2. DINOv3는 RoPE(상대 위치)를 쓴다. 그런데도 관측되는 "절대 좌표" 활성의 메커니즘이 불분명하다. noise-SVD 부분공간이 tracking에 쓰이는 위치 신호와 다를 수 있다.
3. 같은/반대 방향 판정이 잡음일 수 있다. SPair viewpoint 주석의 형식은 미확인이다 [미검증]. 대안으로 좌우 짝 keypoint가 모두 보이는 쌍에서 x좌표 차의 부호로 방향을 정하는 방법이 있다 [가설].
4. fp16 수치 문제.
5. 원 논문 대비 프로토콜 불일치.

**가장 가까운 연구와의 차별화.**
- **INSID3**: ICS와 SPair만 다루고 tracking·좌우 분석이 없다.
- **FoRIS**: 의미 유사도 gate, 분할 중심이고 SPair에서는 AD와 OD가 같다.
- **Preserve-Then-Resolve**: wide-baseline 매칭에서만 시험했고 결과는 개선이다.
- **"What DINO saw"**: linear probe와 분할만 다룬다 ([arXiv](https://arxiv.org/html/2603.16840)).
- **"Geometry without Position?"**: stereo에서 PE를 0으로 만들거나 섞는 실험이다. 부분공간 제거나 tracking은 없다. DINOv2에서 PE를 0으로 만들면 EPE가 12.64→38.08로 나빠진다 ([arXiv](https://arxiv.org/html/2601.22231)).
- 리뷰어가 "INSID3의 trade-off 문장으로 예상되는 결과"라고 할 위험에 대비해야 한다. **변위 축에서 crossover 지점을 정량화하고, H2로 메커니즘을 보이는 것**이 단순한 sanity check와 이 연구를 가르는 요소다.

**장기 확장.** 백본 계열 확대(DINOv3 H+는 fp16 약 1.7 GB로 들어감, 7B는 불가), regime 확대(AP-10K, SPair-U, TAP-Vid-Kinetics), 위치 성분을 분리하는 학습형 투영, 실제 tracker나 SC 파이프라인에 넣었을 때의 효과, "절대 위치가 유효한 prior가 되는 조건"의 이론화.

### 3.2 C3: LVLM attention-intervention hallucination 감사

**핵심 가설 (반증 가능).**
- **H1**: PAI와 VAR의 P(Yes) 기반 AUROC 이득은 accuracy 이득보다 훨씬 작다. baseline에 yes 비율만 맞추는 scalar logit bias를 걸면 accuracy 이득의 대부분이 복원된다. 판정 예: 복원율이 70% 이상이면 지지, 30% 미만이면 기각 [가설·임계값은 제안].
- **H2**: 방법들의 (CHAIR_i, recall) 점이 baseline의 EOS-bias sweep 곡선 위나 그 근처에 놓인다. 곡선을 엄격히 지배하면 기각한다.
- **H3**: 효과 있는 개입은 hallucination이 나는 위치에서 GT 마스크 영역으로 가는 attention을 선택적으로 높인다. 크기를 맞춘 random-head boost가 같은 이득을 내면, 메커니즘 주장은 기각한다.

**Baseline 재현 가능성.** 공식 repo는 VCD, OPERA, PAI, VAR(VisAttnSink), Mirage의 cd_rethink(PBA/OLM/APC 통제와 POPE acc/F1/yes 비율 스크립트), POPE, AMBER가 존재한다 [검증] ([cd_rethink](https://github.com/ustc-hyin/cd_rethink), [VisAttnSink](https://github.com/seilk/VisAttnSink)). 다만 평가 스크립트 내용과 Qwen2.5-VL 지원 여부는 미확인이다 [미검증]. 공개 수치는 fp16 LLaVA-1.5 기준이라 **NF4에서는 절대값을 재현할 수 없다.** 재현 기준은 같은 NF4 모델에서 방법 간 이득의 방향과 순위, 그리고 Mirage가 보고한 "POPE에서 greedy가 sampling보다 낫다"는 정성적 현상으로 잡아야 한다.

**최소 실험 설계.**
- **독립변수**: 방법(greedy baseline, VCD, PAI, VAR, OPERA subset), 개입 강도 α, 통제(양방향 logit-bias sweep, EOS boost, 크기를 맞춘 random-head boost), 모델(LLaVA-1.5-7B NF4, Qwen2.5-VL-3B fp16).
- **종속변수**: split별 POPE acc/F1/yes 비율/AUROC/AP, CHAIR_s/CHAIR_i/recall/길이, attention 질량.
- **통제**: 모든 방법에 같은 양자화 모델, greedy decoding, RePOPE 레이블로 강건성 확인([RePOPE](https://arxiv.org/html/2504.15707v1): "Yes" 항목의 9.3%가 오류).
- **Ablation**: α, 개입 layer 범위, head 선택, max_new_tokens.

**예상 실패 원인.** NF4가 hallucination 비율을 바꾸는 교란, Qwen의 fp16 inf→NaN, OPERA의 메모리와 시간, Qwen으로 재구현할 때의 drift. CHAIR 레이블 자체도 신뢰도 문제가 있다(">50%의 hallucination 판정이 부당", [Feng et al.](https://arxiv.org/pdf/2406.12663)).

**차별화.**
- Mirage: CD만, AUROC 없음, 길이 통제 없음.
- Rethinking-CD: scalar-bias sweep을 쓰므로 반드시 인용하고 대비시켜야 한다. CD만 다룬다.
- COLM workshop 논문: 자기 방법에만 길이 통제를 했다.
- 2610.13111: head 발견 점수 분석이며 PAI·VAR 감사는 아니다.

**장기 확장.** CVPR 2026 방법(ACG, EnAR 등)을 포함해 약 10개 방법, Qwen3-VL급 모델, 프로토콜 공개, 메커니즘 설명. 모두 비양자화 7B 이상을 돌릴 GPU가 필요하다.

### 3.3 C2: OVSS artifact fix × 다중 host

**핵심 가설 (반증 가능).**
- **H0** (1주차 전제 확인): TTR의 register-neuron 탐색이 OpenAI CLIP ViT-B/16에서 high-norm 이상치를 찾는가? PH-Reg의 low-norm 관찰이 맞다면 찾지 못할 것이고, 그 자체가 보고할 만한 발견이다.
- **H1**: MaskCLIP에 TTR을 적용해도 MaskCLIP→ClearCLIP 격차(약 7 mIoU)의 50% 미만만 복원된다.
- **H2**: TTR 이득은 ClearCLIP·SCLIP에서 0에 가깝고, NACLIP·ProxyCLIP에서는 양수다.
- **H3**: 이미지별 이득과 이상치 attention 질량의 Spearman ρ가 0.3을 넘는다.
- 실패 기준: TTR이 ClearCLIP 이득의 70%를 넘게 복원하면 "attention 재설계" 설명이 약한 쪽이다. 이 경우도 반대 방향의 놀라운 결과로 보고할 수 있다.

**Baseline 재현 가능성.** PH-Reg 표의 수치(MaskCLIP 32.21, SCLIP 40.08, ClearCLIP 39.52, NACLIP 39.41)가 외부 기준점이다. GLA-CLIP은 같은 방법을 4가지 프로토콜 설정으로 평가했고 SCLIP 38.2, ClearCLIP 38.1, NACLIP 39.4, ProxyCLIP 42.3을 보고했다 [검증]. repo 상황: ClearCLIP은 확인했고, SCLIP·ProxyCLIP은 aggregator로만 확인했으며, NACLIP repo 소유자는 미확인이다. TTR repo에는 평가 스크립트가 없다 [검증].

**최소 실험 설계.**
- host 5개 × fix 5개 × 8 benchmark(시간이 부족하면 V21·PC59·C-Stf·City·ADE 5개).
- 프로토콜은 하나로 고정한다(예: SCLIP 설정).
- 진단은 노이즈를 피하려고 fp32로 한다. high-norm activation이 fp16에서 overflow할 수 있다.
- attention-sink 진단 도구를 함께 쓴다 ([arXiv 2606.08105](https://arxiv.org/abs/2606.08105)).

**예상 실패 원인.** TTR 효과가 평평하게 나와 Shi et al.의 확인처럼 보이는 경우. 프로토콜 편차가 효과보다 큰 경우. mmcv 환경. Cityscapes 등록.

**차별화.** PH-Reg(학습형, TTR·ProxyCLIP 없음), Shi et al.(사전학습 단계 수정, raw CLIP만), RF-CLIP(register 없음, 단일 host).

**장기 확장.** SigLIP2와 DINOv3 dino.txt로 백본을 넓히고, 원리에 기반한 수정법을 제안하고, MLMP 프로토콜로 강건성 축을 추가한다.

---

## 4. Step 4, 최종 추천: C1을 고르고 3주차에 직접 학습하는 소형 모듈을 붙인다

### 4.0 우선순위별 정당화

**1순위(연구 경험).** C1은 연구 사이클을 빠짐없이 거친다.
- regime 의존성이라는 문제를 정의한다.
- 부호가 뒤집힌다는 양방향 예측이 있는 반증 가능한 가설을 세운다.
- 구체적 공개 수치(45.0→50.0, 38.9–41.3)를 기준으로 baseline을 재현한다.
- 세 과제를 아우르는 자체 PyTorch harness를 만든다.
- 무작위 부분공간과 DINOv2라는 통제를 둔다.
- ablation과 flip 실패 갤러리로 실패를 분석한다.

빠진 것은 학습 루프 하나다. 5.1절의 소형 모듈로 메운다. C3가 재구현 밀도에서 약간 앞서지만, 그 차이는 소형 모듈로 대부분 상쇄된다.

**2순위(필수 조건).** 상위 후보 중 C1만이 모든 데이터와 가중치를 등록·승인 없이 즉시 확보할 수 있음이 확인됐다. 양자화나 bf16 기본 모델도 쓰지 않는다. 연산 예산(20–40 GPU-시간)도 C3(40–80)의 절반이다.

**3순위(ICCV).** 가설 두 개가 미개척이고, 논문 규모로 확장해도 Titan V로 가능하다. 반대 요인은 "예상된 결과" 비판과 선점 위험이다(4.5절).

### 4.1 (a) 제안서 준비: 11-15 → 11-18

| 날짜 | 목표 | 구체 작업 | 완료 기준 |
|---|---|---|---|
| **11-15 (일, 오늘)** | 환경, 접근 신청, novelty 재확인 | ① `nvidia-smi`로 드라이버 R580 확인 → conda(Python 3.11)에 `torch==2.14.1` cu126 설치 → `torch.cuda.get_arch_list()`에 `sm_70`이 있는지 확인 ② 공식 DINOv3 가중치 신청(병행. 1순위는 게이트 없는 timm) ③ SPair-71k, TAP-Vid-DAVIS, DAVIS-2017 다운로드 시작(약 2.7 GB) ④ novelty 재검색(5.2절 체크리스트) | sm_70 확인, 다운로드 진행 중 |
| **11-16 (월)** | Smoke test | timm DINOv3 S/B/L 로드. 이미지 50장에서 fp16과 fp32 특징의 cosine이 0.999를 넘는지 확인. 512/768/960 px의 peak VRAM 측정. SPair 50쌍에서 NN PCK 계산. **noise-SVD debias를 직접 구현(약 30줄)**하고 s가 정의된 공간 확인. INSID3 3절과 부록 A.4, GeoAware-SC의 geo-aware 정의, FoRIS의 AD 다시 읽기 | 첫 수치가 나옴 |
| **11-17 (화)** | 파일럿과 초안 | SPair test 500쌍에서 B의 raw vs debias 비교(INSID3의 +5.0 방향과 일치하는지). TAP-Vid-DAVIS 비디오 3–5개에서 NN tracking raw vs debias를 Δt 구간별로 비교. 파일럿 그림 1장. 제안서 초안 | 파일럿 수치 2개 |
| **11-18 (수)** | 제출 | 1페이지로 다듬기, related work의 venue와 링크 재확인, AI 사용 고지 문구, **18:00까지 KLMS 업로드**(23:59 마감 대비 여유) | 제출 완료 |

**1페이지 제안서 개요 (C1용 초안).**
1. **Title & Member**: "Position as Prior or Bias? Dissecting the Absolute-Positional Component of DINOv3 Features across Semantic and Temporal Correspondence". 1인(이름·학번).
2. **Problem (Input → Output)**: (source image, target image or video frame, query points) → corresponding target points(+visibility). 분석 대상: debias 강도 × regime × displacement에 따른 accuracy.
3. **Motivation**: frozen DINOv3 dense feature는 교차 인스턴스 매칭과 비디오 tracking 양쪽에 쓰인다. INSID3는 위치 부분공간 제거가 SPair를 +5.0 개선하지만 "trade-off"라고 했다. 언제 debias해야 하는지에 대한 규칙이 없다. 좌우 혼동은 여전히 주요 실패 유형이다.
4. **Related work (top venue, 최근 3년)**: INSID3 (CVPR 2026): 방법과 빠진 것(tracking·좌우 분석 없음). Telling Left from Right (CVPR 2024): geo-aware 평가와 flip 정렬, 위치 요인은 고려하지 않음. Chrono (CVPR 2025) / DINO-Tracker (ECCV 2024): DINO 특징 기반 tracking, 위치 분석 없음. 보조로 FoRIS (arXiv 2026): 적응형 gate, 분할 중심.
5. **Idea**: H1/H2/H3를 각각 한 문장으로. 무작위 부분공간 통제. stretch로 learned soft projection 또는 regime gate.
6. **Evaluation plan**
   - 데이터셋: SPair-71k(전체와 geo-aware), TAP-Vid-DAVIS, DAVIS-2017.
   - baseline: raw NN, INSID3 debias, FoRIS gate, GeoAware flip 정렬, DINOv2.
   - 지표: PCK@0.05/0.10, flip-error, δavg/OA/AJ, J&F, 비디오 단위 bootstrap CI.
   - 파일럿 수치(있으면 1줄).
7. **Resources**: Titan V 12 GB(1장). timm DINOv3 S/B/L(게이트 없음, DINOv3 License). INSID3 코드(Apache-2.0)는 참조용이고 debias와 harness는 직접 구현. tapnet 평가 코드. 1인이 모든 역할 수행. AI 도구(Claude Code: 문헌 조사·코드 보조) 사용 고지.

### 4.2 (b) 주차별 실행 계획: 11-18 → 12-06, 그리고 발표

| 기간 | 주제 | 산출물 | 판정 시점 |
|---|---|---|---|
| **1주차** 11-18(수)–11-24(화) | **Baseline 재현** | config 기반 harness(SPair PCK·geo-aware·flip-error, TAP-Vid δavg/OA/AJ, DAVIS J&F). INSID3 B 재현. DINOv3-B TAP-Vid zero-shot. DINOv2-B 대조군. 무작위 부분공간 통제. seed·환경·로그 기록 | **11-24 재현 게이트** (4.3절) |
| **2주차** 11-25(수)–12-01(화) | **주 실험** | H1: rank × Δt/변위 × 모델 sweep. H2: flip DiD와 GeoAware flip 교란 검정. 11-28(토)에 중간 결과 그림. FoRIS gate baseline 구현과 H3 규칙(변위 gate) 초안 | **12-01 가설 판정** |
| **3주차** 12-02(수)–12-06(일) | **Ablation·분석·슬라이드** | layer·해상도·rank·noise 수와 종류 ablation. 소형 학습 모듈(5.1절, **12-01까지 H1/H2 실행이 끝났을 때만**). 실패 갤러리. 통계 검정. 12-06까지 슬라이드 초안 | 12-05 H3 판정 |
| **발표** 12-07(월) 또는 12-09(수) | 10분 발표 + 2분 Q&A | 리허설 2회. 슬라이드 마감 12-09 23:59. 12-07에 발표한다면 12-06에 사실상 완성 | — |

슬라이드 골격(약 10장): 문제 → 동기 → 관련연구 → 가설 → 설정(2) → 결과(3: H1 곡선, H2 표, ablation) → 실패 사례 → 한계와 다음 단계.

### 4.3 (c) 초기 실험 성공/실패 판정 기준

임계값은 노트의 권고("동일 설정에서 공개 수치 ±2 PCK 이내", "작은 움직임 track에서 δavg 1 이상 손실")를 바탕으로 **우리가 정한 결정 규칙**이다 [추정].

| 시점 | 검사 | 성공 | 실패 시 조치 |
|---|---|---|---|
| 11-16 | 환경 | arch list에 `sm_70`, DINOv3 forward 정상 | `torch==2.14.0` cu126으로 시도. 그래도 안 되면 DINOv2로 진행하고 백업 주제 검토 |
| 11-16 | fp16 | 이미지 50장에서 fp32 대비 cosine > 0.999 | fp32로 진행(메모리 2배지만 S/B/L은 들어갈 것 [추정]) |
| 11-24 | SPair 재현 | DINOv3-B raw PCK@0.10이 45.0(논문 설정) 또는 46.8(README 설정)의 **±2.0 이내**, debias 이득 **+3.0 이상**(논문 +5.0) | loader·해상도·s·정규화 점검. 그래도 어긋나면 재현값을 보고하고 상대 비교로 진행 |
| 11-24 | TAP-Vid 재현 | DINOv3-B zero-shot δavg가 **36–44** 범위(공개 38.9–41.3, ±약 3) [추정] | DiTracker 프로토콜(soft-argmax, 30×52)에 맞춤 |
| 11-24 | 무작위 통제 | 같은 rank의 무작위 투영에서 SPair 변화가 **±0.5 이내** | 무작위도 비슷한 이득을 내면 "debias는 일반적 차원 축소 효과"라는 발견으로 보고 |
| 12-01 | **H1** | 작은 변위 1/3 구간에서 δavg(debias) − δavg(raw) ≤ **−1.0**, 비디오 단위 95% CI가 0을 포함하지 않음. 같은 s에서 SPair 이득은 양수 | 모든 구간에서 \|Δ\| < 0.5이고 CI가 0을 포함하면 **H1 기각** → 부정적 결과로 보고 |
| 12-01 | **H2** | 반대 방향 쌍과 같은 방향 쌍의 flip-error 감소 차이가 **2 %p 이상**, CI가 0 제외 [추정 임계값] | 기각하고 "좌우 오류는 위치 성분 탓이 아니다"로 보고 |
| 12-05 | **H3** | gate가 두 regime 모두에서 최선의 고정 s와 0.3 이내, FoRIS gate보다 종합 점수가 높음 | 부정적 결과: gate가 필요 없다는 결론 |

### 4.4 (d) Fallback

**시나리오 A: H1과 H2가 모두 효과 없음.** 채점 기준은 분석이 있으면 부정적 결과도 인정한다. 이 경우 발표는 다음 네 가지로 구성한다.
- "통합 프로토콜에서 위치 debias는 tracking에 중립이고, 좌우 flip은 위치 성분으로 설명되지 않는다"는 반증 결과
- 논문 간 SPair 수치(58.4/50.6/61.2)가 어디서 갈리는지 해소한 결과
- 무작위 부분공간 통제
- rank·layer·해상도 분석

검정력 분석(비디오 30개로 검출 가능한 최소 효과)을 붙이면 "효과 없음"을 "효과가 X보다 작음"으로 바꿔 말할 수 있다.

**시나리오 B: INSID3 재현 실패.** 불일치 원인(loader, s의 정의, 해상도, 정규화)을 문서화한 재현성 연구(유형 C)로 전환한다. 이것도 정당한 프로젝트다.

**시나리오 C: 환경이나 DINOv3 문제.** fp32, HF 가중치, DINOv2-with-registers 순으로 대체한다. 다만 DINOv2에서는 위치 효과가 +0.3에 불과해 이야기가 성립하지 않을 수 있다. 그때는 백업 주제로 넘어간다.

**백업 주제.**
- **1순위: C5(upsampler × regime).** 같은 harness와 데이터셋을 재사용하므로 전환 비용이 2–3일이다 [추정]. 신규성은 낮지만 기한 안에 끝낼 수 있다.
- **2순위: C2.** 영역이 바뀌어 전환 비용이 크다.
- **주제 전환 마감은 11-24(1주차 끝)**다. 그 이후에는 범위 축소만 허용한다.

### 4.5 (e) 수업 이후 ICCV 2027 로드맵

**공식 사실 [검증].**
- ICCV 2027은 홍콩 HKCEC에서 열린다. Dates 페이지 기준 workshop/tutorial은 2027-10-03~04, 본회의는 **10-05~07**이다. 단 홈페이지 본문은 Oct 3–7과 Oct 2–8로 서로 다르게 적혀 있다 ([ICCV 2027](https://iccv.thecvf.com/), [Dates](https://iccv.thecvf.com/Conferences/2027/Dates)).
- 2026-11-15 현재 **논문 마감은 공지되지 않았고**, ICCV 2027 Author Guidelines 페이지는 404다 ([링크](https://iccv.thecvf.com/Conferences/2027/AuthorGuidelines)).
- ICCV 2025 일정(참고 패턴): 등록 3월 3일, 논문·supplementary 3월 7일, 리뷰 공개 5월 9일, rebuttal 5월 16일, 결정 6월 25일 ([ICCV 2025 Dates](https://iccv.thecvf.com/Conferences/2025/Dates)).

**추정.** 2025년 패턴을 따르면 **ICCV 2027 마감은 2027년 3월 초**다 [추정]. 수업 종료 후 약 12–13주가 남는다. aggregator가 제시한 "2027-03-08"은 비공식 예측이다 ([opencurious](https://www.opencurious.com/ai-conference-deadlines/iccv-2027)). CVPR 2027은 11-16 마감이라 이미 해당이 없다 ([CVPR 2027 Dates](https://cvpr.thecvf.com/Conferences/2027/Dates)).

**최신 CVF 규칙.** ICCV 2027은 CVPR 2027 규칙을 대부분 이어받을 것으로 추정된다 ([CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)).
- 본문 8페이지. 참고문헌과 **필수 LLM Usage Statement**는 페이지 수에서 제외.
- double-blind. arXiv 게시 허용.
- 20% 이상 겹치면 이중투고. **4페이지를 넘는 peer-reviewed workshop 논문도 출판물로 간주**된다. 따라서 수업 결과를 ICCV 전에 4페이지 초과 workshop에 내면 안 된다.
- 존재하지 않는 인용은 심사 없이 reject.
- 모든 저자는 OpenReview 프로필을 마감 전에 완성해야 하고, 모두 reviewer 풀에 들어간다. 무책임한 리뷰는 본인 논문 desk reject로 이어진다.
- **reciprocal-reviewer 할당**: 자격 있는 저자가 한 명도 없으면, 각 저자는 그런 논문을 1편만 공저할 수 있다. 본회의 논문 경험이 없다면 지도교수처럼 자격 있는 공저자가 필요하다.
- ICCV 2025도 "모든 저자는 reviewer"를 요구했다 ([ICCV 2025 Author Guidelines](https://iccv.thecvf.com/Conferences/2025/AuthorGuidelines)).
- CVPR 2027 규칙은 2026-09-15 이후 공개된 연구를 동시기(contemporaneous) 연구로 본다. 이 기준이라면 FoRIS(2026-09-03)와 Preserve-Then-Resolve(2026-04)는 비교해야 할 선행연구가 된다. ICCV 2027의 기준일은 미공지다 [추정].

**수업 수준에서 논문 수준으로 가려면 필요한 것.**
1. **일반성**: 백본 3종 이상(DINOv3 S/B/L/H+, DINOv2 대조, 가능하면 DINO 계열이 아닌 ViT).
2. **regime 폭**: semantic(SPair, AP-10K, SPair-U), temporal(TAP-Vid-DAVIS와 Kinetics. Kinetics는 1,000개 비디오이고 원본 클립을 Kinetics-700-2020에서 받아야 함 [검증]).
3. **메커니즘**: 변위 축의 crossover 정량화, H2와 flip 정렬 교란, 무작위 부분공간 통제.
4. **방법**: 두 regime을 모두 개선하는 regime-aware 규칙. INSID3와 FoRIS gate를 이기고, 실제 tracker나 SC 파이프라인에 넣었을 때의 효과를 보여야 한다.
5. **통계적 엄밀성**.

**일정 [추정].**

| 기간 | 할 일 |
|---|---|
| 12-07 ~ 12-20 | go/no-go 결정(H1 또는 H2가 CI 기준으로 지지되었는가). 교수님·지도교수 면담(공저, reciprocal reviewer) |
| 12월 말 ~ 1월 | 확장 실험(백본, regime, 규칙 개선, downstream 통합). 12월에 추가 GPU가 확보되면 확장에만 쓰고 핵심 경로에는 넣지 않는다 |
| 2월 | 집필, OpenReview 프로필, LLM Usage Statement, supplementary |
| 2027년 3월 초 [추정] | 제출(공식 마감이 공지되면 확정) |
| 탈락 시 | NeurIPS 2027(약 5월, 비공식 예측, [opencurious](https://www.opencurious.com/ai-conference-deadlines/neurips-2027)) 또는 ICCV 2027 workshop(일정 미공지) |

**선점 위험과 시간 민감도.** INSID3는 7.5개월 만에 약 21편에 인용됐다. FoRIS는 이미 적응형 debias를 했고, Preserve-Then-Resolve는 debias를 매칭에 적용했다. "DAVIS/TAP-Vid에 debias를 적용해 본다"는 다음 실험이 뻔히 보이는 위치에 있으므로 **선점 위험은 실재하지만 아직 현실화되지 않았다** [검증: novelty 노트의 검색 결과]. 대응: 매달 arXiv와 OpenReview를 다시 검색한다. 수업 직후 arXiv에 preprint를 올려 시점을 남기는 것도 검토한다(CVF 규칙상 허용).

---

## 5. 비판적 단서

### 5.1 (i) 세 선두 후보는 모두 inference-only다

솔직히 말하면 C1·C2·C3 모두 모델을 학습시키지 않는다. 수업 규정("기존 코드를 그대로 돌리는 것은 프로젝트가 아니다")은 C1으로도 충족된다. debias를 직접 구현하고, 세 과제 harness를 만들고, 새 가설과 통제를 설계하기 때문이다. 그러나 사용자가 원한 "PyTorch 구현과 학습 경험"은 **절반만 충족**된다. 다음 두 모듈 중 하나를 3주차 stretch로 붙이면 실현가능성을 해치지 않고 그 빈칸을 메울 수 있다.

**(1) Learned soft projection (권장).**
- **구조**: noise-SVD 기저 U에 방향별 가중치 σ(θ)를 학습시켜 W = I − U·diag(σ(θ))·Uᵀ로 둔다. 또는 rank 16 정도의 저랭크 잔차 adapter를 쓴다.
- **데이터**: DINOv3는 frozen 상태로 두고, SPair trainval 특징을 fp16으로 한 번 캐시한다. 디스크 수 GB [추정].
- **학습**: GT 대응 patch를 양성으로 하는 InfoNCE. train/val 분리, early stopping, seed 3개. 한 번 학습에 수 분에서 1시간 [추정].
- **핵심 검증**:
  1. 의미 데이터로 학습한 투영을 TAP-Vid에 그대로 옮겼을 때 tracking을 해치는가? H1과 바로 연결된다.
  2. 학습된 θ가 noise-SVD 방향을 억제하는가? 해석 가능성 분석이 된다.
- **주의**: SPair supervised SOTA는 이미 90.6이므로 SOTA 경쟁으로 포장하면 안 된다. 이 모듈의 목적은 분석과 학습 경험이다.

**(2) Learned regime gate.**
- **구조**: 입력은 값싼 단서(debias 없이 한 번 돌린 결과의 평균 변위, mutual-NN 비율, FoRIS의 s_sem). 작은 MLP가 raw와 debias 특징 사이의 보간 계수 α ∈ [0,1]을 출력한다.
- **학습**: SPair train 쌍(GT PCK)과 DAVIS-2017 **train** 비디오(라벨 없이 forward-backward cycle-consistency를 목표로 사용)로 학습한다. TAP-Vid-DAVIS는 DAVIS val이라 데이터 누수가 없다.
- **의미**: H3를 "손으로 만든 규칙 대 학습한 규칙"으로 확장한다.

C3와 C2에도 같은 원리를 적용할 수 있다. C3에서는 hidden state 기반 hallucination linear probe나 head별 학습 gate, C2에서는 held-out split에서 채널별 억제 마스크를 학습하는 식이다. **대형 네트워크를 end-to-end로 학습하는 경험이 반드시 필요하다면** C6가 유일한 선택지다. 하지만 그 대가로 필수 조건(2순위)이 위험해진다. 그 경험은 수업 이후 GPU가 확보될 때로 미루는 편이 합리적이다.

### 5.2 (ii) 검증하지 못한 것과 제출 전 체크리스트

**검증하지 못한 것.**
- **CVF PDF 접근 불가(HTTP 403).** INSID3 수치는 arXiv v1 HTML 기준이라 camera-ready와 다를 수 있다. PEARL, OV-Stitcher, Power-of-Prior, EnAR, ACG의 표는 snippet 수준이다.
- **Google Scholar 접근 불가.** INSID3, Mirage, TTR의 "cited by"는 Semantic Scholar로만 확인했다(INSID3 약 21편, Mirage 8편, TTR 62편). 실제 인용은 더 많을 것이다.
- **목록 전수 조사 못 함.** ECCV 2026과 NeurIPS 2026 채택 목록은 제목 단위로 다 보지 못했다. ICLR 2027 OpenReview는 키워드 검색만 했고, 노트 직접 조회는 403이었다.
- **비공식 집계.** papernotes의 CVPR 2026 분류(hallucination 33편, VLM efficiency 63편)와 opencurious의 마감 예측은 비공식이다.
- **지식 범위 밖의 주장.** PyTorch 2.15(2026-10-28)가 Volta를 지원하지 않는다는 내용과 2.14가 마지막 cu126 릴리스라는 내용은 노트가 GitHub 릴리스 노트와 공식 공지를 인용한 것이다. 이 보고서 작성 모델의 지식 범위(2026년 6월) 밖이라 독립적으로 확인하지 못했다. `torch==2.14.1`의 cu126 wheel 존재도 추론이다(공식 공지 예시는 2.14.0).
- **수치 충돌.**
  - DiTracker의 DINOv3 TAP-Vid 수치는 표 파싱이 불확실하다.
  - DINOv3 DAVIS J&F가 노트마다 다르다(L 79.9 vs 83.3).
  - INSID3의 s=500과 ViT-S 384차원이 충돌한다.
  - SPair test 쌍 수(약 12k로 기억)와 viewpoint 주석 형식을 모른다.
  - timm 가중치와 공식 가중치의 동등성은 미확인이다.
  - DINOv3의 fp16 overflow는 V100 관련 snippet 하나뿐이다.
- **C2·C3 관련.** TTR checkpoint가 OpenAI CLIP 기반인지 LAION 기반인지, MLMP 부록의 host 범위, 교수님의 정확한 KAIST 소속은 확인하지 못했다.

**제출(11-18) 전 체크리스트.**
- [ ] arXiv 검색: `positional AND DINOv3 AND (tracking OR DAVIS)`, "INSID3 positional bias tracking", "DINOv3 positional left right"
- [ ] Google Scholar에서 INSID3와 Telling Left from Right의 "cited by" 확인
- [ ] OpenReview ICLR 2027 키워드 검색(positional, DINOv3, point tracking, semantic correspondence)
- [ ] INSID3 코드에서 s의 정의 공간(채널 vs 토큰)과 기본값 확인
- [ ] download.pytorch.org/whl/cu126에서 torch 2.14.x와 torchvision 짝 확인 → `torch.cuda.get_arch_list()`에 `sm_70`이 있는지
- [ ] 이미지 50장에서 fp16과 fp32 특징의 cosine 비교
- [ ] DiTracker PDF 표에서 DINOv3-B 수치 확인, DINOv3 원문에서 DAVIS J&F 확인
- [ ] SPair 다운로드 후 test split 크기와 viewpoint 주석 확인
- [ ] 제안서 related work의 venue와 링크를 CVF/ECVA 페이지에서 재확인
- [ ] 수업의 AI 사용 고지 형식 확인

---

## 결론: 측정 설계가 곧 기여다

노트 7개가 공통으로 보여 준 것은 **새 방법을 하나 더 얹는 일은 거의 모든 인접 영역에서 이미 수익이 낮다**는 점이다. 남은 기회는 "잘 알려진 개입이 언제, 왜 작동하는가"를 통제된 프로토콜로 재는 데 있다. 이런 프로젝트는 frozen inference로 돌아가고 부정적 결과도 발표할 수 있어서 Titan V와 3주 일정에 잘 맞는다. 대신 "예상된 결과"라는 비판과 선점에 가장 취약한 유형이기도 하다. C1이 C2·C3보다 나은 이유는 단순히 싸고 안전해서가 아니다. **부호가 뒤집힌다는 양방향 예측(H1)과 메커니즘 예측(H2)을 함께 갖고 있어서, 결과가 어느 쪽으로 나와도 정보량이 크기 때문**이다. 반대로 C1을 고르면 학습 루프 경험을 의식적으로 따로 확보해야 한다. 3주차의 소형 학습 모듈이 단순한 부록이 아니라 1순위 목표를 지키는 장치인 이유다.

---

## 6. 부록

### 6.1 Titan V(Volta, sm_70) 환경 메모

**드라이버.** R580 브랜치는 CC 7.5 미만 GPU를 지원하는 마지막 브랜치이고, 2028년 중반까지 LTS로 지원된다. 590 이상으로 올리지 않는다 ([NVIDIA blog](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers)). Volta는 open kernel module을 지원하지 않으므로 proprietary module을 쓴다(제3자 출처).

**PyTorch 고정.**
- `pip install torch==2.14.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`, Python 3.10–3.13. 2.14.1의 cu126 wheel 존재는 추론이고, 공식 공지 예시는 2.14.0이다.
- **버전 고정 없이 `pip install torch`를 하면 안 된다.** 2.15+cu132가 설치되는데, 2.15 wheel은 cu132/cu134만 제공하고 Turing(sm_75) 이상을 요구한다 ([2.15 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.15.0), [공식 공지](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)).
- custom op을 빌드할 때는 CUDA toolkit 12.6–12.9와 `TORCH_CUDA_ARCH_LIST="7.0"`을 쓴다. CUDA 13은 sm_75 미만을 컴파일하지 못한다.

**라이브러리 버전.** DINOv3 학습·평가 코드는 PyTorch ≥2.7.1, HF transformers ≥4.56.0, timm ≥1.0.20이 필요하다 ([DINOv3 README](https://github.com/facebookresearch/dinov3)). C2용 mmcv/mmseg는 오래된 torch를 쓰는 별도 환경을 권장한다 [미검증].

**bf16 → fp16.**
- Volta에는 bf16 tensor core도 TF32도 없다. bf16은 fp32로 에뮬레이션되어 빨라지지 않는다.
- `torch.cuda.is_bf16_supported()`는 기본값 `including_emulation=True` 때문에 True를 반환할 수 있으니 믿지 않는다 ([docs](https://docs.pytorch.org/docs/2.15/generated/torch.cuda.is_bf16_supported.html), [forum](https://discuss.pytorch.org/t/bfloat16-on-nvidia-v100-gpu/201629)).
- DINOv3, SAM 2, SD3.5 예제의 `torch.bfloat16`을 `torch.float16`으로 바꾸고 fp32 결과와 비교해 검증한다. softmax와 LayerNorm은 fp32로 유지한다(autocast 기본 동작).
- 학습은 fp16 autocast와 GradScaler로 하고, bf16 학습은 하지 않는다. bf16으로 사전학습된 모델을 fp16으로 fine-tuning하면 범위 문제가 생길 수 있다 ([PyTorch blog](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)).

**Attention.**
- FlashAttention v1–v4는 모두 Volta를 지원하지 않는다 ([repo](https://github.com/Dao-AILab/flash-attention)).
- PyTorch SDPA에서 flash와 cuDNN backend는 sm80 이상이 필요하다. memory-efficient backend는 sm50–sm121을 지원하지만 sm80 미만에서는 fp16/fp32만 받는다 ([sdp_utils.cpp](https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)).
- `attn_implementation="sdpa"` 또는 `"eager"`를 쓴다(hook에는 eager가 필요). xformers 의존성은 피한다.

**양자화와 서빙.** bitsandbytes NF4/FP4는 CC 6.0 이상이라 사용할 수 있다. LLM.int8은 CC 7.5 이상이라 쓸 수 없다. CUDA 13용 bnb wheel에는 sm70이 빠져 있다 ([docs](https://huggingface.co/docs/bitsandbytes/main/en/installation)). vLLM은 CC 7.5 이상이라 지원되지 않는다 ([docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)).

**게이트 가중치와 미러.**
- `facebook/dinov3-*`는 수동 승인이 필요하고, 인증 없이 접근하면 401이 난다 ([HF](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)).
- **`timm/vit_{small,base,large}_patch16_dinov3.lvd1689m`은 게이트가 없다**(HTTP 200, ViT-B safetensors 약 342.6 MB) ([HF timm](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m)). 어느 경로로 받든 DINOv3 License가 적용된다.
- INSID3는 공식 `.pth`를 기대하므로 loader를 고쳐야 한다.
- 다른 후보 관련: SD3/SD3.5-medium은 게이트(C4) ([HF](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)), PIE-Bench는 Google Form(C4), Cityscapes와 ACDC는 등록이 필요하다(C2·C7).

### 6.2 12 GB에 들어가는 모델 (fp16 가중치 기준 추정, activation 별도)

| 모델 | fp16 가중치 [추정] | 12 GB 적합 여부 [추정] |
|---|---|---|
| DINOv3 ViT-S/B/L | 약 0.04 / 0.17 / 0.6 GB | 예(추론과 경량 학습) |
| DINOv3 ViT-H+ (840M) | 약 1.7 GB | 추론은 가능, fine-tuning은 빠듯 |
| DINOv3 ViT-7B | 약 13.4 GB | **불가**(fp16) |
| CLIP ViT-B/16 OVSS (448 px) | 3 GB 미만 | 예 |
| LLaVA-1.5-7B | 약 14 GB | fp16 불가, **4-bit에서 8 GB 미만(공식)** |
| Qwen2.5-VL-3B | 약 7.5 GB | max_pixels를 제한하면 가능할 듯, NaN 점검 필요 |
| SD3.5-Medium | 9.9 GB(공식, text encoder 제외) | 빠듯함. T5 offload 필요 |

출처: [constraints 노트의 공식 수치](https://github.com/haotian-liu/LLaVA), [Stability AI](https://stability.ai/news-updates/introducing-stable-diffusion-3-5).

### 6.3 데이터셋 크기

| 데이터셋 | 크기 | 상태 |
|---|---|---|
| SPair-71k | 약 227 MB | [검증] 직접 다운로드 |
| TAP-Vid-DAVIS | 약 1.67 GB (비디오 30개, pickle) | [검증] 직접 다운로드 |
| DAVIS-2017 trainval 480p | 약 833 MB | [검증] 직접 다운로드 |
| VOC2012 / ADE20K val / COCO val2017 | 약 2 / 1 / 1 GB | [미검증] 기억에 의존 |
| Cityscapes (gtFine + leftImg8bit) | 약 11 GB | [미검증], 등록 필요 |
| COCO val2014 (POPE·CHAIR) | 약 6 GB | [미검증] |

### 6.4 AI 사용 고지

이 주제 조사 보고서는 AI 도구(Claude Code)의 도움으로 작성되었다. 모든 논문, 수치, 링크, 일정은 제출 전에 원문으로 직접 확인해야 한다. 수업 제안서와 향후 논문에도 AI 사용 내역을 밝혀야 한다(CVPR 2027 규칙상 LLM Usage Statement 필수).
