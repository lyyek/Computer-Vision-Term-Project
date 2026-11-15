<!--
C3 Topic Proposal — DRAFT v0 (2026-11-15)
- 제출: 1-page PDF, KLMS, 2026-11-18 (Wed) 23:59
- 필수 항목(강의 PDF p.9): Title & team members / Problem (input→output) / Motivation /
  Related work (≥2 top-venue papers, recent 3 years) / Your idea / Evaluation plan / Resources (+ role of each member)
- 본문 약 880 단어. 10pt, 2cm 여백 1페이지에는 약 750 단어까지가 안전하므로(추정) 약 15% 줄여야 함.
  줄이는 순서: Related work의 OPERA/VCD 문장 → Ablations 줄 → Motivation 마지막 두 문장.
- [TODO] 표시는 제출 전에 반드시 채우거나 확인할 항목. 세부 체크리스트는 HANDOFF.md 5절.
-->

# Real Fix or Shifted Threshold? Auditing Attention-Intervention Methods for Object Hallucination in LVLMs

**Team:** [TODO: Name, Student ID] (solo) · **Project type:** D (Analysis) + C (Reproduce & extend)

**Problem (Input → Output).** Large vision-language models (LVLMs) often mention objects that are not in the image. Many training-free fixes change the model's attention at inference time, and we ask whether their benchmark gains come from better visual grounding or only from a shifted operating point.
- *Input:* an image with a yes/no probe ("Is there a {object} in the image?") or a captioning prompt, given to an LVLM with or without a mitigation method.
- *Output:* the answer or the caption.
- *Our deliverable:* a controlled audit that reports, for each method, the share of its POPE/CHAIR gain that survives threshold-free metrics, operating-point matching and "fake-fix" controls.

**Motivation.** POPE (balanced yes/no probing) and CHAIR (hallucinated objects in captions) are the de-facto hallucination metrics, and both can improve without any gain in grounding:
- shifting the yes-ratio changes POPE accuracy;
- shorter, less detailed captions lower CHAIR.

Yin et al. (NeurIPS 2025) showed that this happens for contrastive decoding. If attention-based fixes behave the same way, reported progress is inflated. If their gains survive the controls, that would be the first controlled evidence that they improve grounding. Either outcome is informative.

**Related work.**
- *Attention interventions.*
  - PAI [Liu et al., ECCV 2024] amplifies attention to image tokens (α) and adds an "image-centric logit refine" step that contrasts the output with text-only logits (γ).
  - VAR [Kang et al., ICLR 2025] finds "visual attention sink" tokens and moves their surplus attention to other image tokens within image-centric heads.
  - OPERA [Huang et al., CVPR 2024] penalizes over-trust attention patterns during beam search.
- *Contrastive decoding.* VCD [Leng et al., CVPR 2024] contrasts the outputs for the clean image with those for a distorted image.
- *Audits.*
  - Mirage [Yin et al., NeurIPS 2025] shows that the POPE gains of VCD, ICD and SID come from unidirectional output shifts and a collapse of sampling to greedy decoding, and that purpose-built fake fixes match those gains.
  - A 2026 workshop follow-up (Rethinking-CD, arXiv 2607.25196) adds scalar-bias sweeps.
- *What is missing.*
  - Both audits cover contrastive decoding only.
  - Both report threshold-dependent accuracy.
  - Neither controls for caption length.
  - Recent preprints raise warning signs for attention-side methods: a head-level intervention lowers CHAIR but also lowers recall (Sandhu et al., 2026), and discovered "visual heads" steer no better than random heads (Kadir & Sonntag, 2026).
  - No one has audited the *published* attention-intervention methods in this way.

**Proposed idea: a three-test audit protocol.**
- **T1. Operating point vs. discrimination (POPE).**
  - We score each probe by s = logit(Yes) − logit(No) and report the threshold-free AUROC/AP, plus accuracy at the baseline's yes-ratio.
  - Fake fix: the vanilla model with a scalar bias on s, tuned to match each method's yes-ratio.
  - *H1: this bias recovers most (≥70%) of a method's accuracy gain, and AUROC barely changes.*
- **T2. Grounding vs. brevity (CHAIR).**
  - We trace the vanilla model's frontier of CHAIR_i against object recall by sweeping an EOS bias and the token budget.
  - *H2: the methods lie on this frontier, not below it.*
- **T3. Mechanism specificity.**
  - We re-run PAI and VAR with magnitude-matched *random* heads or tokens.
  - We measure the attention mass inside the COCO mask of the queried object and test whether its change predicts per-sample correctness.
  - *H3: the random variants recover a large part of the gain.*
- **Controls.**
  - VCD serves as a *positive control*: the protocol must reproduce Mirage's conclusion on it.
  - We split PAI into α-only and γ-only variants, because its γ term is itself a form of contrastive decoding.
- **What we build ourselves.**
  - A unified HF-transformers harness that re-implements PAI, VAR and VCD as attention and logit hooks, validated against the official code on a subset.
  - All controls, sweeps and analyses.

**Evaluation plan.**
- *Data:*
  - POPE-COCO: random, popular and adversarial splits, each 500 val2014 images × 6 questions.
  - RePOPE corrected labels as a robustness check.
  - CHAIR on 500 COCO val2014 images (detailed captions, ≤512 new tokens).
  - COCO instance masks for T3.
- *Baselines:* vanilla greedy decoding, the three fake fixes (yes/no logit bias, EOS bias, random-head boost) and VCD.
- *Methods audited:* PAI and VAR (core); OPERA on a 100-image subset (stretch).
- *Metrics:*
  - POPE: Acc, F1, yes-ratio, AUROC, AP, Acc at matched yes-ratio, gain-recovery ratio.
  - CHAIR: CHAIR_s, CHAIR_i, recall, length, distance to the frontier.
  - Grounding: attention-in-mask ratio.
  - All with paired-bootstrap 95% CIs.
- *Ablations:* intervention strength α, layer range, head selection, numeric precision (fp16 vs. 4-bit).

**Resources.**
- *GPU:* 1× NVIDIA Titan V (12 GB, Volta), with an estimated budget of 40–80 GPU-hours.
- *Model:* LLaVA-1.5-7B. Its fp16 weights (~14 GB) exceed the GPU's memory, so:
  - POPE runs in fp16 with CPU offload, one forward pass per probe;
  - CHAIR and the sweeps run in 4-bit NF4;
  - the gap between fp16 and 4-bit is measured on POPE and reported as a control.
  - [TODO: insert the measured fp16-offload speed in s/probe]
- *Optional second architecture:* Qwen2.5-VL-3B (fp16).
- *Code (all cited):* the official repositories of PAI, VAR (VisAttnSink), VCD, OPERA, POPE and Mirage (cd_rethink).
- *Team:* solo; all roles are mine.
- *AI tools:* Claude Code and ChatGPT were used for literature search, drafting and coding assistance. I verified all content.
