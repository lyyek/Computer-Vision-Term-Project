# Novelty Check: A Mirage-Style Audit of Training-Free ATTENTION-INTERVENTION Hallucination Methods (PAI / VAR / ICT / OPERA) with Threshold-Free POPE, Length-Controlled CHAIR, Fake-Fix Controls and Grounding Analysis

Date context: today is 2026-11-15. Scope: ECCV 2026, NeurIPS 2026, ICLR 2027 submissions, EMNLP 2026, and arXiv Jul–Nov 2026, plus the older work needed as background.

Labels: **[Verified]** = confirmed in this session from a primary page (arXiv abstract or HTML, the official repo, the NeurIPS/ICLR/CVF page, or the arXiv API). **[Snippet]** = from a search-engine summary and not checked in the full text. **[Inferred]** = my reasoning, not stated by any source. **[Unverified]** = venue or details from prior knowledge, not re-checked this session.

Search method: about 30 tool calls. They covered web searches (standard and extended); arXiv abstract and HTML fetches; Semantic Scholar (S2) citation lists for Mirage (8 citing papers), OPERA (686), PAI (254), VAR (172) and ICT (37); keyword screening of those lists for audit-style titles (rethink, revisit, really, illusion, mirage, pitfall, evaluat, audit, fair, reproduc, spurious, benchmark, yes-bias, length, calibrat, fail, ...); and the arXiv API for candidate abstracts. I could not query Google Scholar or the OpenReview ICLR 2027 submission search directly, and the S2 keyword search was rate-limited (HTTP 429). See the Gaps sections.

---

## Q0. Background check: what exactly did "The Mirage of Performance Gains" audit?

### Takeaway
The paper is confirmed: Hao Yin, Guangzong Si, Zilei Wang, NeurIPS 2025, arXiv 2504.10020, code at ustc-hyin/cd_rethink. It audited only three **contrastive-decoding** methods: VCD, ICD and SID. Its controls were prompt- or logit-level "spurious improvement" baselines and greedy decoding. It used no attention-intervention methods, no threshold-free POPE metric and no caption-length analysis.

### Cited Findings
- Exact title: "The Mirage of Performance Gains: Why Contrastive Decoding Fails to Mitigate Object Hallucinations in MLLMs?" Authors: Hao Yin, Guangzong Si, Zilei Wang. arXiv 2504.10020: v1 14 Apr 2025, v2 18 Apr 2025, v3 7 Oct 2025, v4 4 Jun 2026. [Verified] — [arXiv abs](https://arxiv.org/abs/2504.10020)
- Venue: NeurIPS 2025 main track (poster 115342), included in the NeurIPS 2025 proceedings. [Verified] — [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/115342); [NeurIPS proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/2f89a23a19d1617e7fb16d4f7a049ce2-Abstract-Conference.html)
- Core claim: POPE gains from contrastive decoding come from (i) "crude, unidirectional adjustments to the model's output distribution" and (ii) the adaptive plausibility constraint, which "reduces the sampling strategy to greedy search". [Verified, abstract] — [arXiv abs](https://arxiv.org/abs/2504.10020)
- **Methods audited: VCD, ICD and SID** ("three leading hallucination mitigation methods"). HALC and VaLiD appear only in related work. [Verified, HTML v3] — [arXiv HTML v3](https://arxiv.org/html/2504.10020v3)
- Models: LLaVA-v1.5-7B, LLaVA-v1.5-13B, Qwen-VL-Chat-7B. Benchmarks: POPE (COCO Random/Popular/Adversarial, GQA, A-OKVQA), MME, CHAIR (CHAIR-S/CHAIR-I), NoCaps, LLaVA-Bench. [Verified, HTML v3] — [arXiv HTML v3](https://arxiv.org/html/2504.10020v3)
- **Controls ("spurious improvement" methods):**
  - Prompt-Based Adjustment (PBA): appends an instruction that pushes answers toward "Yes".
  - Output Layer Modification (OLM): forces the prediction to "Yes" when |p(Yes) − p(No)| < τ.
  - Greedy decoding as the reference.
  - The adaptive plausibility constraint applied alone under sampling ("sample†").
  
  The authors report that greedy search significantly outperforms direct sampling on POPE. [Verified, HTML v3] — [arXiv HTML v3](https://arxiv.org/html/2504.10020v3)
- The official repo README describes the three spurious-improvement methods (PBA, OLM, APC alone) and POPE scripts that report accuracy, F1 and Yes-ratio. VCD, ICD and SID are switched by flags. The README does not mention CHAIR. [Verified] — [GitHub ustc-hyin/cd_rethink](https://github.com/ustc-hyin/cd_rethink)
- Metrics: accuracy, F1 and "Yes (%)". **No AUROC or threshold-free metric, and no formal yes-ratio-matched comparison.** OLM itself depends on a threshold τ. CHAIR-S/I are reported, and contrastive methods are often *worse* than greedy on them. **No caption-length analysis.** **No attention-intervention methods (PAI, OPERA, VAR, ICT) appear, not even as future work.** [Verified, HTML v3] — [arXiv HTML v3](https://arxiv.org/html/2504.10020v3)

### Inferences
- Mirage's fake fixes push toward **"Yes"** because the baseline LLaVA-1.5 setup it studied answers "No" too often relative to POPE's 50/50 balance. For attention interventions the direction of the shift has to be measured per model and method: PAI, for example, might push toward "No" or toward "Yes". A *bidirectional* fake fix (a sweep of a scalar logit bias) is the cleaner control. [Inferred]
- Mirage's methodology carries over to other method families, but its evidence covers only contrastive decoding. Attention-intervention methods change internal attention, not only the output distribution, so whether they also act mainly as "operating-point shifters" is an open empirical question. [Inferred]

### Gaps
- I did not read Mirage's v4 (June 2026) changes in full. The method list above comes from v3. The v4 abstract is unchanged in substance.

---

## Q1. Has any paper already audited attention-based hallucination interventions with threshold-free or yes-ratio-matched POPE, or length-controlled CHAIR?

### Takeaway
**I found no such paper.** No verified 2025–2026 work audits PAI, VAR, ICT, OPERA or similar attention-intervention methods with AUROC-from-P(yes), yes-ratio matching, length- or recall-controlled CHAIR, or fake-fix baselines. The closest works cover parts of the idea:
- a Mirage follow-up that still covers contrastive decoding only;
- a COLM 2026 workshop paper that applies length budgets and random-head controls to its *own* attention-head intervention;
- an October 2026 preprint showing that discovered "visual heads" steer no better than random heads.

Individual ingredients, such as a recall floor, a fixed token budget, or reporting the yes ratio, already appear as good practice inside *method* papers. None of them is a cross-method audit.

### Cited Findings
**Closest audit (contrastive decoding only)**
- "Rethinking the Effectiveness of Contrastive Decoding in Mitigating Hallucinations in MLLMs." Bendre, Gupta, Modi, Grover, Aggarwal. arXiv 2607.25196: v1 28 Jul 2026, v2 3 Oct 2026, "prepared for NeurIPS VLM4RWD 2026" (a workshop). v2 "removed the reproduction component". [Verified] — [arXiv abs](https://arxiv.org/abs/2607.25196)
  - Methods: VCD, ICD, SID. Models: LLaVA-v1.5-7B, Qwen2.5-VL-7B-Instruct, InternVL3-8B. Benchmarks: POPE, CHAIR, LLaVA-Bench.
  - Controls: APC-S, a Gaussian perturbation matched in magnitude, an oracle shift, and a scalar-bias sweep.
  - Analysis: logit lens and linear probes. AUC is used for probes and the selectivity test, **not for POPE answer quality**.
  - **No PAI, OPERA, VAR or ICT. No length or recall control on CHAIR. No attention-map analysis.**
  - Future work: "Applying the same treatment to newer methods and larger models would establish how far these conclusions generalise." [Verified, HTML] — [arXiv HTML](https://arxiv.org/html/2607.25196)

**Closest attention-side papers (methods with controls, not audits of PAI, VAR, ICT or OPERA)**
- "Targeting the Attention Heads Behind Object Hallucination in LLaVA." Sandhu, Senapati, Kammachi. arXiv 2608.24966, 25 Aug 2026, accepted at the Actionable Interpretability Workshop, COLM 2026. [Verified] — [arXiv abs](https://arxiv.org/abs/2608.24966)
  - Method: a 32-head set on LLaVA-1.5-7B with a head-sliced LoRA plus an inference-time grounding controller.
  - Results: CHAIRs 0.370→0.230 and CHAIRi 0.156→0.096, but **object recall 0.78→0.70**. A layer-matched random-head LoRA control does no better than the baseline. The CHAIR reduction persists under fixed 64/128-token budgets, but the method's captions stay shorter. It does not compare against PAI, OPERA, VAR or ICT, and it does not use POPE AUROC or the yes ratio. [Verified]
  - Average caption length reportedly fell from 61.1 to 29.6 words, and the authors call a fully length-matched evaluation "the most important missing control". [Snippet] — [arXiv HTML](https://arxiv.org/html/2608.24966)
- "What Do Visual-Head Discovery Scores Actually Find? Attention Source, Necessity, and Random-Control Pitfalls in Vision-Language Models." Md Abdul Kadir, Daniel Sonntag. arXiv 2610.13111, 9 Oct 2026, no venue listed. [Verified] — [arXiv abs](https://arxiv.org/abs/2610.13111)
  - Setup: Qwen3-VL-8B, compared against layer-matched random head sets.
  - Finding: **"no discovered set steers better than its layer-matched random sets"**, so steerability does not separate discovered heads from random ones, though necessity does.
  - Mentions hallucination reduction only as motivation. The abstract shows no POPE or CHAIR evaluation of PAI, VAR or ICT.
- "System-Mediated Attention Imbalances Make Vision-Language Models Say Yes." Chan, Suresh, Saha, Hahn, Demberg. arXiv 2601.12430, ACL Findings 2026. Redirecting attention away from system tokens toward image and text reduces **yes-bias**, "often outperforming existing approaches". The abstract does not name its baselines. [Verified] — [arXiv abs](https://arxiv.org/abs/2601.12430)
- Dual-path attention intervention (arXiv 2511.09018, AAAI 2026 per snippet): "Solely enhancing visual attention consistently reduces hallucinations but leads to shorter outputs." [Snippet] — [arXiv PDF](https://arxiv.org/pdf/2511.09018)
- "Encoded but Disconnected: Decomposing VLM Failures under a Patching Null." Genpei Zhang. arXiv 2610.00024, Sep 2026. On POPE with LLaVA-1.5-7B, Qwen2.5-VL-7B and InternVL3-8B, mid layers encode the right answer in 68–91% of errors, yet residual patching gives 0% non-trivial flips at the layer level on all three models. A mechanistic negative result, but it is not an audit of mitigation methods. [Verified, arXiv API abstract] — [arXiv abs](https://arxiv.org/abs/2610.00024)

**Length, recall and yes-ratio controls used inside method papers (ingredients, not audits)**
- RUDDER (arXiv 2511.10292) restricts comparisons to configurations that keep at least 95% of the vanilla model's recall, arguing that aggressive steering can lower hallucination scores by producing overly simple captions. [Snippet] — [arXiv PDF](https://arxiv.org/pdf/2511.10292)
- ACG (arXiv 2601.13707, CVPR 2026 Findings) chooses its operating point by requiring F1 within about 5% of baseline and no degenerate short captions. [Snippet] — [arXiv HTML](https://arxiv.org/html/2601.13707); [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Jo_Attention-Space_Contrastive_Guidance_for_Efficient_Hallucination_Mitigation_in_LVLMs_CVPRF_2026_paper.pdf)
- A diffusion-VLM reliability study (arXiv 2609.01318) reads one model's apparent CHAIR gain as shorter outputs rather than better grounding. [Snippet] — [arXiv PDF](https://arxiv.org/pdf/2609.01318)
- The AOD paper (arXiv 2605.25377) explicitly warns that a conservative "no" bias "artificially inflates precision at the cost of recall". [Snippet] — [arXiv HTML](https://arxiv.org/html/2605.25377v1)
- MARINE (arXiv 2402.08680) and Energy-Guided Decoding (arXiv 2507.07731) report movement of the yes ratio toward 50% as part of their gains. [Snippet] — [MARINE](https://arxiv.org/html/2402.08680v2); [EGD](https://arxiv.org/html/2507.07731)
- AUROC on POPE does appear, but for *hallucination detectors*, not for scoring mitigation methods: Efficient CD with probabilistic hallucination detection (arXiv 2504.12137) and HALP (EACL 2026). [Snippet] — [arXiv 2504.12137](https://arxiv.org/pdf/2504.12137); [HALP, ACL Anthology](https://aclanthology.org/2026.eacl-long.287.pdf)

**Negative search results (as evidence of absence)**
- Extended web searches found no paper that critiques PAI, VAR or ICT, or tests whether their gains come from yes-bias or caption shortening. Queries included "do attention intervention methods really reduce hallucination ... PAI OPERA re-evaluation", "PAI VAR ICT ... yes bias or caption shortening critique", "sanity checks for training-free LVLM hallucination mitigation", and "rethinking evaluation training-free hallucination mitigation ... attention methods 2026". [Verified as search outcome] — e.g., [search hits included PAI](https://arxiv.org/pdf/2407.21771), [CAI](https://arxiv.org/pdf/2506.23590), [SPIN/EMNLP 2025](https://aclanthology.org/2025.emnlp-main.631.pdf)
- S2 citation lists, screened by title with audit-style keywords: **PAI (254 citing papers), VAR (172), ICT (37), OPERA (686)**. No title suggested an audit of attention-intervention hallucination methods. Hits were benchmarks, new methods, or tangential work such as "What Do Visual-Head Discovery Scores Actually Find?" and "Breaking the Illusion: When Positive Meets Negative in Multimodal Decoding". The latter (arXiv 2605.06679, CVPR 2026) proved to be a *new method* (PND), not an audit. [Verified] — [arXiv 2605.06679](https://arxiv.org/abs/2605.06679)

### Inferences
- The specific package has not been published as of 2026-11-15, as far as public indexes show. The package: attention-intervention family + yes-ratio-matched or AUROC POPE + length- or recall-controlled CHAIR + fake-fix baselines (a 'no'-logit bias, forced brevity or EOS boosting) + per-token grounding analysis. [Inferred]
- Indirect evidence supports the hypothesis that some attention-intervention CHAIR gains partly reflect shorter or more conservative output: the AAAI 2026 snippet ("enhancing visual attention ... leads to shorter outputs"), the COLM 2026 workshop result (recall 0.78→0.70) and RUDDER's recall floor. It has not been tested systematically for PAI, VAR or ICT. That makes the project's hypothesis plausible *and* still untested. [Inferred]
- The October 2026 head-discovery paper is close to idea (iv) and to the fake-fix logic: random heads steer as well as discovered heads. It should be cited and could serve as a control: random-head versions of the attention boosts in VAR and ICT. [Inferred]

### Gaps
- I could not search OpenReview's ICLR 2027 submission pool directly. ICLR 2027 submissions, roughly late Sep to early Oct 2026, would be the most likely place for a hidden competitor. Web search returned no ICLR 2027 OpenReview hits for this topic, but indexing of fresh submissions is weak.
- I did not scan the NeurIPS 2026 accepted-paper list or the EMNLP 2026 proceedings title by title. I relied on web search and S2 citations, which surfaced one EMNLP 2026 paper (SAFE, below) and no audit.
- I did not open the full texts of RUDDER, AOD, the dual-path paper or the diffusion-VLM paper. Their quoted claims are [Snippet].

---

## Q2. Papers critiquing the POPE and CHAIR metrics themselves, and whether they re-rank existing mitigation methods

### Takeaway
There is a solid body of metric critique:
- POPE annotation errors (RePOPE);
- benchmark reliability and validity (HQM/HQH);
- POPE's incompleteness and Type I vs. Type II hallucination (THRONE);
- CHAIR's fixed vocabulary (VALOR-EVAL);
- CHAIR's length and detail sensitivity (Feng et al.; "The Devil is in the EOS");
- QA-based evaluation and COCO contamination (Geigle et al.).

**None of the verified ones re-ranks training-free *mitigation methods*.** RePOPE re-ranks *models*. Geigle et al. re-assess *grounding training objectives*. Mirage and Rethinking-CD are the only works that re-assess mitigation methods, and both cover contrastive decoding only.

### Cited Findings
- **RePOPE** (Neuhaus & Hein, arXiv 2504.15707, Apr 2025): re-annotated the 500 COCO images used by POPE. Among "Yes" items, 9.3% are annotation errors and 13.8% are ambiguous; among "No" items the figures are 1.7% and 4.3%. Corrected labels lower precision, raise recall and **change model rankings by F1**. Code at YanNeu/RePOPE. It re-ranks **LVLMs, not mitigation methods**. [Snippet; multiple consistent summaries] — [arXiv HTML](https://arxiv.org/html/2504.15707v1); [HF papers](https://huggingface.co/papers/2504.15707)
- **HQM/HQH**, "Evaluating the Quality of Hallucination Benchmarks for LVLMs" (Yan et al., arXiv 2406.17115): applies psychometric reliability and validity tests and finds that closed-ended benchmarks show test-retest reliability problems and misalignment with humans. It proposes the HQH benchmark. It does not re-rank mitigation methods. Venue: one search summary associates HQH with ICLR 2025 [Unverified]. [Snippet] — [arXiv HTML](https://arxiv.org/html/2406.17115v1)
- **THRONE** (arXiv 2405.05256): argues POPE gives an incomplete picture because Type I (free-form) and Type II (yes/no) hallucinations are disconnected. Venue CVPR 2024 [Unverified]. [Snippet] — [arXiv HTML](https://arxiv.org/html/2405.05256v1)
- **VALOR-EVAL** (arXiv 2404.13874): CHAIR's predefined vocabulary limits detection, so it proposes an LLM-based open-vocabulary extension with coverage. [Snippet] — [arXiv PDF](https://arxiv.org/pdf/2404.13874v3)
- **Feng et al.**, "Do More Details Always Introduce More Hallucinations in LVLM-based Image Captioning?" (arXiv 2406.12663, Jun 2024): reports "technical flaws in existing metrics" and that ">50% of identified object hallucinations are unjust". Proposes CLIP-Precision, CLIP-Recall and CLIP-F1, plus DBD decoding. It does not re-rank PAI-type methods. [Snippet] — [arXiv PDF](https://arxiv.org/pdf/2406.12663)
- **Geigle, Timofte, Glavaš**, "Does Object Grounding Really Reduce Hallucination of LVLMs?" (arXiv 2406.14492; EMNLP 2024 per S2): argues that earlier evaluation protocols are flawed because they rely on COCO, which appears in training, and use QA instead of open captioning. Under a better protocol, grounding objectives have "little to no effect". This is a "really?" re-assessment, but of **training objectives**, not training-free inference methods. [Verified, arXiv API abstract] — [arXiv abs](https://arxiv.org/abs/2406.14492)
- **"The Devil is in the EOS"** (COLM 2025): longer captions raise recall and detail, and hallucination rises with length. It is a length–hallucination link, not a re-ranking of attention methods. [Snippet] — [OpenReview PDF](https://openreview.net/pdf?id=oSub7DiyjL)
- **DO-Bench** (arXiv 2604.22822, Apr 2026): a diagnostic benchmark separating prior override from perceptual limits, with PriorRobust and PerceptionAbility metrics. It evaluates VLMs, not mitigation methods. [Verified, arXiv API abstract] — [arXiv abs](https://arxiv.org/abs/2604.22822)
- **HallusionBench** defines a Yes/No bias test (Yes Percentage Difference, False Positive Ratio), a ready-made yes-bias diagnostic. [Snippet] — [arXiv PDF](https://arxiv.org/pdf/2310.14566)
- **SAFE** (Zhao et al., arXiv 2609.16646, EMNLP 2026): a contrastive "probe" decoding framework. It states that results beyond MMHalBench "are more mixed". It is a method plus analysis, not an audit of attention interventions. [Verified, arXiv API abstract] — [arXiv abs](https://arxiv.org/abs/2609.16646)

### Inferences
- The metric-critique work and the method-audit work have not been combined for attention interventions. The proposed project would be the first to put RePOPE-style label awareness, threshold-free scoring and length control together on the attention family. Using RePOPE labels in addition to POPE labels is a cheap robustness check. [Inferred]

### Gaps
- I did not verify RePOPE's or HQM's final venues.
- I did not exhaustively check whether any 2026 benchmark paper, such as ReactBench (arXiv 2605.29579) or KnowHal (arXiv 2608.03782), includes a table re-evaluating mitigation methods. Their abstracts and snippets do not suggest one.

---

## Q3. Papers citing Mirage that extend it to attention-based methods

### Takeaway
S2 lists **8 citing papers** for Mirage as of 2026-11-15. **None extends the audit to attention interventions.** The only audit-style follow-up is Rethinking-CD (arXiv 2607.25196), which stays within contrastive decoding.

### Cited Findings
- S2 citations of arXiv:2504.10020, retrieved 2026-11-15 [Verified, S2 API] — [S2 API endpoint](https://api.semanticscholar.org/graph/v1/paper/arXiv:2504.10020/citations):
  1. "A Survey of Large Vision-Language Models: An Architectural View" (Information Fusion, Nov 2026). A survey.
  2. "Lost in Contrast: Instruction-Preserving Decoding for Reliable VQA" (arXiv 2610.16722, 12 Oct 2026). Shows that VCD-style candidate filtering and subtraction turn correct abstentions into wrong answers, and proposes IP-VCD and IP-M3ID. A contrastive-decoding method paper. It does not mention PAI, OPERA, VAR or ICT. [Verified] — [arXiv abs](https://arxiv.org/abs/2610.16722)
  3. "Relevant Evidence Decoding for Audio-Visual Hallucination Mitigation" (arXiv 2610.02976). Audio-visual. [title only]
  4. "OPD-Aha: From Linguistic Momentum to Visual Reflection in Multimodal On-Policy Distillation" (arXiv 2609.16459). A training method. [title only]
  5. "Rethinking the Effectiveness of Contrastive Decoding in Mitigating Hallucinations in MLLMs" (arXiv 2607.25196). An audit, CD only (see Q1). [Verified]
  6. "Do We Really Need External Tools to Mitigate Hallucinations? SIRA" (arXiv 2605.14621). A method. [title only]
  7. "The Cost of Language: Centroid Erasure Exposes and Exploits Modal Competition in MLLMs" (arXiv 2604.14363). [title only]
  8. "Text-Guided Layer Fusion Mitigates Hallucination in Multimodal LLMs" (arXiv 2601.03100). A method. [title only]

### Inferences
- The only direct descendant of Mirage explicitly leaves "newer methods" as future work. So the attention-family extension is visibly open, and also an obvious next step that others could take. Scoop risk is real but not yet realized. [Inferred]

### Gaps
- S2 citation counts lag, and a NeurIPS 2025 paper likely has more citations on Google Scholar. I could not access Google Scholar from this environment. For items marked "title only" I did not read abstracts, though their titles indicate method papers.

---

## Q4. Verdict, the three closest works, and the remaining differentiation

### Takeaway
**Verdict: OPEN on the specific idea; PARTIALLY COVERED at the level of methodology.**
- Applying Mirage's audit logic to attention interventions (PAI, VAR, ICT, OPERA) with threshold-free or yes-ratio-matched POPE, length- or recall-controlled CHAIR, fake-fix baselines and a grounding check has **not** been published in any verified source as of 2026-11-15.
- But the audit template (Mirage, plus Rethinking-CD's magnitude-matched and scalar-bias controls) and the length and random-control caveats for attention-head interventions (COLM 2026 workshop; arXiv 2610.13111) already exist.
- So the novelty claim must rest on the **attention family + combined protocol + grounding evidence**, not on "auditing hallucination methods" in general.

### Cited Findings
**The three closest works**
1. **Mirage (NeurIPS 2025)**: same audit philosophy and fake-fix concept (PBA, OLM, greedy), but limited to VCD, ICD and SID, with no AUROC, no length control and no attention analysis. [Verified] — [arXiv HTML v3](https://arxiv.org/html/2504.10020v3); [NeurIPS](https://neurips.cc/virtual/2025/poster/115342)
2. **Rethinking-CD (arXiv 2607.25196, NeurIPS 2026 VLM4RWD workshop)**: extends Mirage with magnitude-matched Gaussian perturbations, a scalar-bias sweep and logit-lens/probe mechanistic analysis on Qwen2.5-VL-7B and InternVL3-8B. Still CD only, with no length control and no attention maps. Lists newer methods as future work. [Verified] — [arXiv HTML](https://arxiv.org/html/2607.25196)
3. **Attention-side controls**:
   - (a) "Targeting the Attention Heads Behind Object Hallucination in LLaVA" (arXiv 2608.24966, COLM 2026 Actionable Interpretability workshop): random-head control, fixed decoding budgets, and a recall drop showing the CHAIR-vs-length/recall confound for an attention-head intervention.
   - (b) "What Do Visual-Head Discovery Scores Actually Find?" (arXiv 2610.13111): discovered heads steer no better than random heads.
   
   Neither audits the published attention-intervention methods. [Verified] — [arXiv 2608.24966](https://arxiv.org/abs/2608.24966); [arXiv 2610.13111](https://arxiv.org/abs/2610.13111)

**Status of the target methods** (these are the methods to audit; their own papers are not audits)
- PAI, "Paying More Attention to Image: A Training-Free Method for Alleviating Hallucination in LVLMs", arXiv 2407.21771, ECCV 2024. [Title verified in search results; venue per sibling notes from the official repo] — [arXiv PDF](https://arxiv.org/pdf/2407.21771)
- VAR / "See What You Are Told: Visual Attention Sink in Large Multimodal Models", arXiv 2503.03321, ICLR 2025. ICT, arXiv 2411.15268, CVPR 2025. OPERA, arXiv 2311.17911, CVPR 2024. [I used these arXiv IDs to pull citation lists from S2. Venues confirmed in the sibling notes file vlm_generation_ct.md; not re-checked here.]

### Inferences
**Remaining differentiation, ranked by how strongly it separates the project from the closest works**
1. **Method family.** The first audit of *attention-intervention* methods (PAI, VAR, ICT, OPERA, optionally ACG or CAI-style ones). The closest audits cover only contrastive decoding. [Inferred]
2. **Threshold-free and matched-operating-point POPE.**
   - AUROC from P(yes) and accuracy/F1 at a matched yes ratio. Neither Mirage nor Rethinking-CD scores POPE without a threshold. Rethinking-CD's scalar-bias sweep is the nearest relative, so cite it and position against it.
   - Framing claim to test: *a method helps only if it improves AUROC, not just the operating point.* [Inferred]
3. **Length- and recall-controlled CHAIR curves.** Plot CHAIR vs. object recall, or CHAIR at matched caption length or token budget, across methods. Mirage and Rethinking-CD report raw CHAIR only. The COLM 2026 workshop paper did this for its own method only. [Inferred]
4. **Attention-specific fake fixes**, beyond Mirage's:
   - a scalar 'no'-logit bias tuned to match each method's yes ratio;
   - EOS boosting or length truncation tuned to match each method's caption length;
   - *random-head / random-token attention boosts* matched in magnitude to PAI and VAR, following 2610.13111's random-control logic. This last one is the most attention-specific and novel control. [Inferred]
5. **Grounding check (iv).** Measure whether the intervention increases attention or attribution on the GT object regions (COCO masks) at the tokens that matter, and whether that increase predicts per-sample correctness. None of the closest works does this for published attention methods. Rethinking-CD uses probes only, and Encoded-but-Disconnected suggests internal signals can be causally inert, a useful framing. [Inferred]
6. **Optional robustness.** RePOPE labels alongside POPE, and Qwen2.5-VL-3B as a non-LLaVA architecture, since many attention methods were tuned on LLaVA-1.5. [Inferred]

**Risk assessment for a 3-week course project and an ICCV 2027 submission (deadline about early March 2027)** [Inferred]
- Course project: safe. Even a negative result ("PAI's gains survive the controls") is a valid finding, and the protocol is the contribution.
- ICCV 2027: viable but exposed.
  - (a) Reviewers may call it "Mirage for attention methods" unless the results are striking. Examples: gains largely vanish at matched yes ratio or length, or random-head boosts match VAR or PAI.
  - (b) Scoop risk over the next 3–4 months is moderate, because Rethinking-CD's authors explicitly flagged "newer methods" and the ICLR 2027 pool is unchecked.
  - (c) Reviewers will expect 7B-scale models. A 4-bit LLaVA-1.5-7B introduces a quantization confound to disclose. Methods' official hyperparameters were tuned for fp16.
- **Practical caveat**: OPERA relies on beam search with an attention-matrix penalty, which is memory- and time-heavy. On a 12 GB GPU, with 4-bit LLaVA-1.5-7B and long captions, OPERA may need a reduced beam count or a subset of COCO images. Plan PAI and VAR first; they are single-pass. [Inferred]

### Gaps
- **ICLR 2027 OpenReview submissions not directly searchable here.** Before committing, the student should search OpenReview's ICLR 2027 venue for keywords such as "hallucination", "PAI", "attention sink", "POPE", "CHAIR", "spurious" and "mirage". Highest-priority check, about 10 minutes.
- **Google Scholar "Cited by" for Mirage** was not checked. Also do this manually.
- I did not scan the NeurIPS 2026 and ECCV 2026 accepted-paper lists title by title. ECCV 2026 papers that appeared in the OPERA citation scan were method papers, e.g., "Context Blindness in DPO ...", arXiv 2608.12158, and "Looking Back and Forth ...", arXiv 2603.07048.
- I could not confirm from full text the ICLR 2025 modular-attribution paper's reported finding that boosting image-token attention had no effect while down-weighting text attention helped. It appeared only in a search summary. The verified abstract (Yang, Li, Cao, Xu) says only that "hallucination heads" attend strongly to text tokens and that mitigation reduces reliance on text tokens — [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/30556). If the claim holds, it is supporting evidence that "paying more attention to image" is not the operative mechanism.
