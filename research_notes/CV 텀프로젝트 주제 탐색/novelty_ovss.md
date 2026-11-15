# Novelty Check: ViT-Artifact Fixes (Test-Time Registers etc.) x Training-Free CLIP OVSS, a Head-to-Head Test of Mechanistic Explanations

Checked on 2026-11-15. Search channels used: arXiv API (abstract search plus id lookup), Semantic Scholar citation graph (papers citing test-time registers, Shi et al. 2026, SCLIP, ClearCLIP, CLIPtrase, SC-CLIP, ProxyCLIP, NACLIP), OpenReview search API (which surfaces ICLR 2027 submissions and TMLR), CVF/NeurIPS pages, and web search. Not accessed: Google Scholar, and a full scan of the ECCV 2026 and NeurIPS 2026 accepted-paper lists (see Gaps).

## Q1. Has any paper applied test-time registers (or other training-free register/artifact fixes) to the standard OVSS benchmarks across multiple training-free OVSS hosts?

### Takeaway
No paper found applies test-time registers (Jiang et al.) to the standard 8 OVSS benchmarks or to any training-free OVSS host. The TTR paper itself does not do it. The closest work is PH-Reg (NeurIPS 2025). PH-Reg is not training-free: it evaluates a self-distilled register model and the DVT baseline on 4 hosts across all 8 benchmarks, but it does not compare against TTR.

### Cited Findings
- **Test-time registers paper.** "Vision Transformers Don't Need Trained Registers", Nick Jiang, Amil Dravid, Alexei Efros, Yossi Gandelsman. NeurIPS 2025 spotlight. — [arXiv 2506.08010](https://arxiv.org/abs/2506.08010); [OpenReview](https://openreview.net/forum?id=bA02DmQN5d)
  - Its only zero-shot segmentation test is [CLS] attention-map thresholding on ImageNet-segmentation (4,276 images). It runs no text-prompted OVSS: no VOC, Context, ADE20K or COCO, and no MaskCLIP, SCLIP, ClearCLIP or NACLIP host. — [arXiv HTML](https://arxiv.org/html/2506.08010)
  - Reported numbers (mIoU / PixAcc / mAP):
    - DINOv2 ViT-L/14: 38.3 / 87.6 / 80.6. With trained registers: 33.9 / 91.4 / 79.0. With TTR: 38.9 / 87.6 / 81.1.
    - OpenCLIP: 34.7 / 76.0 / 79.2, rising to 40.0 / 74.2 / 82.1 with TTR. The table labels this model "ViT-B/14", which looks like a typo for B/16.
    - Source: [arXiv HTML](https://arxiv.org/html/2506.08010)
  - Linear-probe ADE20K mIoU for OpenCLIP ViT-B/16 goes from 40.1 to 40.3 with TTR, and NYUd RMSE from 0.603 to 0.596. On LOST, OpenCLIP is essentially unchanged (30.8 to 30.9 on VOC07). — [arXiv HTML](https://arxiv.org/html/2506.08010)
  - The models tested are OpenCLIP (LAION-2B) and DINOv2, plus a CLIP-L inside LLaVA. **OpenAI CLIP ViT-B/16, the backbone that SCLIP, ClearCLIP and NACLIP use, is not tested.** — [arXiv HTML](https://arxiv.org/html/2506.08010)
- **Closest prior work: PH-Reg.** "Vision Transformers with Self-Distilled Registers", Yinjie Chen, Zipeng Yan, Chong Zhou, Bo Dai, Andrew F. Luo. NeurIPS 2025 spotlight. — [arXiv 2505.21501](https://arxiv.org/abs/2505.21501); [code](https://github.com/0raiser0/PH-Reg)
  - Table 1 runs OVSS on OpenAI CLIP ViT-B/16 across all 8 benchmarks. The rows are CLIP, MaskCLIP, MaskCLIP+DVT, SCLIP, ClearCLIP, NACLIP, NACLIP+DVT and PH-Reg. — [arXiv HTML v3](https://arxiv.org/html/2505.21501v3)
  - Averages (mIoU over VOC21/PC60/Object/VOC20/PC59/Stuff/City/ADE): CLIP 13.77; MaskCLIP 32.21; MaskCLIP+DVT 30.96 (DVT *hurts* MaskCLIP); SCLIP 40.08; ClearCLIP 39.52; NACLIP 39.41; NACLIP+DVT 39.91; PH-Reg 41.85. — [arXiv HTML v3](https://arxiv.org/html/2505.21501v3)
  - Per-dataset rows:
    - NACLIP: 58.88 / 32.20 / 33.15 / 79.70 / 35.16 / 23.30 / 35.48 / 17.42
    - NACLIP+DVT: 60.25 / 32.73 / 32.89 / 80.26 / 35.91 / 23.41 / 36.31 / 17.54
    - PH-Reg: 63.01 / 34.52 / 35.27 / 83.05 / 37.88 / 24.66 / 37.17 / 19.22
    - Source: [arXiv HTML v3](https://arxiv.org/html/2505.21501v3)
  - Gaps in its comparison:
    - TTR (Jiang et al.) is cited but not compared.
    - DVT is applied only to MaskCLIP and NACLIP, not to SCLIP or ClearCLIP.
    - ProxyCLIP is not included.
    - Source: [arXiv HTML v3](https://arxiv.org/html/2505.21501v3)
  - PH-Reg requires self-distillation training. It updates the registers, the positional embeddings, the patch-embedding conv and the final block, for 100 epochs on 4 GPUs. — [arXiv HTML v3](https://arxiv.org/html/2505.21501v3)
  - Ablation: about half of the gain comes from registers and half from TTA denoising. — [arXiv HTML v3](https://arxiv.org/html/2505.21501v3)
  - **Key mechanistic note:** PH-Reg reports that artifact tokens have *lower* norm in OpenAI CLIP and OpenCLIP, and higher norm in DFN-CLIP and DINOv2. — [arXiv HTML v3](https://arxiv.org/html/2505.21501v3)
- **arXiv abstract search for "test-time register(s)" returns only 3 papers** (as of 2026-11-15), none on OVSS: the original TTR paper (2506.08010), "Test-Time Registers as Global Priors for Tokenized Image Generation" (2607.16824, listed as ECCV 2026 in Semantic Scholar), and "Taming Outlier Tokens in Diffusion Transformers" (2605.05206). — [arXiv 2607.16824](https://arxiv.org/abs/2607.16824); [arXiv 2605.05206](https://arxiv.org/abs/2605.05206)
- **arXiv abstract search for "open-vocabulary" + "segmentation" + (high-norm | register(s) | attention sink | massive activation)** finds nothing relevant to 2D OVSS. The 3 hits are a remote-sensing dataset, a 3D paper and a robotics paper. (Query run 2026-11-15 via the arXiv API.)
- **Semantic Scholar list of the 62 papers citing TTR (2506.08010):** no OVSS method or analysis paper among them, except possibly "DisenCLIP: Semantic Segmentation via Concept-Dimension Masking" (ACM MM 2026). Its content could not be retrieved, so it is **unverified**. — [Semantic Scholar API citations of arXiv:2506.08010](https://api.semanticscholar.org/graph/v1/paper/arXiv:2506.08010/citations)
- **Training-free "artifact/outlier token" fixes built into OVSS methods** (these are method papers, not studies of the registers family):
  - SC-CLIP: "Self-Calibrated CLIP for Training-Free Open-Vocabulary Segmentation". Identifies anomaly tokens (high norm, sparse activation) and replaces them with neighbour averages, at the penultimate layer only. — [arXiv 2411.15869](https://arxiv.org/html/2411.15869)
  - SFP: "Feature Purification Matters: Suppressing Outlier Propagation for Training-Free OVSS", Jin et al., ICCV 2025 Highlight. Detects and mitigates outliers at each layer, on the argument that they arise in intermediate layers and propagate. Reports +4.6 mIoU on average across 8 benchmarks. — [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Jin_Feature_Purification_Matters_Suppressing_Outlier_Propagation_for_Training-Free_Open-Vocabulary_Semantic_ICCV_2025_paper.html)
  - LHT-CLIP: "Improving Visual Discriminability of CLIP for Training-Free OVSS", Zhou, Jiang, Zhu; accepted at TMLR. Finds that the last layers sacrifice visual discriminability "partly due to the emergence of anomalous tokens" and proposes abnormal-token replacement. — [arXiv 2510.23894](https://arxiv.org/abs/2510.23894); [OpenReview TMLR](https://openreview.net/forum?id=9spNW3DXg5)
  - RF-CLIP: "Target Refocusing via Attention Redistribution for OVSS: An Explainability Perspective", AAAI 2026. "Distraction tokens" arise from dimension-specific over-activation. For ViT-B/16 the over-activated channels are {4, 162, 189, 326, 429, 474, 633, 713}. — [arXiv 2511.16170](https://arxiv.org/html/2511.16170v1)
  - OAR: "Outlier-aware feature reconstruction for training-free OVSS", Neurocomputing 2026. Title known from Semantic Scholar only; content **unverified**. — [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2312.01597/citations)
  - "Sparse Attention for Dense Open-Vocabulary Prediction in CLIP", Zohra, Zhao, Liu, Ghanem, July 2026. Swaps softmax for α-entmax in the final attention layers at inference. Evaluated on VOC, Context and ADE20K; reports that the gain is proportional to how much baseline attention spreads off-target. — [arXiv 2607.07135](https://arxiv.org/abs/2607.07135)

### Inferences
- The exact cell "TTR or register-neuron zeroing × {MaskCLIP, SCLIP, ClearCLIP, NACLIP, ProxyCLIP} × 8 benchmarks" appears **unoccupied** as of 2026-11-15.
- PH-Reg already shows that "registers/denoising + OVSS host" improves the 8-benchmark average (about +2.4 over NACLIP), and that DVT can *hurt* MaskCLIP. A new study must therefore position itself as **training-free vs. PH-Reg's trained registers**, not as "first to bring registers to OVSS".
- **Technical risk worth turning into a finding:** TTR identifies register neurons via high-norm outliers. But PH-Reg reports that OpenAI CLIP's artifacts are *low-norm*, and TTR never tested OpenAI CLIP ViT-B/16. TTR or high-norm masking may therefore do little on the standard OVSS backbone.
  - This is a falsifiable early check for week 1. The answer is publishable either way.
  - RF-CLIP's "dimension-specific over-activation" channels look like a close analogue of "register neurons". This overlap should be checked and cited.

### Gaps
- DisenCLIP (ACM MM 2026) and OAR (Neurocomputing 2026) could not be read. Either might use register- or outlier-style dimension masking on OVSS hosts.
- Semantic Scholar citation lists looked incomplete; for example ClearCLIP showed only 172 citers. Recent ECCV 2026 and NeurIPS 2026 camera-ready papers may not yet be indexed.
- Unverified whether TTR's project page or repository ships register-neuron indices for OpenAI CLIP ViT-B/16.

## Q2. Has any paper systematically compared the competing mechanistic explanations (ClearCLIP residual/FFN, CLIPtrase [CLS]-global patches, SC-CLIP anomaly tokens, Shi et al. lazy aggregation) under a unified protocol?

### Takeaway
No unified "what actually matters" study was found that tests these explanations head-to-head on training-free OVSS hosts. Fragments exist: each newer method paper rebuts one prior explanation inside its own ablations. RF-CLIP (AAAI 2026) comes closest by directly pitting the [CLS]-proxy explanation against its own distraction-token view on 8 benchmarks. A benchmark-style TMLR submission compares methods, not mechanisms.

### Cited Findings
- **The competing explanations, as stated in the original papers:**
  - ClearCLIP (ECCV 2024) blames the residual path in the last block and the FFN. — [arXiv 2407.12442](https://arxiv.org/abs/2407.12442); summarized in [ResCLIP CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_ResCLIP_Residual_Attention_for_Training-free_Dense_Vision-language_Inference_CVPR_2025_paper.pdf)
  - CLIPtrase: "Explore the Potential of CLIP for Training-Free OVSS", ECCV 2024. — [arXiv 2407.08268](https://arxiv.org/abs/2407.08268); [OpenReview](https://openreview.net/forum?id=fLZQod8ksj)
  - SC-CLIP: anomaly tokens. — [arXiv 2411.15869](https://arxiv.org/html/2411.15869)
  - Shi et al., CVPR 2026: lazy aggregation; high-norm tokens are a symptom, not the cause. — [arXiv 2602.22394](https://arxiv.org/abs/2602.22394)
- **RF-CLIP (AAAI 2026) explicitly contests the [CLS]-proxy view of CLIPtrase and DeCLIP.**
  - It argues that distraction tokens mainly proxy *foreground target* tokens.
  - Its Table 6 shows that reallocating attention budget to [CLS] helps less than reallocating to defocused tokens (34.2 vs 38.2 average mIoU).
  - It contrasts its tokens with Darcet-style registers: data-agnostic and not confined to background.
  - It compares suppression strategies: −∞ masking and low-pass filtering hurt badly, while mean and median filtering help slightly.
  - Host: layer-averaged key-key attention. 8-benchmark ViT-B/16 average: RF-CLIP 44.2 (45.5 with PAMR), SC-CLIP 43.9, NACLIP 39.4, ClearCLIP 38.1.
  - Source: [arXiv 2511.16170](https://arxiv.org/html/2511.16170v1)
- **GCLIP: "Rethinking the Global Knowledge of CLIP in Training-Free OVSS"** (IEEE TMM per Semantic Scholar; title variant "Globality Strikes Back"). Argues that the common modifications weaken CLIP's ability to aggregate global context. It proposes a method rather than a cross-explanation test. — [arXiv 2502.06818](https://arxiv.org/abs/2502.06818)
- **"A Multifaceted Evaluation Protocol for Training-Free Open-Vocabulary Segmentation"** (OpenReview, "Under review for TMLR", created 2026-07-17).
  - A benchmark that jointly evaluates "effectiveness, robustness, cross-dataset performance, and efficiency".
  - Findings: CorrCLIP has the highest mean mIoU on four compact targets, and methods are "sensitive to vocabulary ambiguity" with hidden worst-target fragility.
  - It compares methods as black boxes. Its abstract does not mention artifact, register or mechanistic interventions.
  - Source: [OpenReview 6LG7oFBbLH](https://openreview.net/forum?id=6LG7oFBbLH)
- **"What Holds Back Open-Vocabulary Segmentation?"** (Šarić et al., ICCV 2025 Workshop). Uses oracle components to decouple bottlenecks behind the performance plateau. This is a bottleneck analysis, not a test of the mechanistic explanations or artifact fixes. — [arXiv 2508.04211](https://arxiv.org/abs/2508.04211)
- **"A Unifying View of Attention Sinks: From Mechanisms to Architectural Interventions"** (Fesser, Jacobs, Fel, Keller, Kakade, June 2026).
  - Distinguishes "nop" sinks from "broadcast" sinks in pretrained ViTs.
  - Reports that sinks move from [CLS] in early layers to patches in deeper layers, and that "registers relocate rather than remove sink computation".
  - Uses dense probes only; no OVSS. A useful diagnostic toolkit.
  - Source: [arXiv 2606.08105](https://arxiv.org/abs/2606.08105)
- **"Register and [CLS] tokens yield a decoupling of local and global features in large ViTs"** (Lappe & Giese; NeurIPS 2025 per Semantic Scholar). Shows that the [CLS] token causes a register-like decoupling in models without registers. It cites MaskCLIP and NACLIP, but whether it runs OVSS benchmarks is **unverified**. — [arXiv 2505.05892](https://arxiv.org/abs/2505.05892)
- **No hits** in arXiv abstract searches combining "training-free" + "open-vocabulary" + "segmentation" with "rethinking", "revisit", "empirical study", "unified", "closer look" or "analysis", or in OpenReview searches (including ICLR 2027 submissions), for a mechanism-comparison study. The only hits were method papers such as GLA-CLIP (CVPR 2026), RF-CLIP and CorrCLIPv2. — [arXiv 2603.23030](https://arxiv.org/abs/2603.23030); [arXiv 2610.24757](https://arxiv.org/abs/2610.24757)

### Inferences
- The "head-to-head of explanations under one protocol" framing is still **open as a study design**. The field holds conflicting single-paper claims:
  - CLIPtrase/DeCLIP: [CLS]-proxy.
  - RF-CLIP: foreground proxy, not background registers.
  - SC-CLIP and SFP: anomaly or outlier tokens at different layers.
  - Shi et al.: high-norm is not the root cause.
  - PH-Reg: low-norm artifacts in OpenAI CLIP.
- No single controlled study reconciles these. That gives a concrete motivation paragraph for a proposal.
- The bar for reviewers is higher than "nobody did it": the student must show that the explanations make *different, testable predictions*. For example, removing [CLS] attention should help host A but not host B, and TTR should fix anomaly tokens but not lazy aggregation.

### Gaps
- RF-CLIP's full reference list could not be checked (the text was truncated), so it is unknown whether it cites TTR or SC-CLIP's anomaly framing.
- The TMLR benchmark's full method list and its meaning of "robustness" could not be confirmed beyond the abstract. Authors are anonymous.

## Q3. Verify Shi et al., CVPR 2026 ("lazy aggregation"): title, authors, link, and exactly what it tested. How much does it overlap with the idea?

### Takeaway
Verified as "Vision Transformers Need More Than Registers" by Cheng Shi, Yizhou Yu (HKU) and Sibei Yang (SYSU), CVPR 2026. It argues that high-norm tokens are a symptom of lazy aggregation. Its fix, LaSt-ViT, is applied during pretraining. It evaluates raw CLIP zero-shot segmentation on 6 of the 8 benchmarks, but **never tests a training-free OVSS host** (MaskCLIP, SCLIP, ClearCLIP, NACLIP or ProxyCLIP) and never compares numerically against TTR or DVT. Overlap is conceptual (moderate), not experimental (low).

### Cited Findings
- **Title:** "Vision Transformers Need More Than Registers".
- **Authors:** Cheng Shi and Yizhou Yu (School of Computing and Data Science, The University of Hong Kong); Sibei Yang (Sun Yat-sen University).
- **Links:** arXiv 2602.22394, with v2 dated 14 Apr 2026. — [arXiv abs](https://arxiv.org/abs/2602.22394); [arXiv HTML v2](https://arxiv.org/html/2602.22394v2)
- **Venue:** CVPR 2026 poster. — [CVPR 2026 virtual poster 39203](https://cvpr.thecvf.com/virtual/2026/poster/39203)
- **Core claim:** "ViT uses semantically irrelevant background patches as shortcuts to represent global semantics". This is driven by global attention and coarse-grained supervision. Table 1 caption: "Register reduces high-norm tokens but high-norm is not the root cause of artifacts." — [arXiv HTML v2](https://arxiv.org/html/2602.22394v2)
- **Method:** LaSt-ViT / LazyStrike, a frequency-guided selective aggregation of patch features into [CLS].
  - The paper says it "tackles the problem directly during pretraining, avoiding architectural changes, post-hoc fine-tuning, and test-time interventions". It is therefore **not** a training-free plug-in.
  - Whether the CLIP, MetaCLIP and EVA-CLIP "+LazyStrike" results were retrained from scratch or fine-tuned is ambiguous in the text.
  - Source: [arXiv HTML v2](https://arxiv.org/html/2602.22394v2)
- **Zero-shot segmentation (Table 4)** covers COCO-Object, ADE20K, Cityscapes, VOC20, Context59 and COCO-Stuff. **VOC21 and Context60 are absent.** It compares base CLIP, MetaCLIP and EVA-CLIP against +LazyStrike.
  - CLIP ViT-B/16, base → +LazyStrike: COCO-Obj 8.8→13.3; ADE 3.1→8.3; City 6.5→12.1; VOC20 49.0→75.0; Ctx59 11.2→15.2; Stuff 7.2→11.8.
  - Source: [arXiv HTML v2](https://arxiv.org/html/2602.22394v2)
- **Other tasks:** Point-in-Box (CLIP 39.8→50.1), OV detection and instance segmentation on frozen-CLIP F-ViT/F-VLM, unsupervised object discovery (CorLoc) and VOC12 coarse segmentation. — [arXiv HTML v2](https://arxiv.org/html/2602.22394v2)
- **Registers baseline:** trained registers appear only in Table 1 Point-in-Box: OpenCLIP 39.8 → 37.6 with registers, and ViT 42.7 → 41.5. TTR is described only in related work, with no numbers. DVT is not mentioned. — [arXiv HTML v2](https://arxiv.org/html/2602.22394v2)
- **Training-free OVSS methods** appear only in related work. None is used as a host. — [arXiv HTML v2](https://arxiv.org/html/2602.22394v2)

### Inferences
- **Overlap with the proposed idea:**
  - Shi et al. already makes the headline argument "high-norm or register fixes are insufficient". A study concluding only "TTR doesn't help much" would therefore look like confirmation, not discovery.
  - Shi et al. tests raw CLIP patch-text similarity (MaskCLIP-like without the value trick; inferred from the 49.0 VOC20 baseline matching the raw-CLIP row in PH-Reg) and not the hosts the community uses.
  - Its fix needs pretraining, so it cannot be compared fairly as a test-time intervention.
- **The gap a student can fill:** test whether lazy aggregation explains anything *on top of* ClearCLIP, SCLIP or NACLIP-style last-layer surgery, using cheap training-free proxies such as restricting [CLS] aggregation. Shi et al. does not test this.
- **Risk:** the LaSt-ViT CLIP weights may not be public, which would make a "trained" lazy-aggregation baseline unavailable.

### Gaps
- Did not check whether LaSt-ViT code or weights are released, or the exact training budget for the CLIP experiments.

## Q4. Any benchmark or robustness study comparing training-free OVSS design families under distribution shift, beyond MLMP?

### Takeaway
Shift-robustness work on OVSS exists, but it is almost entirely about **test-time adaptation methods**: MLMP, DAF, TF-CTTA and TPA. These usually sit on a single host. No paper found compares training-free OVSS design families (attention surgery vs. VFM-proxy vs. artifact fixes) under corruptions or ACDC. The TMLR benchmark covers "robustness", but its abstract emphasizes vocabulary ambiguity rather than image corruption.

### Cited Findings
- **MLMP:** "Test-Time Adaptation of Vision-Language Models for OVSS", NeurIPS 2025. Entropy minimization across layers and prompt templates, plus an OVSS-TTA benchmark.
  - Scenario counts differ by version: the NeurIPS abstract says 9 datasets, 15 corruptions and 87 scenarios; arXiv v1/v2 say 7 datasets and 82 scenarios.
  - A secondary summary says the base model is NACLIP ViT-L/14 (single host).
  - Sources: [NeurIPS](https://papers.neurips.cc/paper_files/paper/2025/hash/6c5b82193c5d8e6aa5806239676ddc97-Abstract-Conference.html); [arXiv 2505.21844](https://arxiv.org/abs/2505.21844); [code](https://github.com/dosowiechi/MLMP)
- **DAF:** "Towards Continual Test-Time Adaptation of VLMs in OVSS", Aug 2026, under review.
  - 5 datasets with corrupted variants. Reports over +8 mIoU on Pascal VOC20-C and over +3 on Foggy Cityscapes versus the source model.
  - Its diagnosis: entropy minimization causes patch-level class collapse.
  - Sources: [arXiv 2608.29923](https://arxiv.org/abs/2608.29923); [code](https://github.com/chandlerbing65nm/DAF.git)
- **"When to Adapt: Multi-Signal Domain Shift Detection for Efficient Training-Free Adaptation in OVSS"**, Sept 2026. Robotics continual TTA with indoor/outdoor real-robot data. — [arXiv 2609.37602](https://arxiv.org/abs/2609.37602)
- **TPA:** "Test-Time Prototype Adaptation for OVSS", Aug 2026. Output-level plug-in evaluated with 5 OVSS hosts (attention-redesign and VFM-injection), 3 CLIP backbones and 8 benchmarks. This is clean-benchmark multi-host evaluation, not shift. — [arXiv 2608.08290](https://arxiv.org/abs/2608.08290)
- **TMLR submission:** "A Multifaceted Evaluation Protocol for Training-Free OVS". Covers effectiveness, robustness, cross-dataset performance and efficiency; the robustness findings are framed around vocabulary ambiguity and worst-target fragility. — [OpenReview 6LG7oFBbLH](https://openreview.net/forum?id=6LG7oFBbLH)
- **"Open-Vocabulary Segmentation in the Dark"**, an ICLR 2027 submission. Only the title was found; content **unverified**. It likely concerns low-light shift. — [OpenReview RxznlxGg8J](https://openreview.net/forum?id=RxznlxGg8J)
- **OpenBench** (from "Stepping Out of Similar Semantic Space for OVS"). A semantic-shift benchmark of novel categories, not corruptions. — [arXiv 2506.16058](https://arxiv.org/pdf/2506.16058)

### Inferences
- **The secondary axis is crowded at the TTA level.** "Do artifact fixes and design families change corruption or ACDC robustness *without* adaptation?" is still an open slice.
- That slice is a modest add-on, not a standalone contribution. Reusing MLMP's corruption suite keeps the protocol credible and cheap.

### Gaps
- Did not read the full MLMP benchmark table to confirm which hosts beyond NACLIP were run.
- Did not confirm whether the TMLR benchmark includes image corruptions.

## Q5. Verdict and closest works

### Takeaway
**PARTIALLY COVERED.** The specific experimental cell, "training-free register/artifact fixes (TTR, register-neuron zeroing, high-norm masking, [CLS]-attention removal) × multiple training-free OVSS hosts × standard 8 benchmarks, used to adjudicate the competing explanations", appears **OPEN**. Its components are covered piecemeal by PH-Reg (registers × OVSS hosts, trained), Shi et al. (high-norm is not the cause; raw CLIP only) and RF-CLIP ([CLS]-proxy vs. distraction tokens). Shift robustness is covered at the TTA level (MLMP, DAF).

### Cited Findings
- **Closest work 1: PH-Reg** (NeurIPS 2025).
  - Already covers: registers plus TTA denoising on MaskCLIP, SCLIP, ClearCLIP and NACLIP across all 8 benchmarks with OpenAI CLIP ViT-B/16; DVT baseline; the observation that CLIP artifacts are low-norm.
  - Missing: TTR, any training-free register fix, ProxyCLIP, any adjudication of explanations, robustness.
  - Source: [arXiv 2505.21501](https://arxiv.org/html/2505.21501v3)
- **Closest work 2: Shi et al.** (CVPR 2026).
  - Already covers: the lazy-aggregation thesis that high-norm is not the root cause; zero-shot segmentation on 6 of the 8 benchmarks for raw CLIP, MetaCLIP and EVA-CLIP.
  - Missing: training-free hosts, TTR/DVT numbers, test-time interventions (its fix is applied in pretraining).
  - Source: [arXiv 2602.22394](https://arxiv.org/html/2602.22394v2)
- **Closest work 3: RF-CLIP** (AAAI 2026).
  - Already covers: a mechanistic claim (dimension-specific over-activation causing distraction tokens) tested against the [CLS]-proxy view on 8 benchmarks; a comparison of suppression strategies.
  - Missing: registers, TTR, multi-host evaluation, robustness.
  - Source: [arXiv 2511.16170](https://arxiv.org/html/2511.16170v1)
- **Honourable mentions:**
  - TMLR Multifaceted Evaluation Protocol (benchmark of methods). — [OpenReview](https://openreview.net/forum?id=6LG7oFBbLH)
  - MLMP (shift benchmark). — [arXiv 2505.21844](https://arxiv.org/abs/2505.21844)
  - SFP (layer-wise outlier propagation). — [CVF ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Jin_Feature_Purification_Matters_Suppressing_Outlier_Propagation_for_Training-Free_Open-Vocabulary_Semantic_ICCV_2025_paper.html)
  - LHT-CLIP (layer, head and token analysis). — [arXiv 2510.23894](https://arxiv.org/abs/2510.23894)
  - "Unifying View of Attention Sinks" (diagnostics). — [arXiv 2606.08105](https://arxiv.org/abs/2606.08105)

### Inferences
- **Differentiation that remains**, ranked by defensibility:
  1. **Training-free vs. trained registers on OVSS hosts.** TTR, register-neuron zeroing and high-norm masking vs. PH-Reg's released numbers and DVT, across 5 hosts. PH-Reg's own table serves as an external anchor.
  2. **Transferability test of TTR's premise to OpenAI CLIP ViT-B/16.** PH-Reg's low-norm observation and RF-CLIP's over-activated channels predict that TTR's high-norm neuron criterion may misfire. This can be checked in a few days, and either outcome is reportable.
  3. **A factorial "explanation adjudication":**
     - Factors: residual/FFN removal × [CLS] attention removal × anomaly-token fix × register.
     - Measure interactions per host (for example, whether TTR's gain vanishes once ClearCLIP removes the residual), which no paper reports.
     - Use attention-sink diagnostics from 2606.08105 as mechanistic evidence.
  4. **Robustness slice.** Do these fixes change the corruption or ACDC gap without adaptation? Reuse MLMP's corruption protocol. This is a small add-on only.
- **For a 3-week course project:** feasible and novel enough. Everything is inference-only on frozen CLIP ViT-B/16 with public host code.
- **For ICCV 2027:** a pure "we combined X and Y" table is weak. It needs (a) a sharp finding, such as "explanation E predicts host-dependent effects that hold or fail", or "TTR's register-neuron premise does not hold in OpenAI CLIP, and here is the actual artifact mechanism", and ideally (b) a small principled fix derived from it. Reviewers will cite PH-Reg, Shi et al. and RF-CLIP as prior art, so the framing must be set up against them explicitly.
- **Time-sensitivity:** high. TTR (June 2025) and Shi et al. (Feb 2026) are prominent, and OVSS gets many training-free method papers per month. A competing register-on-OVSS method paper could appear during the ICLR 2027 or CVPR 2027 cycles.

### Gaps
- ECCV 2026 and NeurIPS 2026 accepted-paper lists were not scanned title by title. Coverage relied on arXiv, Semantic Scholar and OpenReview search. A NeurIPS 2026 paper not on arXiv could be missed.
- The Google Scholar "cited by" list for TTR, which is likely more complete than Semantic Scholar's 62 citations, was not accessible.
- ICLR 2027 OpenReview coverage came only from keyword search. The OpenReview note API returned a challenge (403) for direct note fetches, so ICLR 2027 abstracts could not be read in bulk.
- DisenCLIP (ACM MM 2026), OAR (Neurocomputing 2026) and "OVS in the Dark" (ICLR 2027 submission) were seen as titles only. Each should be skimmed before the proposal is finalized.
