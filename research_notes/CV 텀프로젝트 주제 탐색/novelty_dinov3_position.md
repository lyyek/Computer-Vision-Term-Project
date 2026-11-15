# Novelty Check: Positional Bias in DINOv3 Dense Features for Correspondence (H1 tracking trade-off / H2 left-right flips / H3 adaptive debiasing)

Search date: 2026-11-15. Sources searched: arXiv (web plus the export.arxiv.org API, abstract-keyword queries sorted by date), OpenReview API (keyword search inside the `ICLR.cc/2027/Conference` group; the NeurIPS 2026 group query returned nothing, probably because the group naming differs), the Semantic Scholar "citations" API for INSID3 (search endpoint was rate-limited), CVF Open Access, GitHub READMEs, and the Hugging Face model API. I did not check Google Scholar "cited by", which cannot be accessed automatically. Everything below is labeled "verified" (read in the primary source) or "unverified".

---

## Q0. Background check: is INSID3 as described? (title, authors, venue, repo, tasks, numbers)

### Takeaway
Confirmed. INSID3 is a CVPR 2026 Oral (Cuttano et al.) with Apache-2.0 code. It evaluates only one-shot / in-context segmentation and SPair-71k semantic correspondence; it has no tracking, video, DAVIS, or left/right analysis. The PCK@0.10 gains from debiasing are +1.9 (S), +5.0 (B) and +1.4 (L), against +0.3 for DINOv2. The brief's "+1.4 to +5.0" is correct.

### Cited Findings
- Title: "INSID3: Training-Free In-Context Segmentation with DINOv3". Authors: Claudia Cuttano, Gabriele Trivigno, Christoph Reich, Daniel Cremers, Carlo Masone, Stefan Roth. arXiv:2603.28480, v1 dated 30 Mar 2026, comment "CVPR 2026" — [arXiv abs](https://arxiv.org/abs/2603.28480); [CVF Open Access PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Cuttano_INSID3_Training-Free_In-Context_Segmentation_with_DINOv3_CVPR_2026_paper.pdf); [CVF supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Cuttano_INSID3_Training-Free_In-Context_CVPR_2026_supplemental.pdf)
- Oral status: the README title reads "[CVPR 2026 Oral]", and the news entry dated 2026/04/09 says it was "selected for Oral presentation at CVPR 2026". The README also lists the paper and code release (2026/03/29), a Colab demo (04/16), and **semantic-correspondence inference support (04/21)**. License: Apache-2.0 — [GitHub visinf/insid3](https://github.com/visinf/insid3). OpenReview also lists the paper as "VisCon 2026 Oral", a CVPR 2026 workshop — [OpenReview search result, id 9jOcjLRMIe](https://openreview.net/forum?id=9jOcjLRMIe)
- Method: the authors pass a Gaussian noise image through DINOv3, run SVD on its features, and take the top-s right singular vectors as the positional subspace B. Reference and target features are projected onto the orthogonal complement. The paper fixes s = 500 for all datasets (Table 6). One noise image gives 57.6 mIoU on COCO-20i, and 5 or 10 noise images give 57.7 (Table 5) — [arXiv HTML](https://arxiv.org/html/2603.28480v1)
- The README's keypoint-matching example uses `inference_keypoint_matching.py --debiased --image-size 768 --model-size base --svd-comps 20`, i.e., rank 20, not 500. The README's SPair table (DINOv3-B) reports PCK@0.10 46.8 → 52.6 and PCK@0.20 61.2 → 68.7 — [GitHub README](https://github.com/visinf/insid3)
- Paper Table 2 (SPair-71k, original → debias), verified from the arXiv HTML:
  - Small: @0.05 26.8→27.9, @0.10 43.8→45.7, @0.15 53.2→55.6, @0.20 59.8→62.6
  - Base: @0.05 29.2→32.3, @0.10 45.0→50.0, @0.15 54.0→59.9, @0.20 59.8→66.4
  - Large: @0.05 32.7→33.6, @0.10 50.6→52.0, @0.15 60.3→62.1, @0.20 66.4→68.6
  - Source: [arXiv HTML](https://arxiv.org/html/2603.28480v1)
- Table 4 (DINOv2 vs DINOv3; model size not stated, but the DINOv3 row matches Base). DINOv2: 50.5→50.8 (+0.3) @0.10. DINOv3: 45.0→50.0 (+5.0) @0.10 and 59.8→66.4 (+6.6) @0.20. Quote: "DINOv3 encodes more positional information" — [arXiv HTML](https://arxiv.org/html/2603.28480v1)
- Evaluated tasks: in-context segmentation on COCO-20i, LVIS-92i, ISIC, Chest X-Ray, iSAID, SUIM, PASCAL-Part, PACO-Part and PerMIS, plus semantic correspondence on SPair-71k. The only "video" content is a SAM 3 baseline adapted with "video-style propagation" for ICS (26.7 mIoU). The paper has **no point tracking, no DAVIS, and no TAP-Vid** — [arXiv HTML](https://arxiv.org/html/2603.28480v1)
- Left/right: it appears only as an illustration: "features from the left side of two images tend to spuriously match regardless of semantics". The paper has no symmetry or flip-error analysis. It cites "Telling Left from Right" [71] but does not connect it to positional bias — [arXiv HTML](https://arxiv.org/html/2603.28480v1)
- Trade-off statement in App. A.4: "positional information remains a trade-off… in semantic segmentation, predicting sky can benefit from absolute positional information, whereas correspondence tasks require features that generalize across spatial positioning". The Fig. 7 caption says the bias "has broader implications for tasks that rely on semantic image alignment…". The paper hypothesizes, without testing, that Gram anchoring "may inadvertently amplify absolute spatial correlations" — [arXiv HTML](https://arxiv.org/html/2603.28480v1)

### Inferences
- INSID3 itself names a "task-dependent trade-off" but tests only the side where debiasing helps (cross-image semantic matching). The side where debiasing should hurt, such as video or small-motion matching, is untested in INSID3. That gap is H1.
- The rank mismatch (500 in the paper vs 20 in the README keypoint script) means rank is a free parameter worth sweeping in any follow-up, and a natural knob for an adaptive rule (H3).

### Gaps
- I could not render the CVF PDF (HTTP 403 to automated fetch). All numbers come from the arXiv v1 HTML, which matches the CVPR camera-ready only if no changes were made.

---

## Q1. Has any paper evaluated positional debiasing of DINO features (INSID3-style, DVT, PE removal) on point tracking or video label propagation?

### Takeaway
No. I found no paper, on arXiv through mid-Nov 2026, in the ICLR 2027 OpenReview submissions, or among the papers citing INSID3, that applies positional debiasing (INSID3 projection, DVT, channel blanking, ALiBi re-training, or PE zeroing) and measures TAP-Vid δavg or DAVIS label-propagation J&F. Every evaluation of debiasing that I found uses segmentation, SPair-71k, or wide-baseline relative pose.

### Cited Findings
- **"What DINO saw: ALiBi positional encoding reduces positional bias in Vision Transformers"** (Pawlowsky, Vamvakeros, Weiss, Bielefeld, Cooper, Docherty). arXiv:2603.16840, v1 17 Mar 2026, v2 11 Aug 2026, no venue stated. The authors measure positional bias with linear probes on ramp functions, using micrographs, DTD textures and white noise. DINOv3-S+ has the highest positional R² (≈0.97), and its positional correlation grows with depth while DINOv2's shrinks. They attribute this to RoPE being applied in every layer. Fixes tested: channel blanking, the DVT denoiser, and **fine-tuning with ALiBi** (micrograph R² falls from 0.97 to −0.51 for Dv3). Evaluation covers linear-probe segmentation on VOC07/VOC12/ADE20K and microscopy segmentation, with **no correspondence, no tracking, no video** — [arXiv HTML](https://arxiv.org/html/2603.16840)
- **"Preserve, Then Resolve: Many-to-Many Association and Robust Estimation with General-Purpose Visual Features"** (Haodong Jiang, Mingzhe Li, Junfeng Wu, CUHK-SZ). arXiv:2604.23670; the v2 title was "Zero-Shot DINOv3-Based Image Matching via Many-to-Many Association", and the current v3 uses "Preserve, Then Resolve". It applies INSID3's null-space projection at layer 19 with rank s = 200 and reports that the correction "consistently improves correspondence quality". Benchmarks: MegaDepth-1500, ScanNet-1500, NAVI-Multi/Wild, METU-CC/CS (relative pose). **No video or tracking**, and no failure cases for debiasing are discussed — [arXiv HTML v2](https://arxiv.org/html/2604.23670v2); [arXiv abs/API (v3 title)](https://arxiv.org/abs/2604.23670)
- **DVT, "Denoising Vision Transformers"** (Yang, Luo, Li, … Wang, ECCV 2024). It attributes grid artifacts to positional embeddings and evaluates semantic segmentation, depth, object detection, object discovery, and classification. Its text has **no tracking, DAVIS, SPair, or label-propagation evaluation** (keyword counts are all zero) — [arXiv HTML v2](https://arxiv.org/html/2401.02957v2); [project page](https://jiawei-yang.github.io/DenoisingViT/)
- **"Geometry without Position? When Positional Embeddings Help and Hurt Spatial Reasoning"** (Shi, Birsak, Cui, Li, Wonka, KAUST). arXiv:2601.22231, Jan 2026. This is an inference-time intervention on PEs (zero / shuffle / pairwise shuffle) across 13–14 ViTs, including DINOv3, evaluated on **stereo correspondence (Spring)**. Zeroing PEs badly hurts DINOv2 (EPE 12.64→38.08). Pairwise-consistent shuffling preserves correspondence for absolute-PE models, while DINOv3 (RoPE) collapses under pairwise shuffle (EPE 60.36). The paper has no frozen-feature subspace removal and no tracking or video — [arXiv HTML](https://arxiv.org/html/2601.22231)
- **"Probing and Leveraging Video Diffusion Transformer Features for Robust Point Tracking"** (Son, An, Nam, …, Hur, Seungryong Kim; KAIST AI and Google DeepMind). arXiv:2512.20606, v2 29 Jun 2026. Its zero-shot frozen-feature tracking baselines on TAP-Vid-DAVIS (δavg, query-first, soft-argmax, 30×52 feature resolution) report DINOv2-B/14 40.5, DINOv3-B/16 41.3 and DINOv3-7B 42.6. The fetcher flagged the column parsing as uncertain, so check these against the PDF. **No positional-bias discussion** — [arXiv HTML](https://arxiv.org/html/2512.20606)
- **"Emergent Region-Level Facial Correspondence in Frozen Vision Foundation Models"** (Al-Zyoud, El Saddik). arXiv:2607.14423, Jul 2026. Frozen DINOv3 reaches ≈0.955 temporal face-part tracking accuracy. The paper does **not** discuss positional bias or apply INSID3 debiasing. It mentions RoPE only as an explanation for its attention-facet results — [arXiv HTML](https://arxiv.org/html/2607.14423v1)
- **"Visualizing Dense Neural Representations"** (ICLR 2027 submission, anonymous). It reports that PCA visualizations of DINOv3 "overestimate the spatial smoothness… by encoding a strong spatial bias", and that Laplacian PCA removes this bias. Evaluation covers semantic segmentation and depth; the abstract mentions no tracking — [OpenReview tSTdpsm63c](https://openreview.net/forum?id=tSTdpsm63c)
- **"How Vision Transformers Represent Position: A Mechanistic Comparison of Absolute and Rotary Positional Encodings"** (ICLR 2027 submission). An SAE and causal-intervention study of APE vs RoPE position encoding, evaluated on classification only — [OpenReview PonKoYvIGl](https://openreview.net/forum?id=PonKoYvIGl)
- **"Unsupervised Semantic Segmentation Facilitates Model Understanding"** (Yu, Mais, …, Kainmüller), arXiv:2605.29691, camera-ready accepted at **ECCV 2026**. A visualization protocol that reveals "distinct positional biases", e.g., "strong boundary artifacts in DINOv3-Large model tokens", and separates positional effects from locality bias. It does not study matching or tracking — [arXiv abs](https://arxiv.org/abs/2605.29691)
- The DINOv3 paper itself evaluates non-parametric DAVIS label propagation with frozen features (it reports 83.3 J&F on DAVIS-L, +6.7 over DINOv2). The search summary found no discussion of positional bias in that protocol; I did not read the DINOv3 paper directly — [DINOv3 arXiv HTML](https://arxiv.org/html/2508.10104v1)
- OpenReview ICLR 2027 keyword sweep ("positional", "DINOv3", "point tracking", "semantic correspondence", "label propagation") turned up DINOv3-based tracker papers (DASTrack, id FUwflY0fT1; a withdrawn Siamese DINOv3 tracker, id ILoSA1J1Tv). These are VOT-style trackers, and their titles and abstracts do not mention positional debiasing — [OpenReview FUwflY0fT1](https://openreview.net/forum?id=FUwflY0fT1)

### Inferences
- H1 (debiasing helps SPair but hurts TAP-Vid / DAVIS propagation) appears **untested anywhere**. This is the strongest open piece.
- Indirect support for H1's mechanism: "Geometry without Position?" shows that positional consistency across views helps stereo correspondence, which is small-baseline like adjacent video frames. FoRIS (Q3) shows that positional cues help when reference and target are spatially aligned. Neither tests video.
- DINOv3's zero-shot TAP-Vid δavg baseline (~41 at the reported settings) and DINOv3's strong DAVIS J&F mean both tasks are easy to set up with frozen features. The pipeline is just cosine nearest neighbour or soft-argmax, so applying the INSID3 projection costs one matrix multiply.

### Gaps
- I read only abstracts of NeurIPS 2026 accepted papers and ECCV 2026 accepted lists that surfaced through search. I did not read a complete ECCV 2026 or NeurIPS 2026 accepted-paper list, so a non-arXiv paper there is possible but unverified.
- OpenReview keyword search is term-based and may miss submissions with unusual wording.
- The DINOv3 TAP-Vid numbers in 2512.20606 need confirmation from the PDF table.

---

## Q2. Has any paper linked positional bias to left/right (symmetric) correspondence errors?

### Takeaway
No. I found no paper that attributes left/right flip errors (the "Telling Left from Right" / SPair geo-aware failure mode) to the positional component of DINO features. Existing explanations are "lack of geometric/pose awareness" or "bilateral symmetry of appearance". H2 is open.

### Cited Findings
- "Telling Left from Right: Identifying Geometry-Aware Semantic Correspondence" (Zhang, Herrmann, Hur, Chen, Jampani, Sun, Yang), CVPR 2024. The geo-aware subset covers **82.4% of SPair-71k pairs and 59.6% of keypoints**. DINOv2+NN scores PCK@0.10 53.9 on the full set vs 42.0 on geo-aware; SD+DINO scores 59.9 vs 49.3. The authors attribute the errors to limited geometric/pose awareness ("DINOv2 struggles with left/right (L/R) distinction") and **do not mention positional embeddings or image location**. Their test-time pose alignment (horizontal flip and other transforms, chosen by instance matching distance) raises SD+DINO geo-aware from 49.3 to 55.0 — [arXiv HTML](https://arxiv.org/html/2311.17034); [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Telling_Left_from_Right_Identifying_Geometry-Aware_Semantic_Correspondence_CVPR_2024_paper.pdf); [project page](https://telling-left-from-right.github.io/)
- INSID3 uses "left side matches left side" only as an illustration of absolute-position matching. It has no flip analysis — [arXiv HTML](https://arxiv.org/html/2603.28480v1)
- "Preserve, Then Resolve / Zero-Shot DINOv3-Based Image Matching" uses a left/right ear example as motivation for the geometric ambiguity of DINOv3 features, and does not tie it to debiasing — [arXiv HTML v2](https://arxiv.org/html/2604.23670v2)
- "Geometry Matters: 3D Foundation Priors for Learning Semantic Correspondence" (Jesslen, Dünkel, Kortylewski), arXiv:2605.30093, also submitted to ICLR 2027 (OpenReview m5Ux3AzvGH). It says 2D foundation features "often confuse symmetric object sides" and fixes this with SAM3D and PartField 3D priors plus a trained adapter. It **does not attribute the confusion to positional encoding** — [arXiv abs](https://arxiv.org/abs/2605.30093); [OpenReview](https://openreview.net/forum?id=m5Ux3AzvGH)
- "Improving Semantic Correspondence with Viewpoint-Guided Spherical Maps" (arXiv:2312.13216) reports that DINOv2 gives the left and right sides of a car the same features, framing this as a symmetry/viewpoint issue rather than a positional one — [arXiv HTML](https://arxiv.org/html/2312.13216v1)
- The facial-correspondence paper finds elevated r_eye↔l_eye (0.492) and r_brow↔l_brow (0.436) confusions with frozen DINOv3, and attributes them to "bilateral symmetry, not feature noise". It does not test position — [arXiv HTML](https://arxiv.org/html/2607.14423v1)
- "What DINO saw" measures left-to-right **ramp** channels (positional probes) but does not connect them to symmetric keypoint matching — [arXiv HTML](https://arxiv.org/html/2603.16840)

### Inferences
- The mechanism H2 predicts is testable and specific. When source and target objects face the same direction, the absolute-position prior agrees with the correct match. When they face opposite directions (mirror pose), it pushes the match toward the mirror-side keypoint. So debiasing should reduce flip errors mainly on opposite-orientation pairs. SPair-71k gives viewpoint and truncation annotations, and the geo-aware subset gives the flip-candidate structure.
- A confound worth flagging (my inference, not tested by anyone): the "Telling Left from Right" flip-based test-time pose alignment flips the source image, which also mirrors absolute positions. Part of its gain could therefore come from realigning the positional component rather than geometry. This would be a novel, falsifiable sub-claim.
- Because INSID3 shows the positional component is far weaker in DINOv2 (+0.3 PCK), H2 is mostly a DINOv3 story. DINOv2 flip errors are probably dominated by appearance symmetry.

### Gaps
- I did not systematically scan everything citing "Telling Left from Right" (2024–2026). A minor paper making this link could exist, but none surfaced in arXiv API, web, or OpenReview searches.

---

## Q3. Papers citing INSID3 that extend it toward tracking, left/right, or adaptive debiasing

### Takeaway
Semantic Scholar lists about 21 citing papers as of 2026-11-15. Almost all are ICS or few-shot segmentation, detection, or deepfake work. One is directly relevant to H3: **FoRIS (arXiv 2609.03384)**, which already proposes **adaptive, per-pair positional debiasing**. None extends INSID3 to tracking or video, and none addresses left/right.

### Cited Findings
- Semantic Scholar citations of arXiv:2603.28480 (retrieved 2026-11-15) include: FoRIS (2609.03384), REBASE (2607.09082), SADe (2607.24706), FROST (2606.31136), Example-Based Object Detection (2605.04501), Preserve-Then-Resolve (2604.23670), What DINO saw (2603.16840), VocaDet (2607.08541), Exemplar (2609.03080), From Visual Matching to Textual Conditioning (2611.04084), FrontVeg V2 (2610.06575), VINO (2611.07176), Imagine3D-LLM, SegBanana, PATE-Forensics and others. **None has a tracking, video, or symmetry title** — [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2603.28480/citations?fields=title,year,venue,externalIds,publicationDate&limit=100)
- **FoRIS: "Progressive Foreground Refinement for Training-Free In-Context Segmentation"** (Ming Hu, Jianfu Yin, Mingyu Dou, Miaomiao Zhang, Yao Wang, Cong Hu, Bingliang Hu, Quan Wang). arXiv:2609.03384v1, 3 Sep 2026, no venue stated.
  - It argues that positional information "is not purely harmful". Quotes: "useful spatial regularities can facilitate correspondence when reference and target images are already well aligned", and in lung CT positional cues "can serve as informative priors rather than noise".
  - **Adaptive debiasing (AD):** the decision is per reference–target pair, based on a semantic alignment score s_sem (mean reference-foreground feature vs mean target feature). Debiasing is applied only when s_sem < θ, with θ = 0.80. The subspace comes from SVD of features of a zero-input image.
  - Results: Chest X-ray is 84.9 with always-on debiasing vs 87.6 with AD. On ISIC, unconditional debiasing costs 5.6 points (46.8→41.2). COCO and PACO are the same for OD and AD (60.9 / 42.3).
  - SPair-71k, DINOv3-B (App. C): PCK@0.10 46.80 → 52.60 for **both** always-on and adaptive, and PCK@0.20 68.72 vs 68.73. So on SPair the adaptive rule simply matches always-on.
  - **No video, tracking, DAVIS, TAP-Vid, or left/right analysis** in the text read, which was truncated around App. F.
  - Source: [arXiv HTML](https://arxiv.org/html/2609.03384)
- "Training-free Cross-domain Few-shot Segmentation via Robust Semantic Representation and Matching" (Sun, Ren, Zhang), **accepted to ECCV 2026**, uses DINOv3. A search snippet says it finds position-aware channels take a larger share of variance in deeper DINOv3 layers. I did not verify this in the full text; the abstract does not mention it — [arXiv abs](https://arxiv.org/abs/2606.24297)
- "DINO-KNN: Training-Free Multiclass In-Context Segmentation via Nearest Neighbors" (ICLR 2027, withdrawn) uses INSID3 as its baseline; ICS only — [OpenReview qGev1rtvOm](https://openreview.net/forum?id=qGev1rtvOm)

### Inferences
- **H3 is partly scooped in concept** ("adaptive per-pair debiasing beats always-on/always-off") by FoRIS, but only for segmentation. Its gate is semantic similarity, not displacement, motion, or confidence, and it shows no gain on SPair. A displacement- or temporal-gap-aware rule that switches positional removal by task regime (video vs semantic) remains open. Any H3 claim must cite FoRIS and beat it as a baseline, for example by running FoRIS's s_sem gate on TAP-Vid and DAVIS.
- The citation velocity (~21 citing papers in 7.5 months, several applying INSID3 debiasing) shows the area is active. The risk that someone runs the obvious "debiasing on DAVIS/TAP-Vid" experiment before an ICCV 2027 submission (deadline expected around early March 2027; not verified here) is real but not yet realized.

### Gaps
- Semantic Scholar citation coverage of very recent arXiv papers is incomplete, and I did not check Google Scholar's citation list, which is usually larger.
- I read FoRIS only up to about App. F. A later appendix could contain extra experiments.

---

## Q4. Verdict, closest works, and remaining differentiation

### Takeaway
**PARTIALLY COVERED (mostly OPEN).** H1 (debiasing hurts small-motion tracking and video propagation while helping semantic correspondence) and H2 (left/right flips attributable to the positional component) are **open**; I found no paper testing either. H3 (adaptive debiasing) is **partially covered** by FoRIS (per-pair semantic-similarity gate, segmentation and SPair only). The defensible contribution is a cross-task characterization: semantic vs temporal correspondence, plus a flip-error attribution, with a displacement/task-aware gate benchmarked against FoRIS's gate.

### Cited Findings
- Closest work #1: **INSID3** (CVPR 2026 Oral). It discovers the bias, proposes noise-image SVD projection, and reports gains on SPair. It calls position a "trade-off" but tests only segmentation and SPair — [arXiv](https://arxiv.org/abs/2603.28480); [GitHub](https://github.com/visinf/insid3)
- Closest work #2: **FoRIS** (arXiv 2609.03384). Adaptive per-pair debiasing gated by semantic alignment. It shows positional cues help when layouts align (X-ray, ISIC), and AD = OD on SPair — [arXiv HTML](https://arxiv.org/html/2609.03384)
- Closest work #3: **Preserve, Then Resolve** (arXiv 2604.23670). INSID3 debiasing (layer 19, rank 200) on wide-baseline geometric matching and relative pose, where it improves results. No small-motion or video regime — [arXiv HTML v2](https://arxiv.org/html/2604.23670v2)
- Adjacent: **What DINO saw** (positional bias across DINO/DINOv2/DINOv3, ALiBi fine-tune, segmentation only) — [arXiv](https://arxiv.org/html/2603.16840). **Geometry without Position?** (PE interventions on stereo; positional consistency matters) — [arXiv](https://arxiv.org/html/2601.22231). **DVT** (PE artifacts; no correspondence or tracking evaluation) — [arXiv](https://arxiv.org/html/2401.02957v2)

### Inferences
- Remaining differentiation for the student project, in order of novelty strength:
  1. **H1, the first evaluation of positional debiasing on temporal correspondence.** Run TAP-Vid-DAVIS δavg (query-first, frozen DINOv3 S/B/L, soft-argmax, as in 2512.20606) and DAVIS-2017 label propagation J&F (DINOv3 protocol), sweeping rank s (0, 20, 200, 500) and frame gap Δt (1, 2, 4, 8, 16, and query-first long gaps). Prediction: the harm from debiasing shrinks or reverses as Δt and displacement grow. Pair this with SPair-71k (and SPair-U / AP-10K if time allows) to show the sign flip across tasks.
  2. **H2, flip-error attribution.** On SPair geo-aware keypoints, measure the flip-error rate (match landing on the symmetric counterpart) before and after debiasing, stratified by relative orientation (same-facing vs opposite-facing pairs). Also test whether the "Telling Left from Right" flip-alignment gain is partly a positional-realignment effect.
  3. **H3, a regime-aware gate.** Gate debiasing strength by estimated displacement (e.g., a no-debias first pass's mean match offset, or mutual-NN confidence) and compare against always-on, always-off, and the FoRIS s_sem gate.
- Realistic framing: an "analysis plus simple rule" paper. For ICCV 2027 the risk is that reviewers call H1 "expected" given INSID3's trade-off sentence. Quantifying where the crossover happens (in displacement or Δt) and the flip-error mechanism (H2) is what makes it more than a sanity check.

### Gaps
- Not verified: whether any NeurIPS 2026 accepted (non-arXiv) or ECCV 2026 paper not surfaced by search covers H1 or H2. I recommend re-running an arXiv search ("positional" AND "DINOv3" AND ("tracking" OR "DAVIS")) just before committing, and again before submission.

---

## Q5. Practical: DINOv3 weight access, mirrors, dataset availability

### Takeaway
The official `facebook/dinov3-*` Hugging Face checkpoints are **gated with manual approval**, and the official GitHub weights need a request form plus an approval email. The **timm re-hosts are ungated** and downloadable right away (verified HTTP 200). TAP-Vid-DAVIS, SPair-71k and DAVIS-2017 are all directly downloadable.

### Cited Findings
- `facebook/dinov3-vits16-pretrain-lvd1689m`, `facebook/dinov3-vitb16-pretrain-lvd1689m` and `facebook/dinov3-vitl16-pretrain-lvd1689m` report `"gated": "manual"` from the HF API, library `transformers`, `license:other`. Unauthenticated `config.json` returns HTTP 401 — [HF ViT-B card](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m); [HF API](https://huggingface.co/api/models/facebook/dinov3-vitb16-pretrain-lvd1689m)
- Official repo: "Please follow the link provided below to get access to all the model weights: once accepted, an e-mail will be sent with the complete list of URLs…". The download link is ai.meta.com/resources/models-and-libraries/dinov3-downloads. Transformers support started in v4.56.0 — [facebookresearch/dinov3 README](https://github.com/facebookresearch/dinov3)
- **Ungated timm checkpoints** (HF API `gated: False`): `timm/vit_small_patch16_dinov3.lvd1689m`, `timm/vit_base_patch16_dinov3.lvd1689m` and `timm/vit_large_patch16_dinov3.lvd1689m`, plus S+/H+/7B and `_qkvb` variants. The license metadata is "dinov3-license" pointing to ai.meta.com/resources/models-and-libraries/dinov3-license. `config.json` returns HTTP 200, and `model.safetensors` for ViT-B resolves (≈342.6 MB) without a token — [HF timm ViT-B](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m); [HF timm ViT-L](https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m); [HF timm ViT-S](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m)
- Other ungated re-uploads exist: `onnx-community/dinov3-vitb16-pretrain-lvd1689m-ONNX` and various personal mirrors such as `zeromodels/dinov3-vitb16-pretrain-lvd1689m`. These are unofficial, so their provenance and license compliance are unverified — [HF model search](https://huggingface.co/api/models?search=dinov3-vitb16&limit=40)
- INSID3's README expects the **official `.pth`** files (`lvd1689m` ViT-S/B/L) in `pretrain/`. Using the timm weights would need a small loader change, since key names differ — [GitHub README](https://github.com/visinf/insid3)
- TAP-Vid-DAVIS: `https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip` returns HTTP 200, ≈1.67 GB (pickle with 30 DAVIS val videos) — [tapnet TAP-Vid README](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid/README.md)
- SPair-71k: `http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz` redirects (301) and then returns HTTP 200, ≈227 MB — [SPair-71k](http://cvlab.postech.ac.kr/research/SPair-71k/)
- DAVIS-2017 trainval 480p: `https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip` returns HTTP 200, ≈833 MB — [DAVIS zip](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip)

### Inferences
- There is **no approval-delay blocker**. Use the timm weights (or request official access in parallel). Feature equivalence between timm and the official weights is expected but should be sanity-checked by reproducing INSID3's SPair Base number (45.0 → 50.0 PCK@0.10 at their settings).
- Compute is light: frozen-feature cosine matching for 30 TAP-Vid-DAVIS videos, 30 DAVIS-2017 val videos and about 12k SPair test pairs fits a 3-week single-GPU budget (inference).

### Gaps
- I did not check how long manual approval takes for the official gated repos (it varies), or the exact DINOv3 license terms for academic redistribution.
- I did not verify download links for SPair-U or AP-10K.
