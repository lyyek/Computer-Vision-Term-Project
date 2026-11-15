# Visual Features / Correspondence / Matching / Tracking with Foundation-Model Features — Term-Project Candidates (researched 2026-11-15)

> **Scope (after the user's mid-task change):** Culture-technology applications (heritage, artwork, historical photo matching) and 3D-vision or 3D-reconstruction projects are **excluded**. Seed direction 5's cross-era and painting angle has been dropped and appears only in the rejected list. Direction 2 (3D-awareness of 2D features) is kept **only as a feature-representation / correspondence problem evaluated with correspondence metrics** (PCK, δavg, correspondence recall). It is **not** a 3D-reconstruction project. Priority order: direction 1 (semantic correspondence), direction 3 (feature upsampling), direction 4 (point tracking), then general image matching.
>
> Labels used throughout: **[V]** = verified this session from a primary or official source (cited inline). **[E]** = my estimate. **[H]** = hypothesis or proposal. **[inferred]** = a limitation I inferred that the authors did not state. "unverified" = not confirmed this session.

---

## Q0. Cross-cutting platform constraints (Titan V, 12 GB, Volta sm_70) that apply to every direction

### Takeaway
Frozen-backbone inference with DINOv2/DINOv3 up to ViT-L or ViT-H+ fits a 12 GB Volta GPU comfortably. **DINOv3 ViT-7B does not fit**, video-DiT backbones (CogVideoX-5B) are impractical, and the biggest practical risk is the **software stack**. Volta has been dropped from most PyTorch wheels, and only the CUDA 12.6 builds of PyTorch ≤2.14 still ship sm_70.

### Cited Findings
- PyTorch notice dated 2026-09-04: CUDA 12.6 is the only remaining prebuilt variant covering sm_50–sm_70. CUDA 12.8/12.9 had already dropped Volta, and CUDA 13.x does not support anything below sm_75. **PyTorch 2.14 is the last release with a prebuilt Volta wheel.** Options are to stay on ≤2.14 (cu126 index) or to build ≥2.15 from source with `TORCH_CUDA_ARCH_LIST` including 7.0 — [PyTorch dev-discuss notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)
- Volta was removed from the CUDA 12.8/12.9 binaries in release 2.11 so that cuDNN 9.15.1 could be adopted. Some mirrors of the 2.8 release notes claim an earlier removal, so the exact timing conflicts between sources — [PyTorch dev-discuss 2.11](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290)
- DINOv3 model zoo: ViT-S/16 (21M), S+/16 (29M), B/16 (86M), L/16 (300M), H+/16 (840M), **7B/16 (6,716M)**, plus ConvNeXt T/S/B/L. Weights are gated (Meta request form) under the DINOv3 License. Training and eval code needs **PyTorch ≥2.7.1**, while `torch.hub` loading needs only PyTorch. HF Transformers supports it from ≥4.56.0 and timm from ≥1.0.20. The examples use **bfloat16 autocast**. The README does not mention xformers or flash-attention — [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- ViT-L/16 pretraining used 4 nodes × 8 H100-80GB (~14 h), so retraining any DINOv3 backbone is out of scope — [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- GeoAware-SC (SD + DINOv2 features) needs PyTorch 1.13.1 / CUDA 11.6, with optional xformers 0.0.16 "significantly reducing VRAM". Pre-extracting SPair-71k features (original + flipped) takes **~2 h on one RTX 3090 and ~90 GB of disk** — [GeoAware-SC GitHub](https://github.com/Junyi42/GeoAware-SC)
- Chrono's environment requires **xformers** (CUDA 11.8 index) — [Chrono GitHub](https://github.com/cvlab-kaist/Chrono)

### Inferences
- [E] The workable stack window is **PyTorch 2.7.1–2.14 with cu126 wheels**. It satisfies DINOv3's ≥2.7.1 and still ships sm_70. Pin this in week 1 and do not upgrade. Old repos (GeoAware-SC, sd-dino on torch 1.13.1) would need a separate conda environment.
- [E] DINOv3 ViT-7B: 6.7B params × 2 bytes ≈ 13.4 GB in fp16 for the weights alone, which exceeds 12 GB, so it is infeasible without 8-bit/4-bit quantization (untested and not recommended). ViT-H+ (840M) ≈ 1.7 GB fp16 and ViT-L ≈ 0.6 GB fp16. A forward pass at 960×960 (60×60 = 3,600 tokens) should need well under 6 GB even with materialized attention (16 heads × 3600² × 2 B ≈ 0.4 GB per layer, transient). Uncertainty is about ±2×, depending on implementation.
- [E] No bf16 on Volta: run DINOv3 in fp16 autocast or fp32. **Risk (unverified):** fp16 overflow in large-norm activations of DINOv3/DINOv2 layers is plausible. Validate fp16 against fp32 features on about 50 images (cosine similarity > 0.999) before running full evaluations.
- [E] Titan V fp16 tensor-core throughput is in the same class as an RTX 3090, but memory is half. Expect roughly 1–2.5× the RTX-3090 runtimes quoted in papers (rough estimate).
- FlashAttention-2 needs sm_80+ (constraint from the assignment, not re-verified). PyTorch SDPA's memory-efficient (cutlass) path is the practical fallback. Whether xformers memory-efficient attention works on sm_70 is unverified; test it in week 1 if Chrono or GeoAware-SC is used.

### Gaps
- I did not verify whether DINOv3 HF checkpoints are stored in fp32 or bf16 (the README does not say).
- No published Titan V / V100 memory or runtime numbers exist for DINOv3 at high resolution. Measure them in week 1.

---

## Q1. Direction 1 — Zero-shot / lightly supervised SEMANTIC CORRESPONDENCE with foundation features (DINOv3 era)

### Takeaway
This direction is **very crowded and largely "solved" on headline PCK**. Fine-tuning the last four DINOv3 blocks reaches **90.6 PCK@0.1 on SPair-71k**. Annotation-free pipelines (DIY-SC, ICCV 2025; 3D-SC, arXiv May 2026) reach 69–73, and CVPR 2026 already has an oral (MARCO). A new adapter is **not** a defensible contribution. What remains open and cheap is **analysis of the frozen DINOv3 feature space**. DINOv3 has a documented **absolute-positional bias** (INSID3, CVPR 2026 Oral) that hurts cross-instance matching. Left/right confusion remains a listed open failure mode. **No one (found) has connected the two, or tested how the positional component behaves in temporal (tracking) versus semantic correspondence.**

### Problem (Input → Output)
A source image, a target image (different instance, same category), and a set of query keypoints on the source → predicted target locations. Metric: PCK@α (α·max(bbox h,w)).

### Cited Findings — key papers (2023–2026)
- **DIFT** — "Emergent Correspondence from Image Diffusion", Tang et al., NeurIPS 2023 — [arXiv 2306.03881](https://arxiv.org/abs/2306.03881). DIFT scores 52.9 PCK@0.1 on SPair-71k and 50.3/46.0/35.0 on AP-10K (intra-species / cross-species / cross-family), as tabulated in the 3D-SC paper — [3D-SC](https://arxiv.org/html/2605.30093). The repo (github.com/Tsingularity/dift) was not re-verified this session.
- **SD+DINO** — "A Tale of Two Features: Stable Diffusion Complements DINO for Zero-Shot Semantic Correspondence", Zhang et al., NeurIPS 2023 — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8e9bdc23f169a05ea9b72ccef4574551-Abstract.html). Official code: [Junyi42/sd-dino](https://github.com/Junyi42/sd-dino) (PyTorch 1.13.1, SPair/PF-Pascal/TSS scripts, `pck_spair_pascal.py`). Core claim: DINOv2 gives "sparse but accurate" matches, while SD gives better spatial information — [arXiv 2305.15347](https://arxiv.org/pdf/2305.15347)
- **GeoAware-SC** — "Telling Left from Right: Identifying Geometry-Aware Semantic Correspondence", Zhang et al., CVPR 2024 — [arXiv 2311.17034](https://arxiv.org/abs/2311.17034). Code and checkpoints: [Junyi42/GeoAware-SC](https://github.com/Junyi42/GeoAware-SC). The repo has eval configs for SPair, PF-Pascal and AP-10K (including cross-species and cross-family subsets) and zero-shot flags (`--ADAPT_FLIP`, `--MUTUAL_NN`). **`--COMPUTE_GEOAWARE_METRICS`** evaluates the geometry-aware subset (symmetric/left-right keypoints), and `pose_awareness_test.py` tests global pose awareness. It reports 85.6 PCK@0.1 supervised on SPair. The zero-shot figure is 65.4 or 64.2 depending on the arXiv version — [GeoAware-SC GitHub](https://github.com/Junyi42/GeoAware-SC), [search summary of arXiv versions](https://arxiv.org/pdf/2311.17034)
- **Unified benchmark** — "Semantic Correspondence: Unified Benchmarking and a Strong Baseline", K. Zhang, X. Li, J. Lu, K. Han (HKU/Oxford), arXiv 2505.18060 (v4, Jun 2026; no venue listed) — [arXiv](https://arxiv.org/html/2505.18060). Code: [Visual-AI/Semantic-Correspondence](https://github.com/Visual-AI/Semantic-Correspondence). Hardware: a single **RTX 3090**. Key numbers (SPair-71k PCK@0.1, per-image):
  - Zero-shot **DINOv3 ViT-L/16**: 58.4 @960 px (60×60 tokens), 54.9 @512, 45.3 @256. **DINOv2**: 54.4 @840, 52.9 @448, 43.1 @224.
  - **SD2.1+DINOv3 = 58.1, below DINOv3 alone (58.4).** SD2.1+DINOv2 = 60.4. The authors suggest the features "are not complementary" for DINOv3.
  - Fine-tuning the last four DINOv3 blocks gives **90.6 @960**, 86.8 @512, 74.3 @256 (fine-tuned DINOv2 @840 = 81.7). On AP-10K: 90.3 versus GeoAware-SC's 87.5.
  - Failure modes the authors list (Appendix E): extreme intra-class variation, **left-right confusion of bilaterally symmetric parts**, cross-category confusion, and incomplete views. The fixes they suggest are stronger positional encodings, explicit structural modeling, orientation-aware modules, auxiliary classification, and lightweight 3D or multi-view information.
- **DIY-SC** — "Do It Yourself: Learning Semantic Correspondence from Pseudo-Labels", Dünkel, Wimmer, Theobalt, Rupprecht, Kortylewski, ICCV 2025 — [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Dunkel_Do_It_Yourself_Learning_Semantic_Correspondence_from_Pseudo-Labels_ICCV_2025_paper.html). Code is linked from the project page: [odunkel/DIY-SC](https://github.com/odunkel/DIY-SC), with an HF demo — [project page](https://genintel.github.io/DIY-SC). It trains a lightweight adapter on pseudo-labels (3D-aware chaining, relaxed cyclic consistency, spherical-prototype filtering). Reported: SPair-71k 71.6, Geo-Aware subset 67.5, SPair-U 67.9 — [3D-SC table](https://arxiv.org/html/2605.30093). Stated limitation: zero-shot performance drops with larger viewpoint change — [arXiv 2506.05312](https://arxiv.org/pdf/2506.05312)
- **Jamais Vu** — "Exposing the Generalization Gap in Supervised Semantic Correspondence", Mariotti, Du, Bhalgat, Mac Aodha, Bilen, NeurIPS 2025 — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/17826a22eb8b58494dfdfca61e772c39-Abstract-Conference.html). Claims that supervised SC methods behave like keypoint detectors and fail to generalize beyond annotated keypoints. Introduces the **SPair-U** benchmark (new unseen keypoints). Code: github.com/VICO-UoE/JamaisVu (per the search summary; the repo was not opened).
- **GECO** — "Geometrically Consistent Embedding with Lightspeed Inference", Hartwig, Muhle, Marin, Cremers, ICCV 2025 — [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Hartwig_GECO_Geometrically_Consistent_Embedding_with_Lightspeed_Inference_ICCV_2025_paper.html). Uses optimal-transport training to separate left and right parts. Reports ~30 fps, +6.0/+6.2/+4.1 PCK on PF-Pascal/AP-10K/CUB, and argues **PCK is insufficient** to analyze geometric properties.
- **MARCO** — "Navigating the Unseen Space of Semantic Correspondence", Cuttano, Trivigno, Masone, Roth, **CVPR 2026 Oral** — [arXiv 2604.18267](https://arxiv.org/abs/2604.18267), [CVPR page](https://cvpr.thecvf.com/virtual/2026/oral/40305). Single DINOv2 backbone, coarse-to-fine Gaussian RBF loss, and self-distillation to dense pseudo-correspondences. Claims SOTA on SPair/AP-10K/PF-PASCAL and to be ~3× smaller and ~10× faster than diffusion-based methods. Introduces the **MP-100** transfer benchmark. Code: github.com/visinf/MARCO (per the search summary).
- **Bridging Viewpoint Gaps** (CVPR 2025): KNN-based geometric matching plus sparse semantic matching, with up to +7.4 zero-shot on the SPair rigid subset — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Qian_Bridging_Viewpoint_Gaps_Geometric_Reasoning_Boosts_Semantic_Correspondence_CVPR_2025_paper.pdf)
- **3D-SC** — "Geometry Matters: 3D Foundation Priors for Learning Semantic Correspondence", Jesslen, Dünkel, Kortylewski, arXiv 2605.30093 (28 May 2026; no venue) — [arXiv](https://arxiv.org/html/2605.30093). Code: [GenIntel/3D-SC](https://github.com/GenIntel/3D-SC). It uses SAM 3D meshes, Orient Anything V2, and PartField, with a 5M-parameter adapter on frozen DINO+SD features. Results without human annotation: **73.0 on SPair-71k, 70.8 on SPair-Geo-Aware**, 67.3 on SPair-U. DINOv3 adds +0.9 over DINOv2. Hardware: a single **L40 (40 GB)**, ~12.4 s of mesh reconstruction per object. Stated limitations: dependence on SAM 3D pose and shape errors, and coarse PartField cues. It also names related 2025–26 works: C3PO (3DV 2026), SemAlign3D (CVPR 2025), Common3D (CVPR 2025), DenseMatcher (ICLR 2025), MATCHA (2025), and the WACV 2025 distillation paper.
- **DINOv3 itself** (Siméoni et al., arXiv 2508.10104) — [arXiv](https://arxiv.org/html/2508.10104v1). Under the Probe3D protocol with frozen last-layer features (correspondence recall): DINOv3-7B scores NAVI 64.4 / SPair 58.7, versus DINOv2-g 60.1 / 56.1. Distilled models on SPair: S 50.4, S+ 55.2, B 57.2, **L 61.2, H+ 56.3**, so **H+ is below L and S is below DINOv2-S (51.6)**. The paper says it adopted axial RoPE plus a regularizer to avoid positional artifacts.
- **INSID3** — "Training-Free In-Context Segmentation with DINOv3", Cuttano, Trivigno, Reich, Cremers, Masone, Roth, **CVPR 2026 Oral** — [arXiv 2603.28480](https://arxiv.org/html/2603.28480), [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Cuttano_INSID3_Training-Free_In-Context_Segmentation_with_DINOv3_CVPR_2026_paper.pdf), code [visinf/insid3](https://github.com/visinf/insid3) (Apache-2.0).
  - Finding: DINOv3 similarity maps show **structured activations at the same absolute coordinates as the reference point**, mostly in low-semantic regions. The cause (Gram anchoring amplifying absolute spatial correlation) is the authors' unverified hypothesis.
  - Fix: pass Gaussian noise through the encoder, take an SVD, and project features onto the orthogonal complement of the top-s singular vectors (**s = 500**). The cost is one matrix multiply at inference.
  - SPair-71k PCK@0.1 for DINOv3 orig → debiased: **S 43.8 → 45.7, B 45.0 → 50.0, L 50.6 → 52.0**. For DINOv2 only ~+0.3.
  - Stated limitations: positional information is a trade-off (it helps "sky"-type segmentation), they do not claim *all* positional bias is removed, and hyperparameters were tuned on COCO-20i.
  - **No per-category or left/right/geo-aware analysis was done.**
- Other 2026 positional-bias work: "What DINO saw: ALiBi positional encoding reduces positional bias in Vision Transformers" (Pawlowsky et al., arXiv 2603.16840, v2 Aug 2026; no venue). It detects positional bias by linear probing and fine-tunes models to ALiBi, with a microscopy segmentation application — [arXiv](https://arxiv.org/abs/2603.16840)
- "A Patch is More Than a Token" (Wang & Ma, Shaoxing Univ., arXiv 2610.41814, 30 Oct 2026; no venue) is a training-free frozen-DINOv3 multi-view-of-patch representation. It is evaluated on SPair, DAVIS-2017, MegaDepth-1500, ScanNet-1500, NYUv2, and ADE20K, and uses bf16 autocast. Code: github.com/zixiongwang-ai/PMTT — [arXiv](https://arxiv.org/html/2610.41814v1). This is low-profile but the closest "frozen DINOv3 across many correspondence types" work.
- DINOv3 features are "semantically meaningful but often feature artifacts, making the correspondences imprecise" (claim from Densemarks, arXiv 2511.02830, via search snippet) — [arXiv](https://arxiv.org/html/2511.02830v1)
- SPair-71k has 70,958 pairs over 18 categories — [SPair paper](https://arxiv.org/pdf/1908.10543)

### Inferences
- **Saturation verdict:** Supervised and pseudo-label SC on SPair is saturated: 90.6 (fine-tuned DINOv3), 73.0 (annotation-free), plus an ICCV 2025 / NeurIPS 2025 / CVPR 2026 Oral cluster. Proposing "another adapter / pseudo-label filter" is a small variant and **would not be paper-level**. The SD-complements-DINO story (NeurIPS 2023) appears **not to hold for DINOv3** (58.1 vs 58.4, single third-party source), which weakens every SD+DINO fusion idea.
- **Protocol hazard [inferred]:** zero-shot DINOv3-L on SPair is reported as 58.4 (Zhang et al., 960 px, per-image PCK), 50.6 (INSID3, resolution not stated), and 61.2 (DINOv3 paper, Probe3D recall). Differences in resolution, normalization (per-image vs per-keypoint, bbox), and soft-argmax make cross-paper comparisons unreliable. A course project must re-run **all** baselines in one harness. That is itself a useful contribution for the course.
- **Open angle (not found in literature):** INSID3 shows that removing an absolute-position subspace helps semantic matching. GeoAware-SC and Zhang et al. list left/right confusion as unsolved. Nobody (found) asks whether (a) left/right errors on opposite-facing pairs are partly *caused* by absolute-position leakage, or (b) the same positional component *helps* temporal correspondence (tracking), where absolute position is a valid prior because inter-frame motion is small. → developed as **Candidate A** below.
- [inferred] INSID3's noise-image subspace estimate is a global, input-independent projection with rank 500 of 384–1024 dims (for S/B/L). That is a large fraction of the space for ViT-S (384 dims), so it must remove semantic signal too. This is a natural ablation target (rank versus model width).

### Gaps
- I could not find SPair-71k download size, the exact test-pair count, or the viewpoint/scale difficulty annotation format this session (believed to be ~12k test pairs with difficulty flags; **unverified**).
- MARCO's and DIY-SC's exact numbers and checkpoint availability were not opened. Jamais Vu and MARCO repos were found via search only.
- DistillDIFT (WACV 2025) and CleanDIFT (CVPR 2025) were not re-verified this session. They are only referenced as "Distillation of diffusion features for semantic correspondence (WACV 2025)" in 3D-SC.

---

## Q2. Direction 3 — Feature upsampling for dense ViT features: is there room left?

### Takeaway
For segmentation and depth this area is **saturated**. Gains over bilinear are ≤1–3 mIoU, and Upsample Anything (CVPR 2026) shows near-zero gains on Cityscapes. For **correspondence**, the 2026 evidence is striking: popular learned upsamplers (JAFAR, AnyUp) are **~14 PCK *worse* than bilinear** on SPair-71k with DINOv3 (ViT-Up, Jun 2026). The SPair evaluation is now taken (ViT-Up, WRC), but **point tracking (TAP-Vid) with upsampled features is not evaluated** by any upsampler paper I found. That leaves a narrow, analysis-type course project. Paper-level novelty is low because the field is crowded.

### Problem (Input → Output)
An image plus low-resolution patch features (H/16 × W/16 × C) → high-resolution features (H × W × C) that preserve the feature distribution. Downstream evaluation is via frozen probes or matching.

### Cited Findings
- **FeatUp** (Fu et al., ICLR 2024) is a baseline in all later papers. Its extra cost at 448 px with 2× upsampling (A100) is +0.17M params and 65 ms — [WRC](https://arxiv.org/html/2605.17472)
- **LoftUp** (Huang et al., ICCV 2025) is a coordinate-based upsampler — [cvlibs PDF](https://www.cvlibs.net/publications/Huang2025ICCV.pdf). Cost: +4.3M params, +1066 GMACs, 135 ms at 448 px — [WRC](https://arxiv.org/html/2605.17472). AnyUp reports that LoftUp **shifts the feature distribution** (ADE20k with a pre-trained DINOv2 probe: LoftUp 4.27 vs bilinear 39.73 mIoU) — [AnyUp](https://arxiv.org/html/2510.12764v2)
- **JAFAR** (Couairon et al., 2025; venue unverified this session) uses single-step high-to-low attention — [AnyUp](https://arxiv.org/html/2510.12764v1)
- **AnyUp** — "Universal Feature Upsampling", Wimmer, Truong, Rakotosaona, Oechsle, Tombari, Schiele, Lenssen, arXiv 2510.12764 (listed under ICLR 2026 by a secondary source, [papernotes](https://en.papernotes.org/ICLR2026/others/anyup_universal_feature_upsampling/); not confirmed on OpenReview this session) — [arXiv](https://arxiv.org/html/2510.12764v2). It is encoder-agnostic, with code and weights at github.com/wimmerth/anyup. Training: ImageNet, 100k steps, batch 4, **~5 h on one H100**, 0.8M params. **Peak backward memory at 448×448 is ~12.9 GB** (LoftUp 21.1, JAFAR 22.3). Tasks: segmentation, depth, and normals only, with **no SPair, keypoint, or tracking evaluation**. Stated limitations: it assumes outputs are linear combinations of low-res features (it can miss sub-patch detail) and it does **not remove positional-encoding artifacts**.
- **Upsample Anything** — "A Simple and Hard to Beat Baseline for Feature Upsampling", Seo, Hamilton, Kim, arXiv 2511.16301. A CVPR 2026 supplement exists on CVF — [CVF supp](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Seo_Upsample_Anything_A_CVPR_2026_supplemental.pdf), [arXiv](https://arxiv.org/html/2511.16301v2). It is per-image test-time optimization of anisotropic Gaussian kernels (50 Adam iterations), ~0.419 s per 224² image (its Table 4 says 0.0419 s, a 10× text/table inconsistency).
  - COCO mIoU: bilinear 60.43, AnyUp 61.25, ours 61.41.
  - On Cityscapes, **all methods are within ~0.03 mIoU of bilinear**. The authors say gains over bilinear are "modest" with strong backbones.
  - Correspondence is only shown qualitatively. Stated limitation: it overfits under low SNR or noise. No code repo is given, only a project page.
- **ViT-Up** — "Faithful Feature Upsampling for Vision Transformers", Wandel, Wang, Wang (SJTU), arXiv 2606.14024 (12 Jun 2026; no venue) — [arXiv](https://arxiv.org/html/2606.14024v1). Code: github.com/krispinwandel/vit-up.
  - **SPair-71k PCK@0.10 with DINOv3-S+: bilinear 51.27, JAFAR 36.82, AnyUp 37.63, UpLiFT 46.87, NAF 48.68, ViT-Up 55.44.** With DINOv3-B: bilinear 50.01, NAF 47.19, ViT-Up 58.10.
  - Also evaluates NAVI geometric correspondence. **Does not evaluate point tracking or video propagation.**
  - Trained 1 epoch of IN-1k on a single RTX 5090 with LoRA rank 16 on the backbone.
  - Appendix B: ViT-Up *reconstructs* positional/grid artifacts, while image-guided upsamplers suppress them. It attributes DINOv3's cleaner features to RoPE.
- **WRC** — "Weighted Reverse Convolution for Feature Upsampling", W. Li et al. (PolyU/NUAA/NJU), arXiv 2605.17472 (May 2026) — [arXiv](https://arxiv.org/html/2605.17472). Code: github.com/PolyU-VCLab/WRC. It is a closed-form FFT Tikhonov inverse upsampler.
  - SPair keypoint PCK@**0.01** (DINOv2-S, 448 px): bilinear 3.52, JAFAR 2.58, AnyUp 4.05, LiFT 4.32, WRC 5.21.
  - DAVIS VOS label propagation J&F @672: bilinear 62.52, JAFAR 63.59, AnyUp 71.21, WRC 73.57.
  - COCO20K CorLoc: **AnyUp 28.89 and JAFAR 25.91, versus bilinear 40.35**.
- **RaysUp** (arXiv 2606.22749) is another 2026 "ultra-light universal" upsampler. It describes AnyUp as JAFAR with windowed attention — [arXiv](https://arxiv.org/html/2606.22749)

### Inferences
- **Saturation verdict:** at least 8 upsamplers were published Jan 2024–Jun 2026 (FeatUp, LiFT, LoftUp, JAFAR, AnyUp, Upsample Anything, NAF, UpLiFT, WRC, ViT-Up, RaysUp). The SPair evaluation niche was taken in May–June 2026. Building **a new upsampler to beat ViT-Up is not feasible** on a Titan V in 3 weeks. AnyUp's backward pass needs ~12.9 GB at 448² on an H100, which already exceeds 12 GB without lowering resolution or using fp16 and checkpointing [E].
- **Remaining course-level gap [H]:** "Do learned upsamplers help or hurt *temporal* point correspondence?" Here the reference and query share an instance, so the image-guided smoothing that blurs part distinctiveness on SPair may hurt less, or may help sub-patch localization. A conflicting signal already exists: AnyUp is −14 PCK on SPair (ViT-Up) but +8.7 J&F on DAVIS label propagation (WRC). **That contradiction is a clean analysis question:** is matching *discriminativeness* (NN-ratio / distinctiveness) traded for *smoothness*, and does the task regime (cross-instance vs same-instance) decide which wins?
- [inferred] Every upsampler is trained with reconstruction or probe-style objectives that reward smooth, locally averaged features. None trains for **descriptor distinctiveness**, which explains the SPair losses. A "matching-aware" test-time variant would be a small method contribution, but Upsample Anything plus ViT-Up make it incremental.

### Gaps
- FeatUp and LoftUp SPair numbers under the same protocol were not found (ViT-Up omits them from its SPair table).
- JAFAR's venue (possibly NeurIPS 2025) was not verified. LoftUp, JAFAR, and AnyUp checkpoints for DINOv3 specifically were not checked (AnyUp claims encoder-agnostic weights).

---

## Q3. Direction 4 — Point tracking with foundation / temporal features

### Takeaway
This is an active and fast-moving area, with KAIST's own groups very visible (Chrono, DiffTrack, DiTracker). **Zero-shot frozen DINOv3 is far behind** trained trackers: δavg ≈ 39–41 on TAP-Vid-DAVIS, versus 46.9–49.7 for video-DiT features and 48–53.5 for lightly adapted backbones. Lightweight *adaptation* of DINOv3 is already claimed by M2P (Mar 2026, LoRA plus VOS masks, 2× RTX 4090 for ~1 day, code not released). Video-DiT methods are infeasible on 12 GB. A Titan-V-sized project is viable as **zero-shot feature analysis** or a **scaled-down weak-supervision reproduction**, but paper-level novelty is hard.

### Problem (Input → Output)
A video plus query points (t, x, y) → per-frame 2D track positions and visibility. Metrics: δavg (position accuracy averaged over pixel thresholds), OA (occlusion accuracy), and AJ (Average Jaccard). Protocol: TAP-Vid, query-first or strided, at 256×256.

### Cited Findings
- **TAP-Vid**: DAVIS has **30 videos** (DAVIS val), Kinetics **1000 videos** (raw clips must be fetched from Kinetics-700-2020), RGB-Stacking 50 synthetic videos, and Kubric is the synthetic training set. Modes are `first` and `strided`, and evaluation is at 256×256 (full-resolution results are a separate category) — [TAP-Vid README](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid/README.md)
- **Chrono** — "Exploring Temporally-Aware Features for Point Tracking", Inès Hyeonsu Kim, Seokju Cho, Jiahui Huang, Jung Yi, Joon-Young Lee, Seungryong Kim, **CVPR 2025** — [arXiv 2501.12218](https://arxiv.org/abs/2501.12218), [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_Exploring_Temporally-Aware_Features_for_Point_Tracking_CVPR_2025_paper.pdf). DINOv2 plus a temporal adapter, refiner-free. Code: [cvlab-kaist/Chrono](https://github.com/cvlab-kaist/Chrono).
  - **ViT-S and ViT-B checkpoints are released (Google Drive)**, with eval scripts for DAVIS and RGB-Stacking.
  - Training data is **panning-MOVi-E, ~273 GB** (HF, Git LFS). xformers is required. Training hardware is not stated.
  - Chrono-S/16 δavg (queried-first): DAVIS 48.0 / Kinetics 49.3 (as tabulated by M2P) — [M2P](https://arxiv.org/html/2603.17813)
- **DINO-Tracker** — "Taming DINO for Self-Supervised Point Tracking in a Single Video", Tumanyan, Singer, Bagon, Dekel, **ECCV 2024** — [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3799_ECCV_2024_paper.php). Code: [AssafSinger94/dino-tracker](https://github.com/AssafSinger94/dino-tracker) (linked from the [project page](https://dino-tracker.github.io)). Test-time training per video: TAP-Vid-DAVIS-480 δavg 80.4 / OA 88.1 / AJ 64.6, 7.6M params, **~0.9 h per video** — [M2P table](https://arxiv.org/html/2603.17813)
- **CoTracker3** — "Simpler and Better Point Tracking by Pseudo-Labelling Real Videos", Karaev et al., **ICCV 2025 (Highlight)** — [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Karaev_CoTracker3_Simpler_and_Better_Point_Tracking_by_Pseudo-Labelling_Real_Videos_ICCV_2025_paper.html). Supervised DAVIS AJ/δavg/OA: 64.8 / 76.8 / 91.7 — [DiTracker table](https://arxiv.org/html/2512.20606)
- **DiffTrack** — "Emergent Temporal Correspondences from Video Diffusion Transformers", Nam, Son, Chung, J. Kim, Jin, Hur, S. Kim, **NeurIPS 2025** — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b2fea79b1137d917e8b7cce9434ab5fa-Abstract-Conference.html). Finds that query-key similarity in specific layers drives temporal matching and strengthens with denoising. Zero-shot SOTA among VFMs: with CogVideoX-5B, δavg DAVIS 46.9 / Kinetics 49.2 — [M2P table](https://arxiv.org/html/2603.17813)
- **DiTracker** — "Probing and Leveraging Video Diffusion Transformer Features for Robust Point Tracking", Son, An, Nam, … Hur, S. Kim (KAIST AI / Google DeepMind), arXiv 2512.20606 (v2 29 Jun 2026; no venue) — [arXiv](https://arxiv.org/html/2512.20606). Project page: cvlab-kaist.github.io/DiTracker, with no repo link seen.
  - Zero-shot DAVIS δavg: CogVideoX-5B 49.7, CogVideoX-2B 48.2, WAN-14B 46.6, **DINOv3-7B 42.6, DINOv3-B 41.3, DINOv2-B 40.5**, VGGT 44.7.
  - Supervised DiTracker (CogVideoX-2B + CoTracker3 head + LoRA) is roughly on par with CoTracker3 on DAVIS (AJ 64.8 vs 64.8) and better on ITTO-MOSE and corrupted DAVIS.
  - The authors note a "perfectly controlled ablation is impossible" across VFMs.
- **M2P** — "Improving Visual Foundation Models with Mask-to-Point Weakly-Supervised Learning for Dense Point Tracking", Q. Wu, T. Yang, B. Fang, J. Wan, M. Di Martino, G. Sapiro, A. B. Chan, arXiv 2603.17813 (Mar 2026; no venue) — [arXiv](https://arxiv.org/html/2603.17813).
  - LoRA (rank 64) on DINOv2/DINOv3, trained with VOS masks (YouTube-VOS 2019 + DAVIS-2017 train, 3.4k videos) and three losses: Procrustes local-structure pseudo-labels, mask-label consistency, and mask-boundary constraint.
  - Training: 150k iterations at batch 16 on **2× RTX 4090 for ~1 day**.
  - δavg DAVIS/Kinetics: **DINOv3-B zero-shot 38.9 / 37.4 → M2Pv3-B 53.5 / 49.4**. DINOv3-S zero-shot 38.0 / 36.7 → M2Pv3-S 50.5 / 48.4.
  - M2P-Tracker (TTO) on DAVIS-480: δavg 79.1, AJ 65.4, 0.5M params, ~0.7 h per video.
  - **Code "will be made publicly available upon acceptance"** (not released).
  - Stated limitations: LoRA and the small dataset limit capacity, raw features fail under distractors and severe deformation, online memory may be needed, and patch-16 hurts δ0 on Kinetics.
- Zero-shot DINOv3 tracking numbers differ between papers (DINOv3-B 38.9 in M2P vs 41.3 in DiTracker). The protocols differ — [M2P](https://arxiv.org/html/2603.17813), [DiTracker](https://arxiv.org/html/2512.20606)
- DINOv3's own report gives DAVIS mask-propagation J&F at 960 px for distilled models: S 72.7, S+ 75.5, B 77.2, **L 79.9, H+ 79.3** (DINOv2 S/B/L 73.6/72.9/73.4). It uses last-layer features only, with no layer comparison — [DINOv3 arXiv](https://arxiv.org/html/2508.10104v1)
- Multiview-equivariance fine-tuning (MEF, ICLR 2025) also reports TAP-Vid-DAVIS tracking gains, and its repo ships a TAP-Vid-DAVIS eval script — [3DCorrEnhance GitHub](https://github.com/qq456cvb/3DCorrEnhance)

### Inferences
- **Crowdedness:** high. Temporal adapters (Chrono, CVPR 2025), video-DiT features (DiffTrack NeurIPS 2025, DiTracker 2025/26), weakly supervised DINOv3 adaptation (M2P 2026), and TTO (DINO-Tracker ECCV 2024, M2P-Tracker) all exist. Several are from KAIST's CVLAB. That is good for local mentorship, but it means reviewers will know the space intimately.
- **Feasibility on Titan V [E]:**
  - Zero-shot NN tracking with frozen DINOv3-S/B/L on TAP-Vid-DAVIS (30 videos, at most a few hundred frames each) at 256–480 px takes minutes to hours of inference. Easy.
  - LoRA-fine-tuning DINOv3-S/B on frame pairs at 256–448 px in fp16 should fit in 12 GB (rough). M2P's full recipe (2× 4090 × 1 day ≈ 2–4 Titan-V days) is borderline, and there is **no code**, so a reimplementation would be partial and "unfair" versus their numbers.
  - DINO-Tracker TTO at ~0.9 h per video × 30 videos ≈ 27 GPU-hours on their hardware; on Titan V, likely 1.5–3× that [E]. Feasible only on a subset.
  - Training Chrono from scratch is out (273 GB data). **CogVideoX-5B** at ~10 GB of fp16 weights plus activations is infeasible on 12 GB. CogVideoX-2B is marginal and would need CPU offloading [E].
- [H] Absolute-position priors should **help** tracking (small inter-frame displacement), the opposite of INSID3's semantic-matching finding. This ties directly to Candidate A and is cheap to test with Chrono's or the TAP-Vid eval harness.

### Gaps
- Chrono's training compute and GPU memory are not stated in its README. Whether Chrono runs on sm_70 with xformers is unverified.
- The DiTracker GPU and compute appendix was not read. Its code availability is unconfirmed.
- No 2026 conference version of M2P was found.

---

## Q4. Direction 2 (re-framed) — Multi-view-consistent / "3D-aware" 2D features, evaluated ONLY with correspondence metrics

### Takeaway
This direction is allowed only as a **feature-representation and correspondence** problem: train or probe features, then measure SPair geo-aware PCK, NAVI correspondence recall, and TAP-Vid δavg, with no reconstruction outputs. Even so, it is **crowded in 2025–26** (MEF ICLR 2025, Multi-View Foundation Models Dec 2025, Splat-and-Distill Feb 2026, DINOv3D ECCV 2026). It needs **3D-rendered training data** (MEF renders ~10k Objaverse objects with blenderproc), which conflicts with "avoid data-collection-dominated projects" and the user's de-prioritization of 3D. **Backup only.**

### Cited Findings
- **Probe3D** — "Probing the 3D Awareness of Visual Foundation Models", El Banani et al., **CVPR 2024** — [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Banani_Probing_the_3D_Awareness_of_Visual_Foundation_Models_CVPR_2024_paper.html), [arXiv 2404.08636](https://arxiv.org/pdf/2404.08636). Code: github.com/mbanani/probe3d (per search). Tests single-view probes plus multiview consistency (NAVI/SPair correspondence). DINOv3 adopts its protocol — [DINOv3](https://arxiv.org/html/2508.10104v1)
- **FiT3D** — "Improving 2D Feature Representations by 3D-Aware Fine-Tuning", Yue, Das, Engelmann, Tang, Lenssen, **ECCV 2024** — [mlanthology](https://mlanthology.org/eccv/2024/yue2024eccv-improving), [HF papers 2407.20229](https://huggingface.co/papers/2407.20229). It lifts features to 3D Gaussians, re-renders them, and fine-tunes the 2D model. Evaluated on segmentation and depth.
- **MEF** — "Multiview Equivariance Improves 3D Correspondence Understanding with Minimal Feature Finetuning", Yang You, Yixin Li, Congyue Deng, Yue Wang, Leonidas Guibas, **ICLR 2025** — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/923a141d7c65adf0b1823127dd74032d-Abstract-Conference.html). Code: [qq456cvb/3DCorrEnhance](https://github.com/qq456cvb/3DCorrEnhance) (MIT). Fine-tuned DINOv2 S/B/L/G (plus registers) and CLIP checkpoints are on HF. Training needs **Objaverse GLBs (10k subset) rendered with blenderproc**. Eval covers OnePose++ pose, **TAP-Vid-DAVIS tracking**, and **PF-PASCAL transfer**. The README gives no GPU or time figures. The paper claims "finetuning on a single object for one iteration" already yields substantial gains — [search summary of ICLR page](https://www.iclr.cc/virtual/2025/poster/30515)
- **Multi-View Foundation Models** (Segre, Hirschorn, Avidan, arXiv 2512.15708) enforces multiview consistency in feature space on frozen backbones, including DINOv3, and reports improved 3D awareness across backbones — [arXiv](https://arxiv.org/pdf/2512.15708)
- **"How Much 3D Do Video Foundation Models Encode?"** (arXiv 2512.19949) finds that **cross-view feature similarity is a weak or biased proxy** for 3D awareness: DINOv2 and V-JEPA show strong cross-view correspondence but do poorly on 3D probes — [arXiv](https://arxiv.org/pdf/2512.19949)
- **Splat and Distill** (arXiv 2602.06032) distills feed-forward-3D-reconstruction-augmented teachers into DINOv2 and evaluates multi-view correspondence — [arXiv](https://arxiv.org/pdf/2602.06032)
- **DINOv3D** (ECCV 2026 poster) does 2D-3D joint optimization, adapting VGGT with small parameter updates — [ECCV page](https://eccv.ecva.net/virtual/2026/poster/4298). **VGGT-Ω** (arXiv 2605.15195) uses an unfrozen DINOv3 backbone — [arXiv](https://arxiv.org/html/2605.15195v1)
- Feat2GS (CVPR 2025) was **not verified this session**.

### Inferences
- [H, correspondence-framed] "Does MEF-style multiview-equivariance fine-tuning still help DINOv3, which already has RoPE and Gram anchoring and better NAVI recall (64.4 vs 60.1)? Does it fix left/right errors on SPair-Geo-Aware?" This is testable with released code and only a handful of rendered objects, since MEF reports one-object/one-iteration gains. **However**, Multi-View Foundation Models (Dec 2025) already applies multiview consistency to frozen DINOv3, so novelty is weak. Rendering requires a Blender toolchain (data-engineering risk).
- Framing flag for the report writer: if pursued, every result must be a **correspondence metric** (SPair Geo-Aware PCK, NAVI recall, TAP-Vid δavg). No mesh or point-cloud outputs. Otherwise it violates the user's scope.

### Gaps
- MEF training GPU and time are unknown (not in README). Objaverse 10k render size is unknown.
- Whether anyone applied MEF directly to DINOv3 was not exhaustively checked (Multi-View Foundation Models is the closest).

---

## Q5. General image matching (RoMa / MASt3R / LightGlue) without the culture-tech angle

### Takeaway
State-of-the-art dense matchers now sit on frozen DINOv3 (RoMa v2) or feed-forward 3D backbones (RoMa-Ω, Sep 2026, +8.1 mAA over RoMa v2 on WxBS). The "what do 3D/foundation models know about matching" analysis question was **just answered** by RoMa-Ω. **Training or fine-tuning matchers is beyond a 3-week Titan-V budget.** Inference-only evaluation is feasible but leaves little room for a contribution. **Not recommended as a primary direction.**

### Cited Findings
- **MASt3R** — "Grounding Image Matching in 3D with MASt3R", ECCV 2024 — [arXiv 2406.09756](https://arxiv.org/pdf/2406.09756), [ECVA PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09080.pdf)
- **RoMa v2** (arXiv 2511.15706; venue unverified) uses a **frozen DINOv3** backbone and reports gains over RoMa and MASt3R on MegaDepth and ScanNet pose estimation (secondary summary) — [emergentmind](https://www.emergentmind.com/papers/2511.15706)
- **RoMa-Ω** — "What Feed-Forward 3D Models Know About Image Matching", Nordström, Zhang, Loiseau, Lepetit, Kahl, arXiv 2609.09507 (8 Sep 2026) — [arXiv](https://arxiv.org/abs/2609.09507). VGGT and MASt3R features **match poorly zero-shot (especially in later layers)** but probe and train well. The authors retrain RoMa v2 with VGGT-Ω replacing DINO, which gives +8.1 mAA on WxBS. No code link seen.
- "Mismatched: Evaluating the Limits of Image Matching Approaches and Benchmarks" (arXiv 2408.16445) evaluates RoMa, MASt3R, and others on harder benchmarks and notes that MASt3R is the slowest — [arXiv](https://arxiv.org/html/2408.16445v1)
- The "A Patch is More Than a Token" frozen-DINOv3 study reports MegaDepth-1500 P@10 at ViT-S of 20.83 (center token) → 28.65 (its manifold variant) — [arXiv](https://arxiv.org/html/2610.41814v1)

### Inferences
- [E] RoMa-family training (MegaDepth, multi-GPU) is not feasible. MASt3R inference at 512 px on 12 GB is likely fine for pairs but slow. Contribution space for a 3-week solo project is limited to evaluation, which does not satisfy "running existing code as-is is NOT a project."
- Optional reuse: MegaDepth-1500 / HPatches could serve as a **third, geometric regime** in Candidate A's positional-bias analysis (frozen-feature NN matching only, not training a matcher).

### Gaps
- RoMa v2's venue and checkpoint details were not verified from the primary source. LightGlue 2025–26 successors were not surveyed (de-prioritized).

---

## Q6. Recommended candidate projects (2–3) and rejected directions

### Takeaway
**Top pick: Candidate A**, a training-free **analysis plus small method** on the **positional component of DINOv3 dense features across correspondence regimes** (semantic SPair vs temporal TAP-Vid), including its link to left/right errors. It is cheap (frozen inference), clearly evaluable, novel as a question (no direct prior found), and degrades gracefully. **Candidate B** (do upsamplers help or hurt correspondence, including tracking) is a safe analysis project with low paper potential. **Candidate C** (scaled-down weak adaptation of DINOv3 for tracking) has the most hands-on training experience but the highest risk and overlap (M2P, Chrono). Supervised SC adapters, new upsamplers, video-DiT tracking, matcher training, culture-tech matching, and 3D-reconstruction framings are rejected.

### Cited Findings (evidence base for the candidates; details in Q1–Q5)
- DINOv3 positional bias and the noise-SVD debias (s = 500) give +0.9 to +5.0 PCK@0.1 on SPair (S/B/L), versus ~+0.3 for DINOv2. INSID3 did no left/right or geo-aware analysis and calls positional information a task-dependent trade-off — [INSID3](https://arxiv.org/html/2603.28480)
- Left-right confusion of symmetric parts is an explicitly open failure mode — [Zhang et al. 2505.18060](https://arxiv.org/html/2505.18060). The geo-aware subset evaluation flag exists in the released code — [GeoAware-SC](https://github.com/Junyi42/GeoAware-SC)
- Zero-shot frozen DINOv3-B tracking: δavg 38.9 (M2P) / 41.3 (DiTracker) on TAP-Vid-DAVIS — [M2P](https://arxiv.org/html/2603.17813), [DiTracker](https://arxiv.org/html/2512.20606)
- On whether any work benchmarks DINOv3 denoising/debiasing on point tracking, a 2026-focused search found none ("that comparison remains open"). This is a negative search result, not proof of absence — [search context incl. INSID3 and DINOv3](https://arxiv.org/pdf/2603.28480)
- AnyUp and JAFAR are −14 PCK versus bilinear on SPair with DINOv3-S+ — [ViT-Up](https://arxiv.org/html/2606.14024v1). AnyUp is +8.7 J&F over bilinear on DAVIS label propagation (DINOv2-S, 672 px) — [WRC](https://arxiv.org/html/2605.17472). No upsampler paper found evaluates TAP-Vid point tracking — [ViT-Up](https://arxiv.org/html/2606.14024v1), [AnyUp](https://arxiv.org/html/2510.12764v2)
- M2P: LoRA DINOv3 with VOS masks takes DINOv3-B from 38.9 to 53.5 δavg on DAVIS. 2× RTX 4090 for ~1 day. Code not released — [M2P](https://arxiv.org/html/2603.17813)

### Inferences — Candidate A (RECOMMENDED): "Position is a feature, not a bug? The absolute-positional component of DINOv3 across semantic vs. temporal correspondence"
- **Type:** D (analysis) + light A (training-free method). **Course-level contribution:** a controlled multi-regime study plus an adaptive debiasing rule. **Paper-level (stretch):** only if a general principle emerges, for example "absolute-position leakage explains a measurable share of left/right flips and is beneficial exactly when the inter-image displacement prior holds," with a fix that improves both regimes across backbones (DINOv2/v3, possibly SigLIP2 or RADIO). Otherwise, an honest workshop-level analysis.
- **Input → Output:** (image pair or video, query point) → predicted correspondence. The analysis output is accuracy as a function of positional-subspace removal, displacement magnitude, and pose difference.
- **Hypotheses [H]:**
  - H1: debiasing improves cross-instance SC (replicate INSID3 on SPair-71k) but **is neutral or harmful for TAP-Vid tracking**, with harm concentrated in small-displacement tracks.
  - H2: on the SPair **Geo-Aware subset**, a measurable share of left/right flip errors occurs where the wrong-side part lies near the source keypoint's absolute coordinates. Debiasing should reduce these specifically for opposite-facing pairs, using GeoAware-SC's flip/pose test to identify them.
  - H3: an **adaptive** projection (strength or rank chosen per pair from a cheap global cue, such as estimated displacement or pose-flip score) dominates both "always-debias" and "never-debias."
  - Side question: does INSID3's fixed rank of 500 over-remove semantics for ViT-S (384 dims)? Test with a rank sweep normalized by width.
- **Minimum deliverable in 3 weeks:**
  - Frozen DINOv3 S/B/L plus a DINOv2-B control.
  - Own reimplementation of the noise-SVD debias (about 30 lines). This is reimplementation, not running INSID3's code as-is.
  - One unified evaluation harness: SPair-71k PCK@0.05/0.1 per-image and per-keypoint, Geo-Aware subset, and difficulty bins; TAP-Vid-DAVIS δavg/OA with NN + cycle-check tracking, binned by displacement; optional MegaDepth-1500 / HPatches NN matching as a geometric third regime.
  - Ablations: rank s, layer, resolution, and model size.
  - A failure gallery (flip errors, background-anchored matches).
- **Baselines (fair):** raw DINOv3/DINOv2 NN; INSID3 debias; GeoAware-SC's test-time flip adaptation (zero-shot, training-free) for the left/right comparison. Optionally a DVT-style or registers-style variant (DVT was not verified this session).
- **Data:** SPair-71k (70,958 pairs, 18 categories; the test subset alone suffices; download size unverified, roughly hundreds of MB [E]) and TAP-Vid-DAVIS (30 videos, small zip).
- **Compute [E]:** < 6 GB VRAM at 960 px for ViT-L in fp16. SPair-test feature extraction for S/B/L at 2–3 resolutions takes a few GPU-hours. TAP-Vid-DAVIS NN tracking takes < 1 GPU-hour per configuration. No training.
- **Volta risks:** the PyTorch 2.7.1–2.14 cu126 window, fp16 instead of bf16 (validate against fp32), and ViT-7B excluded (use L or H+).
- **Week plan [E]:**
  - W1 (11/18–24): environment, DINOv3 access request (gated; **request on day 1**), SPair NN baseline. Target: within ~2 PCK of a published number at matched settings.
  - W2: debias reproduction, TAP-Vid harness, the H1/H2 experiments.
  - W3: adaptive variant (H3), ablations, failure analysis, slides.
- **Success / failure criteria:**
  - Success = a statistically clear regime-dependent effect, for example a debias gain on SPair together with a ≥1 δavg loss on small-motion DAVIS tracks, or a flip-error reduction on the Geo-Aware subset.
  - If all effects are within noise, the course project still delivers a rigorous negative result ("positional leakage does not explain left/right errors / does not matter for tracking"), and the ICCV angle is dropped.
- **Learning value:** high (feature analysis, evaluation-protocol rigor, linear-algebra interventions). **Career value:** medium-high (DINOv3 dense features are widely used). **Main risks:** overlap with INSID3 (same authors are active: MARCO, INSID3), possible small effect sizes, and DINOv3 gated-access delays. Fallback: use the HF Transformers DINOv3 weights or DINOv2-with-registers.

### Inferences — Candidate B (SAFE): "Do feature upsamplers help or hurt correspondence? Semantic vs. temporal regimes"
- **Type:** D (analysis), optionally plus a tiny A (matching-aware test-time reweighting). **Course-level:** yes. **Paper-level:** low. ViT-Up and WRC took the SPair niche in May–June 2026, and at least 8 competing upsamplers exist.
- **Input → Output:** low-res DINO features plus an image → high-res features → SPair PCK@0.01/0.05/0.1 and TAP-Vid-DAVIS δavg (NN tracking).
- **Hypothesis [H]:** the SPair/DAVIS contradiction (AnyUp −14 PCK on SPair but +8.7 J&F on DAVIS VOS) comes from a smoothness-versus-distinctiveness trade-off. Measure NN-ratio distinctiveness and part-boundary leakage, and test whether point tracking behaves like VOS (helped) or like SPair (hurt).
- **Minimum deliverable:** inference-only comparison of bilinear, FeatUp-JBU, LoftUp, JAFAR, AnyUp, and (if weights exist) ViT-Up and WRC on DINOv2-S/B and DINOv3-S+/B. Run one harness across SPair, TAP-Vid-DAVIS, and the DAVIS VOS sanity check, plus distinctiveness diagnostics and failure cases.
- **Compute [E]:** inference fits easily. Do not retrain upsamplers: AnyUp's backward pass needs ~12.9 GB at 448² (H100 figure).
- **Risks:** checkpoint availability per backbone, results that only restate ViT-Up, and an "evaluation-only" project that may look thin unless the diagnostic plus a small fix is solid.

### Inferences — Candidate C (AMBITIOUS, riskiest): "Weakly-supervised adaptation of frozen DINOv3 for point tracking on a 12 GB budget"
- **Type:** C (reproduce and extend). Reimplement an M2P-lite (LoRA on DINOv3-S with the mask-consistency plus Procrustes losses) trained on DAVIS-2017 train only (60 videos) or a YouTube-VOS subset. Compare against zero-shot DINOv3, released Chrono checkpoints, and DINOv2.
- **Extension [H]:** combine with Candidate A's finding. Does a temporal adaptation re-learn or suppress the positional component? Or add a cheap occlusion head.
- **Minimum deliverable:** zero-shot baselines plus Chrono-checkpoint evaluation (week 1), the M2P-lite training curve and DAVIS δavg (week 2), and data-size and loss ablations (week 3).
- **Compute [E]:** LoRA on DINOv3-S/B at 256–384 px with batch 4–8 fp16 fits in 12 GB. The full M2P recipe (2× 4090 × 1 day) would take ~2–4 Titan-V days, so only a scaled-down version is feasible. Results will not be directly comparable to M2P's 53.5.
- **Novelty:** low (M2P, Chrono, DiTracker, many from KAIST). **Learning value:** highest (actual training loop, LoRA, TAP-Vid metrics). **Risks:** no M2P code, unknown xformers-on-Volta behavior for Chrono, training instability, and the 3-week window.

### Inferences — Backup D (only if 3D-flavored work is acceptable as correspondence)
- MEF-style multiview-equivariance fine-tuning applied to DINOv3, evaluated **only** on SPair Geo-Aware PCK, TAP-Vid δavg, and PF-PASCAL. It reuses the MIT-licensed MEF repo and its one-object finding. Risks: Blender/Objaverse rendering and the 2025–26 overlap (Multi-View Foundation Models, Splat-and-Distill).

### Inferences — Rejected directions (with reasons)
1. **New supervised / pseudo-label SC adapter on SPair.** Saturated: fine-tuned DINOv3 at 90.6, DIY-SC at 71.6, 3D-SC at 73.0, plus GECO, MARCO (CVPR 2026 Oral), and Jamais Vu. A new adapter is a small variant, and 3D-SC needs SAM 3D (3D-heavy, L40 40 GB).
2. **Diffusion-feature SC (DIFT / SD+DINO / distillation).** SD reportedly no longer complements DINOv3 (58.1 vs 58.4). SD feature extraction costs ~2 h on a 3090 plus ~90 GB of disk, and needs an old torch 1.13 stack.
3. **Training a new feature upsampler to SOTA.** At least 8 competitors in 2024–26. ViT-Up and WRC already cover SPair. AnyUp's backward pass needs ~12.9 GB at 448² (too tight for 12 GB). Learned upsamplers give little benefit over bilinear on segmentation (Upsample Anything).
4. **Video-DiT features for tracking (DiffTrack / DiTracker style).** CogVideoX-5B/WAN-14B do not fit in 12 GB [E]. These models are bf16-native, and Volta has no bf16. The area is also dominated by the originating lab.
5. **Training Chrono-style trackers on Kubric.** Requires 273 GB panning-MOVi-E plus xformers. Multi-day training is not feasible.
6. **Training or fine-tuning dense matchers (RoMa v2 / RoMa-Ω / MASt3R).** Multi-GPU training. The analysis question ("what 3D models know about matching") was answered in Sep 2026 by RoMa-Ω.
7. **Cross-era / painting-vs-photo / heritage matching.** Excluded by the user's scope change. Also, no standard benchmark was found: older work uses tiny ad-hoc sets, and the newest, HistReNeRF (ECCVW 2026), uses 230 archival photos across 3 landmarks — [HistReNeRF](https://arxiv.org/html/2608.15420), [ResearchGate benchmark paper](https://www.researchgate.net/publication/333637606_GENERATION_OF_A_BENCHMARK_DATASET_USING_HISTORICAL_PHOTOGRAPHS_FOR_AN_AUTOMATED_EVALUATION_OF_DIFFERENT_FEATURE_MATCHING_METHODS)
8. **3D-awareness fine-tuning framed as 3D reconstruction (FiT3D / Feat2GS-style).** Excluded by scope. Even when correspondence-framed (Backup D), it carries data-rendering overhead and a crowded 2025–26 field.

### Gaps
- No exhaustive check (e.g., Google Scholar "cited-by" on INSID3) was possible, so a very recent (Oct–Nov 2026) paper testing positional debiasing on tracking or left/right could exist. **Re-search "INSID3 positional bias tracking" and "DINOv3 positional left right" right before submitting the proposal.**
- ICCV 2027 dates and submission rules were not checked here (outside this note's scope).
- Exact SPair-71k test-split size, download sizes for SPair and TAP-Vid-DAVIS, and the Volta behavior of xformers and fp16 DINOv3 remain unverified and should be checked on day 1.
