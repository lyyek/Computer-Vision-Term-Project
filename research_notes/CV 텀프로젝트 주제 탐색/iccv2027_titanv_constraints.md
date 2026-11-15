# Practical Constraints: ICCV 2027 Schedule/Policies and Titan V (sm_70) Feasibility

Research date: 2026-11-15. Scope note: per the coordinator's update, 3D-specific checks (3DGS rasterizers, PyTorch3D, VGGT/DUSt3R/MASt3R VRAM) were skipped. Facts below cite official sources unless explicitly marked "aggregator", "forum", or "estimate".

---

## A1. Is ICCV 2027 officially announced (location, dates, deadlines)?

### Takeaway
ICCV 2027 is officially set for **Hong Kong (Hong Kong Convention and Expo Center, HKCEC)**. The official Dates page lists **workshops/tutorials Oct 3–4, 2027 and the main conference Oct 5–7, 2027**. As of 2026-11-15, the official site has **not announced any paper deadline**: no registration, submission, supplementary, rebuttal, or decision dates. ICCV 2025's official timeline is the best guide: registration in early March, the paper deadline about 4 days later, decisions in late June.

### Cited Findings
**ICCV 2027 (official):**
- The official ICCV site (iccv.thecvf.com) is now the ICCV 2027 site. Venue: Hong Kong Convention and Expo Center (HKCEC). — [ICCV 2027 homepage](https://iccv.thecvf.com/)
- **The official homepage contradicts itself on dates:** the header says **Oct 3–7, 2027**, but the Venue section says **Oct 2–8, 2027**. — [ICCV 2027 homepage](https://iccv.thecvf.com/)
- The official Dates page lists only conference dates: **Workshops and Tutorials Oct 3–4, 2027; Main Conference Oct 5–7, 2027** ("Sunday, October 3 – Thursday, October 7, 2027"). It has **no paper registration, submission, supplementary, review, rebuttal, decision, or camera-ready deadlines**. — [ICCV 2027 Dates](https://iccv.thecvf.com/Conferences/2027/Dates)
- The ICCV 2027 Author Guidelines page returns **HTTP 404**, so it has not been published yet. — [ICCV 2027 AuthorGuidelines (404)](https://iccv.thecvf.com/Conferences/2027/AuthorGuidelines)
- The CVF's list of future conferences confirms **ICCV 2027: Hong Kong** (also CVPR 2027 Seattle, WACV 2027 Orlando, ICCV 2029 location TBD, ICCV 2031 Lisbon). — [CVF Future Conferences](https://www.thecvf.com/?page_id=100)
- **Aggregator, unofficial:** opencurious.com lists an "ICCV 2027 Deadline: Mar 8, 2027 AoE (Expected)". This is a projection, not an official date. — [opencurious (aggregator)](https://www.opencurious.com/ai-conference-deadlines/iccv-2027)

**ICCV 2025 official timeline (REFERENCE PATTERN ONLY, not ICCV 2027 dates):**
- Paper Registration Deadline: **Mar 03 '25, 11:59 PM HST** — [ICCV 2025 Dates](https://iccv.thecvf.com/Conferences/2025/Dates)
- Submission **and** Supplementary Materials Deadline (same day): **Mar 07 '25, 11:59 PM HST** — [ICCV 2025 Dates](https://iccv.thecvf.com/Conferences/2025/Dates)
- Reviews Released: **May 09 '25**; Author Rebuttals Due: **May 16 '25** (11:59 PM HST) — [ICCV 2025 Dates](https://iccv.thecvf.com/Conferences/2025/Dates)
- Main Conference Paper Decisions: **Jun 25 '25, 11:59 PM HST** — [ICCV 2025 Dates](https://iccv.thecvf.com/Conferences/2025/Dates)
- Workshop proposals due Mar 13 '25 (AoE). Workshops Oct 19–20, 2025; main conference Oct 21–23, 2025, Honolulu Convention Center. — [ICCV 2025 Dates](https://iccv.thecvf.com/Conferences/2025/Dates)

### Inferences
- **Estimate, not official:** if ICCV 2027 follows the 2025 pattern, the paper deadline would fall around **early March 2027**. That is about 12–13 weeks after the term project ends on 2026-12-06, so roughly 3 months to turn the project into a full paper. Under the same pattern, reviews would come back around early-to-mid May and decisions around late June 2027.
- ICCV 2025 had the supplementary deadline on the same day as the paper (CVPR 2027 gives one extra week). Plan for "supplementary due with the paper" until ICCV 2027 says otherwise.

### Gaps
- No official ICCV 2027 paper deadlines exist as of 2026-11-15. Re-check https://iccv.thecvf.com/Conferences/2027/Dates and the CVF homepage.
- The Oct 2–8 vs Oct 3–7 inconsistency on the official homepage is unresolved. The Dates page (Oct 3–7) is the more specific source.

---

## A2. Submission requirements (most recent official ICCV/CVPR author guidelines)

### Takeaway
The newest official rules are **CVPR 2027's**, which tightened ICCV 2025's: 8-page limit (references and a **mandatory LLM Usage Statement** don't count), double-blind review, arXiv allowed, the 20%-overlap dual-submission rule, mandatory OpenReview profiles, and all authors in the reviewer pool. Irresponsible reviewing can get **all of your own papers desk-rejected**. CVPR 2027 also adds a **reciprocal-reviewer quota**: if no author qualifies as a reciprocal reviewer, an author may co-author at most one paper. ICCV 2027 will probably inherit most of this (inference).

### Cited Findings
**CVPR 2027 Author Guidelines (official, newest):**
- **Page limit:** "Papers are limited to eight pages, including figures and tables." Extra pages are allowed only for references and the mandatory LLM Usage Statement. Rebuttal limit: 1 page. Formatting violations or exceeding the limit → desk rejection. — [CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
- **Double-blind:** "CVPR reviewing is double blind." No identifying names, grant IDs, or links. A concurrent submission of your own must be cited in anonymized form via the supplementary. — [CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
- **Dual submission:** ≥20% content overlap with another submitted or published paper is not allowed. "Concurrent submission is not allowed." A peer-reviewed paper >4 pages (excluding references), including workshop papers, counts as a publication. — [CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
- **arXiv:** arXiv preprints and tech reports are not publications. Posting is permitted, but writing "submitted to CVPR" is strongly discouraged and media discussion of the paper as under submission is prohibited. Work first public **on or after Sep 15, 2026** counts as contemporaneous: cite it, but it is not grounds for rejection by itself. — [CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
- **Reviewer duty:** All authors are automatically added to the reviewer pool ("We expect all authors to be willing to serve as reviewers if asked"). Exemptions only for exceptional circumstances (busyness or reviewing elsewhere don't qualify). Highly irresponsible reviewers or ACs "may have all of their own CVPR 2027 submissions desk-rejected". Examples: missing reviews, careless reviews, uploading submissions to external LLMs. — [CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
- **Reciprocal-reviewer quota (new for 2027):** if no author on a paper is an eligible reciprocal reviewer, each author may co-author at most **one** such paper. Eligibility requires an accepted main-track paper or full journal article at a listed venue (e.g., CVPR, ICCV, NeurIPS, ICRA, SIGGRAPH) by the abstract deadline. — [CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
- **Max submissions:** "Each author may submit at most 25 papers." — [CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
- **LLM policy (authors):** LLM use is allowed, but authors are fully responsible ("an LLM did it" is no defense). LLMs cannot be authors. Fabricated or non-existent citations → rejection without review. A **mandatory "LLM Usage Statement"** goes after the main text and before the references, listing tools, affected parts, and extent, or saying "No LLMs or generative AI tools were used in this work." Missing statement → possible desk rejection. Hidden prompt injection → desk rejection. — [CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
- **LLM policy (reviewers):** Reviewers and ACs may process submissions only through conference-approved LLM tools inside OpenReview. Uploading to any other AI service is misconduct. LLMs don't make decisions or scores. Papers cannot opt out of conference-side LLM processing, and content isn't used for training. — [CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
- **OpenReview:** Submissions go through OpenReview. **Every author** needs a complete OpenReview profile and enrollment (including data-use and AI consent) by the abstract deadline: "Papers with authors who have not completed their profiles will be desk-rejected." Conflicts must be declared. Publications used for reciprocal-reviewer eligibility must be imported into OpenReview (a DBLP link alone isn't enough). Authors can't be added or removed after the abstract deadline. — [CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
- **Supplementary:** Optional and anonymous; reviewers are "encouraged... but not obligated" to read it. No external links that expand content. Rebuttals cannot add new experiments or links. — [CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)

**ICCV 2025 Author Guidelines (official, most recent ICCV-specific):**
- 8 pages excluding references, in ICCV style. Over-length, non-anonymized, or off-template papers are rejected without review. — [ICCV 2025 Author Guidelines](https://iccv.thecvf.com/Conferences/2025/AuthorGuidelines)
- "Every author is required to serve as a reviewer (unless they hold another role in ICCV 2025)." — [ICCV 2025 Author Guidelines](https://iccv.thecvf.com/Conferences/2025/AuthorGuidelines)
- 20% overlap dual-submission rule. arXiv posting allowed. Max 25 submissions per author. All authors need a valid OpenReview profile by the submission deadline or face desk rejection. Authors are locked at registration (Mar 3). The author list is final after the paper deadline. 1-page rebuttal. — [ICCV 2025 Author Guidelines](https://iccv.thecvf.com/Conferences/2025/AuthorGuidelines)
- LLM policy (2025): AI tools allowed but authors are responsible. Citations to non-existent material → rejection. — [ICCV 2025 Author Guidelines](https://iccv.thecvf.com/Conferences/2025/AuthorGuidelines)

### Inferences
- ICCV 2027 will likely adopt the CVPR 2027 rules: mandatory LLM Usage Statement, reviewer-misconduct desk rejections, and the reciprocal-reviewer quota (the CVPR→ICCV carry-over pattern held in 2025). This is not confirmed. **Practical implication:** the student should create and complete an OpenReview profile early. If the student has no prior main-track paper, at least one co-author (e.g., the advisor) should be an eligible reciprocal reviewer so the quota doesn't limit submissions.

### Gaps
- No ICCV 2027 author guidelines are published yet (404). The rebuttal format and reviewer requirements for ICCV 2027 are unknown.

---

## A3. Nearby alternatives (CVPR 2027, ECCV 2026, NeurIPS 2027, ICCV 2027 workshops)

### Takeaway
**CVPR 2027 is effectively closed for this project:** paper registration closed Nov 10, 2026, and full papers are due **Nov 16, 2026 (AoE)**, tomorrow. ECCV 2026 is past. NeurIPS 2027 is listed only as "Europe" with no official dates. ICCV 2027 workshops run Oct 3–4, 2027, but no workshop paper deadlines are announced.

### Cited Findings
- **CVPR 2027 (Seattle, official):** Workshop proposals Nov 04 '26. **Paper Registration Nov 10 '26. Submission Nov 16 '26. Supplementary Nov 23 '26** (all AoE). Reviews released Jan 25 '27. Final decisions Feb 25 '27. Workshops/tutorials June 20–21, 2027; main conference June 22–25, 2027. — [CVPR 2027 Dates](https://cvpr.thecvf.com/Conferences/2027/Dates); location — [CVPR 2027 homepage](https://cvpr.thecvf.com/)
- CVPR 2027 guidelines: "Extensions are not granted"; authors are locked after the Nov 10 abstract deadline. — [CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
- **ECCV 2026:** listed by CVF as Malmö, 2026. — [CVF Future Conferences](https://www.thecvf.com/?page_id=100)
- **NeurIPS 2027:** the official future-meetings page lists "2027 — Europe" (2028 — Western US). No city or dates. — [NeurIPS Future Meetings](https://neurips.cc/Conferences/FutureMeetings) (read via search-result summary, not fetched directly). An aggregator projects an abstract deadline of May 14, 2027 and a paper deadline of May 21, 2027. **Unofficial.** — [opencurious (aggregator)](https://www.opencurious.com/ai-conference-deadlines/neurips-2027)
- **ICCV 2027 workshops:** Oct 3–4, 2027. No workshop proposal or paper deadlines listed. — [ICCV 2027 Dates](https://iccv.thecvf.com/Conferences/2027/Dates). For reference, ICCV 2025 workshop proposals were due Mar 13, 2025. — [ICCV 2025 Dates](https://iccv.thecvf.com/Conferences/2025/Dates)
- **WACV 2027:** Orlando. Dates not checked. — [CVF Future Conferences](https://www.thecvf.com/?page_id=100)

### Inferences
- ECCV 2026's submission window was in early 2026 (CVF lists it as a 2026 event). It is long past for a project starting 2026-11-18.
- The realistic path is term project (Dec 6, 2026) → ICCV 2027 (expected ~early March 2027, not confirmed). If rejected, the next options are NeurIPS 2027 (~May 2027, unofficial projection) or ICCV 2027 workshops.

### Gaps
- ECCV 2026's exact dates weren't fetched (irrelevant, already past). WACV 2027 deadlines weren't checked. NeurIPS 2027 dates are not announced.

---

## B1. PyTorch / CUDA support for sm_70 (Titan V) in late 2026

### Takeaway
**Volta has been dropped from current prebuilt PyTorch.** PyTorch **2.15.0 (Oct 28, 2026)** ships only `cu132`/`cu134` wheels, which need Turing (sm_75) or newer. The **last prebuilt wheels that support a Titan V are PyTorch 2.14.x with `cu126`** (2.14.1, Sep 30, 2026 is the latest patch). CUDA 13.x cannot compile for sm_50–sm_72 offline. **Recommended combo: `torch==2.14.1` with the cu126 index + NVIDIA R580 driver**, or build from source with a CUDA 12.x toolkit and `TORCH_CUDA_ARCH_LIST=7.0`.

### Cited Findings
- **PyTorch 2.15.0** (released Oct 28): "The 2.15 wheels are `cu132` and `cu134`; `cu126` and `cu130` are no longer built." "CUDA 13 wheels need NVIDIA driver 580 or newer and a GPU of Turing (`sm_75`) or newer." "Maxwell, Pascal and Volta GPUs and hosts with a CUDA 12.x driver should stay on 2.14." 2.15 also raises the minimum Python to 3.11. — [PyTorch v2.15.0 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)
- **Official notice (Sep 4, 2026):** "PyTorch 2.14 is the last release to ship any CUDA 12.x wheels." From 2.15, Maxwell (sm_50), Pascal (sm_60), and Volta (sm_70) have no prebuilt support. Recommendations: pin `torch==2.14.0` with the `cu126` index, or build from source with a CUDA 12.6 toolkit and `TORCH_CUDA_ARCH_LIST=7.0`. — [PyTorch dev-discuss: CUDA 12.6 wheels end at 2.15](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432); RFC — [pytorch#190385](https://github.com/pytorch/pytorch/issues/190385)
- Release dates on GitHub: v2.15.0 (28 Oct), v2.14.1 (30 Sep), v2.14.0 (02 Sep), v2.13.0 (08 Jul), v2.12.0 (13 May), v2.11.0 (23 Mar), v2.10.0 (21 Jan), v2.9.0 (15 Oct). — [PyTorch GitHub Releases](https://github.com/pytorch/pytorch/releases)
- **PyTorch 2.11** removed Volta (SM 7.0) from the CUDA 12.8 and 12.9 binaries and pointed Volta users to the CUDA 12.6 builds. PyPI default became CUDA 13.0. — [PyTorch GitHub Releases](https://github.com/pytorch/pytorch/releases). Reason: "Keeping Volta supports prevents us from updating CuDNN" (newer cuDNN dropped Volta). The 2.11 support matrix lists Volta only under CUDA 12.6.3. — [dev-discuss: Dropping Volta from CUDA-12.8 binaries for 2.11 (Jan 15, 2026)](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290); [RFC pytorch#172352](https://github.com/pytorch/pytorch/issues/172352)
- **PyTorch 2.12:** minimum CUDA for source builds is 12.6. CUDA 12.8 builds deprecated, CUDA 13.0 the stable default, CUDA 12.6 kept for older drivers. — [PyTorch GitHub Releases](https://github.com/pytorch/pytorch/releases)
- **PyTorch 2.8:** Maxwell and Pascal were removed from the CUDA 12.8/12.9 builds, while CUDA 12.6 kept them. — [dev-discuss: Maxwell/Pascal removed in CUDA 12.8 and 12.9 builds](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128). **Conflict:** one search summary paraphrased the 2.8 notes as removing "sm50 through sm70". But the separate Volta-from-cu128 removal in 2.11 means Volta stayed in cu128/cu129 for 2.8–2.10. Trust the 2.11 sources.
- **CUDA 13.0:** NVIDIA removed offline compilation and library support for Maxwell, Pascal, and Volta (CC 5.x, 6.x, 7.0, 7.2). NVCC "won't be able to... build an application that targets any GPU architecture before CC 7.5". NVIDIA recommends staying on **driver branch 580 and CUDA Toolkit 12.9** for these GPUs. Apps built with CUDA ≤12.9 "will continue to run on all hardware supported by the NVIDIA driver." — [NVIDIA blog: Navigating GPU architecture support](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers); [CUDA Features Archive](https://docs.nvidia.com/cuda/cuda-features-archive/index.html). CUDA 12.9 had warned in advance: "offline compilation and library support will be removed in the next major CUDA Toolkit version release." — quoted in [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-to-drop-cuda-support-for-maxwell-pascal-and-volta-gpus-with-the-next-major-toolkit-release)

**PyTorch CUDA variant vs sm_70 (verified where cited):**

| Variant | Includes sm_70? | Source |
|---|---|---|
| cu126 | **Yes, through 2.14.x**; not built from 2.15 | [2.11 matrix](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290), [2.15 notes](https://github.com/pytorch/pytorch/releases/tag/v2.15.0) |
| cu128 / cu129 | Yes in 2.8–2.10; **No from 2.11** | [Releases (2.11 notes)](https://github.com/pytorch/pytorch/releases) |
| cu130 / cu132 / cu134 | **No** (CUDA 13 can't target <7.5) | [NVIDIA blog](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers), [2.15 notes](https://github.com/pytorch/pytorch/releases/tag/v2.15.0) |
| cu118 / cu121 / cu124 | Older PyTorch only (≤ ~2.7 era). Not verified this session | — (gap) |

### Inferences
- **Recommended for a Titan V (late 2026):** `pip install torch==2.14.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`, with Python 3.10–3.13 and an R580 driver. (The 2.14.1 patch is inferred to exist on cu126 because 2.14 is the last cu126 release; the official notice's example pins 2.14.0.) Use a CUDA 12.6–12.9 toolkit for compiling custom ops so nvcc can emit sm_70.
- Many 2026 repos will pin torch ≥2.15 or ship cu13 wheels for their extensions. Expect to downgrade or rebuild such dependencies, which takes time out of a 3-week project.
- The pip default (`pip install torch`) on Linux will pull 2.15 + cu132 and **fail on Titan V**, typically with "no kernel image is available" or an unsupported-GPU warning. Always pin.

### Gaps
- I didn't verify sm_70 inclusion in old cu118/cu121/cu124 wheels or the exact torchvision version paired with 2.14.1. Check https://download.pytorch.org/whl/cu126 before setup.

---

## B2. NVIDIA driver support status for Volta

### Takeaway
**R580 is the final driver branch that supports Volta (Titan V).** It is a Long-Term Support branch supported until about mid-2028, so drivers are fine for this project and an ICCV 2027 timeline, but new driver features won't reach Volta.

### Cited Findings
- NVIDIA: branch 580 is "the final branch that supports GPUs before CC 7.5". It is an LTS branch supported for three years, "until mid-year 2028". Recommendation: "Remain on branch 580" and CUDA "12.9". — [NVIDIA Technical Blog](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers)
- NVIDIA's Unix driver deprecation notice says 580 will be the last branch for Maxwell, Pascal, and Volta. The branch is shared with Windows, and press coverage notes Titan V is the only consumer Volta card. — [Tom's Hardware](https://www.tomshardware.com/pc-components/gpu-drivers/nvidia-to-axe-maxwell-pascal-and-volta-gpus-with-end-of-driver-support-580-series-drivers-will-be-the-last-to-support-gtx-900-and-1000-cards) (secondary report of the NVIDIA forum notice)
- Third-party note: NVIDIA's open kernel modules don't support Maxwell, Pascal, or Volta, so Volta needs the proprietary kernel module. — [negativo17 580 LTS repo (third-party)](https://negativo17.org/nvidia-driver-580-lts-repository/)

### Inferences
- With R580, a cu126 PyTorch wheel runs fine: an R580 driver also runs CUDA 12.x applications. Do not upgrade the lab machine to a 590+ driver.

### Gaps
- I didn't fetch NVIDIA's original Unix forum deprecation post directly (I used a secondary report plus the official NVIDIA blog).

---

## B3. bf16 on Volta and handling bf16-trained checkpoints

### Takeaway
Volta has **no native bf16**: no bf16 tensor cores and no TF32. PyTorch lets bf16 tensors exist and **emulates** bf16 math in fp32, so it is usually no faster than fp32. `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True`, so it can return True on Volta. **Practical rule:** load bf16 checkpoints and cast to **fp16** for speed (tensor cores) when activations stay within range. Otherwise fall back to **fp32** (always safe, 2× the memory). Watch for inf/NaN overflow in fp16 with bf16-pretrained models, especially LLM/VLM attention.

### Cited Findings
- Signature: `torch.cuda.is_bf16_supported(including_emulation=True)`, documented as "Return a bool indicating if the current CUDA/ROCm device supports dtype bfloat16." — [PyTorch 2.15 docs](https://docs.pytorch.org/docs/2.15/generated/torch.cuda.is_bf16_supported.html)
- Forum (PyTorch Discuss): on V100, bf16 runs but hardware without bf16 compute emulates it via fp32. Users report bf16 AMP about the same speed as fp32, and slower than fp16 on GV100. `is_bf16_supported()` returned True on a DGX V100. — [PyTorch Forums: Bfloat16 on V100](https://discuss.pytorch.org/t/bfloat16-on-nvidia-v100-gpu/201629)
- PyTorch official blog: "fine-tuning bfloat16-pretrained models in float16 can easily run into range issues in float16... users should stick with bfloat16 fine-tuning if the model was trained in bfloat16." TF32 exists only on Ampere and later. Mixed precision (fp16) on V100 was 1.5–5.5× faster than fp32. — [PyTorch Blog: Mixed Precision](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
- In PyTorch SDPA, the memory-efficient kernel on devices **below sm80 accepts only fp16/fp32, not bf16**. bf16 attention on Volta therefore falls back to the slower, memory-hungry math path. — [PyTorch sdp_utils.cpp (main)](https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)
- An fp16 NaN example in a VLM: transformers PR #33312 fixed Qwen2-VL float16 inference, where inf values in the attention weights turned into NaN after softmax. — [transformers PR #33312](https://github.com/huggingface/transformers/pull/33312); user report of "probability tensor contains inf/nan" with fp16 — [HF Qwen2-VL-7B discussion #20](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/discussions/20)
- Official CV repos default to bf16 in their examples: SAM 2 uses `torch.autocast("cuda", dtype=torch.bfloat16)` — [SAM 2 README](https://github.com/facebookresearch/sam2); DINOv3 examples use bf16 autocast — [DINOv3 README](https://github.com/facebookresearch/dinov3); SD3.5-Medium examples use `torch_dtype=torch.bfloat16` — [SD3.5-Medium model card](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)

### Inferences
- **Guidance (inference):** In every repo, replace `dtype=torch.bfloat16` with `torch.float16`, then validate: compare a few outputs against an fp32 run and check for NaN/inf. Vision backbones (DINOv2/v3 ViT-S/B/L, SAM 2, Depth Anything V2) usually work in fp16. LLM/VLM decoders and large DiTs are riskier. Mitigations: keep LayerNorm/softmax in fp32 (autocast does this), clamp attention logits, or run sensitive blocks in fp32.
- **Training:** use fp16 autocast + GradScaler. Do not plan bf16 training on the Titan V; it is emulated and slow.

### Gaps
- No official NVIDIA or PyTorch matrix lists which individual bf16 ops (e.g., cuBLAS bf16 GEMM) run natively vs emulated on sm_70. A search snippet reported fp16 NaNs from DINOv3 on V100 at larger inputs under TensorRT, but the NVIDIA forum page returned 403 and could not be verified.

---

## B4. FlashAttention, PyTorch SDPA backends, and xformers on sm_70

### Takeaway
**No official FlashAttention version (v1/v2/v3/v4) supports Volta.** On a Titan V, PyTorch SDPA can only use the **memory-efficient (CUTLASS, xformers-derived) backend (sm50–sm121, fp16/fp32 only)** and the math backend. The Flash and cuDNN backends require sm80+. xformers no longer documents Volta support, so use PyTorch SDPA's memory-efficient backend instead.

### Cited Findings
- FlashAttention-2: "Ampere, Ada, or Hopper GPUs"; Turing has a separate repo; bf16 requires Ampere+. FA3: "H100 / H800 GPU, CUDA >= 12.3". FA4: Hopper and Blackwell. Volta/V100 is not mentioned. — [Dao-AILab/flash-attention README](https://github.com/Dao-AILab/flash-attention)
- PyTorch SDPA hardware checks (current main): Flash requires `sm80`–`sm121` ("Flash attention only supports gpu architectures in the range [sm80, sm121]"). **Memory-efficient attention supports `sm50`–`sm121`**. cuDNN attention requires `sm80`–`sm121`. Below sm80, mem-efficient dtypes are `{Half, Float}` (no bf16). — [PyTorch sdp_utils.cpp](https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)
- xformers now: "pure Python, and runs on any PyTorch 2.10+". Attention kernels come from the separate `mslk` package, version-matched to torch. Install examples use the cu132/cu130 indexes. The README names no compute capabilities and doesn't mention Volta. — [xformers README](https://github.com/facebookresearch/xformers)
- Unofficial: a third-party "flash-attention-legacy" project targets Pascal/Volta and notes PyTorch SDPA's memory_efficient backend already covers these GPUs. — [sirCamp/flash-attention-legacy (third-party)](https://github.com/sirCamp/flash-attention-legacy)

### Inferences
- In HF transformers on Titan V, use `attn_implementation="sdpa"` with fp16 (or "eager"). `flash_attention_2`, which Qwen2.5-VL's model card recommends, will not work. Memory use for long sequences will be higher than on Ampere.
- Matching recent xformers/mslk wheels to torch 2.14+cu126 is likely painful or impossible. Avoid xformers dependencies, or patch repos to use `F.scaled_dot_product_attention`.

### Gaps
- I didn't verify which (if any) xformers/mslk release ships sm_70 kernels for torch 2.14+cu126.

---

## B5. Custom CUDA ops and libraries (MSDeformAttn, bitsandbytes, vLLM; 3D ops skipped per scope change)

### Takeaway
**MSDeformAttn** builds from source with nvcc 12.x for sm_70. Deformable DETR was benchmarked on a V100, so it should work. **bitsandbytes** prebuilt Linux wheels for CUDA ≤12.8 include **sm70**, so **NF4/FP4 4-bit and 8-bit optimizers work**, but **LLM.int8() needs CC 7.5+, so 8-bit inference via LLM.int8 does not work on Titan V**. CUDA 13 bitsandbytes wheels omit sm70. **vLLM requires CC ≥7.5, so it is not supported on Titan V.**

### Cited Findings
- Deformable DETR: "Linux, CUDA>=9.2, GCC>=5.4", PyTorch>=1.5.1. The MultiScaleDeformableAttention op compiles via `sh ./make.sh` in `models/ops`. Speed was measured on an "NVIDIA Tesla V100 GPU". — [fundamentalvision/Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
- bitsandbytes: supported on CC 6.0+. **LLM.int8() requires CC 7.5+.** 8-bit optimizers and NF4/FP4 quantization require CC 6.0+. Linux x86-64 wheel targets: CUDA 11.8–12.6 → sm60, **sm70**, sm75…sm90; CUDA 12.8 → **sm70**…sm120; CUDA 13.0–13.2 → sm75+ (no sm70). Building from source is supported down to CUDA 11.8. — [bitsandbytes installation docs](https://huggingface.co/docs/bitsandbytes/main/en/installation)
- vLLM: "GPU: compute capability 7.5 or higher (e.g., T4, RTX20xx, A100, L4, H100, B200, etc.)". — [vLLM GPU installation docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)
- SAM 2 installs a custom CUDA kernel that needs `nvcc` matching the PyTorch CUDA version. If the build fails, install continues with limited post-processing. — [SAM 2 README](https://github.com/facebookresearch/sam2)

### Inferences
- For any custom op (MSDeformAttn, SAM 2's connected-components kernel, etc.): install a CUDA 12.6 toolkit to match the cu126 torch wheel and set `TORCH_CUDA_ARCH_LIST="7.0"` before building.
- For 4-bit VLM/LLM inference on Titan V, use HF transformers + bitsandbytes NF4 (`load_in_4bit=True`, `bnb_4bit_compute_dtype=torch.float16`; bf16 compute would be emulated). Avoid `load_in_8bit` (LLM.int8). Instead of vLLM, use plain HF `generate`, or possibly llama.cpp (not verified here).
- 3DGS rasterizers, gsplat, and PyTorch3D were not checked (excluded by the scope change).

### Gaps
- I didn't verify whether the newest bitsandbytes release still builds sm70 into its CUDA 12.x wheels after torch 2.15. The docs table currently says yes for cu≤12.8.

---

## B6. VRAM needs on a 12 GB Titan V (vision backbones, generative models, VLMs)

### Takeaway
All mid-size **vision backbones fit easily** in fp16: DINOv2/v3 ViT-S/B/L, SAM 2 (all sizes), and Depth Anything V2 S/B/L. **DINOv3 ViT-7B does not fit** in fp16 (~13.4 GB of weights alone). **SD3.5-Medium officially needs 9.9 GB excluding text encoders**, which is tight on 12 GB and requires offloading the text encoders. **7B VLMs in fp16 don't fit.** **LLaVA-1.5-7B in 4-bit uses <8 GB (official).** Qwen2.5-VL-3B in fp16 and Qwen2.5-VL-7B in 4-bit are estimated to fit.

### Cited Findings (official parameter counts and VRAM statements)
- **DINOv3:** ViT-S/16 21M, ViT-S+ 29M, ViT-B/16 86M, ViT-L/16 300M, ViT-H+/16 840M, **ViT-7B/16 6,716M**; ConvNeXt T/S/B/L 29/50/89/198M. No VRAM figures given; requires PyTorch ≥2.7.1. — [DINOv3 README](https://github.com/facebookresearch/dinov3)
- **SAM 2 / 2.1:** hiera_tiny 38.9M, small 46M, base_plus 80.8M, large 224.4M (FPS 91.2/84.8/64.1/39.5 on an A100, torch 2.5.1, CUDA 12.4). No VRAM statement; requires PyTorch ≥2.5.1. — [SAM 2 README](https://github.com/facebookresearch/sam2)
- **Depth Anything V2:** Small 24.8M (Apache-2.0), Base 97.5M, Large 335.3M, Giant 1.3B ("coming soon"); Base/Large/Giant are CC-BY-NC-4.0. Default input size 518. No VRAM statement. — [Depth Anything V2 README](https://github.com/DepthAnything/Depth-Anything-V2)
- **SD3.5 Medium:** "2.5 billion parameters... MMDiT-X". "This model only requires 9.9 GB of VRAM (excluding text encoders)". — [Stability AI announcement](https://stability.ai/news-updates/introducing-stable-diffusion-3-5). The model card suggests 4-bit NF4 via bitsandbytes plus `enable_model_cpu_offload()`. The card's default dtype is bf16. — [SD3.5-Medium model card](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
- **LLaVA-1.5-7B:** "With 4-bit quantization, for our LLaVA-1.5-7B, it uses less than 8GB VRAM on a single GPU." "LLaVA-13B with 4-bit quantization allows you to run on a GPU with as few as 12GB VRAM". The README suggests multiple GPUs if VRAM <24 GB (for full precision). — [LLaVA README](https://github.com/haotian-liu/LLaVA)
- **Qwen2.5-VL:** 3B/7B/72B variants (the HF sidebar shows "8B params" for 7B). Recommends flash_attention_2 (not usable on Volta). Visual tokens per image range 4–16384, and the card suggests limiting via `min_pixels=256*28*28`, `max_pixels=1280*28*28` to balance memory. No VRAM figures. — [Qwen2.5-VL-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

### Inferences (ESTIMATES: weights-only = params × 2 bytes for fp16, ~0.5–0.6 bytes for NF4; activations extra; Titan V usable ≈ 11.5–11.8 GiB)
| Model | fp16 weights (est.) | Fits 12 GB? (est.) |
|---|---|---|
| DINOv2/v3 ViT-S / B / L | ~0.04 / 0.17 / 0.6 GB | **Yes**, inference and fine-tuning at moderate batch sizes |
| DINOv3 ViT-H+ (840M) | ~1.7 GB | Yes (inference); fine-tuning is tight |
| DINOv3 ViT-7B (6.7B) | ~13.4 GB | **No in fp16.** Only 4-bit NF4 (~4 GB) is plausible, inference only |
| SAM 2.1 tiny → large | ~0.08 → 0.45 GB | **Yes**. Video memory bank grows with frames, but should fit |
| Depth Anything V2 S / B / L | ~0.05 / 0.2 / 0.67 GB | **Yes** at 518 px; high-res inputs raise activation memory |
| SD 1.5 (~1.1B total, est.) | ~2 GB | **Yes** (512²), including LoRA fine-tuning with fp16 + gradient checkpointing |
| SDXL (~3.5B total, est.) | ~7 GB | **Inference yes** at 1024² with VAE slicing/tiling; full fine-tuning no; LoRA tight |
| SD3.5-Medium | 9.9 GB official (excl. text encoders) | **Tight.** Needs text-encoder CPU offload or dropping T5, or NF4. Also bf16→fp16 risk |
| Qwen2.5-VL-3B (~3.8B) | ~7.5 GB | **Likely yes** with capped `max_pixels`, SDPA, fp16 (validate for NaN) |
| Qwen2.5-VL-7B (~8.3B) | ~16.6 GB | **No in fp16; yes in NF4** (~5–6 GB) |
| LLaVA-1.5-7B (~7B + 0.3B ViT) | ~14 GB | **No in fp16; yes in 4-bit** (<8 GB, official) |

- SD1.5/SDXL parameter totals and the SDXL fp16 VAE NaN issue come from general knowledge, not fetched this session. Treat them as estimates.
- No 7B-class model can be fine-tuned on a single Titan V, except perhaps QLoRA-style NF4 + LoRA at very small batch and resolution (fp16 compute only; bf16 would be emulated). Projects should centre on ≤1B vision backbones, or use VLMs as frozen inference-only components.

### Gaps
- No official VRAM numbers for DINOv2/v3, SAM 2, Depth Anything V2, SD1.5/SDXL, or Qwen2.5-VL were found. All of those fits are estimates. DINOv2 parameter counts weren't fetched (the DINOv3 README doesn't list them; DINOv2 ViT-S/B/L/g are commonly cited as 21M/86M/300M/1.1B, unverified here).

---

## B7 (Bonus). Prof. Sunghoon Im's research areas

### Takeaway
Prof. Im's lab page lists the **KAIST Robotics and Vision Lab** (KAIST, Daejeon; the footer links to KAIST GSCT). Areas: **computer vision, robot learning, multimodal generative AI, 3D perception & scene understanding, and vision-language-action (VLA) models** ("Physical AI"). His July 2025 CV lists DGIST EECS with expertise in 3D vision, multimodal video understanding, and multimodal generative AI.

### Cited Findings
- Lab homepage: "KAIST Robotics and Vision Lab", N5 KAIST, Daehak-ro 291, Daejeon; footer links to KAIST GSCT. Areas: computer vision, robot learning, multimodal generative AI, 3D perception and scene understanding, VLA models for robots/autonomous systems. — [sunghoonim.github.io](https://sunghoonim.github.io)
- CV (updated July 2025): Associate Professor, Dept. of EECS, DGIST. Expertise: "3D Vision, Multi-modal Video Understanding, Multi-modal Generative AI". — [CV_SunghoonIM.pdf](https://sunghoonim.github.io/assets/CV_SunghoonIM.pdf) (via search summary)
- DGIST repository profile (previous affiliation). — [DGIST Scholar](https://scholar.dgist.ac.kr/handle/20.500.11750/12135)

### Inferences
- Since 3D-vision topics are excluded, the instructor's non-3D overlap areas are multimodal generative AI, multimodal video understanding, and VLA/robot-perception. These are good alignment targets, though 7B-class VLAs/VLMs are heavy for a Titan V (see B6).

### Gaps
- I didn't confirm his exact KAIST title or department (GSCT vs EE) from an official KAIST faculty directory.
