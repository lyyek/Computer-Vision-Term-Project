#!/usr/bin/env python3
"""bench.py 결과(results/*.json)를 표로 정리하고, fp16과 NF4를 비교해 판정한다.

    python summarize.py

표준 라이브러리만 쓴다. 결과는 화면에 출력되고 results/summary.md에도 저장된다.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 판정 기준 (HANDOFF.md 6절의 제안값). 단위: 질문 1개당 초 (전처리 + forward)
FAST_SEC = 1.0
SLOW_SEC = 3.0


def auroc(scores: list[float], labels: list[int]) -> float | None:
    """무작위로 고른 yes 문항의 점수가 no 문항의 점수보다 클 확률. 동점은 0.5로 센다."""
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    n_pos = sum(label for _, label in pairs)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum_pos = sum(rank for rank, (_, label) in zip(ranks, pairs) if label)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def run_metrics(r: dict) -> dict:
    timing = r["config"]["timing"]
    valid = [q for q in r["per_question"] if q["s"] is not None]
    fwd = r["probe"]["sec_forward"]["median"]
    per_probe = fwd + r["probe"]["sec_preprocess"]["median"]

    # 생성: 첫 토큰은 프롬프트 전체를 forward하고, 이후 토큰은 한 스텝씩 추가로 든다.
    steps = [(g["sec"] - fwd) / (g["new_tokens"] - 1) for g in r["generation"] if g["new_tokens"] > 1]
    step = statistics.median(steps) if steps else None
    chair_hours = None
    if step is not None:
        chair_hours = timing["chair_images"] * (fwd + timing["chair_avg_tokens"] * step) / 3600

    m = {
        "name": r["mode"] + (f"-{r['tag']}" if r.get("tag") else ""),
        "kind": r["config"]["modes"][r["mode"]]["kind"],
        "n": len(r["per_question"]),
        "sec_per_probe": per_probe,
        "sec_forward_p90": r["probe"]["sec_forward"]["p90"],
        "pope_hours": per_probe * timing["pope_full_size"] / 3600,
        "sec_per_token": step,
        "chair_hours": chair_hours,
        "peak_vram": r["memory"]["gpu"].get("peak_allocated_gib"),
        "cpu_rss": r["memory"]["cpu_rss_gib"],
        "nonfinite": r["probe"]["nonfinite_count"],
        "synthetic": r.get("synthetic", False),
        "yes_ratio": None, "acc": None, "auroc": None,
    }
    if valid:
        m["yes_ratio"] = sum(q["pred"] == "yes" for q in valid) / len(valid)
        if not m["synthetic"]:
            m["acc"] = sum(q["pred"] == q["label"] for q in valid) / len(valid)
            m["auroc"] = auroc([q["s"] for q in valid], [int(q["label"] == "yes") for q in valid])
    return m


def compare(a: dict, b: dict) -> dict | None:
    """같은 문항에 대한 두 실행(fp16, NF4)의 출력 차이."""
    sa = {q["question_id"]: q for q in a["per_question"] if q["s"] is not None}
    sb = {q["question_id"]: q for q in b["per_question"] if q["s"] is not None}
    common = sorted(set(sa) & set(sb))
    if not common:
        return None
    da = [sa[i]["s"] for i in common]
    db = [sb[i]["s"] for i in common]
    flips = [i for i in common if sa[i]["pred"] != sb[i]["pred"]]
    yes_a = sum(sa[i]["pred"] == "yes" for i in common) / len(common)
    yes_b = sum(sb[i]["pred"] == "yes" for i in common) / len(common)
    return {
        "n": len(common),
        "agreement": 1 - len(flips) / len(common),
        "flips": flips,
        "mean_abs_ds": statistics.fmean(abs(x - y) for x, y in zip(da, db)),
        "max_abs_ds": max(abs(x - y) for x, y in zip(da, db)),
        "mean_ds": statistics.fmean(y - x for x, y in zip(da, db)),  # NF4 - fp16
        "pearson": pearson(da, db),
        "yes_ratio_diff": yes_b - yes_a,                             # NF4 - fp16
    }


def fmt(x, spec: str = ".3f", none: str = "-") -> str:
    return none if x is None else format(x, spec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--include-dryrun", action="store_true", help="개발용 dry-run 결과도 포함")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    runs = []
    for path in sorted(rdir.glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        if r.get("dry_run") and not args.include_dryrun:
            continue
        runs.append(r)
    if not runs:
        raise SystemExit(f"{rdir}에 결과 파일이 없습니다. 먼저 bench.py를 실행하세요.")

    metrics = [run_metrics(r) for r in runs]
    env = runs[0]["env"]
    L: list[str] = []
    L.append("# fp16 offload vs NF4 벤치마크 요약\n")
    L.append(f"- GPU: {env.get('gpu_name')} ({env.get('gpu_total_gib')} GiB), 드라이버 {env.get('nvidia_driver')}")
    L.append(f"- torch {env.get('torch')}, transformers {env.get('transformers')}, "
             f"accelerate {env.get('accelerate')}, bitsandbytes {env.get('bitsandbytes')}")
    L.append(f"- 모델: {runs[0]['config']['model']['id']} (commit {runs[0].get('model_commit')})\n")

    L.append("## 1. 속도와 메모리\n")
    L.append("| 실행 | 문항 수 | s/문항 (중앙값) | forward p90 (s) | POPE 전체 환산 (h) | s/생성 토큰 | CHAIR 환산 (h) | peak VRAM (GiB) | CPU RSS (GiB) | NaN/inf |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for m in metrics:
        L.append(f"| {m['name']} | {m['n']} | {fmt(m['sec_per_probe'])} | {fmt(m['sec_forward_p90'])} | "
                 f"{fmt(m['pope_hours'], '.1f')} | {fmt(m['sec_per_token'])} | {fmt(m['chair_hours'], '.1f')} | "
                 f"{fmt(m['peak_vram'], '.2f')} | {fmt(m['cpu_rss'], '.2f')} | {m['nonfinite']} |")
    t = runs[0]["config"]["timing"]
    L.append(f"\n환산 기준: POPE {t['pope_full_size']}문항, CHAIR {t['chair_images']}장 × 평균 {t['chair_avg_tokens']}토큰 "
             "(평균 토큰 수는 가정값). 방법 1개·설정 1개당 시간이다.\n")

    L.append("## 2. 출력 품질 (문항 수가 적어서 참고용)\n")
    L.append("| 실행 | yes 비율 | accuracy | AUROC |")
    L.append("|---|---|---|---|")
    for m in metrics:
        L.append(f"| {m['name']} | {fmt(m['yes_ratio'])} | {fmt(m['acc'])} | {fmt(m['auroc'])} |")
    if any(m["synthetic"] for m in metrics):
        L.append("\n합성 이미지로 돌린 실행은 accuracy와 AUROC를 계산하지 않았다.")
    L.append("")

    fp16_runs = [(m, r) for m, r in zip(metrics, runs) if m["kind"] == "fp16_offload"]
    nf4_runs = [(m, r) for m, r in zip(metrics, runs) if m["kind"] == "nf4"]

    L.append("## 3. fp16과 NF4의 출력 차이 (양자화 영향)\n")
    if fp16_runs and nf4_runs:
        best_fp16 = min(fp16_runs, key=lambda mr: mr[0]["sec_per_probe"])
        c = compare(best_fp16[1], nf4_runs[0][1])
        if c is None:
            L.append("두 실행에 공통 문항이 없어서 비교하지 못했다.\n")
        else:
            L.append(f"- 비교 대상: `{best_fp16[0]['name']}` vs `{nf4_runs[0][0]['name']}`, 공통 문항 {c['n']}개")
            L.append(f"- 답 일치율: {c['agreement'] * 100:.1f}% (답이 바뀐 문항 {len(c['flips'])}개)")
            L.append(f"- s = logit(Yes) − logit(No)의 차이: 평균 |Δs| {c['mean_abs_ds']:.3f}, 최대 |Δs| {c['max_abs_ds']:.3f}, "
                     f"평균 Δs(NF4 − fp16) {c['mean_ds']:+.3f}")
            L.append(f"- s의 Pearson 상관: {fmt(c['pearson'], '.4f')}")
            L.append(f"- yes 비율 차이 (NF4 − fp16): {c['yes_ratio_diff'] * 100:+.1f}%p")
            if c["flips"]:
                L.append(f"- 답이 바뀐 question_id (최대 20개): {c['flips'][:20]}")
            L.append("\n해석 주의: 문항이 100여 개면 1문항이 약 1%p다. 차이가 작게 나와도 "
                     "`--num-questions 600` 정도로 다시 재서 확인하는 편이 안전하다.\n")
    else:
        L.append("fp16_offload와 nf4 결과가 둘 다 있어야 비교할 수 있다.\n")

    L.append("## 4. 판정\n")
    if fp16_runs:
        m = min(fp16_runs, key=lambda mr: mr[0]["sec_per_probe"])[0]
        sec = m["sec_per_probe"]
        if m["nonfinite"] > 0:
            L.append(f"- **주의**: `{m['name']}`에서 NaN/inf가 {m['nonfinite']}개 문항에서 나왔다. "
                     "fp16 결과를 기준점으로 쓰기 전에 원인을 먼저 확인해야 한다.")
        if sec <= FAST_SEC:
            L.append(f"- fp16 offload가 {sec:.2f} s/문항으로 **빠르다 (≤ {FAST_SEC} s)**. "
                     f"POPE 전체를 fp16으로 돌릴 수 있다 (설정당 약 {m['pope_hours']:.1f}시간).")
        elif sec <= SLOW_SEC:
            L.append(f"- fp16 offload가 {sec:.2f} s/문항으로 **중간 ({FAST_SEC}–{SLOW_SEC} s)** 이다. "
                     f"POPE 전체는 설정당 약 {m['pope_hours']:.1f}시간이 든다. "
                     "fp16은 split 1개나 1,000문항 subset에서 기준점·양자화 비교용으로 쓰고, 본 실험은 NF4로 한다.")
        else:
            L.append(f"- fp16 offload가 {sec:.2f} s/문항으로 **느리다 (> {SLOW_SEC} s)**. "
                     "본 실험은 NF4로 하고, fp16은 500문항 subset에서 양자화 차이만 잰다.")
        if m["chair_hours"] is not None:
            L.append(f"- fp16 offload로 CHAIR를 돌리면 설정당 약 {m['chair_hours']:.0f}시간으로 환산된다. "
                     "수십 시간 이상이면 CHAIR는 NF4로만 돌린다.")
        L.append(f"- 기준값({FAST_SEC} s, {SLOW_SEC} s)은 제안값이다. 전체 GPU 예산(40–80시간)에 맞춰 조정할 수 있다.")
    else:
        L.append("- fp16_offload 결과가 아직 없다.")
    L.append("")

    text = "\n".join(L)
    print(text)
    out = rdir / "summary.md"
    out.write_text(text, encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
