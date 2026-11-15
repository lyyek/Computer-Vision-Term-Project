#!/usr/bin/env python3
"""LLaVA-1.5-7B: fp16(CPU offload)와 4-bit NF4의 속도·메모리·출력 차이 벤치마크.

한 번 실행에 한 가지 mode만 돌린다. 프로세스를 나눠야 VRAM 측정이 섞이지 않는다.

    python bench.py --mode nf4
    python bench.py --mode fp16_offload

결과는 results/<mode>[-<tag>].json에 저장되고, summarize.py가 두 결과를 비교해 판정한다.

측정 항목
- POPE 형식 yes/no 질문 1개당 forward 시간 (질문당 forward 1회)
- s = logit(Yes) - logit(No), P(Yes), 답, logit의 NaN/inf 여부
- peak VRAM, CPU RAM
- 캡션 생성 속도 (CHAIR 소요 시간 환산용)
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- 설정

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, help="config.yaml의 modes 아래 이름 (fp16_offload 또는 nf4)")
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    p.add_argument("--num-questions", type=int, help="POPE 문항 수 (config의 data.num_questions를 덮어씀)")
    p.add_argument("--gpu-mem", help="fp16_offload의 GPU 가중치 상한, 예: 10.5GiB")
    p.add_argument("--cpu-mem", help="fp16_offload의 CPU RAM 가중치 상한, 예: 24GiB")
    p.add_argument("--attn", choices=["eager", "sdpa"], help="attention 구현 (기본: 라이브러리 기본값)")
    p.add_argument("--tag", help="결과 파일 이름에 붙일 꼬리표, 예: 10.5g")
    p.add_argument("--no-generation", action="store_true", help="캡션 생성 속도 측정을 생략")
    p.add_argument("--synthetic", action="store_true",
                   help="POPE/COCO를 받지 않고 무작위 이미지로 측정 (속도·메모리만 유효)")
    p.add_argument("--dry-run", action="store_true",
                   help="개발용: GPU 없이 CPU에서 소형 더미 모델로 스크립트 로직만 확인")
    return p.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.mode not in cfg["modes"]:
        sys.exit(f"[오류] --mode {args.mode}가 config의 modes에 없습니다: {list(cfg['modes'])}")
    if args.num_questions:
        cfg["data"]["num_questions"] = args.num_questions
    mode = cfg["modes"][args.mode]
    if args.gpu_mem:
        mode["gpu_max_memory"] = args.gpu_mem
    if args.cpu_mem:
        mode["cpu_max_memory"] = args.cpu_mem
    if args.attn:
        cfg["model"]["attn_implementation"] = args.attn
    if args.no_generation:
        cfg["generation"]["num_images"] = 0
    return cfg


# --------------------------------------------------------------------------- 데이터

def download(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "offload-benchmark"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001 - 네트워크 오류 종류와 무관하게 재시도
            last_err = e
            time.sleep(2 * attempt)
    raise RuntimeError(f"다운로드 실패: {url} ({last_err})")


def prepare_pope(cfg: dict) -> list[dict]:
    """POPE 질문 파일과, 앞 num_questions개 문항에 필요한 COCO 이미지만 받는다."""
    d = cfg["data"]
    data_dir = HERE / d["dir"]
    qfile = data_dir / Path(d["pope_url"]).name
    if not qfile.exists():
        print(f"[data] POPE 질문 파일 다운로드: {d['pope_url']}")
        download(d["pope_url"], qfile)
    rows = [json.loads(line) for line in qfile.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[: d["num_questions"]]

    items, n_new = [], 0
    for r in rows:
        name = Path(r["image"]).name  # 원격 파일의 값이므로 파일 이름 부분만 쓴다
        path = data_dir / "images" / name
        if not path.exists():
            download(d["coco_image_url"].format(image=name), path)
            n_new += 1
        items.append({
            "question_id": r["question_id"],
            "image": name,
            "image_path": str(path),
            "question": r["text"],
            "label": r["label"].strip().lower(),
        })
    n_img = len({it["image"] for it in items})
    print(f"[data] POPE 문항 {len(items)}개, 이미지 {n_img}장 (새로 받은 이미지 {n_new}장)")
    return items


def prepare_synthetic(cfg: dict) -> list[dict]:
    """네트워크가 막힌 서버용. 무작위 노이즈 이미지라서 정확도·AUROC는 의미가 없다."""
    import numpy as np

    n = cfg["data"]["num_questions"]
    out_dir = HERE / cfg["data"]["dir"] / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    objects = ["person", "car", "dog", "chair", "cup", "bottle"]
    items = []
    for i in range(n):
        img_idx = i // len(objects)
        path = out_dir / f"noise_{img_idx:04d}.png"
        if not path.exists():
            Image.fromarray(rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)).save(path)
        items.append({
            "question_id": i + 1,
            "image": path.name,
            "image_path": str(path),
            "question": f"Is there a {objects[i % len(objects)]} in the image?",
            "label": "yes" if i % 2 == 0 else "no",
        })
    print(f"[data] 합성 문항 {len(items)}개 (정확도·AUROC는 의미 없음)")
    return items


# --------------------------------------------------------------------------- 환경 기록

def pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def env_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    for name in ("transformers", "accelerate", "bitsandbytes", "tokenizers"):
        info[name] = pkg_version(name)
    if torch.cuda.is_available():
        _, total = torch.cuda.mem_get_info(0)
        info.update(
            gpu_name=torch.cuda.get_device_name(0),
            gpu_capability=list(torch.cuda.get_device_capability(0)),
            gpu_total_gib=round(total / 2**30, 2),
            arch_list=torch.cuda.get_arch_list(),
        )
        try:
            info["nvidia_driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20, check=True,
            ).stdout.strip().splitlines()[0]
        except Exception:  # noqa: BLE001 - nvidia-smi가 없어도 측정은 계속한다
            info["nvidia_driver"] = None
    try:
        import psutil
        info["ram_total_gib"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        info["ram_total_gib"] = None
    return info


def rss_gib() -> float | None:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2**30, 2)
    except ImportError:
        return None


def gpu_mem() -> dict:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info(0)
    g = 2**30
    return {
        "allocated_gib": round(torch.cuda.memory_allocated(0) / g, 2),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(0) / g, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved(0) / g, 2),
        "free_now_gib": round(free / g, 2),
        "total_gib": round(total / g, 2),
    }


# --------------------------------------------------------------------------- 모델

def build_tiny_model(processor):
    """--dry-run용 소형 무작위 LLaVA. 구조만 같고 출력은 의미가 없다."""
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

    tok = processor.tokenizer
    vision = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                              num_attention_heads=2, image_size=336, patch_size=14, projection_dim=32)
    text = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096,
                       pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
                       eos_token_id=tok.eos_token_id)
    config = LlavaConfig(vision_config=vision, text_config=text,
                         image_token_index=tok.convert_tokens_to_ids("<image>"),
                         vision_feature_layer=-2, vision_feature_select_strategy="default")
    return LlavaForConditionalGeneration(config)


def load_model(cfg: dict, mode_name: str, dry_run: bool):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    m, mode = cfg["model"], cfg["modes"][mode_name]
    processor = AutoProcessor.from_pretrained(m["id"], revision=m.get("revision"))

    t0 = time.perf_counter()
    if dry_run:
        model, device, dtype = build_tiny_model(processor), "cpu", torch.float32
    else:
        kwargs = {"revision": m.get("revision"), "low_cpu_mem_usage": True, "torch_dtype": torch.float16}
        if m.get("attn_implementation"):
            kwargs["attn_implementation"] = m["attn_implementation"]

        if mode["kind"] == "fp16_offload":
            # GPU 예산을 넘는 layer는 가중치를 CPU RAM에 두고, 그 layer를 계산할 때만 GPU로 복사한다.
            kwargs.update(device_map="auto",
                          max_memory={0: mode["gpu_max_memory"], "cpu": mode["cpu_max_memory"]})
        elif mode["kind"] == "nf4":
            from transformers import BitsAndBytesConfig
            kwargs.update(device_map={"": 0}, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=mode.get("quant_type", "nf4"),
                bnb_4bit_use_double_quant=bool(mode.get("double_quant", True)),
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_skip_modules=mode.get("skip_modules"),  # 이름과 달리 4-bit에도 적용된다
            ))
        else:
            sys.exit(f"[오류] 알 수 없는 mode kind: {mode['kind']}")

        try:
            model = LlavaForConditionalGeneration.from_pretrained(m["id"], **kwargs)
        except ValueError as e:
            if "offload" in str(e).lower():
                sys.exit("[오류] 가중치 일부가 disk로 밀려났습니다. CPU RAM 예산이 부족하다는 뜻이므로 "
                         f"--cpu-mem을 늘리세요 (현재 {mode.get('cpu_max_memory')}).\n원래 오류: {e}")
            raise
        device, dtype = "cuda:0", torch.float16

    model.eval()
    return model, processor, device, dtype, time.perf_counter() - t0


def describe_placement(model) -> dict:
    """accelerate가 각 모듈을 어디에 뒀는지 요약한다. 'cpu'가 offload된 부분이다."""
    dm = getattr(model, "hf_device_map", None)
    if not dm:
        return {"by_device": None, "offloaded_modules": []}
    by_dev: dict[str, dict] = {}
    for name, dev in dm.items():
        sub = model.get_submodule(name) if name else model
        nbytes = sum(p.numel() * p.element_size() for p in sub.parameters())
        e = by_dev.setdefault(str(dev), {"modules": 0, "param_gib": 0.0})
        e["modules"] += 1
        e["param_gib"] = round(e["param_gib"] + nbytes / 2**30, 3)
    offloaded = [n for n, d in dm.items() if str(d) in ("cpu", "disk")]
    return {"by_device": by_dev, "offloaded_modules": offloaded}


def quant_coverage(model) -> dict:
    """Linear layer가 부분별로 양자화됐는지 센다. skip_modules가 의도대로 먹었는지 확인하는 용도."""
    cov: dict[str, dict] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if "vision_tower" in name:
            group = "vision_tower"
        elif "multi_modal_projector" in name:
            group = "projector"
        elif name.endswith("lm_head"):
            group = "lm_head"
        else:
            group = "language_model"
        kind = "full_precision" if type(mod).__name__ == "Linear" else "quantized"
        g = cov.setdefault(group, {"quantized": 0, "full_precision": 0})
        g[kind] += 1
    return cov


def answer_token_ids(tokenizer) -> dict:
    """'ASSISTANT:' 뒤에 오는 Yes/No의 첫 토큰 id."""
    out = {}
    for word in ("Yes", "No"):
        tid = tokenizer("ASSISTANT: " + word, add_special_tokens=False).input_ids[-1]
        piece = tokenizer.convert_ids_to_tokens(tid)
        if piece.replace("▁", "").strip() != word:
            sys.exit(f"[오류] '{word}'가 한 토큰으로 나뉘지 않습니다 (마지막 토큰: {piece!r}). 토크나이저를 확인하세요.")
        out[word.lower()] = {"id": int(tid), "piece": piece}
    return out


# --------------------------------------------------------------------------- 측정

def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def fnum(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def summarize_times(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {
        "median": round(statistics.median(xs), 4),
        "mean": round(statistics.fmean(xs), 4),
        "p90": round(xs[max(0, math.ceil(0.9 * len(xs)) - 1)], 4),
        "min": round(xs[0], 4),
        "max": round(xs[-1], 4),
    }


def build_prompt(cfg: dict, question: str) -> str:
    p = cfg["prompt"]
    return p["probe"].format(system=p["system"], question=question)


@torch.no_grad()
def probe_once(model, processor, cfg, item, device, dtype, ids) -> dict:
    t0 = time.perf_counter()
    image = Image.open(item["image_path"]).convert("RGB")
    inputs = processor(images=image, text=build_prompt(cfg, item["question"]), return_tensors="pt")
    t1 = time.perf_counter()

    inputs = inputs.to(device, dtype)        # dtype 변환은 부동소수 텐서(pixel_values)에만 적용된다
    logits = model(**inputs).logits[0, -1].float()   # 마지막 위치 = 답의 첫 토큰 분포
    sync(device)
    t2 = time.perf_counter()

    finite = bool(torch.isfinite(logits).all())
    probs = torch.softmax(logits, dim=-1)
    ly, ln = logits[ids["yes"]["id"]].item(), logits[ids["no"]["id"]].item()
    s = ly - ln
    return {
        "question_id": item["question_id"],
        "image": item["image"],
        "label": item["label"],
        "n_input_tokens": int(inputs["input_ids"].shape[1]),
        "logit_yes": fnum(ly),
        "logit_no": fnum(ln),
        "s": fnum(s),
        "p_yes": fnum(probs[ids["yes"]["id"]].item()),
        "p_no": fnum(probs[ids["no"]["id"]].item()),
        "top1": processor.tokenizer.convert_ids_to_tokens(int(logits.argmax())) if finite else None,
        "pred": ("yes" if s > 0 else "no") if math.isfinite(s) else None,
        "finite": finite,
        "sec_preprocess": round(t1 - t0, 4),
        "sec_forward": round(t2 - t1, 4),
    }


def run_probes(model, processor, cfg, items, device, dtype, ids) -> list[dict]:
    warmup = min(cfg["timing"]["warmup"], len(items))
    for it in items[:warmup]:
        probe_once(model, processor, cfg, it, device, dtype, ids)
    print(f"[probe] warm-up {warmup}회 완료. 본 측정 {len(items)}문항 시작")

    records = []
    for i, it in enumerate(items, 1):
        records.append(probe_once(model, processor, cfg, it, device, dtype, ids))
        if i % 20 == 0 or i == len(items):
            med = statistics.median(r["sec_forward"] for r in records)
            print(f"[probe] {i}/{len(items)}  forward 중앙값 {med:.3f} s/문항")
    return records


@torch.no_grad()
def run_generation(model, processor, cfg, items, device, dtype) -> list[dict]:
    g = cfg["generation"]
    paths = list(dict.fromkeys(it["image_path"] for it in items))[: g["num_images"]]
    records = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        prompt = build_prompt(cfg, cfg["prompt"]["caption_question"])
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype)
        n_in = inputs["input_ids"].shape[1]
        sync(device)
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=g["max_new_tokens"], do_sample=False)
        sync(device)
        sec = time.perf_counter() - t0
        n_new = int(out.shape[1] - n_in)
        records.append({
            "image": Path(path).name,
            "new_tokens": n_new,
            "sec": round(sec, 3),
            "text": processor.tokenizer.decode(out[0, n_in:], skip_special_tokens=True)[:300],
        })
        print(f"[gen] {Path(path).name}: {n_new} 토큰, {sec:.2f} s")
    return records


# --------------------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    env = env_info()
    print("[env]", json.dumps(env, ensure_ascii=False))
    if not args.dry_run and not torch.cuda.is_available():
        sys.exit("[오류] CUDA GPU가 보이지 않습니다. 먼저 python check_env.py로 환경을 확인하세요.")

    items = prepare_synthetic(cfg) if args.synthetic else prepare_pope(cfg)

    print(f"[model] {cfg['model']['id']} 로딩 (mode={args.mode}) ...")
    try:
        model, processor, device, dtype, load_sec = load_model(cfg, args.mode, args.dry_run)
    except torch.cuda.OutOfMemoryError:
        sys.exit("[오류] 모델 로딩 중 GPU 메모리 부족. fp16_offload라면 --gpu-mem을 낮추세요 (예: 9GiB). "
                 "nvidia-smi로 다른 프로세스가 GPU를 쓰고 있는지도 확인하세요.")
    placement = describe_placement(model)
    coverage = quant_coverage(model)
    mem_after_load = gpu_mem()
    print(f"[model] 로딩 {load_sec:.1f} s | 배치: {json.dumps(placement['by_device'])}")
    print(f"[model] offload된 모듈 {len(placement['offloaded_modules'])}개 | Linear 양자화 현황: {json.dumps(coverage)}")
    if "disk" in (placement["by_device"] or {}):
        print("[경고] 일부 가중치가 disk에 있습니다. 매우 느려지니 --cpu-mem을 늘리세요.")

    ids = answer_token_ids(processor.tokenizer)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    try:
        probes = run_probes(model, processor, cfg, items, device, dtype, ids)
        generation = run_generation(model, processor, cfg, items, device, dtype) \
            if cfg["generation"]["num_images"] > 0 else []
    except torch.cuda.OutOfMemoryError:
        sys.exit("[오류] 추론 중 GPU 메모리 부족. --gpu-mem을 0.5GiB씩 낮춰 다시 실행하세요.")

    n_bad = sum(not r["finite"] for r in probes)
    result = {
        "mode": args.mode,
        "tag": args.tag,
        "dry_run": args.dry_run,
        "synthetic": args.synthetic,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg,
        "env": env,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "answer_tokens": ids,
        "load": {"seconds": round(load_sec, 1), "placement": placement, "quant_coverage": coverage,
                 "gpu_mem_after_load": mem_after_load},
        "probe": {
            "n": len(probes),
            "sec_forward": summarize_times([r["sec_forward"] for r in probes]),
            "sec_preprocess": summarize_times([r["sec_preprocess"] for r in probes]),
            "nonfinite_count": n_bad,
        },
        "memory": {"gpu": gpu_mem(), "cpu_rss_gib": rss_gib()},
        "generation": generation,
        "per_question": probes,
    }

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stem = ("dryrun-" if args.dry_run else "") + args.mode + (f"-{args.tag}" if args.tag else "")
    out_path = out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    fwd = result["probe"]["sec_forward"]
    print("\n===== 결과 요약 =====")
    print(f"mode: {args.mode} | 문항 {len(probes)}개")
    print(f"forward: 중앙값 {fwd['median']} s, p90 {fwd['p90']} s")
    print(f"peak VRAM: {result['memory']['gpu'].get('peak_allocated_gib')} GiB | CPU RSS: {result['memory']['cpu_rss_gib']} GiB")
    print(f"NaN/inf가 나온 문항: {n_bad}개")
    print(f"저장: {out_path}")
    print("두 mode를 모두 돌린 뒤 python summarize.py를 실행하세요.")


if __name__ == "__main__":
    main()
