#!/usr/bin/env python3
"""모델을 받기 전에 GPU 환경이 벤치마크를 돌릴 수 있는지 확인한다 (다운로드 없음, 1분 이내).

    python check_env.py

확인 항목: GPU 인식, PyTorch 빌드가 이 GPU 아키텍처를 지원하는지, fp16 연산,
bitsandbytes 4-bit 연산, accelerate의 CPU offload 동작, RAM, 디스크 여유 공간.
"""
from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_GB = 14.13          # llava-hf/llava-1.5-7b-hf 전체 크기 (HF API로 확인한 값)
MIN_RAM_GIB = 16          # offload된 가중치 약 3–4 GiB + 로딩 여유 [추정]

results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"[{status:4s}] {name}" + (f": {detail}" if detail else ""))


def version(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    print(f"python {sys.version.split()[0]}")
    for pkg in ("torch", "transformers", "accelerate", "bitsandbytes"):
        v = version(pkg)
        record("OK" if v else "FAIL", f"패키지 {pkg}", v or "설치되지 않음")

    try:
        import torch
    except ImportError:
        record("FAIL", "torch import", "PyTorch를 먼저 설치하세요")
        return finish()

    # 1. GPU 인식
    if not torch.cuda.is_available():
        record("FAIL", "CUDA GPU", "torch.cuda.is_available()가 False. 드라이버나 PyTorch CUDA 빌드를 확인하세요")
        return finish()
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info(0)
    record("OK", "GPU", f"{name}, compute capability {cap[0]}.{cap[1]}, "
                        f"VRAM {total / 2**30:.2f} GiB (지금 비어 있는 양 {free / 2**30:.2f} GiB)")
    if free / 2**30 < 11:
        record("WARN", "VRAM 여유", "다른 프로세스가 GPU 메모리를 쓰고 있습니다. nvidia-smi로 확인하세요")
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=True).stdout.strip().splitlines()[0]
        record("OK", "NVIDIA 드라이버", drv)
    except Exception as e:  # noqa: BLE001
        record("WARN", "NVIDIA 드라이버", f"nvidia-smi 실행 실패 ({e})")

    # 2. 이 PyTorch 빌드가 GPU 아키텍처를 지원하는가 (Titan V는 sm_70)
    arch = f"sm_{cap[0]}{cap[1]}"
    archs = torch.cuda.get_arch_list()
    record("OK" if arch in archs else "FAIL", f"PyTorch {torch.__version__} (CUDA {torch.version.cuda})의 {arch} 지원",
           f"빌드에 포함된 아키텍처: {archs}")

    # 3. fp16 연산이 실제로 도는가
    try:
        x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        y = torch.nn.functional.softmax(x @ x, dim=-1)
        ok = bool(torch.isfinite(y).all())
        record("OK" if ok else "FAIL", "GPU fp16 연산", "matmul + softmax 정상" if ok else "NaN/inf 발생")
    except Exception as e:  # noqa: BLE001
        record("FAIL", "GPU fp16 연산", str(e).splitlines()[0])

    # 4. bitsandbytes 4-bit (NF4) 연산
    try:
        import bitsandbytes as bnb
        lin = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.float16, quant_type="nf4").to("cuda")
        y = lin(torch.randn(4, 64, device="cuda", dtype=torch.float16))
        ok = bool(torch.isfinite(y).all())
        record("OK" if ok else "FAIL", "bitsandbytes NF4 연산", "Linear4bit forward 정상" if ok else "NaN/inf 발생")
    except Exception as e:  # noqa: BLE001
        record("FAIL", "bitsandbytes NF4 연산", str(e).splitlines()[0])

    # 5. accelerate CPU offload: 'cpu'로 지정한 모듈도 계산은 GPU에서 하는가
    try:
        from accelerate import dispatch_model
        net = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.Linear(32, 32)).half()
        net = dispatch_model(net, device_map={"0": 0, "1": "cpu"})
        y = net(torch.randn(2, 32, device="cuda", dtype=torch.float16))
        exec_dev = getattr(getattr(net[1], "_hf_hook", None), "execution_device", "확인 불가")
        stored = next(net[1].parameters()).device   # offload된 가중치는 forward 밖에서는 GPU에 없다
        ok = y.is_cuda and bool(torch.isfinite(y).all()) and stored.type != "cuda"
        record("OK" if ok else "FAIL", "accelerate CPU offload",
               f"offload된 모듈의 실행 장치 = {exec_dev}, 평소 가중치 위치 = {stored}, 출력 장치 = {y.device}")
    except Exception as e:  # noqa: BLE001
        record("FAIL", "accelerate CPU offload", str(e).splitlines()[0])

    # 6. RAM과 디스크
    try:
        import psutil
        ram = psutil.virtual_memory()
        record("OK" if ram.total / 2**30 >= MIN_RAM_GIB else "WARN", "CPU RAM",
               f"전체 {ram.total / 2**30:.1f} GiB, 사용 가능 {ram.available / 2**30:.1f} GiB "
               f"(권장 {MIN_RAM_GIB} GiB 이상)")
    except ImportError:
        record("WARN", "CPU RAM", "psutil이 없어 확인하지 못함")

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    probe = hf_home
    while not probe.exists():
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    record("OK" if free_gb >= MODEL_GB + 3 else "FAIL", "모델 저장 위치의 디스크 여유",
           f"{hf_home}: {free_gb:.0f} GB 비어 있음 (모델 {MODEL_GB} GB 필요). "
           "부족하면 HF_HOME 환경변수로 다른 디스크를 지정하세요")

    finish()


def finish() -> None:
    n_fail = sum(s == "FAIL" for s, _, _ in results)
    n_warn = sum(s == "WARN" for s, _, _ in results)
    print()
    if n_fail:
        print(f"결과: FAIL {n_fail}개, WARN {n_warn}개. FAIL 항목을 해결한 뒤 모델을 받으세요.")
        sys.exit(1)
    print(f"결과: 통과 (WARN {n_warn}개). 모델을 받아 bench.py를 실행해도 됩니다.")


if __name__ == "__main__":
    main()
