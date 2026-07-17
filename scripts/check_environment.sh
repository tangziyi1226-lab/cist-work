#!/usr/bin/env bash
set -euo pipefail

echo "System: $(uname -a)"
PYTHON_BIN="$(command -v python || command -v python3 || true)"
echo "Python: ${PYTHON_BIN:-not found}"
if [[ -n "$PYTHON_BIN" ]]; then
  "$PYTHON_BIN" --version
fi
echo "Git: $(git --version)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "NVIDIA GPU: not detected (CUDA experiments require a Linux NVIDIA host)"
fi

if [[ -n "$PYTHON_BIN" ]]; then
  "$PYTHON_BIN" - <<'PY'
import importlib
for name in ("torch", "transformers", "datasets", "flash_attn"):
    try:
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'installed')}")
    except Exception as exc:
        print(f"{name}: unavailable ({exc})")
PY
fi
