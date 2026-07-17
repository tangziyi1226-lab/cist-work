# 复现材料清单

## 已准备

- `external/AdaKV/`：README 指定的 AdaKV 基线仓库，固定在提交
  `04497abac4c1a58426f3daf1014578990e225cc5`。其中同时包含 fixed SnapKV、
  adaptive SnapKV、Llama monkeypatch 和 LongBench 预测/评分脚本。
- `external/LongBench/`：LongBench 官方评测仓库的浅克隆。
- `papers/SnapKV.pdf`：SnapKV 论文（arXiv:2404.14469）。
- `papers/AdaKV.pdf`：AdaKV 论文（arXiv:2407.11550）。
- `papers/EntroKV.pdf`：EntroKV 论文（OpenReview `xhAMjsnWUe`，ICML 2026）。
- `requirements-repro.txt`：CUDA 主机上的 Python 依赖基线。
- `scripts/download_assets.py`：按需缓存 4 个候选 LongBench 任务及下载模型。
- `scripts/check_environment.sh`：记录报告所需的软件、CUDA 和 GPU 环境。

EntroKV 官方页面：<https://openreview.net/forum?id=xhAMjsnWUe>。

## 仍需用户提供/确认

1. **Hugging Face access token (`HF_TOKEN`)**：Llama 3.1 是 gated model。先在
   <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> 接受许可，再提供一个
   具有 read 权限的 token。不要把 token 写入 Git；复制 `.env.example` 为 `.env`。
2. **Linux + NVIDIA CUDA GPU**：当前机器是 Apple Silicon macOS，没有 NVIDIA GPU，
   无法编译 AdaKV CUDA kernel/FlashAttention 或完成目标实验。8B BF16 权重约 16 GB，
   加上 KV cache 和运行开销，建议至少 24 GB 显存；长上下文 Full Cache 更适合
   40/48/80 GB。显存不足时用 Llama 3.2 1B/3B 做开发验证，再到服务器跑 8B。
3. **服务器访问方式**：如希望我继续安装并跑实验，需要 CUDA 服务器终端环境（或在
   当前工作区挂载/切换到该服务器）以及足够磁盘空间。模型、数据与输出建议预留
   30–50 GB，完整长上下文实验应更多。

## 建议执行顺序（在 CUDA 服务器）

```bash
cp .env.example .env                  # 填写 HF_TOKEN，不提交 .env
bash scripts/check_environment.sh

# 创建 Python 3.10 环境后，先按服务器 CUDA 安装匹配的 PyTorch，再执行：
pip install -r requirements-repro.txt
pip install flash-attn==2.4.0 --no-build-isolation
pip install -e external/AdaKV

set -a; source .env; set +a
python scripts/download_assets.py --data-only
python scripts/download_assets.py --model-only
```

之后以 `external/AdaKV/experiments/LongBench/pred.py` 和 `eval.py` 为基线入口，先跑
固定预算 SnapKV，再接 EntroKV 熵预算。注意：AdaKV 的 adaptive budget 本身不是
README 所述的 EntroKV 熵公式，不能直接把 AdaKV 结果冒充 EntroKV；需要修改预算分配
逻辑，同时保持 SnapKV 的观察窗口、pooling 和 top-k 流程一致。
