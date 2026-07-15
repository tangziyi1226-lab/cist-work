# Llama + SnapKV 的 KV Cache 压缩复现

## 1. 作业背景

本作业基于论文 **EntroKV: Entropy-Guided Dynamic Budget Allocation for KV-Cache Compression**。论文研究长上下文大语言模型推理中的 KV cache 显存瓶颈，并提出一种基于注意力熵的动态预算分配方法：对注意力分布更分散、压缩更敏感的 attention head 分配更多 KV 保留预算；对注意力更集中的 head 进行更激进压缩。

本次夏令营作业不要求完整复现论文中的全部模型、全部数据集和全部压缩算子。作业范围限定为：

```text
Llama-3.1-8B-Instruct + SnapKV + LongBench 子集
```

学生只需要比较 **固定预算 SnapKV** 和 **基于注意力熵动态分配预算的 SnapKV**，验证论文中“entropy-guided budget allocation 可以提升固定预算 SnapKV”的主要趋势。

## 2. 学习目标

完成本作业后，应能：

1. 解释长上下文推理中 KV cache 的显存开销来源。
2. 理解 SnapKV 的观察窗口、attention score 聚合和 top-k KV 保留机制。
3. 理解注意力熵为什么可以作为压缩敏感度信号。
4. 在 Llama 模型上实现或复现 layer-head 级动态 KV 预算分配。
5. 在 LongBench 子集上完成固定预算 SnapKV 与 entropy-guided SnapKV 的对比实验。

## 3. 参考材料

- 论文：`example_paper.pdf`
- 论文链接：`https://openreview.net/forum?id=xhAMjsnWUe`
- 可参考公开基线仓库(**建议基于该仓库进行实现**)：`https://github.com/FFY0/AdaKV.git`
- LongBench数据集地址：`https://huggingface.co/datasets/zai-org/LongBench`

## 4. 论文重点

论文将 KV cache 压缩拆成两个问题：

1. **压缩算子**：如何根据重要性分数保留或丢弃 KV cache。作业中只考虑 SnapKV。
2. **预算分配**：每一层、每个 attention head 应该保留多少 KV。作业中重点复现 entropy-guided 分配。

核心预算分配公式为：

```text
B_l^h = beta * [(1 - alpha) * H_bar + alpha * H_l^h]
```

其中：

- `H_l^h`：第 `l` 层第 `h` 个 attention head 的归一化注意力熵。
- `H_bar`：校准集上的全局平均熵 (Llama 默认使用0.3)。
- `alpha`：控制全局统计与当前样本动态信号的权重。
- `beta`：控制总体 KV cache 保留比例。

直观理解：

- 高熵 head 的注意力更分散，小预算下更容易丢失信息，因此应分配更多 KV 预算。
- 低熵 head 的注意力更集中，可以更激进压缩，把预算释放给高熵 head。

## 5. 复现前置要求

开始实验前，需要先满足以下要求，保证后续结果可以被复查和公平比较。

1. **基础模型**：使用 Llama-3.1-8B-Instruct 作为主模型，并在报告中注明具体模型来源、模型路径和精度设置。
2. **基线方法**：需要先跑通固定预算 SnapKV。EntroKV 的实现应在 SnapKV 的基础上只改变预算分配逻辑，尽量保持观察窗口、pooling、top-k 保留等 SnapKV 主流程一致。
3. **对比方法**：至少准备三种推理模式：Full Cache、固定预算 SnapKV、EntroKV。其中 Full Cache 作为不压缩上界；如果显存不足无法完整运行 Full Cache，需要在报告中说明。
4. **数据集**：使用 LongBench 子集，且所有方法必须使用相同任务、相同样本数量和相同 prompt 模板。
5. **预算设置**：固定预算 SnapKV 和 EntroKV 应使用相同总体 KV 保留比例。建议以 30% 保留比例作为主要设置；Llama 的全局基线熵 `H_bar` 可按论文默认设置为 0.3。
6. **解码设置**：允许使用贪婪搜索代替论文中的多次采样取均值。无论使用贪婪搜索还是采样，Full Cache、SnapKV、EntroKV 三种方法必须使用完全相同的解码设置。
7. **运行环境**：需要记录 GPU 型号、CUDA、PyTorch、Transformers、FlashAttention 等关键版本。若硬件条件不足，可以减少任务或样本数量，但不能改变不同方法之间的对比条件。
8. **实验输出**：需要保存预测结果、任务分数、平均 KV 预算、EntroKV 的 layer-head 预算分配数据，以及推理效率统计所需的显存和延迟记录。

## 6. 必做任务

### 6.1 实现 EntroKV 压缩逻辑

基于公开基线或自行实现，完成 Llama + SnapKV 场景下的 EntroKV 压缩逻辑：

1. 使用 Llama-3.1-8B-Instruct 作为主模型。
2. 实现或接入固定预算 SnapKV。
3. 在 SnapKV 的观察窗口内读取或复用 attention weights。
4. 计算每层每个 attention head 的归一化注意力熵。
5. 根据论文公式计算每个 layer-head 的 KV 保留预算。
6. 将动态预算接入 SnapKV 的 top-k KV 保留流程。
7. 输出每个样本的预测结果、任务分数和平均预算。

### 6.2 性能提升验证

在 LongBench 子集上比较固定预算 SnapKV 与 EntroKV，验证 EntroKV 是否相比 SnapKV 取得性能提升。至少完成 3 个 LongBench 任务，建议从以下任务中选择：

- `narrativeqa`
- `hotpotqa`
- `qasper`
- `lcc` 或 `passage_retrieval_en`

至少比较以下方法：

1. Full Cache 或不压缩上界。
2. 固定预算 SnapKV。
3. EntroKV，即 entropy-guided SnapKV。



### 6.3 动态分配可视化

可视化 EntroKV 对不同任务、不同层和不同 attention head 的动态预算分配。至少包含：

1. 一个 layer-head 预算热力图，横轴为 head，纵轴为 layer。
2. 至少两个 LongBench 任务的预算分配对比。
3. 对可视化结果的解释：哪些层或 head 分配到更多预算，是否与任务类型相关。

### 6.4 推理效率对比

比较 Full Cache、SnapKV、EntroKV 三者的推理效率关系。至少记录以下指标中的两项：

1. Peak GPU memory。
2. Prefill latency。
3. Decode latency 或 tokens/s。
4. 平均 KV 保留预算。

要求报告中说明：EntroKV 相比 SnapKV 是否带来额外开销；相比 Full Cache 是否降低显存占用。

## 7. 选做任务

1. 对 `alpha`、`beta` 或观察窗口长度做消融。
2. 如算力充足，可补充更长上下文任务验证。
3. 尝试分析 EntroKV 在不同任务上提升或失败的原因。

## 8. 实验报告要求

提交一份实验报告，建议包含：

1. **论文理解**：说明 SnapKV、注意力熵和动态预算分配的关系。
2. **实现说明**：说明固定预算 SnapKV 和 entropy-guided SnapKV 分别如何实现。
3. **实验环境**：GPU、CUDA、PyTorch、Transformers、FlashAttention 版本。
4. **模型与数据**：Llama 模型名称、LongBench 任务列表、样本数量。
5. **实验命令**：给出可复查的运行命令或脚本入口，并注明解码方式，例如贪婪搜索或采样。
6. **结果表格**：包括每个任务分数、平均分、平均预算。
7. **对比分析**：说明 entropy-guided SnapKV 相比固定预算 SnapKV 是否提升，以及可能原因。
8. **误差分析**：解释与论文参考值不同的原因，如硬件、任务子集、随机性、实现细节等。
