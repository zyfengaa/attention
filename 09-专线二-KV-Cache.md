[第 8 章](08-专线一-位置编码.md) ｜ [← 返回目录](README.md) ｜ [第 10 章 →](10-核心公式演变.md)

---

# 第 9 章 专线二：KV Cache 的独立演进史

> **这条线解决的问题**：自回归解码时，每生成一个 token 都要重算全部历史 K/V。缓存它们 → 换来 $O(n)$ 的显存开销。而长上下文 + 高并发的产品形态，让这个开销变成**第一成本**。
>
> 五年里出现了**四层**优化，每一层的对象不同：
>
> | 层 | 优化对象 | 代表 | 典型收益 |
> |---|---|---|---|
> | **第一层：份数** | 存几份 K/V | MQA（2019）→ GQA（2023） | KV ÷4～÷8 |
> | **第二层：字节** | 每个数占几 bit | INT8/FP8 KV 量化 | KV ÷2～÷4 |
> | **第三层：参与量** | 每次看多少个 | H2O/StreamingLLM/SnapKV/Quest | 参与量 ÷10～÷100 |
> | **第四层：不存** | 架构上取消 cache | MLA（2024）→ DSA/NSA → KDA（2025） | KV ÷10～÷70 或**常数** |
>
> **读法**：这四层是**可叠加**的。2026 年的顶级模型通常是"第四层 + 第三层 + 第二层"的组合，而不是只做一层。

---

## 9.1 第一层：份数——MQA 与 GQA

### 9.1.1 先把公式钉死

$$
\text{KV bytes per token}=2\times L\times h\times d_h\times b
$$

$L$ 层、$h$ 头、$d_h$ 维、$b$ 字节/数（FP16 时 $b=2$）。

### 9.1.2 MQA（2019）：共享到极限

$K,V$ 只有 1 份，所有 query 头共用。$h$ 从 32 降到 1，**KV 直接 ÷32**。

代价（见 3.3）：所有头的 K 分布被强制相同，表达力受损、训练不稳。

### 9.1.3 GQA（2023）：折中，并给出"如何从 MHA 改造"

**构造**：把 $h$ 个 query 头分成 $g$ 组，每组共享一份 K/V（$1<g<h$）。

$$
\text{KV}\propto g\quad\Longrightarrow\quad
\text{压缩比}=h/g
$$

**最实用的贡献是"uptraining"**：不用从头训，而是把已有的 MHA 检查点**均值池化**成 $g$ 组（组内 K/V 取平均），再继续预训练约 **5%** 的原始步数。

**为什么均值池化能当初始化？** 因为同一组内的 K/V 本来高度相关（3.4 节的头剪枝实验已证明），平均后信息损失小，微调即可恢复。

**实验**：T5-XXL 上，GQA 的质量**接近 MHA**、速度**接近 MQA**——这就是它 2023 年成为行业默认值的原因（Llama 2 70B、Mistral、Qwen2 全系）。

### 9.1.4 第一层的天花板

$g\ge1$，所以最多 ÷$h$。**再往下压必须换维度**——这正是 MLA 出现的原因。

---

## 9.2 第二层：字节——量化 KV

思路：K/V 的数值分布比权重更集中，可以压到 8 bit 甚至 4 bit。

| 方案 | 压缩 | 关键问题 |
|---|---|---|
| FP8 / INT8 KV | ÷2 | **outlier**：少数通道的极大值会把整组量化步长拉大 |
| 分组量化（per-channel / per-token） | ÷2～÷4 | 需要额外存缩放因子 |
| 4 bit / 2 bit（KIVI 等） | ÷4～÷8 | 长上下文下误差累积，检索类任务先坏 |

**为什么 KV 量化比权重量化更难？** 因为 KV 是**动态生成**的：分布随输入变化，不能像权重那样离线校准。这也是为什么工业界更倾向于"架构级压缩（MLA）+ 8 bit"而不是"极致量化"。

---

## 9.3 第三层：参与量——淘汰、窗口、选择

### 9.3.1 H2O：重击者（Heavy Hitter）

**观察**：注意力分数高度偏斜，少数 token 承担了大部分权重。

**做法**：为每个 token 维护**累积注意力分数**，缓存满时淘汰分数最低的。

$$
\text{score}_t=\sum_{s\le t}A_{s,t}\quad\Longrightarrow\quad
\text{保留 top-}k
$$

**实验**：batch 提升约 5×，精度接近全量。

**失效**：**train-free**——模型训练时从没见过"部分 token 消失"，长文上分布偏移。

### 9.3.2 Scissorhands：重要性持久性假设

**假设**：一个 token 一旦重要，会在后续若干步内持续重要。
**做法**：用滑动窗口统计重要性，配合"重要 token 保留 + 其余淘汰"。
**意义**：它把 H2O 的"静态选择"变成了"动态维护"，但仍是 train-free。

### 9.3.3 StreamingLLM：注意力汇聚点（attention sink）

**观察**：**最开始的几个 token** 会吸收大量注意力，即使它们语义无关。删掉它们，模型立刻崩。

**解释**：softmax 必须把概率分完，当 query 与所有 key 都不相似时，模型需要"一个地方放剩余概率"——初始 token 充当了这个**汇聚点**。

**做法**：永远保留前 4 个 token + 最近窗口，中间全丢。**4M token 上稳定生成**，比"滑动窗口 + 重算"快 22.2×。

**意义**：它解释了为什么"简单滑窗"会崩，也让"attention sink"成为后续所有稀疏方法必须处理的对象（DSA/NSA 的"必须保留初始 token"就是这条结论）。

### 9.3.4 SnapKV：在 prefill 阶段就选

**观察**：生成阶段的注意力模式在 prefill 时已基本确定。
**做法**：prefill 时用最后几个 query 的注意力分布选出重要 KV，只保留这些。
**收益**：解码速度约 3.6×，且**不改模型**。
**失效**：需要"最后几个 query 与未来 query 相似"的假设；多轮对话中每轮都要重选。

### 9.3.5 Quest：把"选 token"升级为"选页"

**做法**：KV 按页（page）组织，用 query 与页的**元数据**（每页 K 的最大/最小值）估计相关性，只加载高分页。

**意义**：它与 PagedAttention 的页式存储天然配合——**选择粒度 = 存储粒度 = 访存粒度**，这是"硬件对齐"思想在缓存管理上的第一次落地。

---

## 9.4 第四层：不存——架构级取消 cache

### 9.4.1 MLA：压成潜向量（2024）

存 $d_c=512$ 维潜向量 + 64 维位置键 = **576 个数/token/层**，用时解压（见 7.1）。

### 9.4.2 DSA / NSA：按需取（2025）

- **NSA**：三分支并行，KV 只按块加载被选中的部分；
- **DSA**：indexer 选 top-$k$=2048，主注意力只算这些。

### 9.4.3 线性层 / KDA：用状态替代 cache（2025）

$$
S_t=f(S_{t-1},k_t,v_t)\quad\Longrightarrow\quad
\text{状态大小}=\text{常数}
$$

KDA 在 1M 上下文下 KV **−75%**，解码 **6×**——因为"历史"被压进一个固定大小的状态，不再随 $n$ 增长。

### 9.4.4 CSA + HCA（2026）：两级压缩

近处 CSA（压缩 + 选择）、远处 HCA（极重压缩），1M 下 KV 降至 V3.2 的 **10%**。

---

## 9.5 系统层：cache 怎么存、怎么复用

| 技术 | 解决的问题 | 关键点 |
|---|---|---|
| **PagedAttention**（vLLM，2023） | 显存碎片 | 把 KV 切成固定大小页，按需分配，显存浪费从 60–80% 降到 <4% |
| **前缀缓存 / RadixAttention**（SGLang，2024） | 相同前缀重复计算 | 用前缀树共享 KV，多轮对话/少样本场景大幅省算力 |
| **Chunked prefill**（Sarathi-Serve，2024） | prefill 阻塞 decode | 把长 prompt 切块混批，降低尾延迟 |
| **树注意力**（投机解码） | 多候选共享前缀 | 分支共享 KV，掩码要按树结构构造 |

**这一层不改变注意力数学，但决定了"同样的 KV 预算能服务多少并发"**——工程上往往比算法层收益更大。

---

## 9.6 预算表：128K 上下文，四种方案的显存

假设 $L=32$、$h=32$、$d_h=128$、FP16、单条序列：

| 方案 | 每 token 每层 | 每 token 全模型 | 128K 总量 | 相对 MHA |
|---|---|---|---|---|
| **MHA** | 8192 个数（16 KB） | 512 KB | **64 GB** | 1× |
| **GQA（$g$=8）** | 2048 个数（4 KB） | 128 KB | **16 GB** | ÷4 |
| **MQA** | 256 个数（0.5 KB） | 16 KB | **2 GB** | ÷32 |
| **MLA** | 576 个数（1.1 KB） | 36 KB | **4.6 GB** | ÷14 |
| **KDA（线性）** | — | — | **常数（约 1 MB）** | 与 $n$ 无关 |

**这张表解释了整条演进线的经济逻辑**：

- MHA→GQA 是"**省 4 倍**"，够用一阵；
- GQA→MLA 是"**再省 3.5 倍**"，代价是实现复杂；
- MLA→线性 是"**从线性降到常数**"，但表达力掉得更多，所以必须混合。

---

## 9.7 五个容易踩的坑

1. **只压份数不压字节，很快到顶**：$g$ 最小是 1，MQA 就是极限。
2. **淘汰类方法大多 train-free**：分布偏移是硬伤，长文上先崩（H2O/Scissorhands 都有此问题）。
3. **attention sink 必须保留**：任何稀疏方案丢掉初始 token 都会崩，这不是玄学，是 softmax 归一化的必然结果。
4. **页式存储是前提**：没有 PagedAttention 式的分页，"按需加载"无法实现（选择粒度必须等于存储粒度）。
5. **cache 优化要和 kernel 一起做**：MLA 不做 absorb 优化会比 GQA 更慢；稀疏没有块对齐会打不满带宽。

---

## 9.8 本章参考文献

1. **2019.01** · Dai, Z., Yang, Z., Yang, Y., et al. — *Transformer-XL*（**跨段缓存首次提出**）— ACL 2019 ｜ [arXiv:1901.02860](https://arxiv.org/abs/1901.02860)
2. **2019.11** · Shazeer, N. — *Fast Transformer Decoding: One Write-Head is All You Need*（**MQA 首次提出**）｜ [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
3. **2022.06** · Pope, R., et al. — *Efficiently Scaling Transformer Inference* ｜ [arXiv:2211.05102](https://arxiv.org/abs/2211.05102)
4. **2023.05** · Ainslie, J., Lee-Thorp, J., de Jong, Z., et al. — *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*（**GQA 首次提出**）｜ [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
5. **2023.06** · Zhang, Z., Sheng, Y., Zhou, T., et al. — *H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs* ｜ [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)
6. **2023.05** · Liu, Z., Desai, A., Liao, F., et al. — *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression* ｜ [arXiv:2305.17118](https://arxiv.org/abs/2305.17118)
7. **2023.09** · Xiao, G., Tian, Y., Chen, B., et al. — *Efficient Streaming Language Models with Attention Sinks*（**StreamingLLM**）｜ [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
8. **2023.09** · Kwon, W., Li, Z., Zhuang, S., et al. — *Efficient Memory Management for Large Language Model Serving with PagedAttention*（**vLLM**）— SOSP 2023 ｜ [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
9. **2024.04** · Li, Y., et al. — *SnapKV: LLM Knows What You are Looking for Before Generation* ｜ [arXiv:2404.14469](https://arxiv.org/abs/2404.14469)
10. **2024.05** · DeepSeek-AI — *DeepSeek-V2*（**MLA 首次提出**）｜ [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
11. **2024.06** · Tang, J., et al. — *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference* ｜ [arXiv:2406.10774](https://arxiv.org/abs/2406.10774)
12. **2024.07** · Chen, Y., et al. — *MInference 1.0* ｜ [arXiv:2407.02490](https://arxiv.org/abs/2407.02490)
13. **2025.02** · DeepSeek-AI — *Native Sparse Attention*（NSA）— ACL 2025 ｜ [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)
14. **2025.10** · Moonshot AI — *Kimi Linear*（**KDA**）｜ [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)
15. **2026.04** · DeepSeek-AI — *DeepSeek-V4*（**CSA + HCA**）｜ [arXiv:2606.19348](https://arxiv.org/abs/2606.19348)

---

**下一章**：把注意力从 2014 到 2026 的**核心公式**排成一条线，看每一步到底改了什么符号、为什么改。
