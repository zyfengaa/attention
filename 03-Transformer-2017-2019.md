[第 2 章](02-软对齐革命-2014-2016.md) ｜ [← 返回目录](README.md) ｜ [第 4 章 →](04-高效注意力-2020-2021.md)

---

# 第 3 章 2017–2019：Transformer 与自注意力，把循环删掉

> **这一章不再是"某年提出了什么"，而是回答四个问题**：
> 1. 为什么必须删掉循环？——用**每层的 FLOPs 与串行步数**算给你看。
> 2. 自注意力凭什么能替代？——三个设计（缩放、多头、位置编码）各自在修什么 bug，逐个推导。
> 3. 删掉循环之后新增了什么债？——$O(n^2)$、$O(n)$ KV cache、位置外推。
> 4. 这三笔债怎么还？——2019 年的稀疏注意力与 MQA 是第一次还款尝试。

---

## 3.1 深度解读：Attention Is All You Need（2017）

**论文**：Vaswani, Shazeer, Parmar, et al.，NeurIPS 2017 ｜ [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) ｜ [PDF](https://arxiv.org/pdf/1706.03762)
**首次提出的模型**：Transformer（base 65M / big 213M，均为 6 层）

### 3.1.1 困境的定量刻画：不是"RNN 慢"，而是"RNN 的串行步数 = 序列长度"

先看三种层在**单层、序列长 $n$、模型宽 $d$** 下的账：

| 层类型 | 每层计算量 | 最长路径长度 | 每层串行步数 | 能否并行 |
|---|---|---|---|---|
| 循环（RNN/LSTM） | $O(n\cdot d^2)$ | $O(n)$ | $n$ | 否 |
| 卷积（ConvS2S，$k$ 宽） | $O(k\cdot n\cdot d^2)$ | $O(\log_k n)$ | $O(1)$ | 是 |
| 自注意力 | $O(n^2\cdot d)$ | $O(1)$ | $O(1)$ | 是 |

三个关键结论，全部是这张表的直接推论：

**结论 1（计算量交叉点）**：自注意力 vs 循环的比较是 $n^2 d$ vs $n d^2$，即**当 $n<d$ 时自注意力更便宜，$n>d$ 时更贵**。论文 Table 1 就是这张表。代入 $d=512$：$n=512$ 时两者打平；$n=8192$ 时自注意力贵 **16 倍**。这一条直接预言了 2020 年之后所有高效注意力研究的存在理由——**Transformer 的成本随 $n$ 平方增长，而它又被设计来处理长序列**。

**结论 2（为什么不是 CNN）**：ConvS2S 的路径长度是 $O(\log_k n)$，理论上也能做到对数级，但**每层只能看 $k$ 个邻居**，要覆盖全序列需要 $\log_k n$ 层堆叠；自注意力一层就把任意两点连起来。所以论文标题才是 "Attention Is All You Need" 而不是 "Convolution Is All You Need"——**它比的是"路径长度"，不是"参数效率"**。

**结论 3（真正被删掉的是什么）**：删掉的不是"循环"这个算子，而是**时间步之间的数据依赖**。RNN 的第 $t$ 步必须等第 $t-1$ 步算完，GPU 上千个 CUDA core 只能利用一个；自注意力把整个序列变成一个矩阵乘，**并行度从 1 变成 $n$**。这才是 2017 年真正的工程胜利。

### 3.1.2 缩放点积：$\sqrt{d_k}$ 是从方差推出来的，不是拍脑袋

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

**为什么必须除以 $\sqrt{d_k}$？** 设 $q,k\in\mathbb{R}^{d_k}$ 各分量独立同分布、均值 0、方差 1（训练初期近似成立），则

$$
q\cdot k=\sum_{i=1}^{d_k}q_i k_i
\quad\Longrightarrow\quad
\mathbb{E}[q\cdot k]=0,\qquad
\mathrm{Var}[q\cdot k]=d_k
$$

即**点积的标准差是 $\sqrt{d_k}$**。$d_k=64$ 时标准差 8，$d_k=512$ 时 22.6。softmax 的输入若是这种量级，会立刻进入**饱和区**：

$$
\mathrm{softmax}(z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
\ \xrightarrow[\ \|z\|\ \text{很大}\ ]{}\
\text{one-hot 化}
\quad\Longrightarrow\quad
\frac{\partial\,\mathrm{softmax}}{\partial z}\to 0
$$

梯度趋于 0 → 训练不动。除以 $\sqrt{d_k}$ 后方差归一为 1，softmax 落在敏感区。

**三个常被误解的点：**

1. **不是除以 $d_k$**。方差是 $d_k$ 的一次方，标准差才是 $\sqrt{d_k}$；除 $d_k$ 会让方差变成 $1/d_k$，softmax 反而过于平坦。
2. **加性注意力也饱和，但饱和方式不同**。Bahdanau 的 $v^\top\tanh(Wq+Uk)$ 有 $\tanh$ 天然把值压在 $(-1,1)$，但代价是多一次 MLP 与 $d\times d$ 参数；点积没有任何非线性，靠**显式缩放**控尺度。这是"用结构换参数"与"用缩放换结构"的取舍。
3. **缩放不改变注意力的排序**，只改变分布的尖锐度。所以它不影响"看哪里"，只影响"梯度能不能传"。

### 3.1.3 多头：不是"多个注意力取平均"，而是"多个独立的 softmax 归一化"

维度账（base 配置）：

$$
d_{\text{model}}=512,\quad h=8,\quad d_k=d_v=d_{\text{model}}/h=64
$$

每个头有独立的 $W_Q^i,W_K^i,W_V^i\in\mathbb{R}^{512\times 64}$，输出拼回 512 维再过 $W^O\in\mathbb{R}^{512\times 512}$。

**为什么多头有效？** 关键不在"多组参数"（那等价于一个大矩阵），而在**每个头有自己的 softmax**。一次 softmax 只能产生**一个概率分布**；$h$ 个 softmax 产生 $h$ 个分布，可以在同一层同时表达"该看句法父节点"与"该看指代对象"两种互相冲突的分配。这正是论文 Figure 3–4 可视化的内容。

**等价视角（重要）**：多头可看成对注意力矩阵做**秩为 $h$ 的分块低秩分解**——

$$
\mathrm{MHA}(X)=\sum_{i=1}^{h}\underbrace{A_i(X W_V^i)}_{\text{每头一个 }\mathbb{R}^{n\times d/h}}\;W^O_i
$$

总计算量与"单头 $d_k=512$"相同（$h\cdot d_k=d$），但**归一化次数从 1 变成 $h$**。多头几乎不增加 FLOPs 却明显提升表达力，增加的是**非线性（softmax）的数量**，不是参数量。

**论文消融（Table 3，base 模型，WMT14 EN-DE dev BLEU）**：

| 配置 | BLEU | 解读 |
|---|---|---|
| $h=1$（单头） | 24.9 | 单头明显不足 |
| $h=4$ | 25.5 | 收益主要来自 1→4 |
| $h=8$ | 25.8 | 默认配置 |
| $h=16$ | 25.8 | 饱和 |
| $h=32$ | 25.4 | 过头，反而掉 |

**结论**：多头收益**边际递减**，$h>16$ 后开始反噬（每头维度太小）。这条曲线是 2023 年 GQA 的理论基础——既然 $h=32$ 不比 $h=8$ 好，那 32 个头里的 K/V 就是**冗余的**，可以共享。

### 3.1.4 位置编码：正弦编码里藏着 RoPE 的种子

$$
PE_{(pos,\,2i)}=\sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad
PE_{(pos,\,2i+1)}=\cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

**为什么自注意力必须加位置？** 因为自注意力是**置换等变**的：打乱输入 token 顺序，输出只是跟着打乱。证明一行即可：$\mathrm{softmax}(QK^\top)V$ 中 $Q,K,V$ 都随 token 排列做同一置换，结果只是被同一置换重排。

**正弦编码的真正性质（常被忽略）**：对任意固定偏移 $k$，存在一个**只依赖 $k$ 的旋转矩阵 $R_k$**，使得

$$
PE(pos+k)=R_k\cdot PE(pos)
$$

因为每对 $(\sin,\cos)$ 就是一个二维旋转。**"相对位置可以用线性变换表示"——这正是 2021 年 RoPE 的全部思想**：既然位置差是旋转，那就把旋转直接作用到 $q,k$ 上，点积自动只依赖相对距离（见 4.12）。**2017 年的位置编码里已经包含这个数学事实，只是当时没人把它用到 $q,k$ 上。**

**消融**：论文 Table 3 显示**正弦编码与学习式位置嵌入结果几乎相同**（dev BLEU 差 <0.1）。这说明 2017 年的位置上界不在"怎么编码"，而在"能不能外推"——而论文没测这一点，成了 2019–2023 年最大的一笔债。

### 3.1.5 被低估的部分：训练配方才是它能跑起来的原因

| 技巧 | 取值 | 解决什么 |
|---|---|---|
| **warmup** | 前 4000 步线性升温 | 初期梯度方差大，直接大 lr 会发散 |
| **Adam $\beta_2$** | 0.98（而非默认 0.999） | 二阶矩估计要更快跟上梯度变化 |
| **label smoothing** | 0.1 | 缓解过度自信导致的 BLEU 停滞 |
| **dropout** | 0.1（base）/ 0.3（big） | WMT 仅 450 万句对，防过拟合 |
| **Post-LN** | $\mathrm{LN}(x+\mathrm{Sublayer}(x))$ | 当时的标准做法 |

**Post-LN 的隐患**：梯度要穿过每层 LayerNorm，深层时梯度尺度不稳，必须靠 warmup 兜住。这就是 2019 年 GPT-2 改用 **Pre-LN**（$x+\mathrm{Sublayer}(\mathrm{LN}(x))$）的原因——**不 warmup 也能训深**，代价是表达能力略降（后来的 RMSNorm + QK-norm 在补这个洞）。

### 3.1.6 实验证据：不是"略好"，是"训练成本低一个量级还更好"

| 指标 | 数值 |
|---|---|
| WMT14 EN-DE | **28.4 BLEU**（big，比此前最佳单模型 +2.0） |
| WMT14 EN-FR | **41.8 BLEU**（big，超过此前最佳集成模型） |
| 训练成本 | base：8×P100 **12 小时**；big：**3.5 天** |
| 对比 | 同期 ConvS2S / GNMT 需数倍算力才能接近 |

论文 Figure 2 是"BLEU vs 训练 FLOPs"曲线：**Transformer 在同等算力下始终在上方**。这张图才是它取代 RNN 的真正原因——不是精度，是**算力效率**。

### 3.1.7 它同时欠下的三笔债（后面十年的研究地图）

| 债 | 数学形式 | 后果 | 偿还者 |
|---|---|---|---|
| **二次复杂度** | 时间 $O(n^2d)$、激活 $O(n^2)$ | 8K token 就爆显存 | 稀疏（2019）、线性（2020）、FlashAttention（2022） |
| **KV cache 线性增长** | 每 token 存 $2hd_h$ 个数 | 32K 上下文的 KV 比模型还大 | MQA（2019）、GQA（2023）、MLA（2024） |
| **位置外推** | 绝对编码无外推性 | 训练 512、测试 1024 即崩 | 相对位置（2018）、T5 bias（2019）、RoPE（2021）、ALiBi（2021） |

**这三笔债就是本文第 4–9 章的全部内容。** 2017 年之后所有注意力研究，都是在对 Transformer"还债"。

### 3.1.8 落地谱系（谁在哪一代用了它）

| 年份 | 模型 | 注意力配置 | 相对 Transformer 的改动 |
|---|---|---|---|
| 2018 | BERT-base | 12 层 / 12 头 / $d$=768 | 双向（去掉因果掩码）+ 绝对位置 |
| 2018 | GPT-1 | 12 层 / 12 头 | 因果掩码 + 学习式位置 |
| 2019 | GPT-2 | 48 层 / 25 头 / $d$=1600 | **Pre-LN**、可学习位置、上下文 1024 |
| 2020 | GPT-3 | 96 层 / 96 头 | **交替 dense + 局部带状稀疏**（见 3.2） |
| 2023 | Llama 1 | 32 层 / 32 头 | **RoPE** 替代绝对位置 |
| 2023 | Llama 2 70B | 80 层 | **GQA**（8 组 KV） |
| 2024 | Llama 3 8B | 32 层 / 32 头 / 8 组 KV | GQA + RoPE + 128K 词表 |

**读法**：从 2017 到 2024，Transformer 的**主干数学几乎没变**（还是 scaled dot-product + 多头），变的全是"位置怎么编"和"KV 怎么省"。这就是为什么说注意力研究的重心在 2020 年后从"表达力"转向了"成本"。

---

## 3.2 深度解读：Sparse Transformer（2019）——第一次试图把 $O(n^2)$ 变成 $O(n\sqrt n)$

**论文**：Child, Gray, Radford, Sutskever，2019 ｜ [arXiv:1904.10509](https://arxiv.org/abs/1904.10509) ｜ [PDF](https://arxiv.org/pdf/1904.10509)
**首次提出的模型**：Sparse Transformer

### 3.2.1 观察：注意力矩阵天生稀疏

作者先做了一件朴素的事——**统计训练好的 Transformer 注意力矩阵的行熵**。结论：绝大多数位置的注意力集中在少数 token 上，**大部分 $(i,j)$ 的权重接近 0**。既然看了也不用，那就不算。

问题变成：**哪些 $(i,j)$ 值得算？** 三种候选：

| 模式 | 连接方式 | 适合的数据 | 每行边数 |
|---|---|---|---|
| **strided（跨步）** | 位置 $i$ 连到 $i, i\pm s, i\pm 2s,\dots$ | 图像、音频这类**周期性**结构 | $n/s$ |
| **fixed（固定）** | 前 $s$ 个邻居 + 每 $s$ 个的"汇总点" | 文本这类**局部强、全局弱**的结构 | $s+n/s$ |
| **full（全）** | 全连 | 小序列 | $n$ |

取 $s=\sqrt n$，两种模式各 $O(\sqrt n)$ 条边/行 → 总边数 $O(n\sqrt n)$。

### 3.2.2 数学：把一个大注意力分解成两个稀疏注意力的复合

不是"把矩阵挖空"再算，而是**换计算图**：每层算两次稀疏注意力，让**信息经过两跳到达**。

$$
\text{第 }l\text{ 层}:\quad
A^{(l)}=\mathrm{SparseAttn}_{\text{strided}}\big(\mathrm{SparseAttn}_{\text{fixed}}(X)\big)
$$

**关键洞察**：两层 $O(n\sqrt n)$ 的稀疏注意力，其**有效感受野可以覆盖全序列**——第一跳让每个位置看到 $\sqrt n$ 个"列代表"，第二跳让"列代表"看到所有行。效果近似一层全注意力，成本从 $n^2$ 降到 $n\sqrt n$。

用图论语言：**把完全图换成了一个直径很小的稀疏图**。这正是 2020 年 BigBird 用"扩展图"给出理论保证的地方（见 4.3）。

### 3.2.3 为什么 $s=\sqrt n$ 是最优的

总边数 $E(s)=n\big(\underbrace{n/s}_{\text{strided}}+\underbrace{s}_{\text{local}}\big)$，对 $s$ 求极小：

$$
\frac{dE}{ds}=n\left(-\frac{n}{s^2}+1\right)=0
\;\Longrightarrow\;
s=\sqrt n,\qquad E_{\min}=2n^{3/2}
$$

这就是 $O(n\sqrt n)$ 的来历，也是"局部 + 跨步"组合的**理论上限**——想再降必须换思路（低秩、核函数、循环），这就是 2020 年那批工作的起点。

### 3.2.4 实验证据，以及"纸面收益 vs 实际收益"

| 任务 | 结果 |
|---|---|
| enwik8 字符级语言建模 | **0.99 bpc**（首次低于 1.0） |
| 序列长度 | 12,288 token（同期 dense 约 1K–2K） |
| 图像生成 | 64×64 级别的像素级自回归 |

**但这里有一个必须讲清楚的历史教训**：Sparse Transformer **纸面上是 $O(n\sqrt n)$，GPU 上并不快**。因为：

1. 稀疏索引导致**随机访存（gather）**，无法用连续大矩阵乘；
2. 需要**手写 CUDA kernel**，不可移植到新硬件；
3. 不同任务要换不同稀疏模式，**没有通用性**。

**结果**：GPT-3（2020）确实用了"交替 dense + 局部带状稀疏"，但 2023 年之后模型**几乎全部退回 dense**——因为 FlashAttention 把 dense 优化到接近硬件峰值，稀疏实现跟不上。**"算法复杂度"输给了"硬件适配度"**，这是本文最重要的教训之一（第 5 章会再见到）。

### 3.2.5 失效条件

- 序列不够长（$n<\sim 2\text{K}$）时，稀疏收益被索引开销吃掉；
- 需要**精确全局检索**（如"比较首尾两个数"）时，两跳稀疏图会丢信息；
- 硬件不支持块稀疏时，理论加速比拿不到。

### 3.2.6 遗产

- **思想**：注意力可分解为"局部 + 全局" → Longformer（2020）、BigBird（2020）、NSA（2025）、CSA/HCA（2026）全部沿用；
- **教训**：稀疏必须与硬件对齐（block size、连续访存），否则不如 dense —— 这条到 2025 年 NSA 提出 "hardware-aligned" 才被真正解决。

---

## 3.3 深度解读：MQA（2019）——第一个把"推理是带宽瓶颈"讲清楚的工作

**论文**：Shazeer，*Fast Transformer Decoding: One Write-Head is All You Need*，2019 ｜ [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) ｜ [PDF](https://arxiv.org/pdf/1911.02150)
**首次提出的模型**：MQA

### 3.3.1 定量：解码为什么慢？用 roofline 算一遍

A100 规格：算力约 312 TFLOPS（FP16），显存带宽约 2 TB/s。两者相除得**平衡点**：

$$
\text{arithmetic intensity}^*=\frac{312\times10^{12}}{2\times10^{12}}=156\ \text{FLOP/byte}
$$

即：**每读 1 字节，至少要做 156 次浮点运算，才不浪费算力**。

自回归解码（batch=1）的实际情况：生成 1 个 token 要把**全部权重（约 $2P$ 字节，$P$ 为参数量）读一遍**，只做约 $2P$ 次 FLOP。所以

$$
\text{实际 intensity}\approx\frac{2P}{2P}=1\ \text{FLOP/byte}\ \ll\ 156
$$

**结论：解码阶段 99% 的时间在搬数据，不在算。** 这不是"优化一下"能解决的，是结构性的。

### 3.3.2 KV cache 的读取量，才是长上下文的真正成本

生成第 $t$ 个 token 时需读取的历史 K/V 量：

$$
\underbrace{2}_{\text{K 和 V}}\times\underbrace{n}_{\text{历史长度}}\times\underbrace{h}_{\text{头数}}\times\underbrace{d_h}_{\text{每头维度}}\times\underbrace{2}_{\text{FP16 字节}}
$$

代入 32 层 / 32 头 / 128 维、$n=32\text{K}$：每层 $2\times32768\times32\times128\times2=536\ \text{MB}$，**32 层合计 17 GB**——而模型权重才 14 GB。**长上下文推理的成本主要不是权重，是 KV。**

### 3.3.3 MQA 的做法：让所有 Q 头共享一份 K/V

$$
\underbrace{Q\in\mathbb{R}^{n\times h\times d_h}}_{\text{保留 } h \text{ 个查询头}},\qquad
\underbrace{K,V\in\mathbb{R}^{n\times 1\times d_h}}_{\text{只有 1 份}}
$$

读取量从 $2nhd_h$ 降到 $2nd_h$，**降 $h$ 倍**（$h=32$ 时，17 GB → 0.5 GB）。

论文的另一贡献是给出"为什么可以共享"的论证：不同查询头需要不同的"查询方向"，但**键值空间的冗余度很高**——32 个头学到的 K/V 高度相关。这一点被 3.4 节的"16 个头剪枝"实验独立证实。

### 3.3.4 实验证据与代价

| 指标 | 结果 |
|---|---|
| 解码加速 | **1.7–3×**（翻译任务，beam search 下更明显） |
| BLEU | 小幅下降（约 0.1–0.5，视任务） |
| 训练稳定性 | **变差**，需特殊初始化与更长 warmup |

**为什么会掉点？** 从 3.1.3 的结论看：多头的价值在于"多个独立 softmax"。MQA 没减少 softmax 的数量（Q 头还是 $h$ 个），但**它们的 K 分布被强制相同**，等于给所有头加了"必须看同样东西"的约束。表达力损失来自这里。

### 3.3.5 失效条件与遗产

- **失效**：小模型（<1B）、需要细粒度多头分工的任务（代码、多语言）掉点明显；
- **遗产**：Falcon（2023）、StarCoder（2023）、ChatGLM 系列采用；**2023 年下半年被 GQA 全面取代**——GQA 是"MQA 与 MHA 的插值"（见 5.3）；
- **真正的贡献**：把"**KV cache 是长上下文第一成本**"变成行业共识，从此注意力优化主战场从"算得快"转向"存得少"。

---

## 3.4 其余节点：每条只留"为什么"，细节见总表

| 时间 | 技术 | 论文 | 一句话的"为什么" |
|---|---|---|---|
| 2018.03 | **相对位置嵌入**（Shaw） | [arXiv:1803.02155](https://arxiv.org/abs/1803.02155) | 绝对编码学不到"距离"，而句法关系本质是相对的；做法是把位置变成注意力打分里的加性偏置，裁到 $[-k,k]$ 的可学习向量。 |
| 2018.06 | **GPT** | OpenAI 技术报告（无 arXiv） | 因果掩码 + 自回归：牺牲双向信息换来"可生成"，这条路在 GPT-3 后胜出。 |
| 2018.10 | **BERT** | [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) | 双向自注意力 + MLM：每个位置同时看左右，代价是 `[MASK]` 造成预训练/微调不一致。 |
| 2019.01 | **Transformer-XL** | [arXiv:1901.02860](https://arxiv.org/abs/1901.02860) | 上下文碎片化：段与段之间没有信息流。解法是缓存上一段隐状态（**KV cache 的雏形**）+ 相对位置避免缓存冲突。 |
| 2019.05 | **Adaptive Span** | [arXiv:1905.07799](https://arxiv.org/abs/1905.07799) | 不同头需要的上下文长度不同；用可学习 soft mask 截断，但**只省显存不省算力**（仍算全矩阵）。 |
| 2019.05 | **头剪枝分析** | [arXiv:1905.10650](https://arxiv.org/abs/1905.10650) | 训练后大量头可剪而 BLEU 几乎不掉，说明多头冗余；但**训练时剪会崩**——多头是"训练期的探索多样性"，不是"推理期的必需品"。 |
| 2019.06 | **XLNet 双流** | [arXiv:1906.08237](https://arxiv.org/abs/1906.08237) | 自回归不能看右侧、BERT 不能生成；用内容流 + 查询流分离 + 排列语言建模调和。 |
| 2019.10 | **T5 相对分桶** | [arXiv:1910.10683](https://arxiv.org/abs/1910.10683) | 相对距离按对数分 32 桶加进 logits，零额外参数、可外推一点，是 ALiBi 之前的工业标准。 |
| 2019.10 | **BP-Transformer** | [arXiv:1911.04070](https://arxiv.org/abs/1911.04070) | 二叉分区树：把 token 组织成树，粗粒度节点代表区间，$O(n\log n)$。 |
| 2019.11 | **Compressive Transformer** | [arXiv:1911.05507](https://arxiv.org/abs/1911.05507) | 被挤出窗口的旧记忆不丢弃，压缩成粗粒度表示存起来——2026 年 HCA 的思想祖先。 |

---

## 3.5 这一章的三条"反直觉"结论

1. **Transformer 赢 RNN 靠的不是精度，是算力效率**（Figure 2 的 BLEU/FLOPs 曲线）。理解了这点，才能理解为什么 2022 年后"用更多算力换更好结果"会成为主线。
2. **多头的价值在"多个 softmax"，不在"多组参数"**。这条推理解释了三件事：为什么多头几乎不增加 FLOPs、为什么头剪枝在推理时可行、为什么 MQA 会掉点。
3. **$O(n\sqrt n)$ 的稀疏注意力输给了 $O(n^2)$ 的 dense 注意力**，只因为后者对硬件友好。**从 2019 年起，"算法复杂度"不再是唯一指标，"能否映射到 tensor core" 同等重要。**

---

## 3.6 本章参考文献

按首次公开时间排序，链接指向 arXiv 摘要页（"PDF"为直链）。作者只列前三，`et al.` 表示更多。

1. **2017.06** · Vaswani, A., Shazeer, N., Parmar, N., et al. — *Attention Is All You Need* — NeurIPS 2017 ｜ [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) ｜ [PDF](https://arxiv.org/pdf/1706.03762)
2. **2018.03** · Shaw, P., Uszkoreit, J., Vaswani, A. — *Self-Attention with Relative Position Representations* — NAACL 2018 ｜ [arXiv:1803.02155](https://arxiv.org/abs/1803.02155) ｜ [PDF](https://arxiv.org/pdf/1803.02155)
3. **2018.06** · Radford, A., Narasimhan, K., Salimans, T., Sutskever, I. — *Improving Language Understanding by Generative Pre-Training*（GPT-1）— OpenAI 技术报告，无 arXiv
4. **2018.10** · Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* — NAACL 2019 ｜ [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) ｜ [PDF](https://arxiv.org/pdf/1810.04805)
5. **2019.01** · Dai, Z., Yang, Z., Yang, Y., et al. — *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context* — ACL 2019 ｜ [arXiv:1901.02860](https://arxiv.org/abs/1901.02860) ｜ [PDF](https://arxiv.org/pdf/1901.02860)
6. **2019.04** · Child, R., Gray, S., Radford, A., Sutskever, I. — *Generating Long Sequences with Sparse Transformers* — 2019 ｜ [arXiv:1904.10509](https://arxiv.org/abs/1904.10509) ｜ [PDF](https://arxiv.org/pdf/1904.10509)
7. **2019.05** · Sukhbaatar, S., Grave, E., Bojanowski, P., Joulin, A. — *Adaptive Attention Span in Transformers* — ACL 2019 ｜ [arXiv:1905.07799](https://arxiv.org/abs/1905.07799) ｜ [PDF](https://arxiv.org/pdf/1905.07799)
8. **2019.05** · Michel, P., Levy, O., Neubig, G. — *Are Sixteen Heads Really Better than One?* — NeurIPS 2019 ｜ [arXiv:1905.10650](https://arxiv.org/abs/1905.10650) ｜ [PDF](https://arxiv.org/pdf/1905.10650)
9. **2019.06** · Yang, Z., Dai, Z., Yang, Y., et al. — *XLNet: Generalized Autoregressive Pretraining for Language Understanding* — NeurIPS 2019 ｜ [arXiv:1906.08237](https://arxiv.org/abs/1906.08237) ｜ [PDF](https://arxiv.org/pdf/1906.08237)
10. **2019.10** · Raffel, C., Shazeer, N., Roberts, A., et al. — *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*（T5）— JMLR 2020 ｜ [arXiv:1910.10683](https://arxiv.org/abs/1910.10683) ｜ [PDF](https://arxiv.org/pdf/1910.10683)
11. **2019.11** · Ye, Z., Guo, Q., Gan, Q., et al. — *BP-Transformer: Modelling Long-Range Context via Binary Partitioning* — NeurIPS 2019 ｜ [arXiv:1911.04070](https://arxiv.org/abs/1911.04070) ｜ [PDF](https://arxiv.org/pdf/1911.04070)
12. **2019.11** · Shazeer, N. — *Fast Transformer Decoding: One Write-Head is All You Need*（MQA）— 2019 ｜ [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) ｜ [PDF](https://arxiv.org/pdf/1911.02150)
13. **2019.11** · Rae, J. W., Potapenko, A., Jayakumar, S. M., et al. — *Compressive Transformers for Long-Range Sequence Modelling* — ICLR 2020 ｜ [arXiv:1911.05507](https://arxiv.org/abs/1911.05507) ｜ [PDF](https://arxiv.org/pdf/1911.05507)

> **引用口径**：年份取 arXiv v1 的提交月；会议信息取正式发表版。少数工作（GPT-1）只有技术报告，已在条目中注明。

---

**下一章**：2020–2021 年，四条路同时试图还第一笔债（$O(n^2)$）：稀疏图（Longformer/BigBird）、低秩（Linformer）、核函数（Performer）、循环（Linear Transformer）。我们会逐条推导它们**为什么在数学上成立、在实践中为什么大多失败**。
