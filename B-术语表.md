[附录 A](A-参考文献总表.md) ｜ [← 返回目录](README.md)

---

# 附录 B 术语表

| 术语 | 英文 | 一句话解释 |
|---|---|---|
| 注意力 | Attention | 用 query 与 key 的相似度加权取回 value |
| 交叉注意力 | Cross-Attention | Q 来自解码器，K/V 来自编码器 |
| 自注意力 | Self-Attention | Q/K/V 来自同一序列 |
| 多头 | Multi-Head | 把 $d$ 维拆成 $h$ 份并行算注意力再拼接 |
| MQA | Multi-Query Attention | 所有 Q 头共享 1 份 K/V |
| GQA | Grouped-Query Attention | 每 $g$ 个 Q 头共享 1 份 K/V |
| MLA | Multi-head Latent Attention | 把 K/V 压成低秩隐向量缓存 |
| KV Cache | — | 自回归解码时缓存历史 K/V |
| 稀疏注意力 | Sparse Attention | 只计算部分 $(i,j)$ 对 |
| 线性注意力 | Linear Attention | 用核函数 + 结合律把复杂度降到 $O(n)$ |
| 状态空间模型 | SSM | 用递推状态 $h_t=Ah_{t-1}+Bx_t$ 建模序列 |
| Delta 规则 | Delta Rule | 写入前先擦除旧值的联想记忆更新规则 |
| RoPE | Rotary Position Embedding | 用旋转把相对位置编码进 Q/K 内积 |
| ALiBi | Attention with Linear Biases | 在注意力 logits 上加距离线性偏置 |
| 注意力汇聚 | Attention Sink | 大量注意力被分配给少数无意义 token |
| 位置插值 | Position Interpolation | 压缩位置索引以扩展 RoPE 上下文 |
| IO 感知 | IO-Awareness | 以显存读写为第一优化目标的 kernel 设计 |
| 分页 KV | Paged KV Cache | 像虚拟内存一样按块分配 KV |
| 序列并行 | Sequence Parallelism | 把序列维切到多卡 |
| 有效上下文 | Effective Context | 模型真正能用（而非只是能跑）的上下文长度 |

---

> **最后一句**：注意力四十年的历史，是"**如何用有限的内存，近似无限的全对全交互**"的历史。每一次迭代都在回答同一个问题——**我可以少看多少，还能不丢信息？** 2026 年的答案是：**分层次地看**（近处细、远处粗）、**有选择地看**（indexer 选块）、**压着看**（低秩/压缩），再留一小部分**全看**（全局注意力层）兜底。

*文档生成时间：2026-11-19。论文编号已尽力核对；标注"技术报告/博客"的条目为无 arXiv 正式论文的工作，引用时请以官方发布为准。*
