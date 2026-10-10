# HCAM V2 技术报告：121M 中文混合卷积-注意力掩码语言模型

**HCAM V2 / 121M — Technical Report**  
作者：`qilimary`  
日期：2026-10-10

> HCAM V2 是 HCAM 中文 MLM 的第二代版本。  
> 本报告重点记录模型结构、数据规模、训练预算、预训练目标和最终评测结果。  
> V2 当前仅公开技术报告与结果，不公开权重和完整训练代码；仓库现有可运行代码与 Release 权重仍对应 HCAM V1。
> 本报告是对 V1 主技术报告的第二代补充：V2 沿用已经验证的 HCAM 主体路线，并在模型规模、数据、词表和训练目标上继续扩展。

---

## 摘要 / Abstract

HCAM V2 是一个约 **121.08M 参数**的中文双向掩码语言模型。模型延续 HCAM 的混合卷积-注意力路线：11 层主干中包含 **8 个 Local 层与 3 个 GQA 全局注意力层**，全局层位于 L3 / L7 / L11。

V2 使用 **23K 中英混合词表**，中文保持“一字一 token”，Latin 片段允许最多约 2.8K BPE 子词。预训练语料约 **3.5B prepared tokens**，总训练量约 **3.4 passes**，即约 **11.9B token exposures**。按 624 context、有效 global batch 384 计算，整个训练约对应 **49.7K 次 optimizer updates**。

训练目标以动态多粒度 MLM 为主，辅以轻量 Self-Critic RTD；优化器采用 Muon + AdamW，并配合 AdaGC。最终固定验证集 MLM Top-1 达到 **70.14%**；在外部语言学基准 **SLING-38K** 上达到 **75.579%（28,720 / 38,000）**，比 V1 的 64.924% 提高 **10.655 个百分点**。

**English summary.** HCAM V2 is a 121.08M-parameter Chinese bidirectional masked language model with 8 local blocks and 3 global GQA layers. It was trained on a 3.5B-token prepared mixture for about 3.4 passes (~11.9B token exposures, ~49.7K optimizer updates), using dynamic multi-granularity MLM plus a lightweight Self-Critic RTD objective. Final SLING-38K accuracy is 75.579%, a +10.655 pp gain over HCAM V1.

---

## 1. V1 → V2

| Item | HCAM V1 | HCAM V2 |
|---|---:|---:|
| Parameters | 102.78M | **121.08M** |
| Layers | 10 | **11** |
| Local / Global | 7 + 3 | **8 + 3** |
| Hidden size | 1056 | **1088** |
| Attention layers | 3 / 7 / 10 | **3 / 7 / 11** |
| Vocabulary | 18K character | **23K mixed** |
| Context | 600 | **624** |
| Dropout | 0.10 | **0.05** |
| Main objective | multi-granularity MLM | **dynamic MLM + light RTD** |
| Prepared training data | ~1.638B chars | **3.5B tokens** |
| Effective passes | 2 | **3.4** |
| SLING-38K | 64.924% | **75.579%** |
| Reflexive-560 | **67.857%** | 66.071% |

V2 的主要变化不是简单增加参数量，而是同时扩大了数据规模、训练预算、词表覆盖和 MLM 课程，并增加轻量 RTD 辅助目标。

---

## 2. 模型结构 / Architecture

| 项目 | HCAM V2 |
|---|---:|
| Parameters | **121.08M** |
| Layers | **11** |
| Hidden size | **1088** |
| Local layers | **8** |
| GQA layers | **3（L3 / L7 / L11）** |
| Query / KV heads | **16 / 4** |
| Context length | **624** |
| Vocabulary | **23,000** |
| Latin BPE budget | **≤2,800** |
| Dropout | **0.05** |
| Embedding dropout | **0.0** |
| Local expand | **1.50** |
| FFN multiplier | **2.75** |
| Attention gate | per-token / per-head sigmoid gate |
| H-gate | rank-16 |
| Dynamic Local | L6 / L10 |
| Norm | zero-centered RMSNorm variant |
| Position | RoPE |
| Weight tying | input embedding = MLM head |

11 层中只有 3 层使用全局 GQA，其余 8 层继续承担局部模式建模。L6 / L10 在局部卷积基础上增加动态短卷积分支；三个 GQA 层保留 headwise gate，并加入 rank-16 H-gate。

这仍然延续 HCAM 的基本思路：**大部分计算用于高效局部建模，少量全局注意力层负责跨位置交互。**

---

## 3. Tokenizer

V2 使用 **23K** 混合词表。

中文部分仍保持：

- 一个汉字对应一个 token；
- 不把常见中文词永久合并成输入 token；
- 整词 / 子词边界只用于指导 MLM mask。

Latin 片段允许 BPE：

- Latin BPE 预算不超过约 **2,800**；
- 保留基础大小写字符；
- 对普通大小写做适度归一化，同时保留有意义的缩写和混合大小写形式。

相比 V1，V2 的词表覆盖更广，同时继续保持中文字符级预测的可解释性。

---

## 4. 数据与训练预算 / Data & Training Budget

V2 的 prepared training mixture 约为 **3.5B tokens**，包含通用网页、教育文本、百科、文学以及法律、医疗、生物医药、数学、科研、金融等专业长尾内容。

主要来源规模：

| 来源 | 约定预算 |
|---|---:|
| Ultra | 1.632B |
| SkyPile | 391M |
| FineEdu | 578M |
| Cosmopedia | 340M |
| Wiki | 272M |
| Wikisource / literature | 170M |
| 六类专业长尾 | ~101M |
| 用户语料 | ≤16M |

### 4.1 训练量怎么理解

训练日志内部使用了 **34 个 virtual epochs**，但这并不表示完整数据训练了 34 遍。

更直观的口径是：

- prepared corpus：**3.5B tokens**
- 总训练：**约 3.4 passes**
- 总暴露量：**约 11.9B tokens**
- context：624
- effective global batch：384 sequences
- 每次 optimizer update 约处理：`624 × 384 = 239,616` tokens
- 总 optimizer updates：**约 49.7K**

因此可以把整个预训练理解为：

> **3.5B 数据训练约 3.4 遍，共约 11.9B token exposures / 49.7K optimizer steps。**

最后约 0.4 pass 使用温和的质量重加权，略提高 FineEdu、Wiki、Wikisource 和专业长尾数据的抽样比例，同时降低部分普通网页来源。它更接近“高质量内容在收尾阶段多复习一点”，而不是大量重复某些固定句子。

---

## 5. 预训练目标 / Objectives

### 5.1 动态多粒度 MLM

V2 的主任务仍然是 MLM。

最终 mask 粒度约为：

- **30%** 单字符
- **50%** 整组 / whole-piece
- **20%** 相邻或 span

相邻分支覆盖 2～6 字的连续 span。

总遮蔽率在训练中动态变化，后期固定在 **15.5%**。训练越往后，整词和跨度 mask 的比例逐渐提高，使模型不只学习孤立字符恢复，也需要利用更完整的上下文结构。

被选 token 的替换方式最终为：

- 80% learned mask embedding
- 8% 同类别随机替换
- 2% 全类别随机替换
- 10% 保持原样

### 5.2 Self-Critic RTD

V2 加入轻量 Self-Critic RTD，让 encoder 额外判断一个位置是否像被替换进去的假 token。

RTD 始终只是辅助目标：

- 中段最高权重约 **0.20**
- 训练末期逐渐下降
- 最终权重 **0.12**

最终 RTD balanced accuracy 为 **83.96%**。

MLM 负责“这里应该是什么”，RTD 更偏向“当前 token 在上下文里是否自然 / 一致”。最终推理可以完全不调用 RTD critic。

---

## 6. 优化与训练环境 / Optimization

| 项目 | 配置 |
|---|---:|
| GPU | **Tesla T4 × 2** |
| Mixed precision | FP16 |
| Per-GPU microbatch | **24** |
| Gradient accumulation | **8** |
| Effective global batch | **384** |
| Optimizer | **Muon + AdamW** |
| Gradient control | AdaGC |
| Peak LR | **7e-4** |
| Final LR | **7e-5** |
| LR schedule | warmup + cosine |
| Gradient clip | 1.5 |

整个训练跨多个 Kaggle 会话完成，并使用精确 checkpoint 续接模型、优化器和训练进度。工程细节不作为本报告重点。

---

## 7. 最终预训练结果

第 34 个 virtual epoch 结束时：

### 固定验证集

- MLM loss：**1.3717**
- masked PPL：**3.94**
- Top-1：**70.14%**

### 困难 25% mask

- MLM loss：**1.9054**
- masked PPL：**6.72**
- Top-1：**60.90%**
- Top-5：**77.50%**

### 固定三粒度测试

| Granularity | Loss | PPL | Top-1 | Top-5 |
|---|---:|---:|---:|---:|
| Character | **0.9466** | **2.58** | **78.38%** | **90.69%** |
| Whole-piece | **1.6663** | **5.29** | **63.92%** | **81.21%** |
| Multi-span | **2.5249** | **12.49** | **50.56%** | **68.48%** |

结果显示，V2 的普通单字符恢复已经明显强于连续多字符联合恢复。多字符 span 仍然是模型最清晰的剩余弱点之一。

---

## 8. SLING-38K

最终结果：

- **75.579%**
- **28,720 / 38,000**
- ties：24
- OOV / `[UNK]`：**0 / 945,495**

![SLING-38K score comparison](sling_38k_leaderboard.png)

> 上图用于直观展示 HCAM V1 / V2 在 SLING-38K 参考模型中的分数位置。

按语言现象：

| Phenomenon | Accuracy |
|---|---:|
| AlternativeQuestion | **85.60%** |
| Anaphor | **75.69%** |
| Aspect | **62.44%** |
| Classifier-Noun | **91.22%** |
| DefinitenessEffect | **96.65%** |
| PolarityItem | **65.53%** |
| RelativeClause | **33.20%** |
| WhFronting | **99.65%** |

V1 为 **64.924%**，V2 提高到 **75.579%**：

- **+10.655 percentage points**
- 多答对 **4,049** 对

这也是 V2 相比 V1 最明显的整体提升。

同时，V2 的能力仍然不完全均衡。WhFronting、DefinitenessEffect、Classifier-Noun 很强，而 RelativeClause 仍然是明显短板。

---

## 9. Reflexive-560

最终结果：

- **66.071%**
- **370 / 560**

分项：

| Group | Accuracy |
|---|---:|
| reflexive_auto | **58.93%** |
| reflexive_natural | **73.21%** |

V1 在同一测试上的总分为 **67.857%**，因此 V2 并没有在所有语言学子任务上都超过 V1。

V2 在 natural 子集上的表现明显高于 auto 模板，也说明更强的通用预训练能力并不会自动转化为所有人工形式规则上的同步提升。

---

## 10. 下游字符纠错与量化部署

预训练结束后，HCAM V2 从基座重新微调了一个联合 **的 / 地 / 得 / 做 / 作** 字符纠错模型。任务设计保持为 masked prediction：目标字符位置被遮蔽，模型根据其余双向上下文直接预测正确字符。

### 10.1 基座到微调模型

微调前，V2 基座在 500 句、1,082 个目标位置的五字测试上：

- Accuracy：**88.262%**
- Macro-F1：**85.841%**
- Macro-F0.5：**86.373%**

微调后的最终 FP32 模型：

| Strategy | Accuracy | Macro-F1 | Macro-F0.5 |
|---|---:|---:|---:|
| all | **98.969%** | **96.730%** | **97.035%** |
| single | **99.062%** | **96.791%** | — |

在 FreshNatural“的 / 地 / 得”自然文本测试的 **4,980 个目标位置**上：

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Older HCAM fine-tune | 99.096% | 99.199% |
| **HCAM V2 fine-tune** | **99.438%** | **99.470%** |

这一结果与 V1 时期已经观察到的字符纠错优势方向一致，并显示 V2 在自然字符恢复任务上还有进一步提升。

### 10.2 量化

最终移动端格式为 **ExecuTorch `.pte` + XNNPACK Dynamic INT8 per-channel**：

- FP32 导出：约 **462.66 MiB**
- PTE INT8：约 **117.24 MiB**
- 体积约缩小 **3.95×**

在 2,000 句 / 6,501 个目标位置的完整 PTE Runtime 等价性验证中：

- Accuracy：**99.000%**
- Macro-F1：**96.823%**
- Macro-F0.5：**97.168%**

与 FP32 `all` 结果相比没有观察到可测的精度下降。另有 **116.97 MiB** 的 ONNX Runtime dynamic INT8 版本，可用于 Windows / CPU 路线。

### 10.3 为什么不能只看 SLING 总分

SLING-38K 更适合观察一组广泛语言学现象，但它不是所有下游任务的统一代理指标。

HCAM 的两个结构特征——**中文字符级 tokenization** 与 **多数 Local 层形成的局部归纳偏置**——天然更贴近字符恢复、局部搭配、近邻语法和规范用字判断。V1 已经在“的 / 地 / 得”下游实验中显示出很强的这一类能力；V2 的联合五字纠错和 FreshNatural 结果又进一步延续了这一趋势。

因此，HCAM V2 在 SLING 排名中仍低于部分经典中文 encoder，并不意味着它在所有中文任务上都更弱。对于字符级恢复和强局部依赖任务，HCAM 更有可能发挥自身结构偏置的优势。

---

## 11. 结论与后续方向

HCAM V2 将第一代扩大到 **121.08M 参数、3.5B prepared tokens、约 3.4 passes / 11.9B token exposures / 49.7K optimizer steps**。

最终结果中最明显的一点是：

> **SLING-38K：64.924% → 75.579%**

这说明 HCAM 的混合卷积-注意力路线在第二代更大数据和训练预算下仍然能够继续获得明显收益，并没有在 V1 水平附近提前停止提升。

同时，联合字符纠错与量化后的部署结果表明，V2 的提升不只体现在预训练基准上：在字符级恢复和局部搭配任务中，HCAM 的字符粒度建模与局部偏置仍具有很强的实用竞争力。

V2 仍然存在几个清晰的弱项：

- RelativeClause
- 部分 reflexive 形式规则
- 连续多字符联合恢复

因此未来如果继续做 V2.1，更有价值的方向是加入新的、高质量、去重中文语料，特别是复杂长句、关系从句、正式书面语、学术 / 法律 / 技术文本和更丰富的长尾实体，而不是简单重复当前 3.5B 数据。

V2 当前更适合作为 HCAM 的长期第二代基座。后续可以在保留现有架构与 tokenizer 的情况下继续预训练，也可以用于新的中文纠错、分类、检索等下游实验。

---

## 12. 发布状态

### HCAM V1
- 代码：公开
- 权重：公开
- tokenizer：公开
- 技术报告：公开

### HCAM V2
- 技术报告与结果：公开
- 权重：**暂不公开**
- 完整训练代码：**暂不公开**

仓库现有 `hcam.py`、`train_pretrain.py`、Release 权重与相关许可仍对应 HCAM V1。
