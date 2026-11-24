# MoECLIP 改动说明（双侧 MoE → 视觉引导文本路由）

> 本文档记录基于 **MoECLIP（CVPR 2026）** 的**两次修改**，以及每次修改背后的研究动机。
> 基线代码：[CoCoRessa/MoECLIP](https://github.com/CoCoRessa/MoECLIP)。
> 相关文献：`MoECLIP`（CVPR 2026，视觉侧专家化）、`PromptMoE`（AAAI 2026，文本侧提示组合化）、`HiPL`（AAAI 2026，文本侧高阶提示学习）。

| 修改 | 名称 | 核心内容 | 状态 |
|---|---|---|---|
| 第一次 | 双侧 MoE（v1） | 在视觉 MoE 基础上，为文本编码器增加独立 MoE | 已完成并跑出结果 |
| 第二次 | 视觉引导的文本路由（v2） | 用全局视觉上下文引导文本 MoE 的专家路由，实现跨模态协同 | 本次修改（当前默认） |

通过 `--no_visual_router` 开关可回退到 v1（独立双侧 MoE），便于做消融对比。

---

## 一、研究背景与总体动机

零样本异常检测（ZSAD）的核心矛盾是：**既要让模型专门化于异常检测，又要保留 CLIP 的泛化能力**。三篇代表性工作围绕"单一/一刀切表征存在瓶颈"给出了两条互补但**彼此割裂**的技术路径：

1. **视觉侧专家化（MoECLIP）**：在 CLIP 视觉编码器多层引入 MoE，按 patch 特性把每个图像 patch 动态路由到专门的 LoRA 专家。但它的**文本侧仍是单一简单投影**，未做专家化。
2. **文本侧组合化（PromptMoE / HiPL）**：在文本提示侧维护一池提示专家，按视觉输入动态组合。但它们的**视觉侧是冻结的**，未做专家化。

**关键空白**：视觉侧专家化与文本侧组合化从未被同时打通。本工作的两次修改正是沿着"先打通两侧、再让两侧协同"的路线推进：

- 第一次修改回答"**能不能两侧都做专家化**"——把文本侧也升级为 MoE；
- 第二次修改回答"**两侧如何不孤立、而是协同**"——用视觉上下文引导文本路由。

---

## 二、第一次修改：双侧 MoE（v1）

### 2.1 动机：为什么要搞双侧

1. **文献空白**：MoECLIP 只动视觉、PromptMoE/HiPL 只动文本，两侧专家化从未结合。
2. **文本侧同样存在"单一表征瓶颈"**：PromptMoE 指出，固定 prompt 向量无法刻画未见类别正常/异常模式的多样性。MoECLIP 虽然把视觉 patch 做了差异化适配，但文本侧仍是单一投影，无法按语义动态调整。
3. **对称直觉**：如果 patch 级专家化对视觉有效（MoECLIP 已验证），那么 token 级专家化对文本也应有益——不同类别、不同异常语义需要不同的文本表征。
4. **可行性**：MoECLIP 的 `BaseIndependentMoE`（LoRA 专家 + 路由 + FOFS + ETF）只吃 `(B, L, d_model)` 的 hidden states，与模态无关，可镜像复用到文本侧；且它是三篇中唯一开源的，改造风险最低。

**结论**：把文本侧从单一投影升级为独立 MoE，形成"视觉 patch 级 + 文本 token 级"的对称双侧专家特化，是一个空白明确、风险可控的切入点。

### 2.2 结构改动

涉及文件：`model/moe_adapter.py`

- 新增构造参数：`text_moe_layers`（默认 `[8, 11]`）、`text_adapt_weight`（默认 0.1）、`text_moe_r`、`text_moe_lora_alpha`、`text_moe_num_experts`、`text_moe_top_k`。
- 文本侧通过独立的 `MixLoraConfig` 与 `BaseIndependentMoE` 创建 Router 与 LoRA 专家；文本维度取自 `clipmodel.transformer.width`（768），不再沿用视觉侧 1024。
- `text_adapter` 由单个 `SimpleProj` 改为 `ModuleDict`，包含 `proj`（原文本投影）与 `moe_adapters`（文本 MoE）。

默认配置：

| 配置项 | 视觉侧 | 文本侧 |
|---|---|---|
| MoE 插入层（零基索引） | `[5, 11, 17, 23]` | `[8, 11]` |
| Transformer 宽度 | 1024 | 768 |
| 专家数 / Top-k | 4 / 2 | 4 / 2 |
| LoRA rank / alpha | 8 / 16 | 8 / 16 |
| 残差融合权重 | 0.1 | 0.1 |
| 防冗余机制 | FOFS + ETF | FOFS + ETF |

文本 MoE 只放在后两层，以减少对原始文本语义的扰动、降低过拟合风险并控制显存。

### 2.3 文本前向与 padding 屏蔽

- `encode_text()` 在指定文本 block 后执行：token 表征送入文本 MoE → 输出范数缩放到与原表征匹配 → 按 `text_adapt_weight` 残差融合 → 继续后续 block → 最后仍用原 `ln_final`、EOT 提取与文本投影。
- `encode_text()` 新增 `return_aux`，训练时返回文本特征、负载均衡损失、ETF 损失。
- `BaseIndependentMoE.forward()` 新增 `token_mask`：文本侧用 `text != 0` 构造有效 token 掩码，Router/Expert/负载均衡/ETF 只统计有效 token，padding 位置保持原特征。视觉侧不传掩码。

### 2.4 训练流程（v1）

涉及文件：`train.py`、`forward_utils.py`

- 总损失：
  \[
  L = L_{det} + L_{seg} + \lambda_v^{lb}L_v^{lb} + \lambda_t^{lb}L_t^{lb} + \lambda_v^{etf}L_v^{etf} + \lambda_t^{etf}L_t^{etf}
  \]
  文本侧用独立的 `--text_balance_loss_lambda` / `--text_etf_loss_lambda`（默认均 0.01），并按文本 MoE 层数、类别数、编码调用次数归一化。
- **FOFS 参数冻结修正**：启用 FOFS 时，文本与视觉两侧的 `lora_A` 都保持冻结，避免文本 `lora_A` 被误设为可训练。

### 2.5 v1 实验结果（证据）

14 个数据集、训练 20 epoch、逐 epoch 评测。相对原论文基准（各指标最佳 checkpoint 口径）：

| 指标 | 原始 MoECLIP | 双侧 MoE (v1) | Δ |
|---|---|---|---|
| Image AUROC | 89.6 | 90.1 | +0.5 |
| Image AP | 90.6 | 91.5 | +0.9 |
| Pixel AUROC | 94.3 | 94.5 | +0.2 |
| Pixel AP | 47.5 | 50.1 | **+2.6** |

四项指标全面提升，Pixel AP 提升最明显，**验证了"文本侧也做专家化"这一方向有效**。详细结果见 `results/v1/`。

---

## 三、第二次修改：视觉引导的文本路由（v2，本次）

### 3.1 动机：为什么要搞视觉引导文本路由

1. **v1 的局限**：v1 的双侧是**完全独立**的——文本 Router 只看文本 token，视觉 Router 只看图像 patch，两侧各算各的、没有任何跨模态交互。这使创新叙事停留在"给文本也加了一个 MoE"，容易被质疑为"两个独立模块的简单叠加"。
2. **ZSAD 的本质是视觉-文本对齐**：异常分数来自图像 patch 与正常/异常文本 prompt 的相似度对比。既然要按图对齐，文本 prompt 就不应"一刀切"，而应**"按图定制"**——看到不同的图，需要不同的文本表征来对齐。
3. **该思想已被验证有效，但只在文本侧**：PromptMoE 的 VGMoP 用图像门控信号路由文本提示专家池、HiPL 用跨模态门控聚合专家提示，都证明了"视觉引导文本"的价值；但它们视觉侧冻结、无 MoE。v1 的文本路由不看图，恰恰缺失了这一能力。
4. **因此**：把全局视觉上下文注入文本 MoE 的 Router，让"看到什么图"决定"文本怎么路由"，把两侧从"独立"耦合为"协同"。

**差异化定位**：MoECLIP 文本无 MoE、PromptMoE/HiPL 视觉无 MoE；本工作是"**双侧都有 MoE，且路由跨模态耦合**"，是三篇文献都未覆盖的空白。

### 3.2 核心设计

- **视觉上下文来源**：直接复用视觉前向已计算的 `det_token`（768 维全局图像表征），几乎零额外开销。
- **注入位置**：文本 MoE 的 **Router 输入**。新增 `visual_router_proj`（`Linear(768 → 768)`）把视觉上下文投影到文本维度，作为路由偏置 broadcast 加到每个有效 token 的路由输入上。
- **只影响路由、不影响专家计算**：专家仍只处理文本 token，视觉上下文仅改变专家选择，设计干净、语义明确（"视觉引导路由"而非"视觉改特征"）。
- **零初始化 `visual_router_proj`**：初始时路由偏置为 0，模型等价于 v1，训练过程中平滑地学习视觉引导，避免一开始扰乱文本路由。
- **`visual_context`  detach**：视觉上下文仅作为条件输入，不回传梯度到视觉分支，保证训练稳定（视觉表征由自身 det/seg 损失优化）。

### 3.3 结构改动（涉及文件）

- `model/moe_adapter.py`
  - `BaseIndependentMoE.forward()` 新增可选参数 `router_context`：非空时将其作为路由偏置加到路由输入。
  - `MoECLIP.encode_text()` 新增可选参数 `visual_context`：非空时经 `visual_router_proj` 投影后作为 `router_context` 传入文本 MoE。
  - `text_adapter` 新增 `visual_router_proj`（`nn.Linear(768, text_d_model)`），并零初始化。
- `forward_utils.py`
  - `get_adapted_single_class_text_embedding()` 新增 `visual_context` 参数，透传给 `encode_text()`。
- `train.py` / `test.py`
  - 新增 `--no_visual_router` 开关（默认开启视觉引导）；
  - 数据流调整为"先视觉前向、再文本编码"；
  - `text_adapter` 的 `load_state_dict` 改为 `strict=False`，兼容 v1 checkpoint。

### 3.4 训练/测试数据流调整

- **v1**：文本按类预算、与图像无关，可缓存复用。
- **v2**：先 `model(image)` 视觉前向得到 `det_feature`，再**逐样本**用 `det_feature[b]` 作为视觉上下文编码该样本类别的文本，得到随图像变化的文本特征 `(B, 768, 2)`，下游分类/分割计算保持不变。

### 3.5 新增超参与命令

```text
--no_visual_router    # 关闭视觉引导，回退到 v1 独立双侧 MoE（默认开启）
```

```bash
# v2 训练（默认视觉引导）
python train.py --save_path ckpt/dual_moe_v2
# v2 测试
python test.py --save_path ckpt/dual_moe_v2

# v1 消融（关闭视觉引导，得到公平对照）
python train.py --save_path ckpt/dual_moe_v1 --no_visual_router
python test.py  --save_path ckpt/dual_moe_v1 --no_visual_router
```

### 3.6 设计权衡（需在论文中说明）

- **效率代价**：v2 文本特征随图像变化、无法按类缓存，测试时需逐样本文本编码，开销高于 v1。由于文本编码器仅 12 层、MoE 只在后两层，该代价可接受——这也是 PromptMoE 等 per-image 提示方法的共性代价。
- **checkpoint 兼容**：v2 新增 `visual_router_proj`，建议使用新的 `save_path` 重新训练；加载 v1 checkpoint 时该层保持零初始化（`strict=False`），等价于从 v1 平滑出发。

### 3.7 与 v1 的关系

v2 是在 v1 地基上的**增量创新**：v1 提供"双侧都有 MoE"的结构，v2 在其上叠加"跨模态路由协同"。二者通过 `--no_visual_router` 切换，构成 `v1（独立双侧）→ v2（协同双侧）` 的核心消融对比。

---

## 四、验证情况

- `model/moe_adapter.py`、`forward_utils.py`、`train.py`、`test.py` 均通过 Python `py_compile` 语法检查；
- 四个修改文件的 IDE 静态诊断无错误或警告。

**尚未执行完整模型前向/训练/测试**：当前命令行环境未安装 PyTorch，完整运行需 CLIP 预训练权重与实际数据集。请在服务器上先用小数据集（如 MVTec + VisA）验证 v2 训练稳定、结果合理，再跑全量 14 数据集。

---

## 五、后续消融计划

| 阶段 | 实验 | 目的 |
|---|---|---|
| 1 | 仅视觉 MoE（`--text_moe_layers ""`） | 复现原始 MoECLIP，公平 baseline |
| 2 | 仅文本 MoE（关视觉 MoE） | 单独验证文本侧 MoE |
| 3 | v1 独立双侧 MoE（`--no_visual_router`） | 第一次修改，地基 |
| 4 | **v2 视觉引导双侧 MoE（默认）** | 第二次修改，核心创新 |
| 5 | 进阶：路由一致性约束、正常/异常状态感知双专家池 | 后续创新点 |

建议每组记录 image/pixel AUROC、AP、可训练参数量、显存占用、专家利用率，并用多个随机种子验证稳定性；结果按 `results/v1/`、`results/v2/`… 分目录存放，不覆盖旧版。
