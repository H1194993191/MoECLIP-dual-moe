# 双侧 MoE 改造说明

## 1. 修改目标

本次修改在原有 MoECLIP 视觉 MoE 的基础上，为文本编码器增加一套独立的 MoE，使文本和视觉两侧都能进行动态专家路由。

第一版采用“文本、视觉相互独立”的设计：两侧不共享 Router 或 Expert 权重，分别适配各自的特征维度与语义分布。默认配置为：

| 配置项 | 视觉侧 | 文本侧 |
| --- | --- | --- |
| MoE 插入层（零基索引） | `[5, 11, 17, 23]` | `[8, 11]` |
| Transformer 宽度 | 1024 | 768 |
| 专家数 | 4 | 4 |
| Top-k | 2 | 2 |
| LoRA rank | 8 | 8 |
| LoRA alpha | 16 | 16 |
| 残差融合权重 | 0.1 | 0.1 |

文本 MoE 默认只放在后两层，以减少对原始文本语义的扰动、降低过拟合风险，并控制训练显存开销。

## 2. 模型结构修改

涉及文件：`model/moe_adapter.py`

### 2.1 独立文本 MoE

`MoECLIP` 新增以下构造参数：

- `text_moe_layers`：文本 MoE 插入层，默认 `[8, 11]`；传空列表可关闭文本 MoE。
- `text_adapt_weight`：文本 MoE 残差融合权重，默认 `0.1`。
- `text_moe_r`：文本专家 LoRA rank。
- `text_moe_lora_alpha`：文本专家 LoRA alpha。
- `text_moe_num_experts`：文本专家数量。
- `text_moe_top_k`：每个有效文本 token 选择的专家数量。

文本侧通过独立的 `MixLoraConfig` 和 `BaseIndependentMoE` 创建 Router 与 LoRA Experts。文本维度从 `clipmodel.transformer.width` 获取，不再依赖视觉侧的 1024 维设置。

原来的 `text_adapter` 仅包含一个 `SimpleProj`，现在改为 `ModuleDict`，包含：

- `proj`：原有文本投影层；
- `moe_adapters`：新增的文本 MoE 模块列表。

因此文本 MoE 会被现有优化器收集和 checkpoint 保存流程覆盖。

### 2.2 文本前向传播

`encode_text()` 在每个指定的文本 Transformer block 后执行：

1. 将当前 token 表征送入对应的文本 MoE；
2. 对 MoE 输出进行范数缩放，使其与原始 token 表征的范数匹配；
3. 按 `text_adapt_weight` 进行残差融合；
4. 继续后续 Transformer block；
5. 最后仍使用原有的 `ln_final`、EOT 表征提取和文本投影流程。

`encode_text()` 新增可选参数 `return_aux`。训练时设为 `True`，同时返回文本特征、负载均衡损失和 ETF 损失；默认仍为仅返回文本特征，因此原有评测调用不需要修改。

### 2.3 Padding token 屏蔽

`BaseIndependentMoE.forward()` 新增可选参数 `token_mask`。文本侧使用 `text != 0` 构造有效 token 掩码，并在转换成 LND 布局后传入 MoE。

掩码生效后：

- Router 只处理有效 token；
- Expert 只计算有效 token；
- 负载均衡损失只统计有效 token；
- ETF loss 只使用有效 token 的专家输出；
- MoE 输出通过索引写回原始 token 位置；
- padding 位置不执行 MoE 残差缩放，保持原特征不变。

视觉侧不传 `token_mask`，保留原有全部视觉 token 参与路由的行为。

### 2.4 文本辅助损失归一化

文本侧负载均衡损失和 ETF 损失先按文本 MoE 层数求平均，避免损失尺度随插入层数量线性增长。

## 3. 文本提示与辅助损失传递

涉及文件：`forward_utils.py`

`get_adapted_single_class_text_embedding()` 新增可选参数 `return_aux`。

训练时，该函数分别获取正常提示和异常提示的文本 MoE 辅助损失，并按文本编码调用次数求平均。默认 `return_aux=False` 时，返回值的结构与原来一致，不影响测试及其他文本特征构造函数。

没有使用模型属性隐式保存损失，避免不同类别、不同提示调用之间残留计算图或错误累计。

## 4. 训练流程修改

涉及文件：`train.py`

### 4.1 总训练损失

新的训练目标为：

\[
L = L_{det} + L_{seg}
+ \lambda_v^{lb} L_v^{lb}
+ \lambda_t^{lb} L_t^{lb}
+ \lambda_v^{etf} L_v^{etf}
+ \lambda_t^{etf} L_t^{etf}
\]

其中：

- 视觉侧沿用 `--balance_loss_lambda` 和 `--etf_loss_lambda`；
- 文本侧使用独立的 `--text_balance_loss_lambda` 和 `--text_etf_loss_lambda`；
- 文本辅助损失还会按当前 batch 中不同类别的数量求平均，避免类别数量改变损失尺度。

### 4.2 FOFS 参数冻结

原训练流程会重新开放整个 `text_adapter` 的梯度。新增文本 MoE 后，如果继续使用 FOFS，这会把本应固定的文本 `lora_A` 意外设为可训练。

现在文本参数会逐项检查：启用 FOFS 时，文本和视觉侧的 `lora_A` 都保持冻结，优化器只收集实际需要训练的参数。

### 4.3 新增训练参数

```text
--text_moe_layers 8,11
--text_adapt_weight 0.1
--text_moe_r 8
--text_moe_lora_alpha 16
--text_moe_num_experts 4
--text_moe_top_k 2
--text_balance_loss_lambda 0.01
--text_etf_loss_lambda 0.01
```

直接运行训练脚本时默认启用双侧 MoE。例如：

```bash
python train.py --save_path ckpt/dual_moe
```

进行仅视觉 MoE 消融时，可将 `--text_moe_layers` 设为空字符串。

## 5. 测试与 checkpoint

涉及文件：`test.py`

测试脚本同步增加了文本 MoE 的结构参数和残差融合权重。训练与测试必须使用一致的文本插入层、专家数、rank 和 Top-k，保证 checkpoint 参数能够严格加载。

例如：

```bash
python test.py --save_path ckpt/dual_moe
```

注意：`text_adapter` 已从 `ModuleList` 改为包含文本投影和文本 MoE 的 `ModuleDict`，其 state dict key 与旧版本不同。旧视觉单侧 checkpoint 不能直接严格加载到新的默认双侧模型中，建议使用新的保存目录重新训练。

## 6. 验证情况

已完成：

- `model/moe_adapter.py`、`forward_utils.py`、`train.py` 和 `test.py` 的 Python `py_compile` 语法检查；
- 四个修改文件的 IDE 静态诊断检查，无错误或警告；
- Git 空白错误检查。

尚未在当前命令行环境执行完整模型前向、训练或测试：当前可用 Python 环境未安装 PyTorch，完整运行还需要 CLIP 预训练权重和实际数据集。

## 7. 建议的后续消融实验

1. 原始 MoECLIP：仅视觉 MoE；
2. 仅文本 MoE，关闭视觉 MoE；
3. 文本、视觉独立双侧 MoE；
4. 双侧 MoE 加跨模态路由一致性约束；
5. 与参数量接近的文本普通 LoRA/Adapter 对照，验证收益是否来自动态路由而非单纯增加参数。

建议记录每组的 image/pixel AUROC、AP、可训练参数量、显存占用和专家利用率，并使用多个随机种子验证稳定性。
