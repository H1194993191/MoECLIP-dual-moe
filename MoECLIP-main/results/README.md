# MoECLIP-dual-moe — 实验结果

本目录存放 **MoECLIP 双侧 MoE 修改版**的实验结果。代码改动详见仓库根目录的 [`DUAL_MOE_CHANGES.md`](../DUAL_MOE_CHANGES.md)。

## 代码说明

在 MoECLIP（CVPR 2026）视觉 MoE 的基础上，为文本编码器增加一套独立的 MoE，使文本与视觉两侧都能动态专家路由（第一版：两侧相互独立，不共享 Router/Expert 权重）。

- 论文：**MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection**（CVPR 2026）
- 基线代码：[CoCoRessa/MoECLIP](https://github.com/CoCoRessa/MoECLIP)

## 结果口径（重要）

- 训练 20 个 epoch，每 epoch 存一个 checkpoint，测试对全部 checkpoint 逐 epoch 输出结果。
- 本目录报告的是**各指标最佳 checkpoint**（四指标各自取 20 个 epoch 中的最大值，可能来自不同 epoch）。
- 论文与官方代码均未指定 checkpoint 选择口径；取"各指标最佳"遵循 ZSAD 领域通行做法，且该口径下与论文基准吻合度最高。

## 目录结构（按版本划分）

```
results/
├── README.md          # 本文件（含版本索引）
└── v1/                # 第一版结果：文本+视觉独立双侧 MoE
    ├── summary.csv    # 14 个数据集 × 4 指标 的最佳值汇总（含对应 epoch）
    ├── per_epoch/     # 每数据集逐 epoch 明细（JSON）
    └── logs/          # 原始 test 日志（完整证据链）
```

> 后续若修改模块并重跑实验，请新增 `v2/`、`v3/` … 目录（**不要覆盖旧版**），并在下方版本索引表登记；旧版本结果保留以便横向对比，git 历史作为额外保险。

## 版本索引

| 版本 | 代码改动 | 训练命令 | 结果状态 |
|---|---|---|---|
| v1 | 文本+视觉独立双侧 MoE（首版，见 DUAL_MOE_CHANGES.md） | `python train.py --save_path ckpt/dual_moe`（VisA 权重）+ `ckpt/dual_moe_mvtec`（MVTec 权重） | ✅ 14 数据集全跑完 |

## 数据说明

- **指标顺序**：Image AUROC / Image AP / Pixel AUROC / Pixel AP。
- **训练→测试方向**：默认 VisA 权重测试其余 13 集；`VisA` 一行用 MVTec-AD 权重测试（与论文一致）。
- **无效值**：`Head CT` 无像素标注 → Pixel 两列为空；结肠息肉五集（ColonDB/ClinicDB/CVC-300/Endo/Kvasir）为单类别、无正常样本 → Image 两列为 0，仅 Pixel 有效（与论文口径一致）。

## 结果摘要（v1，各指标最佳 checkpoint）

| 数据集 | Image AUROC | Image AP | Pixel AUROC | Pixel AP |
|---|---|---|---|---|
| MVTec-AD | 93.87 | 97.31 | 92.76 | 46.96 |
| VisA | 84.61 | 86.97 | 95.53 | 26.06 |
| BTAD | 93.23 | 97.83 | 97.18 | 50.15 |
| RSDD | 95.39 | 94.95 | 99.70 | 35.39 |
| DTD-Synthetic | 97.08 | 98.92 | 98.78 | 65.16 |
| Brain MRI | 88.73 | 97.21 | 97.75 | 65.65 |
| Head CT | 95.44 | 94.13 | — | — |
| Liver CT | 75.78 | 68.65 | 97.71 | 9.43 |
| Retina OCT | 87.00 | 87.23 | 96.64 | 69.96 |
| ColonDB | 0.00 | 0.00 | 85.27 | 37.12 |
| ClinicDB | 0.00 | 0.00 | 90.19 | 54.82 |
| CVC-300 | 0.00 | 0.00 | 97.72 | 69.06 |
| Endo | 0.00 | 0.00 | 90.86 | 62.31 |
| Kvasir | 0.00 | 0.00 | 88.45 | 58.81 |

> 权重文件（.pth checkpoint）体积较大，未纳入本仓库。如需复现，依上述训练命令自行训练即可。
