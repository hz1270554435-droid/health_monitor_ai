# Health Monitor AI Workspace Context Summary

日期：2026-05-12

## 项目目标

本项目是一个边缘 AI 呼吸健康监测原型，当前主线仍然是：

1. 先做好 MIC 音频 `cough / non_cough` baseline。
2. 再做 LD6002 雷达特征 baseline，输入是解析后的 radar CSV，不直接用 UART raw bytes。
3. 再做音频 + 雷达融合，第一版优先规则状态机。
4. OPERA / teacher-student 蒸馏暂不进入，必须等 MIC baseline 和板端链路稳定后再做。

当前最重要的任务是让 MIC cough 模型在板端真实音频链路上可靠工作。

## 项目硬规则

- 不修改或删除 `data/raw/`。
- 不覆盖 `data/labels/audio_labels_v2.csv`。
- 不手动编辑已经生成的正式标签文件，必须用脚本生成。
- 不提交 raw audio、processed features、trained models、results。
- train / val / test 必须按 `person_id` 或稳定 source identity 分组，避免泄漏。
- YAMNet 只用于 hard sample mining 预筛，不直接改正式标签，不直接训练。
- 当前不进入 OPERA 蒸馏。

## v2 当前状态与板端问题

当前板端实际部署的是 v2 模型：

```text
models/audio/selected/audio_baseline_v2_best_model.pt
models/audio/export/audio_baseline_v2.onnx
models/audio/export/audio_baseline_v2_int8.onnx
threshold = 0.75
```

v2 训练配置：

```text
configs/audio_formal_50_v2.yaml
```

v2 训练时的前处理没有显式设置 `htk: true` 或 `normalize: none`。根据 `src/audio/features.py` 默认行为，v2 实际训练前处理是：

- 16 kHz mono
- 1.0 s window
- 0.5 s hop
- 40-bin Log-Mel
- Slaney Mel
- `power_to_db`
- `per_window_zscore`

但当前板端 / v3 路线使用的是：

```text
board_htk_no_norm_v1
```

即：

- HTK Mel
- no normalization

这和 v2 训练前处理不一致。

## v2 板端漏检原因分析

用户在线下测试中发现：

- 主动咳嗽时，v2 很少报 `cough`。
- 大部分主动咳嗽报 `non_cough`。
- 仅有的两次 `cough` 中还有一次是误报。

已做离线对照实验：用同一个 v2 checkpoint 和同一批 v2 test 窗口，只替换前处理。

结果：

| 前处理 | recall | false negative | 结论 |
| --- | ---: | ---: | --- |
| v2 正确前处理 Slaney + zscore | 0.939 | 2 / 33 | 离线正常 |
| HTK + zscore | 0.939 | 2 / 33 | 还能识别，但误报增加 |
| Slaney + no_norm | 0.000 | 33 / 33 | cough 全漏 |
| HTK + no_norm board | 0.000 | 33 / 33 | cough 全漏 |

结论：

v2 的板端问题最高概率不是模型权重坏了，而是板端前处理和 v2 训练前处理不一致。尤其是缺少 v2 训练时的 per-window z-score，会让 v2 的 cough probability 基本塌到接近 0。

v2 导出本身看起来没坏：

- PyTorch -> ONNX max abs diff 约 `1.9e-6`
- int8 ONNX 离线 test recall 没掉

因此当前优先排查顺序是：

1. 固定 feature vector 直接喂板端模型，确认模型导入、NCHW/NHWC、softmax、class index、threshold 是否正确。
2. 固定 PCM + 板端 Log-Mel，和 PC v2 前处理逐元素比较。
3. 实时 PDM capture，保存 16 kHz PCM，拿回 PC 离线跑 v2/v3 评分。

## v2 数据和模型限制

v2 标签文件：

```text
data/labels/audio_labels_v2.csv
```

规模：

- label rows: 58
- cough clips: 26
- non_cough clips: 32
- processed windows: 868
- cough windows: 369
- non_cough windows: 499

v2 是小数据 PC 前处理模型，不是 board-compatible frontend 模型。即使 PC test 指标高，也不能直接说明板端实时 PDM 链路可靠。

## YAMNet Hard Mining 已完成

目标：用 YAMNet + v2 baseline 扫描大量 non_cough 数据，挖：

1. model false alarm hard negative
2. human cough-like hard negative
3. clean non_cough
4. uncertain
5. pseudo cough candidate

关键脚本：

```text
scripts/yamnet_full_non_cough_scan.py
scripts/create_yamnet_review_plan.py
scripts/export_review_plan_clips.py
scripts/build_v3_candidate_from_manual_reviews.py
```

完整扫描过：

```text
D:\cough_model_train\DATA\non_cough
```

全量扫描结果：

```text
data/mining_candidates/yamnet_non_cough_full_resumable/candidate_pool.csv
```

扫描统计：

- total_files: 91400
- processed_files: 91400
- candidate_rows: 1261602
- error_rows: 0

decision counts:

```text
hard_negative_review          9352
baseline_suspicious_review  251442
pseudo_cough_candidate          36
uncertain_review             85314
pseudo_non_cough_candidate  432804
exclude                     147034
unselected                  335620
```

人工复听导出曾生成：

```text
data/human_review/第一次人工重听/
data/human_review/第二次人工重听/
review/yamnet_full_non_cough_merged/
```

人工复核后合并生成了 v3 candidate 标签。

## v3 标签生成与固化

v3 candidate 生成脚本：

```text
scripts/build_v3_candidate_from_manual_reviews.py
```

输入：

```text
data/human_review/第一次人工重听/review_sheet.csv
data/human_review/第二次人工重听/review_sheet.csv
data/labels/audio_labels_v2.csv
```

输出：

```text
review/yamnet_full_non_cough_merged/manual_review_merged.csv
data/labels/audio_labels_v3_candidate.csv
docs/experiments/audio_labels_v3_candidate_from_yamnet_review.md
```

v3 candidate 统计：

- base rows: 58
- reviewed rows: 2050
- added rows: 2014
- output rows: 2072
- invalid manual_decision: 0
- manual_conflict: 0

新增样本全部来自人工复核，不自动加入未复核伪标签。

固化脚本：

```text
scripts/finalize_audio_labels_v3.py
```

正式标签：

```text
data/labels/audio_labels_v3.csv
```

固化报告：

```text
docs/experiments/audio_labels_v3_finalize_report.md
results/check_audio_labels_v3/report.md
results/check_audio_labels_v3/report.json
```

固化 cleanup：

- `sample_weight` 空值填 `1.0`: 58 行
- hard negative 且 `hard_negative_source` 为空，填 `legacy_hard_negative`: 2 行
- 不改变 `audio_file/start_time/end_time/label/split/person_id`
- 不删除 candidate 中有效行

正式 v3 标签统计：

- rows: 2072
- cough: 26
- non_cough: 2046
- split: train 2050, val 8, test 14
- hard_negative: 1628
- audio missing: 0
- duplicate audio segment: 0
- person_id split leakage: 0
- sample_weight missing: 0
- sample_weight non-positive: 0
- hard_negative empty source rows: 0

hard negative source:

```text
model_false_alarm          1530
human_cough_like             96
legacy_hard_negative          2
boundary_clean_non_cough    296
clean_non_cough              92
```

注意：空 `hard_negative_source` 仍会出现在非 hard-negative 的 base rows 中，这是正常的。

## v3 Smoke 验收

对 `audio_labels_v3.csv` 已重新做训练前检查：

```text
results/pretrain_check_audio_labels_v3/
results/pretrain_preprocess_smoke_v3/
results/pretrain_one_epoch_smoke_v3/
```

三段均通过：

1. dataset check passed
2. `board_htk_no_norm_v1` preprocess smoke passed
3. one-epoch training smoke passed

preprocess smoke 覆盖：

- cough
- clean_non_cough
- model_false_alarm_hard_negative
- human_cough_like_hard_negative

## v3 Board-Compatible 模型训练

当前已训练一个 board-compatible v3 candidate：

```text
audio_baseline_v3_board_htk_hardneg
```

配置：

```text
configs/audio_baseline_v3_board_htk_hardneg.yaml
```

关键设置：

- frontend: `board_htk_no_norm_v1`
- model: `ds_cnn`
- class_weight: `auto`
- sampling mode: `balanced`
- hard_negative_oversample: true
- hard_negative_multiplier: 2.0
- use_sample_weight: true
- no OPERA / no distillation

前处理输出：

```text
data/processed/audio_features_v3_board_htk_hardneg/audio_manifest.csv
```

processed feature windows:

- total: 2882
- train: 2606
- val: 148
- test: 128

训练输出：

```text
results/audio/audio_baseline_v3_board_htk_hardneg/
models/audio/audio_baseline_v3_board_htk_hardneg/best_model.pt
```

训练结果：

- best_epoch: 12
- epochs_run: 17
- early_stop_triggered: true
- test at argmax / 0.50:
  - accuracy: 0.6875
  - precision: 0.4521
  - recall: 1.0000
  - f1: 0.6226

threshold sweep:

```text
results/audio/audio_baseline_v3_board_htk_hardneg/threshold_sweep.csv
```

当前 recall constraint 下推荐：

- threshold: 0.75
- precision: 0.5455
- recall: 0.9091
- f1: 0.6818
- false_positive windows: 25
- false_negative windows: 3

重要解释：

当前 v3 训练结果只能作为 final candidate / board candidate 选择依据，不能宣称最终泛化可靠。原因是 val/test 样本量仍小，且新增 hard negatives 主要进入 train。

## v3 报告产物

```text
results/audio/audio_baseline_v3_board_htk_hardneg/metrics.json
results/audio/audio_baseline_v3_board_htk_hardneg/classification_report.txt
results/audio/audio_baseline_v3_board_htk_hardneg/confusion_matrix.png
results/audio/audio_baseline_v3_board_htk_hardneg/threshold_sweep.csv
results/audio/audio_baseline_v3_board_htk_hardneg/experiment_report.md
results/audio/audio_baseline_v3_board_htk_hardneg/board_test_plan.md
results/audio/audio_baseline_v3_board_htk_hardneg/eval_data_gap_report.md
```

报告生成脚本：

```text
scripts/write_audio_v3_training_reports.py
```

## 当前技术判断

1. v2 在线下板端漏检，最高概率是前处理不一致。
2. 不建议继续用 v2 + board HTK/no_norm 前端直接部署。
3. 如果坚持 v2，板端必须严格复现 v2 的 Slaney + per-window zscore 前处理。
4. 更合理路线是继续推进 v3，因为 v3 是按 `board_htk_no_norm_v1` 训练的。
5. v3 还不能直接视为最终模型，需要板端固定向量验证和独立 val/test 补充。

## 下一步优先级

### 1. 板端固定 feature vector 验证

用 `deploy/test_vectors/audio/` 或重新为 v3 导出固定 feature vectors。

目标：

- 直接喂 `[1, 1, 40, 101]` feature 到板端模型。
- 验证模型导入、输入 layout、softmax、class index、threshold。

如果这一步失败，先查：

- NCHW / NHWC
- class index 是否反了
- 是否把 logits 当 probability
- threshold 是否用对
- 模型导入是否改变输入/输出

### 2. 固定 PCM + 板端前处理验证

同一段 16 kHz PCM：

1. PC 端生成 `board_htk_no_norm_v1` feature。
2. 板端生成 feature。
3. 比较 shape、mean/std/min/max、逐元素误差。

这一步是验证 board frontend 的关键。

### 3. 实时 PDM capture 验证

保存板端实时 PDM 转出的 16 kHz PCM，拿回 PC 跑：

- v2 正确前处理 + v2 模型
- v3 board 前处理 + v3 模型

用来区分：

- 板端模型/前处理问题
- PDM capture / gain / clipping / DC offset 问题
- 数据域问题

### 4. 补充独立 val/test

当前 val/test 不够支撑最终结论。

优先补：

- 主动咳嗽：近/远、强/弱、干咳/湿咳、连续咳、带背景噪声咳
- clean non_cough：安静、说话、音乐、环境噪声
- hard negative：大叫、音乐突变、拍打/敲击、麦克风摩擦、打嗝、清嗓、喷嚏、呼吸/鼾声
- 低质量输入：静音、无信号沙沙声、低 RMS、爆音

### 5. Active Learning 降低复听成本

下一轮不要盲听全部数据。建议：

1. 用 v3 扫更大的未标注池。
2. 优先选 threshold 附近样本和高置信 false alarm。
3. 按 source_audio 去重，避免反复听相邻窗口。
4. 每轮只人工复听 top hard negative、uncertain 和少量 random clean control。
5. 人工确认后再进入训练，独立 val/test 必须 source-disjoint。

## 当前不要做的事

- 不要宣称 v3 泛化可靠。
- 不要直接把 v3 部署成最终模型。
- 不要进入 OPERA 蒸馏。
- 不要继续用 v2 指标解释板端表现，除非先证明板端前处理完全复现 v2 PC 前处理。
- 不要把 processed features、trained models、results 提交到仓库。

