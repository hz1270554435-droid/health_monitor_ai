# Audio V3 Fixed Vector Board Report

日期：2026-05-12

## 结论

`audio_model_v3_board_htk_hardneg` 的固定特征向量板端 smoke test 通过。

这说明：

- v3 模型已经在 CM55 侧正确加载。
- 输入 shape `1x1x40x101` 和模型期望一致。
- class order `non_cough,cough` 没有明显反转。
- softmax / cough probability 计算与 PC 期望一致。
- 固定 feature 直喂推理路径没有明显 layout、cache、alignment 或导出数值问题。

这一步只验证“固定 feature -> CM55 模型推理”，还没有验证实时 MIC / PDM / CM33 audio frontend。

## 输入日志摘要

模型信息：

```text
model_name=audio_model_v3_board_htk_hardneg
model_version=v3_board_htk_hardneg
input_shape=1x1x40x101
class_order=non_cough,cough
threshold=0.750000
frontend_name=board_htk_no_norm_v1
```

固定向量结果：

| sample | board cough_prob | expected_prob | abs diff | infer_ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.999834 | 0.999834 | 0.000000 | 185 |
| 2 | 0.999348 | 0.999348 | 0.000000 | 188 |
| 3 | 0.000020 | 0.000020 | 0.000000 | 188 |
| 4 | 0.000022 | 0.000022 | 0.000000 | 192 |
| 5 | 0.995179 | 0.995179 | 0.000000 | 188 |
| 6 | 0.995179 | 0.995179 | 0.000000 | 184 |
| 7 | 0.993598 | 0.993598 | 0.000000 | 183 |
| 8 | 0.993598 | 0.993598 | 0.000000 | 176 |

推理时间：

- min: 176 ms
- max: 192 ms
- average: 185.5 ms

当前 hop 是 0.5 s，所以单次模型推理时间低于 hop interval。后续 live MIC 仍需同时记录 preprocess time，确认 `preprocess + inference` 总时间低于实时预算。

## Pass / Fail

结果：`PASS`

通过依据：

- 所有 8 个 fixed vector 的 `diff=0.000000`。
- 高 cough probability 样本保持接近 1。
- 低 cough probability 样本保持接近 0。
- 没有出现 class index reversal。
- 没有出现 NCHW/NHWC 明显错位。
- 没有出现 logits 当 probability 使用的迹象。

## 需要注意

样本 5/6、7/8 的输出完全重复，可能是测试包里存在重复向量或重复输入。这不影响本次 smoke pass，但后续建议确认 fixed vector pack 是否覆盖了四类样本：

- cough
- clean_non_cough
- model_false_alarm_hard_negative
- human_cough_like_hard_negative

如果 fixed vector 包覆盖不足，后续应补充更多非重复样本，但不需要因此阻塞 live MIC smoke。

## 当前约束

本次结果不代表最终泛化可靠。

仍然不能做以下结论：

- 不能说明实时 MIC 输入已经正确。
- 不能说明 CM33 `board_htk_no_norm_v1` 前处理已经和 PC 一致。
- 不能说明 PDM gain、PCM scaling、windowing、drop handling 没问题。
- 不能说明 v3 已经达到最终部署质量。

## 下一步

可以进入 live MIC board test。

推荐顺序：

1. 先跑 quiet background，确认静音/环境声不会持续高 cough probability。
2. 跑 intentional cough near board。
3. 跑 intentional cough at normal use distance。
4. 跑 speech、throat clearing、laugh、knock/impact、keyboard/table noise、fan/room noise。
5. 同时测试 threshold `0.50 / 0.75 / 0.90 / 0.95`。

live log 必须记录：

```text
timestamp
feature_seq
infer_seq
cough_prob
non_cough_prob
threshold
decision
dropped_count
preprocess_time
inference_time
scene_label
```

如果 live cough 仍大多是 `non_cough`，不要回头调模型或训练，先 dump live PCM 和 board features，对比 PC `board_htk_no_norm_v1` 前处理。

## 当前 Deployment Gate

Case A fixed vector fail 已排除。

现在进入：

- Case B：fixed vector pass，但 live MIC cough 仍 mostly non_cough，则查 CM33 frontend / PCM scaling / MIC signal / windowing / data domain。
- Case C：fixed vector pass，live cough probability 上升但误报高，则收集 live false positives 做 v3.1 hard negatives。
- Case D：fixed vector pass，live cough 在 threshold `0.75` 或 `0.90` 可接受，则 v3 可以替换 v2 作为 demo firmware candidate。

