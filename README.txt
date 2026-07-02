请修改 C:\Users\tufi\Desktop\代码\师兄 中的训练代码，重点解决重叠峰区域的通道误分配问题。不要只修改 predict 阈值，也不要改模型结构，先从 dataset、loss、metrics/checkpoint 选择逻辑上处理。

背景：
当前面积主导训练后，sample1 中 Glutamine / Glutamate 定量已经比较接近，但 Proline 仍然被错误预测为 47.65 mM。sample1 实际只有 Glutamine 和 Glutamate，Proline 标准为 0。
另外 sample6 中 Leucine 也出现误判，sample6 实际有 Asparagine、Glutamine、Isoleucine、Valine，但 Leucine 标准为空 / 不存在，却被预测为 22.82 mM。
查看 CSV 可知：
- sample1 中 Proline 的 Channel_Calculated_Conc_mM = 47.65，但 NNLS Raw 只有约 6.27，说明主要是模型通道面积错误分配。
- sample6 中 Leucine 的 Channel_Calculated_Conc_mM = 22.82，但 NNLS Raw 只有约 6.86，也说明主要是模型通道面积错误分配。
因此问题不是单纯的定量公式，而是模型在重叠峰区域把 Gln/Glu/Asn 或 Ile/Val 的峰分配到了错误通道。

本次修改目标：
1. 保持模型结构不变。
2. 保持 predict1.py 暂时不变。
3. 保持现有面积主导训练方向。
4. 在训练数据中增加 hard-negative 样本。
5. 在 loss 中增加 absent-channel penalty。
6. 在验证日志中增加 Proline 和 BCAA 组的假阳性指标。
7. 让模型学习：
   - Glutamine + Glutamate 不等于 Proline
   - Asparagine + Glutamine + Glutamate 不等于 Proline
   - Isoleucine + Valine 不等于 Leucine
   - Isoleucine / Leucine / Valine 的重叠区需要正确分配，而不是平均或错误分配

第一部分：修改 dataset.py

在 NMRSepDataset 中加入或扩展 hard-negative 采样逻辑。

如果 experimental_pure_dir 存在，并且当前使用实验纯谱 network_ready 随机叠加，则增加 overlap hard-negative 样本生成。

新增参数建议：
- proline_hard_negative_prob = 0.20
- bcaa_hard_negative_prob = 0.25
- overlap_hard_negative_prob 可以作为总控制参数，如果已有类似参数则复用，不要重复设计太多参数。

Proline hard-negative：
这些样本必须不包含 Proline，但必须包含容易诱发 Proline 假阳性的组合。组合从下面列表随机选：
- Glutamine + Glutamate
- Asparagine + Glutamine + Glutamate
- Glutamine + Glutamate + Isoleucine
- Glutamine + Glutamate + Valine
- Glutamine + Glutamate + Isoleucine + Leucine + Valine

要求：
- Proline target channel 必须为全 0。
- 使用实验纯谱叠加。
- 保留随机 scale、shift、noise。
- 不改变 20 通道顺序。

BCAA hard-negative：
重点解决 sample6 中 Leucine 被 Ile/Val 诱发的问题。组合从下面列表随机选：
- Isoleucine + Valine，不包含 Leucine
- Isoleucine，不包含 Leucine / Valine
- Valine，不包含 Leucine / Isoleucine
- Leucine + Valine，不包含 Isoleucine
- Isoleucine + Leucine，不包含 Valine
- Isoleucine + Valine + Glutamine，不包含 Leucine
- Isoleucine + Valine + Asparagine，不包含 Leucine

要求：
- 当 hard-negative 设定为“不含 Leucine”时，Leucine target channel 必须为全 0。
- 当设定为“不含 Isoleucine”或“不含 Valine”时，对应 target channel 必须为全 0。
- 使用实验纯谱叠加。
- 保留随机 scale、shift、noise。
- 不改变 20 通道顺序。

注意：
不要把 hard-negative 做成固定模板完全重复。仍然要随机 scale、shift、noise，让模型学习重叠区域的通道归属，而不是背固定样本。

第二部分：修改 loss.py

在当前实际使用的面积主导 loss 中加入 absent-channel penalty。如果当前类名是 AreaDominantQuantitativeLoss，则修改它；如果 test_metrics.py 实际使用的是其他 loss 类，请修改实际被调用的类，不要只改未使用的类。

新增 Proline absent penalty：
参数建议：
- proline_absent_w = 3.0
- proline_absent_peak_w = 1.5
- proline_peak_threshold = 0.015

触发条件：
- Proline target_area <= target_presence_threshold
- 且 overlap_trigger_indices 中至少一个通道 target_area > target_presence_threshold

Proline overlap_trigger_indices：
- Glutamine
- Glutamate
- Asparagine
- Isoleucine
- Leucine
- Valine

惩罚：
- Proline predicted positive area
- relu(Proline predicted max peak - proline_peak_threshold)

只在 Proline target 为 0 时惩罚，不要惩罚真实 Proline 存在的样本。

新增 Leucine absent penalty：
参数建议：
- leucine_absent_w = 1.5
- leucine_absent_peak_w = 0.75
- leucine_peak_threshold = 0.015

触发条件：
- Leucine target_area <= target_presence_threshold
- 且 Isoleucine 或 Valine target_area > target_presence_threshold

惩罚：
- Leucine predicted positive area
- relu(Leucine predicted max peak - leucine_peak_threshold)

注意：
Leucine penalty 权重不要一开始设置太高，因为 Leucine 在不少样本中真实存在，过强惩罚会伤害 Leucine recall。

可选增强：
如果代码结构适合，可以把 absent penalty 写成通用函数，支持：
- Proline absent when Gln/Glu/Asn/Ile/Leu/Val active
- Leucine absent when Ile/Val active
- Isoleucine absent when Leu/Val active
- Valine absent when Ile/Leu active

但本次最低要求必须实现 Proline 和 Leucine。

loss_dict 中新增打印：
- l_proline_absent_area
- l_proline_absent_peak
- l_leucine_absent_area
- l_leucine_absent_peak

第三部分：修改 test_metrics.py

继续保留现有：
- external_raw_MAPE
- external_cal_MAPE
- hard_fp_score
- best_by_table_s1_external_raw_mape.pth
- best_by_low_false_positive.pth
- latest_epoch.pth
- epoch_{epoch}.pth

在 per-compound classification 中继续打印：
- recall
- precision
- TP
- FN
- FP

新增或扩展假阳性指标：

1. 保留 Proline hard FP：
hard_fp_score 中继续重点惩罚 Proline：
Proline_FP * 3.0

2. 增加 BCAA confusion / false-positive score：
新增 bcaa_fp_score 或 overlap_fp_score，建议：
bcaa_fp_score =
    Leucine_FP * 2.0
    + Isoleucine_FP * 1.5
    + Valine_FP * 1.5

如果能判断条件性 FP，则更好：
- Leucine_FP_when_Ile_or_Val_active
- Isoleucine_FP_when_Leu_or_Val_active
- Valine_FP_when_Ile_or_Leu_active

最低要求：日志中打印 Leucine_FP、Isoleucine_FP、Valine_FP，并打印 bcaa_fp_score。

3. 修改 hard_fp_score：
建议改成：
hard_fp_score =
    Proline_FP * 3.0
    + Leucine_FP * 2.0
    + Glutamate_FP * 1.5
    + Glutamine_FP * 1.5
    + Isoleucine_FP * 1.5
    + Valine_FP * 1.5
    + Asparagine_FP * 1.0

4. 新增 hard inactive area 打印：
除已有 Proline、Gln/Glu/Pro、Ile/Leu/Val/Pro 外，增加：
- Leucine inactive area
- BCAA inactive area
- Leucine inactive peak p95，如果容易实现

日志中输出类似：
[overlap-fp]
Proline_FP=...
Leucine_FP=...
Isoleucine_FP=...
Valine_FP=...
hard_fp_score=...
bcaa_fp_score=...

[hard inactive area]
Proline=...
Leucine=...
Gln/Glu/Pro=...
Ile/Leu/Val/Pro=...

第四部分：checkpoint 选择逻辑

不要只用 score 保存 best。

保留：
- best_by_score.pth
- best_by_table_s1_external_raw_mape.pth
- best_by_low_false_positive.pth
- latest_epoch.pth
- epoch_{epoch}.pth

保存 best_by_low_false_positive.pth 时：
- hard_fp_score 下降
- serum_recall >= 0.95

如果能加入 bcaa_fp_score，则保存额外权重：
- checkpoints/best_by_overlap_low_fp.pth

保存条件：
- hard_fp_score 或 bcaa_fp_score 改善
- serum_recall >= 0.95
- external_raw_MAPE 没有明显恶化，例如不超过历史最好 external_raw_MAPE 的 1.5 倍，避免为了压假阳性导致定量崩掉。

第五部分：训练后验证建议

修改后先不要长跑，先跑 10-15 轮观察：
1. sample1 中 Proline 是否明显下降。
2. sample6 中 Leucine 是否明显下降。
3. Leucine 真阳性样本，如 sample4 / sample5，是否被压没。
4. external_raw_MAPE 是否没有大幅恶化。
5. Proline recall 不要明显掉。
6. Leucine recall 不要明显掉。

第六部分：代码检查

修改完成后运行：

python -m py_compile C:\Users\tufi\Desktop\代码\师兄\dataset.py
python -m py_compile C:\Users\tufi\Desktop\代码\师兄\loss.py
python -m py_compile C:\Users\tufi\Desktop\代码\师兄\test_metrics.py

如果是在 Linux 虚拟机路径下运行，则对应检查：

python -m py_compile /home/ty/PycharmProjects/数据集双/dataset.py
python -m py_compile /home/ty/PycharmProjects/数据集双/loss.py
python -m py_compile /home/ty/PycharmProjects/数据集双/test_metrics.py

本次不要修改 predict1.py。先让模型训练本身学会降低 Proline 和 Leucine 的重叠误分配。