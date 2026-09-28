# DeepSeek Harness Prompt

Copy the block below into the DeepSeek harness after it has access to this
repository.

```text
你是一个资深 PyTorch 持续学习和量化工程师。请接手现有仓库，继续完成 AQCL
论文在 fault_csv 表格数据上的轻量化部署实现和验证。

工作目录优先使用：

C:\Users\Administrator\Desktop\CKAA_TIP2026-main\CKAA_TIP2026-main

如果当前环境没有这个目录，先 clone：

git clone https://github.com/zeta666-maker/CKAA-useful1.git
cd CKAA-useful1/CKAA_TIP2026-main

强制第一步：先阅读以下文件，不要直接修改代码。

1. DEEPSEEK_HANDOFF.md
2. AQCL.md
3. CKAA_AUDIT.md
4. train_aqcl.py
5. utils/aqcl.py
6. train_eval.py
7. tools/prepare_tabular_dataset.py
8. tools/run_aqcl_comparison.ps1
9. Adaptive_Quantization_for_Stable_Knowledge_Acquisition_in_Quantization-Aware_Continual_Learning.pdf

然后执行 git status、git log 和 git diff，确认当前基线。

当前已知事实：

- CKAA 和 AQCL 必须物理分离。
- train_eval.py 是纯 CKAA 入口，不得加入 AQCL 逻辑，也不得影响第一篇精度。
- AQCL 只能修改 train_aqcl.py、utils/aqcl.py、tools/run_aqcl_comparison.ps1
  和相关 AQCL 文档。
- 只使用 fault_csv 表格数据集，不使用图像数据集或其他数据集。
- 使用 5 个增量任务，类别分配固定为 2/2/2/2/1。
- DataLoader 必须 num_workers=0。
- 量化必须手写 PyTorch，不允许引入 pytorch-quantization、Brevitas
  或类似第三方量化框架。
- 推理 checkpoint 只能保留量化模型、量化位宽和必要推理参数，不能保存
  Fisher、协方差、SVD 子空间或其他训练期状态。

当前 AQCL 结果（只作为基线，不是最终结论）：

FP32:
  Last-acc 92.69%, Avg-acc 95.56%, Forgetting 0.87%

Fixed 8-bit:
  Last-acc 86.71%, Avg-acc 86.41%, Forgetting 9.57%

AQCL RPQ + SAOU target 4/8-bit:
  Last-acc 86.78%, Avg-acc 92.90%, Forgetting 6.80%

Fixed 4-bit:
  Last-acc 21.10%, Avg-acc 23.57%, Forgetting 12.26%

当前实现不一定正确。不要盲目相信当前设计、当前结果或当前指标。

已知缺陷和必须验证的点：

1. RPQ 可能退化为所有层都分配 8-bit，必须打印每层 Fisher 敏感度、分配位宽
   和位宽数量统计，确认是否真的存在 4/8-bit 混合。
2. Fixed 4-bit 和 Fixed 2-bit 精度仍然偏低。
3. SAOU 的 Uo、Up、特征值范围、梯度调制前后 cosine 必须记录和验证。
4. 当前效率统计只统计 Linear/Conv1d MACs，遗漏 attention QK^T 和
   attention @ V，必须修正后再报告 GFLOPs/GBOPS。
5. 当前最终对照只有 2 epoch/task，不是论文的 80 epoch/task。
6. 当前是表格 ViT，不是论文的 ResNet-20，因此绝对数值不能直接对比论文，
   但趋势必须符合论文。

必须继续执行的任务：

Phase 1：只加 instrumentation，不改变训练行为
- 打印每层 weight sensitivity、activation sensitivity、分配位宽。
- 打印 Uo/Up 维度、最小/最大特征值。
- 打印 SAOU 是否触发、gradient cosine 变化。

Phase 2：修复 RPQ
- 确认 Fisher 梯度不是全零、不是全相等。
- 确认 Eq.13 的高低 bit 分配真正产生混合位宽。
- 在准确率不下降的前提下降低部署模型大小。

Phase 3：提升 4-bit 和 2-bit 稳定性
- 分别测试 warm-up epoch、训练 epoch、激活 clipping、逐通道权重、
  SAOU 参数 alpha/theta/lambda。
- 必须对比 Fixed 8-bit、Fixed 4-bit、Fixed 2-bit、RPQ-only、
  RPQ+SAOU。

Phase 4：按论文趋势验收
必须满足：

32-bit >= 8-bit >= AQCL 4-bit > AQCL 2-bit
AQCL 4-bit >> Fixed 4-bit
AQCL 2-bit >> Fixed 2-bit（如果完成 2-bit）
模型大小和 GBOPS 随固定位宽降低而降低

禁止事项：

- 禁止把 task-id 真值作为推理输入来伪造最终精度。
- 禁止修改第一个论文的 train_eval.py CKAA 训练路径。
- 禁止删除现有实现来绕过问题。
- 禁止只报告数字而不给日志路径。
- 禁止用一次 smoke test 宣称达到论文级别。

每次修改后必须运行：

python -m py_compile train_eval.py train_aqcl.py utils\aqcl.py

然后至少运行对应的 AQCL 训练命令。最终输出必须包含：

1. 修改文件清单。
2. 每条论文公式与代码位置的映射。
3. 执行过的完整命令。
4. 结果表：FP32、8-bit、4-bit、2-bit、RPQ-only、RPQ+SAOU。
5. 模型大小、GFLOPs、GBOPS、推理额外内存。
6. 每个结论对应的日志路径。
7. 尚未解决的问题和风险。

最后请明确说明：当前实现是否真正符合论文，哪些结论已经验证，哪些仍然
只是假设。
```
