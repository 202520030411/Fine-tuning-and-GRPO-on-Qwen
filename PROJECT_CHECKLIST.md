# 项目要求对照检查

对照「项目要有」的设定，逐项检查代码与文档是否满足。

---

## 一、链接与资源

| 要求 | 位置 | 状态 |
|------|------|------|
| Github: https://github.com/202520030411/Fine-tuning-and-GRPO-on-Qwen | README § 项目链接 | ✅ 已写 |
| Slides (Overleaf): https://www.overleaf.com/5734841246gzbqxbwkwqct#5e20f5 | README § 项目链接 | ✅ 已写 |
| PPT (Canva): https://www.canva.com/design/DAHCkz-GCEo/YzZ4 | README § 项目链接 | ✅ 已写 |
| Finetuning 教程 4 个链接 | README § 参考资源 | ✅ 已写 |
| Qwen repo: https://huggingface.co/Qwen/Qwen3-0.6B | README § 参考资源 | ✅ 已写 |
| GSM8K: https://huggingface.co/datasets/openai/gsm8k | README 已写；**代码**见下 | ⚠️ 见「微调数据」 |
| MMLU: cais/mmlu, hendrycks/test | README + dataset/mmlu.py 用 cais/mmlu | ✅ 满足 |

---

## 二、关键决策 (Key decisions)

| 要求 | 代码/文档位置 | 状态 |
|------|----------------|------|
| **属性**：事实性知识 + 特定推理形式（算术） | README § 项目设定；dataset 为 GSM8K/SVAMP/MMLU 数学 | ✅ 满足 |
| **模型**：Qwen | train_sft.py / train_grpo.py / eval.py / eval_mmlu.py 默认 `Qwen/Qwen3-0.6B-Base` | ✅ 满足 |
| **Prompt**：题目本身 + Step by step + **定义基本数学公式与规则 (+, -, x, /)** | dataset/gsm8k.py `_format_prompt` | ❌ **缺**：当前只有 “Question” + “Answer”，target 有 “Let’s think step by step”，**没有**在 prompt 中写明「使用 +, −, ×, ÷ 等基本运算」 |
| **评估策略**：Train + Test | prepare 出 train/test，eval 脚本用 test 集 | ✅ 满足 |
| **微调数据**：openai/gsm8k | dataset/gsm8k.py 第 67 行 `load_dataset("gsm8k", "main")` | ⚠️ **不一致**：要求为 **openai/gsm8k**，当前为 **"gsm8k", "main"**（HF 上可能等价，但未显式写 openai/gsm8k） |
| **微调方法**：LoRA 或 DoRA，训练 + 测试验证 | trainer/sft.py 仅用 LoraConfig；无 DoRA | ⚠️ **仅 LoRA**：README 写「LoRA（或 DoRA）」，**代码未实现 DoRA** |

---

## 三、三项评估指标 (3 metrics)

| 指标 | 要求 | 代码位置 | 状态 |
|------|------|----------|------|
| **1. GSM8K Test, Pass@1** | 一次生成，答案对即对 | scripts/eval.py：temperature=0、do_sample=False，按 final_answer 判对错，输出 accuracy | ✅ 满足 |
| **2. SVAMP, Pass@1** | 同概念不同表述，验证推理 | dataset/svamp.py + scripts/prepare_svamp.py 出 JSONL；eval.py 用同一套逻辑，传入 SVAMP 的 test 路径即可 | ✅ 满足 |
| **3. MMLU, Multiple-choice Accuracy** | 选择题，防灾难性遗忘 | scripts/eval_mmlu.py：cais/mmlu，提取 A/B/C/D 与正确答案比对，输出 accuracy | ✅ 满足 |

---

## 四、Presentation 故事

| 要求 | 位置 | 状态 |
|------|------|------|
| **小猿搜题**：图片 → 文字(OCR) → LLM → 出答案 | README § Presentation 故事 | ✅ 已写 |
| 同上（用于汇报） | slides.tex | ❌ **缺**：slides 仅有 “Motivation & Objective”（小模型+数学推理），**没有**「小猿搜题」故事的一页或一段 |

---

## 五、代码与文件逐一简表

| 文件 | 作用 | 与要求的关系 |
|------|------|--------------|
| dataset/gsm8k.py | 加载 GSM8K、构造 prompt/target | 数据源应为 openai/gsm8k；prompt 未写基本运算规则 |
| dataset/svamp.py | SVAMP 预处理 | ✅ 用于 Metric 2 |
| dataset/mmlu.py | MMLU 预处理，cais/mmlu | ✅ 用于 Metric 3 |
| scripts/prepare_gsm8k.py | 拉取 GSM8K 并写 train/test JSONL | 依赖 gsm8k.py 的 load（即 "gsm8k", "main"） |
| scripts/prepare_svamp.py | 拉取 SVAMP 并写 JSONL | ✅ |
| scripts/train_sft.py | SFT 训练，LoRA，默认 Qwen3-0.6B-Base | ✅ 缺 DoRA 选项 |
| scripts/train_grpo.py | GRPO 训练 | ✅ |
| scripts/eval.py | GSM8K/SVAMP 评估，Pass@1 语义 | ✅ |
| scripts/eval_mmlu.py | MMLU 多选准确率 | ✅ |
| scripts/analyze.py | 结果汇总与作图 | ✅ |
| trainer/sft.py | SFT + LoRA 实现 | ✅ 无 DoRA |
| trainer/reward.py | 正确性 + #### 格式 reward | ✅ |
| README.md | 链接、关键决策、3 metrics、小猿搜题故事 | ✅ 文档齐全；方法写 LoRA/DoRA 但代码仅 LoRA |
| slides.tex | 汇报 slides | ❌ 无小猿搜题故事 |

---

## 六、缺失与建议修改汇总

1. **Prompt 设计**  
   - **缺**：在 system/prompt 中**明确写出**「使用基本数学运算 +, −, ×, ÷，逐步推导并给出最终数字答案」。  
   - **建议**：在 `dataset/gsm8k.py` 的 `_format_prompt`（以及如需一致则在 `dataset/svamp.py`）里加上一句说明基本运算与步骤要求。

2. **微调数据源**  
   - **不一致**：要求为 **openai/gsm8k**，代码为 `load_dataset("gsm8k", "main")`。  
   - **建议**：改为 `load_dataset("openai/gsm8k", "main", split=...)`，与文档一致。

3. **DoRA**  
   - **缺**：README 写「LoRA 或 DoRA」，代码只有 LoRA。  
   - **建议**：二选一 —— 要么在 `trainer/sft.py`（及 train_sft 参数）中增加 DoRA 选项，要么在 README 中改为「本项目使用 LoRA」，并注明 DoRA 为后续可选。

4. **Slides 故事**  
   - **缺**：slides.tex 中没有「小猿搜题：图片→文字→LLM→答案」的叙述。  
   - **建议**：在 Motivation 或单独一页加一句/一小节，说明应用场景为小猿搜题式流程，本项目负责 LLM 数学推理部分。

---

## 七、总结

| 类别 | 已满足 | 待补/待改 |
|------|--------|-----------|
| 链接与参考资源 | Github、Overleaf、Canva、教程、Qwen、GSM8K、MMLU 均在 README | — |
| 关键决策 | 属性、模型、评估策略、3 metrics 在代码与 README 中一致 | Prompt 未写基本运算规则；数据源未写 openai/gsm8k；方法写 DoRA 但未实现 |
| 3 metrics | GSM8K Pass@1、SVAMP Pass@1、MMLU 多选 均已实现 | — |
| Presentation 故事 | README 有小猿搜题 | slides.tex 未体现 |

按上述「缺失与建议」修改后，项目即可在代码与文档层面完全对齐当前要求。
