# Awesome MLLM Re-Identification

[![Validate](https://github.com/Reflection0427/Awesome-MLLM-ReID/actions/workflows/validate.yml/badge.svg)](https://github.com/Reflection0427/Awesome-MLLM-ReID/actions/workflows/validate.yml)
[![Discover Papers](https://github.com/Reflection0427/Awesome-MLLM-ReID/actions/workflows/discover-papers.yml/badge.svg)](https://github.com/Reflection0427/Awesome-MLLM-ReID/actions/workflows/discover-papers.yml)

本仓库系统整理多模态大语言模型（MLLM）、大型视觉语言模型（LVLM）与 Re-Identification（ReID）交叉领域的研究工作，重点关注大模型在行人重识别、交互式行人检索、多模态特征学习和跨模态检索中的应用。

论文按照研究方向进行分类，并为每篇论文添加方向标签，方便研究人员快速定位相关工作。

**25 Reviewed Papers | 16 Core ReID Papers | 9 Related Methods**

## 1. Research Categories

### Core MLLM / LVLM ReID

直接将大模型应用于行人或目标重识别任务的研究。

| Research Direction | Description |
|---|---|
| Interactive ReID | 通过多轮交互、主动提问和用户反馈优化目标检索 |
| General ReID | 利用 MLLM/LVLM 提升行人重识别能力 |
| Text-to-Image ReID | 利用语言描述实现跨模态行人检索 |
| Domain Generalization | 提升模型在不同数据集、场景和域之间的泛化能力 |
| Multimodal Alignment | 学习视觉、语言等不同模态之间的语义对应关系 |
| Object ReID | 将大模型应用于多模态目标重识别 |
| ReID Benchmark | 构建大模型行人重识别评测体系 |

### Related MLLM Retrieval / Re-ranking

虽然并非全部直接面向 ReID，但其检索、嵌入、重排序和知识蒸馏方法具有较高的迁移价值。

| Research Direction | Description |
|---|---|
| Multimodal Retrieval | 利用多模态大模型完成跨模态检索 |
| Multimodal Embedding | 学习统一的多模态向量表示 |
| Re-ranking | 对初始检索结果进行二次排序和优化 |
| Composed Image Retrieval | 根据参考图像和文本修改指令检索目标图像 |
| Knowledge Distillation | 将大模型能力迁移到轻量化检索模型 |

---

## 2. Core MLLM / LVLM ReID

### 2.1 Interactive Person Re-Identification

**研究重点：** 通过多轮对话、主动提问和用户反馈，逐步缩小候选身份范围，提高复杂场景下的行人检索准确性。

| Venue | Method | Paper | Code | Direction Tags |
|---|---|---|---|---|
| ICML 2026 | MNEMO | [Paper](https://proceedings.mlr.press/v306/bai26j.html) | [Code](https://github.com/Flame-Chasers/MNEMO) | `Interactive ReID` `Multi-turn Dialogue` |
| ICCV 2025 | ChatReID | [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Niu_ChatReID_Open-ended_Interactive_Person_Retrieval_via_Hierarchical_Progressive_Tuning_for_ICCV_2025_paper.html) | — | `Interactive ReID` `Progressive Tuning` |
| ICML 2025 | LLaVA-ReID | [Paper](https://proceedings.mlr.press/v267/lu25s.html) | [Code](https://github.com/XLearning-SCU/LLaVA-ReID) | `Interactive ReID` `Active Questioning` |
| CVPR 2025 | ICL | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Qin_Human-centered_Interactive_Learning_via_MLLMs_for_Text-to-Image_Person_Re-identification_CVPR_2025_paper.html) | [Code](https://github.com/QinYang79/ICL) | `Interactive ReID` `Human-in-the-loop` |

### 2.2 General MLLM-based ReID

**研究重点：** 探索多模态大模型在 ReID 特征提取、视觉语义理解、推理和检索任务中的应用。

| Venue | Method | Paper | Code | Direction Tags |
|---|---|---|---|---|
| CVPR 2026 | CoT-ReID | [Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Chain-of-Thought_Guided_Multi-Modal_Object_Re-Identification_CVPR_2026_paper.html) | [Code](https://github.com/Gaoya615/CoT-ReID) | `Object ReID` `Chain-of-Thought` |
| ICASSP 2026 | LVLM-ReID | [Paper](https://doi.org/10.1109/ICASSP55912.2026.11460593) | — | `General ReID` `LVLM` |
| CVPR 2024 | Instruct-ReID | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/He_Instruct-ReID_A_Multi-purpose_Person_Re-identification_Task_with_Instructions_CVPR_2024_paper.html) | [Code](https://github.com/hwz-zju/Instruct-ReID) | `General ReID` `Instruction Following` |
| arXiv 2024 | MLLMReID | [Paper](https://arxiv.org/abs/2401.13201) | — | `General ReID` `MLLM` |
| arXiv 2025 | MMReID-Bench | [Paper](https://arxiv.org/abs/2508.06908) | — | `ReID Benchmark` `MLLM Evaluation` |

### 2.3 Text-to-Image Person Re-Identification

**研究重点：** 利用自然语言描述和视觉信息之间的语义关联，实现文本到行人图像的跨模态检索，并提升跨数据集泛化能力。

| Venue | Method | Paper | Code | Direction Tags |
|---|---|---|---|---|
| Findings of ACL 2026 | Auto-ReID | [Paper](https://aclanthology.org/2026.findings-acl.312/) | [Code](https://github.com/GridNexus/Auto-ReID) | `T2I-ReID` `Self-Correction` |
| CVPR 2025 | HAM + UPS | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Jiang_Modeling_Thousands_of_Human_Annotators_for_Generalizable_Text-to-Image_Person_Re-identification_CVPR_2025_paper.html) | [Code](https://github.com/MPI-Lab/HAM) | `T2I-ReID` `Annotator Modeling` |
| Findings of EMNLP 2025 | PVTNL | [Paper](https://aclanthology.org/2025.findings-emnlp.1181/) | — | `T2I-ReID` `Vision-Language Prompting` |
| CVPR 2024 | MLLM4Text-ReID | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Tan_Harnessing_the_Power_of_MLLMs_for_Transferable_Text-to-Image_Person_ReID_CVPR_2024_paper.html) | [Code](https://github.com/MPI-Lab/MLLM4Text-ReID) | `T2I-ReID` `Transferability` |

### 2.4 Domain Generalization and Multimodal Alignment

**研究重点：** 通过视觉与语言的多粒度语义对齐，增强模型对不同场景、不同数据域的适应能力。

| Venue | Method | Paper | Code | Direction Tags |
|---|---|---|---|---|
| IET Computer Vision 2026 | MUVA | [Paper](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cvi2.70062) | [Code](https://github.com/RikoLi/MUVA) | `Domain Generalization` `Multimodal Alignment` |
| arXiv 2026 | MUSE | [Paper](https://arxiv.org/abs/2606.16161) | [Code](https://github.com/RikoLi/MUSE) | `Domain Generalization` `Re-ranking` `MLLM` |

### 2.5 Object Re-Identification

**研究重点：** 探索多模态大模型的推理和视觉上下文学习能力在目标重识别中的应用。

| Venue | Method | Paper | Code | Direction Tags |
|---|---|---|---|---|
| CVPR 2026 | CoT-ReID | [Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Chain-of-Thought_Guided_Multi-Modal_Object_Re-Identification_CVPR_2026_paper.html) | [Code](https://github.com/Gaoya615/CoT-ReID) | `Object ReID` `Chain-of-Thought` |
| ICCV 2025 | VICP | [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_Generalizable_Object_Re-Identification_via_Visual_In-Context_Prompting_ICCV_2025_paper.html) | [Code](https://github.com/Hzzone/VICP) | `Object ReID` `Visual In-Context Prompting` |

### 2.6 ReID Framework and Benchmark

**研究重点：** 构建面向大模型的 ReID 任务体系和评测基准，研究指令理解、检索能力及多任务适应能力。

| Venue | Method | Paper | Code | Direction Tags |
|---|---|---|---|---|
| arXiv 2025 | MMReID-Bench | [Paper](https://arxiv.org/abs/2508.06908) | — | `ReID Benchmark` `MLLM Evaluation` |
| CVPR 2024 | Instruct-ReID | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/He_Instruct-ReID_A_Multi-purpose_Person_Re-identification_Task_with_Instructions_CVPR_2024_paper.html) | [Code](https://github.com/hwz-zju/Instruct-ReID) | `Instruction Following` `Multi-task ReID` |

---

## 3. Related MLLM Retrieval / Re-ranking

### 3.1 Multimodal Retrieval and Embedding

**研究重点：** 利用 MLLM 学习高质量的多模态向量表示，增强图像、文本及跨模态检索能力。

| Venue | Method | Paper | Code / Project | Direction Tags |
|---|---|---|---|---|
| ICLR 2026 | U-MARVEL | [Paper](https://arxiv.org/abs/2507.14902) | [Code](https://github.com/chaxjli/U-MARVEL) | `Multimodal Retrieval` `Embedding Learning` |
| ICML 2026 | FreeRet | [Paper](https://proceedings.mlr.press/v306/zhu26ba.html) | [Code](https://github.com/MCG-NJU/FreeRet) | `Multimodal Retrieval` `Training-Free` |
| ICLR 2025 | MM-Embed | [Paper](https://openreview.net/forum?id=i45NQb2iKO) | [Project / Model](https://huggingface.co/nvidia/MM-Embed) | `Multimodal Retrieval` `Embedding` |
| ICLR 2025 | VLM2Vec | [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/04261fce1705c4f02f062866717d592a-Abstract-Conference.html) | [Code](https://github.com/TIGER-AI-Lab/VLM2Vec) | `Multimodal Embedding` `Vision-Language` |

### 3.2 Re-ranking

**研究重点：** 利用大模型的视觉理解和语义推理能力，对初始检索结果进行重新评估和排序，提高最终检索质量。

| Venue | Method | Paper | Code | Direction Tags |
|---|---|---|---|---|
| arXiv 2025 | CoTRR | [Paper](https://arxiv.org/abs/2509.14746) | [Code](https://github.com/freshfish15/CoTRR) | `Re-ranking` `Chain-of-Thought` |
| arXiv 2026 | MUSE | [Paper](https://arxiv.org/abs/2606.16161) | [Code](https://github.com/RikoLi/MUSE) | `Re-ranking` `MLLM` `Domain Generalization` |

### 3.3 Composed Image Retrieval

**研究重点：** 根据用户提供的参考图像和自然语言修改指令，检索符合组合条件的目标图像。

| Venue | Method | Paper | Code / Project | Direction Tags |
|---|---|---|---|---|
| CVPR 2026 | ReCALL | [Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_ReCALL_Recalibrating_Capability_Degradation_for_MLLM-based_Composed_Image_Retrieval_CVPR_2026_paper.html) | [Code](https://github.com/RemRico/Recall) | `Composed Image Retrieval` `Capability Calibration` |
| AAAI 2025 | CIR-LVLM | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32768) | — | `Composed Image Retrieval` `User Intent` |
| ICCV 2025 | DistillCIR | [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Zhong_Zero-Shot_Composed_Image_Retrieval_via_Dual-Stream_Instruction-Aware_Distillation_ICCV_2025_paper.html) | [Project / Model](https://distillcir.github.io/) | `Composed Image Retrieval` `Knowledge Distillation` |
| arXiv 2025 | SoFT | [Paper](https://arxiv.org/abs/2512.20781) | [Code](https://github.com/jjungyujin/SoFT) | `Composed Image Retrieval` `Zero-Shot Retrieval` |

---

## 4. Framework Figures

本仓库为每篇论文提供作者原始 Framework / Architecture / Pipeline 图，方便快速理解论文整体技术路线。

所有框架图均从对应论文的固定版本官方 PDF 中提取，不使用自绘图、博客截图或 AI 生成图。

框架图存放于：

`assets/frameworks/`

详细提取信息记录于：

`data/papers.json`

包括 PDF 地址、SHA-256、页码、Figure 编号及裁剪区域。

### Figure Extraction

```bash
python -m pip install -r requirements.txt

# Extract all framework figures
python scripts/extract_figure.py --all

# Verify extracted figures
python scripts/extract_figure.py --all --verify-only

# Check README consistency
python scripts/generate_readme.py --check

# Run tests
python -m unittest discover -s tests -v
```

---

## 5. Automatic Paper Tracking

仓库通过定期自动化工作流检索最新论文，数据来源包括：

- arXiv
- OpenAlex
- Crossref

每周一自动检索并创建或更新候选 Pull Request。

自动检索结果不会直接进入正式论文列表。所有候选论文需要经过人工审核，包括：

1. 阅读论文并确认研究方向。
2. 核实论文标题、发表信息和资源链接。
3. 确认与 MLLM-ReID 研究的相关性。
4. 选择作者提供的原始框架图。
5. 提取框架图并更新论文数据。

---

## 6. Contributing

欢迎研究人员提交相关论文、代码资源和改进建议。

请通过 GitHub Issue 或 Pull Request 参与贡献。

详细规范请参考：

[Paper Tracking and Review Guide](docs/PAPER_TRACKER.md)

---

## 7. License

仓库代码及整理者编写的元数据采用 MIT License。

论文原始框架图的版权及其他相关权利仍归论文作者或出版方所有。
