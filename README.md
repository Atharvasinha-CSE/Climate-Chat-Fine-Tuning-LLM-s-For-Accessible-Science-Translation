# 💬 ClimateChat: Fine-Tuning LLMs for Accessible Science Translation

An end-to-end Parameter-Efficient Fine-Tuning (PEFT) pipeline designed to adapt a Large Language Model's style, format, and tone. This system specializes in transforming dense, technical, and jargon-heavy climate findings (such as IPCC reports) into accessible public-facing summaries and digital communication briefs.

---

## 🚀 Project Overview
* **Fine-tuned an open-weights 3B LLM** using QLoRA parameter-efficient techniques to specialize in climate science translation, modifying less than 1% of total parameters to preserve base capabilities.
* **Engineered a custom instruction-tuning dataset** from dense IPCC reports to automate the synthesis of technical climate findings into highly engaging, public-facing digital communication briefs.

---

## 🛠️ Tech Stack & Optimization Matrix

* **Base Large Language Model:** `Qwen/Qwen2.5-3B-Instruct`
* **Fine-Tuning Framework:** Hugging Face TRL (`SFTTrainer` & `SFTConfig`)
* **Parameter Efficiency:** PEFT (`LoraConfig`)
* **Quantization Engine:** `bitsandbytes` (4-bit NF4 precision with Double Quantization)
* **Optimization Configuration:** Paged AdamW 8-bit optimizer
* **Compute Infrastructure:** Hosted NVIDIA T4 GPU (Google Colab Runtime)

---

## 🧠 Architectural & Engineering Highlights

### 1. Custom Instruction-Tuning Schema
The training data is mapped into a strict structural template to teach the model a deterministic layout behavior. By packing instructions, raw scientific data, and multi-format outputs together, the model learns to format outputs uniformly.

```text
### Instruction:
Translate this scientific IPCC finding into a public-facing policy summary and an engaging social media brief.

### Input:
[Dense Scientific / Meteorological Text]

### Response:
### Policy Summary
[Simplified Text]

### Social Media Brief
[Engaging Copy with Hashtags/Emojis]
2. Parameter-Efficient QLoRA ArchitectureInstead of modifying all 3 billion parameters—which would cause out-of-memory errors on commodity hardware—this pipeline implements Quantized Low-Rank Adaptation (QLoRA).The original weights are frozen in 4-bit NormalFloat (NF4) precision.Low-rank adapter matrices ($r=8, \alpha=16$) are injected specifically into the model's core attention modules (q_proj, v_proj, k_proj, o_proj).Trainable parameters are restricted to just 0.1193% of the entire network, keeping the footprint highly lightweight.
3. Mixed-Precision & Library Adaptation
Built with strict compliance for modern Hugging Face ecosystem updates:

Utilizing SFTConfig for unified asset tracking instead of legacy training argument wrappers.

Passing the tokenizer through the modernized multi-modal processing_class field.

Native execution alignment: fp16 scaling is managed entirely by the bitsandbytes compute dtype configuration rather than PyTorch's native GradScaler, eliminating precision runtime clashes on the NVIDIA T4 chip architecture.
📦 Repository Structure & Files
Climate_Science_FineTuning.ipynb: The complete production notebook containing the dataset constructor, quantization configuration, adapter initialization, training loop, and inference validation.

.gitignore: Configured explicitly to mask heavy auto-generated local data structures:

Blocks checkpoint states (checkpoint-*/)

Blocks fine-tuning model weight files (climate_model_results/)

Cleans macOS/Jupyter runtime metadata artifacts (.DS_Store, .ipynb_checkpoints/)
