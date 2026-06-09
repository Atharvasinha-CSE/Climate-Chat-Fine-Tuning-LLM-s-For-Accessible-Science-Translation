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
