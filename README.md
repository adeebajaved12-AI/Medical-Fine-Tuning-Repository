# 🩺 Clinical Context Fine-Tuning of DeepSeek-R1 via Unsloth Configurations

This repository contains the complete enterprise-grade pipeline for parameter-efficient fine-tuning (PEFT) of the **DeepSeek-R1-Distill-Llama-8B** reasoning model. The project focuses on optimizing large language models for localized medical instructions, diagnostic reasoning, and clean clinical response generation with minimal hardware footprints.

---

## 🚀 Key Architectural Highlights
* **Base Model:** `unsloth/deepseek-r1-distill-llama-8b`
* **Optimization Framework:** Unsloth (QLoRA) kernels for 2x faster execution.
* **Precision Format:** FP16 computational registers forced for high compatibility with NVIDIA T4 Tensor Core GPUs.
* **Trainable Parameter Constraining:** Tuned targeted projection layers (`q_proj`, `v_proj`, `k_proj`, etc.), adjusting only **0.52%** (41.9M) of the total 8.07 Billion model parameters.
* **Dataset Strategy:** Structured Local Data Generation Matrix mimicking structured medical Chain-of-Thought (CoT) configurations.

---

## 📊 SFT Training Convergence Matrix
The model was optimized using Supervised Fine-Tuning (SFT) principles over 60 training steps. The loss experienced a steep, stable optimization trajectory down to a near-zero baseline:

| Training Step | Operational Epoch | Empirical Training Loss |
| :---: | :---: | :---: |
| **Step 10** | Epoch 0.80 | **2.5858** |
| **Step 20** | Epoch 1.60 | **0.2067** |
| **Step 30** | Epoch 2.40 | **0.0380** |
| **Step 40** | Epoch 3.20 | **0.0141** |
| **Step 50** | Epoch 4.00 | **0.0148** |
| **Step 60** | Epoch 4.80 | **0.0127** (Convergence Target) |

---

## 🛠️ Deployed Operational Pipeline (Code Overview)

### Hardware Installation & Core Environment
```python
!pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
!pip install --no-deps xformers trl peft loralib
