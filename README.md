
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/im-anuraggupta/LLM_FineTuning_using_LoRA/blob/main/LLM_Fine_Tuning_using_LoRA.ipynb)

# Fine-Tuning Falcon LLM on Yelp Reviews using LoRA (Colab)

This project fine-tunes the `tiiuae/falcon-rw-1b` model using LoRA (Low-Rank Adaptation) on a 1% subset of the Yelp Review dataset in a Colab GPU environment.

---

## 🚀 Features
- Uses **PEFT LoRA adapters** for memory-efficient fine-tuning
- Optimized for **free-tier Google Colab**
- Generates **new Yelp-style reviews** after training
- Loads Falcon-1B model in **8-bit precision** using `bitsandbytes`

---

## 📦 Installation

Run the following in the **first cell of your notebook** to install all dependencies:

```python
!pip install -r requirements.txt
```

---

## 📂 Project Structure

```
LLM_FineTuning_using_LoRA/
├── LLM_Fine_Tuning_using_LoRA.ipynb   # Main training notebook
├── requirements.txt                   # List of dependencies
└── README.md                          # This file
```

---

## 🧠 How It Works

1. **Dataset**: Loads 1% of `yelp_review_full` from Hugging Face Datasets.
2. **Model**: Loads `tiiuae/falcon-rw-1b` in 8-bit using `bitsandbytes`.
3. **Fine-Tuning**: Applies LoRA adapters (via `peft`) to attention layers only.
4. **Training**: Finetunes using Hugging Face `Trainer` API.
5. **Inference**: Uses the fine-tuned model to generate review completions.

---

## 🛠️ Requirements

- Google Colab (GPU-enabled)
- Python ≥ 3.10
- `transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`, `torch`

---
