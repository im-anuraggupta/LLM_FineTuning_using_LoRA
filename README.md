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
