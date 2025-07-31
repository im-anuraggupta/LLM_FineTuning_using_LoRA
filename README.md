# Fine-Tuning Falcon LLM on Yelp Reviews using LoRA (Colab)

This project fine-tunes the `tiiuae/falcon-rw-1b` model using LoRA (Low-Rank Adaptation) on a 1% subset of the Yelp Review dataset in a Colab GPU environment.

## Features
- Uses PEFT LoRA adapters for memory-efficient fine-tuning
- Optimized for free-tier Google Colab
- Generates new Yelp-style reviews after training

## Installation
Run the following in the first cell of your notebook to install all dependencies:
```python
!pip install -r requirements.txt
