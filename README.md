# Offensive Tweet Classification using DistilBERT and LoRA

This project aims to classify tweets from the `tweet_eval` dataset as either **Offensive** or **NotOffensive** using a pre-trained `distilbert-base-uncased` model. To make the fine-tuning process efficient and memory-friendly, we used **LoRA (Low-Rank Adaptation)**, reducing the number of trainable parameters to only **0.93%** of the original model.

## Highlights
- Fine-tuned a transformer model on a binary classification task.
- Used Hugging Face Transformers, Datasets, Evaluate, and PEFT libraries.
- Applied LoRA with `target_modules=['q_lin']` to reduce training cost.
- Achieved accuracy of **~77.9%** on the validation and test sets on just 2 epochs.
- Inference tested on custom tweet examples to classify them as offensive or not.

## Key Techniques Used
- Model: `distilbert-base-uncased` from Hugging Face
- Dataset: `tweet_eval` (Offensive classification subset)
- Tokenization: Max-length padded and truncated text
- Training: Hugging Face `Trainer` API
- Evaluation Metric: Accuracy
- Parameter Efficient Fine-Tuning: LoRA (`r=4`, `lora_alpha=32`, `lora_dropout=0.01`)

## Example Inference
```text
You're such a loser, no one likes you. → Offensive  
I really enjoyed the concert last night! → NotOffensive  
You look disgusting. Stay home. → Offensive  
Just finished a great workout. Feeling amazing! → NotOffensive  
```
