# SigExt - Salient Information Prompting for Abstractive Summarization

Implementation and exploration of the EMNLP 2024 paper:

**"Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization"**

This project explores how salient keyword extraction can improve prompt-based summarization using Large Language Models (LLMs).

---

# Overview

SigExt improves abstractive summarization by extracting important keyphrases from source documents and injecting them into prompts before sending them to an LLM.

Instead of simply prompting:

```bash
Summarize this article.
```

SigExt extracts salient phrases and prompts like:

```bash
Important phrases:
- Apple
- AI chips
- WWDC

Now summarize the article.
```

This helps generate:

* more informative summaries
* better ROUGE recall
* summaries closer to human-written references

---

# Project Pipeline

```text
Dataset
   ↓
Data preprocessing
   ↓
RAKE keyword extraction
   ↓
Longformer keyword scoring model
   ↓
Top-K salient phrase selection
   ↓
Prompt-based summarization
   ↓
ROUGE evaluation
```

---

# Technologies Used

* Python
* PyTorch
* PyTorch Lightning
* HuggingFace Transformers
* Longformer
* HuggingFace Datasets
* NLTK
* RAKE Keyword Extraction

---

# Supported Datasets

* CNN/DailyMail
* XSum
* PubMed
* Arxiv
* Multi-News
* BigPatent
* BillSum
* WikiHow
* AESLC

---

# Installation

## Clone Repository

```bash
git clone https://github.com/nikhil-dtech/SigExt-Research.git
cd SigExt-Research
```

---

# Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install pytorch-lightning
pip install transformers
pip install datasets
pip install sentencepiece
pip install scikit-learn
pip install numpy pandas tqdm
pip install jsonlines
pip install rouge-score
pip install evaluate
pip install rake-nltk
pip install matplotlib seaborn nltk
```

---

# Download NLTK Resources

Open Python shell:

```bash
python
```

Then run:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
exit()
```

---

# Dataset Preparation

Example using CNN/DailyMail dataset:

```bash
python src/prepare_data.py --dataset cnn --output_dir experiments/cnn_dataset/
```

This generates:

* train.jsonl
* validation.jsonl
* test.jsonl

with extracted candidate phrases.

---

# Train Longformer Keyword Extractor

```bash
python src/train_longformer_extractor_context.py --dataset_dir experiments/cnn_dataset/ --checkpoint_dir experiments/cnn_extractor_model/
```

---

# IMPORTANT NOTE

Training Longformer on CPU is extremely slow.

Recommended:

* NVIDIA GPU with CUDA
* Smaller datasets for local experimentation

---

# Run Inference

```bash
python src/inference_longformer_extractor.py --dataset_dir experiments/cnn_dataset/ --checkpoint_dir experiments/cnn_extractor_model/ --output_dir experiments/cnn_dataset_with_keyphrase/
```

---

# Run Summarization

```bash
python src/zs_summarization.py --model_name claude --kw_strategy sigext_topk --kw_model_top_k 15 --dataset cnn --dataset_dir experiments/cnn_dataset_with_keyphrase/ --output_dir experiments/cnn_extsig_predictions/
```

---

# Research Paper

Amazon Science Publication:

https://www.amazon.science/publications/salient-information-prompting-to-steer-content-in-prompt-based-abstractive-summarization

---

# Citation

```text
@inproceedings{xu2024salient,
  title={Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization},
  author={Xu, Lei and Karim, Mohammed Asad and Dingliwal, Saket and Elangovan, Aparna},
  booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track",
  year={2024}
}
```

---

# Notes

This repository is being explored and studied for:

* NLP research understanding
* prompt engineering
* summarization systems
* Longformer-based keyword extraction
* open-source contribution practice

---
