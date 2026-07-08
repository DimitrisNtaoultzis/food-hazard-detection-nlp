# food-hazard-detection-nlp
Multi-label food hazard classification using TF-IDF, BERT, Focal Loss and ensemble methods. SemEval-2025 Task 9 | Best Kaggle ST1: 0.8188

# Food Hazard Detection — SemEval-2025 Task 9

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co)
[![Kaggle](https://img.shields.io/badge/Kaggle-ST1%200.8188-green)](https://kaggle.com)

Multi-label text classification of food safety recall reports.
Given a recall announcement, predict simultaneously:
- **hazard-category** (10 classes): biological, allergens, chemical...
- **product-category** (22 classes): meat & dairy, cereals, seafood...

**Best Kaggle ST1 Score: 0.8188**

---

## Task

**Official metric:**

$$\text{ST1} = \frac{\text{macro-F1}\_{\text{hazard}} + \text{macro-F1}\_{\text{product}|\text{correct hazard}}}{2}$$

Dataset: 5,082 train / 565 validation / 997 test samples

---

## Results

| Method | Kaggle ST1 |
|--------|-----------|
| TF-IDF + LogReg (baseline) | 0.576 |
| TF-IDF + SVM (tuned) | 0.741 |
| DistilBERT + SVM Ensemble | 0.755 |
| DistilBERT (train+valid) | 0.761 |
| Multi-task BERT | 0.771 |
| BERT-base + Focal Loss | 0.804 |
| BERT + Multi-task Ensemble | 0.817 |
| **BERT haz + Ensemble product** | **0.819** |

---

## Key Findings

- **Focal Loss** (γ=2.0) was the single most impactful improvement (+0.043 over CrossEntropy)
- **Training on train+valid** eliminates validation leakage (gap up to 0.08)
- **Ensemble diversity** matters: heterogeneous models outperform same-architecture ensembles
- **TF-IDF > word embeddings** for domain-specific food safety vocabulary

---

## Project Structure

├── notebooks/
│   ├── main.ipynb                 # Main experiments pipeline
│   ├── phase1_classical_ml.ipynb  # TF-IDF + SVM/LogReg baselines
│   ├── phase15_bertbase_focal.ipynb  # BERT + Focal Loss
│   ├── phase20_multitask_bert.ipynb  # Multi-task learning
│   ├── phase_final_comparison.ipynb  # Analysis & visualization
│   └── ...                        # All other experiments
├── report/
│   └── report_fixed.tex           # LaTeX report
└── figures/                       # All generated plots


---

## Methods Explored

**Classical:** TF-IDF + SVM, Logistic Regression, Naive Bayes,
Random Forest, BM25, LSA, Word2Vec, GloVe, FastText

**Neural:** TextCNN, BiLSTM, T5-small

**Transformers:** DistilBERT, BERT-base/large, RoBERTa-base/large,
DeBERTa-v3-large, BERT-multilingual

**Advanced:** Focal Loss, Multi-task learning, Hierarchical modeling,
Ensemble methods, Data augmentation (EDA), Label smoothing, Stacking

---

## Setup

```bash
pip install torch transformers scikit-learn pandas numpy
pip install lightgbm rank_bm25 wordcloud nltk
```

For transformer experiments, a GPU is required (tested on Google Colab A100).

---

## Tool Usage Statement

Claude AI (Anthropic) was used as a coding assistant throughout this
project to help write, debug, and improve Python code. All code was
understood, adapted, and verified by the student. All experimental
decisions, analyses, and written content are the student's own work.

---

## References

- Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV.
- Devlin et al. (2019). BERT. NAACL-HLT.
- Vallat et al. (2025). SemEval-2025 Task 9: Food Hazard Detection.
