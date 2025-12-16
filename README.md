# DL_FinalProject
# Grammatical Error Correction using Synthetic Data Augmentation with T5

## Project Overview

This project explores **Grammatical Error Correction (GEC)** using **synthetic data augmentation** and **T5 transformer models**. Traditional GEC systems depend heavily on manually annotated datasets (e.g., CoNLL-2014, JFLEG), which are expensive to create and limited in coverage.

Our approach constructs a large-scale **synthetic GEC dataset** from clean English text (C4 corpus) by injecting realistic grammatical, syntactic, morphological, and punctuation errors. We then fine-tune pretrained **T5-small** and **T5-base** models in a text-to-text framework to learn to correct these errors.

The key hypothesis is that a model trained *only on synthetic data* can achieve competitive GEC performance without relying on human-labeled datasets.

---

## Authors

* **Meghna Sharma** ([ms16005@nyu.edu](mailto:ms16005@nyu.edu))
* **Preethika Chennareddy** ([pc3521@nyu.edu](mailto:pc3521@nyu.edu))
* **Ritvik Vasantha Kumar** ([rv2459@nyu.edu](mailto:rv2459@nyu.edu))

Department of Computer Science, New York University

---

## Problem Statement

Grammatical Error Correction (GEC) aims to automatically detect and correct grammatical, lexical, and punctuation errors in text. Existing systems rely on curated datasets with limited diversity, reducing generalization to unseen errors.

This project proposes a **synthetic-data-driven GEC pipeline** that reverses artificially injected errors using a pretrained T5 model, eliminating the need for costly human annotation.

---

## Dataset Description

* **Source**: Clean English sentences sampled from the **C4 corpus**
* **Size**: 100,000 sentence pairs
* **Format**: `(noisy_sentence, clean_sentence)`
* **Split**:

  * Training set: 80,000 pairs
  * Test set: 20,000 pairs

### Synthetic Noise Generation

Errors are injected probabilistically, including:

* Morphological errors (tense, plurality)
* Syntactic errors (word order, missing function words)
* Lexical substitutions
* Punctuation and casing noise

**Limitation**: Synthetic noise patterns may not perfectly reflect real human errors, potentially introducing bias.

---

## Model Architecture

We fine-tune pretrained **T5-small** and **T5-base** encoder–decoder models in a sequence-to-sequence setup.

**Pipeline**:

1. Noisy sentence input
2. Tokenization using T5 tokenizer
3. Encoder stack processes contextual representations
4. Decoder generates corrected text

**Training Details**:

* Loss: Token-level cross-entropy with label smoothing
* Optimizer: AdamW
* Learning-rate scheduling
* Gradient clipping
* Regularization: Dropout and early stopping
* Parameter-efficient tuning: **LoRA** explored in ablation studies

---

## Evaluation Metrics

Model performance is evaluated using:

* **BLEU**
* **GLEU**
* **chrF**

Additionally, qualitative analysis is conducted to inspect corrected structures and common failure cases.

---

## Expected Results

We expect models trained entirely on synthetic data to achieve **comparable accuracy** to real-data GEC baselines. We anticipate:

* Noticeable BLEU improvement over baseline T5
* Strong fluency and grammatical consistency
* Insights into the generalization limits of synthetic-only training

---

## Requirements

* Python 3.8+
* PyTorch
* HuggingFace Transformers
* Datasets
* SentencePiece
* NumPy, Pandas

Install dependencies:

```bash
pip install -r requirements.txt
```

---


## Conclusion

This project demonstrates the feasibility of training competitive GEC models using **synthetic data augmentation alone**, reducing reliance on human-annotated datasets and enabling scalable grammar correction systems.

---

## Acknowledgements

* HuggingFace Transformers
* Google T5
* C4 Dataset
