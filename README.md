# Grammatical Error Correction using Synthetic Data Augmentation with T5

This project explores **Grammatical Error Correction (GEC)** using **synthetic data augmentation** and **T5 transformer models**. Traditional GEC systems often rely on manually annotated datasets (e.g., CoNLL-2014, JFLEG), which are costly to create and limited in error coverage. Our approach trains a GEC model using **synthetically corrupted sentences** paired with their clean originals, then fine-tunes **T5-small** in a text-to-text setup to learn to reverse those errors.

The core hypothesis is that a model trained *only on synthetic data* can learn strong correction behavior and generalize beyond the specific injected noise patterns—while also revealing the limitations of synthetic-only training.

---

## Authors

- **Meghna Sharma** — ms16005@nyu.edu  
- **Preethika Chennareddy** — pc3521@nyu.edu  
- **Ritvik Vasantha Kumar** — rv2459@nyu.edu  

Department of Computer Science, New York University

---

## Problem Statement

**Grammatical Error Correction (GEC)** aims to automatically detect and correct grammatical, lexical, and punctuation errors in text. Many existing systems depend on curated human-labeled datasets, which can reduce robustness to unseen styles, domains, and error types.

This project proposes a **synthetic-data-driven GEC pipeline**:
1) sample clean English sentences (C4),
2) inject realistic noise via probabilistic transformations,
3) train a pretrained T5 model to map **noisy → clean**.

---

## Dataset

- **Source**: Clean English sentences sampled from the **C4 corpus**
- **Format**: `(noisy_sentence, clean_sentence)`
- **Splits (as used in this notebook)**: train / validation / test (see notebook for exact counts)

### Synthetic Noise Generation

Noise is injected probabilistically, including:
- Morphological errors (tense, plurality, agreement)
- Syntactic errors (word order, missing function words)
- Lexical substitutions
- Punctuation and casing noise

**Limitation:** synthetic noise may not perfectly match human error distributions, which can bias the model toward specific transformation patterns.

---

## Model

We fine-tune **T5-small** (encoder–decoder) using a sequence-to-sequence objective.

High-level pipeline:
1. **Noisy sentence** input
2. Tokenization via T5 tokenizer
3. Encoder produces contextual representations
4. Decoder generates **corrected text**

Training is done with Hugging Face `Trainer` (standard seq2seq fine-tuning).

---

## Repository Structure

This repo contains:
- `dl_final_project_gec.ipynb`  
  Main training + evaluation notebook (data loading, preprocessing, training, decoding, metrics, analysis plots).

- `t5-small-gec-finetuned-10/`  
  Saved fine-tuned model artifacts:
  - `config.json`, `generation_config.json`
  - `model.safetensors`
  - tokenizer files (`spiece.model`, `tokenizer.json`, etc.)
  - `training_args.bin`

- `t5-small-gec-finetuned-10/gec_eval/`  
  Evaluation outputs saved from the notebook:
  - `pred_texts.txt` (model predictions)
  - `label_texts.txt` (references)
  - `metrics.json` (aggregated metrics)

---

## How to Run

Open and run the notebook:

1. `dl_final_project_gec.ipynb`
2. Train the model (or load from the saved checkpoint folder)
3. Run evaluation to generate:
   - BLEU / chrF
   - copy rate
   - edit-distance based diagnostics
   - qualitative samples and worst cases

> Tip: For faster training and generation, use an A100 GPU if available.

---

## Evaluation

We evaluate with both **standard text overlap metrics** and **edit-based diagnostics** to understand *how the model edits*.

### Primary Metrics
- **BLEU (sacrebleu)**: n-gram overlap with reference corrections  
- **chrF**: character n-gram F-score (often more forgiving for GEC)

### Behavioral / Diagnostic Metrics
- **Copy rate**: fraction where `prediction == input` (useful for detecting under-correction)
- **Edit distance distributions**:
  - baseline: `edit_distance(input, target)`
  - model: `edit_distance(prediction, target)`
- **Edit-distance improvement**:
  - Δ = baseline − model (positive = improvement)
- **Edit amount**:
  - `edit_distance(prediction, input)` (how aggressively the model rewrites)
- **Difficulty buckets** (optional):
  - performance by edit-density bins to see degradation on harder examples

### Reported Results (T5-small, 10 epochs run)
From the final run logged in the notebook:
- **BLEU (sacrebleu)**: ~68.21  
- **chrF**: ~84.04  
- **Copy rate**: ~0.226  
- **Normalized edit distance (pred→target)**: ~0.138  

Exact match is reported as a *secondary* signal only, since multiple correct outputs may not match the single reference exactly.

---

## Qualitative Analysis

The notebook prints random test examples showing:
- cases where the model fixes small grammar/punctuation errors correctly,
- under-correction cases where the model copies the input despite target differences,
- difficult/noisy outliers (“worst examples”) where the model struggles or produces awkward rewrites.

This qualitative section is important because reference-based metrics alone don’t capture “valid alternative corrections.”

---

## Key Takeaways

- The model achieves strong chrF and solid BLEU on synthetic GEC test data, and edit-distance diagnostics show improvements over the raw noisy baseline for many samples.
- The dominant remaining failure mode is typically **undercorrection** (copying too often / making too few edits) rather than heavy overcorrection.
- Synthetic-only training is effective, but distribution mismatch between synthetic noise and real human errors remains a core limitation.

---

## Acknowledgements

- Hugging Face Transformers / Datasets / Evaluate
- Google T5
- C4 dataset
