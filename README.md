# Financial Document Causality Detection 📊🔍

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow?style=for-the-badge)
![Domain](https://img.shields.io/badge/Domain-FinTech_%7C_NLP-8A2BE2?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A state-of-the-art NLP system designed to extract causal relationships and financial figures from complex financial reports. This project treats causality detection as an **Extractive Question Answering (QA)** task, leveraging an ensemble of Transformer models enhanced with **Neuro-Symbolic Guardrails** for high-precision information extraction.

---

## 📖 Project Overview

Financial documents are dense with causal explanations (e.g., *"Profit declined **due to increased operational costs**"*). Standard language models often struggle to distinguish between the *cause* and the *context*, or between similar numerical figures.

This solution approaches the problem by combining **deep learning** (for semantic understanding) with **symbolic logic** (for structural validation).

### Key Features

* **Ensemble Architecture**: Combines **RoBERTa** (Contextual extraction) and **DeBERTa** (Disentangled attention) predictions.
* **Neuro-Symbolic Guardrails**: Post-processing logic that applies negative constraints, regex pattern matching, and question-type detection to filter out "hallucinated" or grammatically incorrect answers.
* **Factoid vs. Causal Detection**: Automatically detects if a question asks for a number ("How much?") or a reason ("Why?") and adjusts extraction probability accordingly.
* **Domain Adaptive Pre-Training (DAPT)**: Models were further pre-trained on financial corpora to understand domain-specific jargon before fine-tuning.

---

## 🛠️ Methodology & Architecture

### 1. Individual Models

We trained and evaluated three distinct standalone models to handle the task:

* **Custom BERT (The Baseline)**
* *Architecture*: A BERT-medium sized model trained completely from scratch.
* *Purpose*: To test if domain-specific pre-training on financial text alone is sufficient without starting from a massive general-purpose checkpoint.
* *Outcome*: Proved that "Transfer Learning" (using pre-trained weights) is essential for high performance on small datasets.


* **RoBERTa-Base (The "Parrot")**
* *Architecture*: A robustly optimized BERT approach.
* *Training*: We applied **Domain Adaptive Pre-Training (DAPT)** on the financial corpus first (100 epochs), then fine-tuned it on the QA pairs.
* *Strength*: Excellent at syntactic matching and extracting answers that look like standard English spans.


* **DeBERTa-v3 (The "Logician")**
* *Architecture*: Uses disentangled attention and a differing pre-training objective (RTD).
* *Training*: Fine-tuned on a strictly filtered "clean" subset of the data to avoid learning noise from the dataset.
* *Strength*: Superior at reasoning over complex sentence structures and long-range dependencies.



### 2. The Ensemble Strategy

The final production system combines the **already trained** RoBERTa and DeBERTa checkpoints to create a stronger predictor.

* **Logit Averaging**: We compute a weighted average of the start/end logits (55% RoBERTa / 45% DeBERTa).
* **Neuro-Symbolic Guardrails**: The raw ensemble output is passed through a logic layer (Regex, Question Type Detection) to filter out common errors.

1. **Is it a Factoid?**
* Uses `Sentence-BERT` (`all-MiniLM-L6-v2`) to compute cosine similarity (> 0.4) against the anchor question: *"What is the financial value or amount?"*.
* If identified as a factoid, the system boosts candidates containing currency symbols ($£€) or percentages (%).


2. **Negative Constraints**
* **Bad Starts**: Penalizes answers starting with explanatory or comparative phrases like *"marking"*, *"reflecting"*, or *"due to"*.
* **Length Rules**: Factoid answers > 8 words are penalized. General answers > 60 tokens are rejected.


3. **Comparative Penalty**
* Prevents selecting "previous year's value" instead of the current value by penalizing phrases like *"previous"*, *"prior"*, or *"vs"*.



---

## ⚙️ Hyperparameters

| Parameter | Custom BERT | RoBERTa (DAPT + QA) | DeBERTa (QA) |
| --- | --- | --- | --- |
| **Model Architecture** | 6 Layers, 8 Heads | `roberta-base` | `deberta-v3-base` |
| **Pre-Training Epochs** | 8 (MLM) | 100 (MLM) | N/A |
| **Fine-Tuning Epochs** | 30 | 10 | 10 |
| **Batch Size** | 16 | 2 | 4 (Grad Accum=2) |
| **Learning Rate** | 5e-5 | 3e-5 | 2e-5 |

---

## 📊 Performance Results

The **Ensemble** approach demonstrates superior reliability, particularly in **Exact Match (EM)** accuracy.

| Model | Exact Match (EM) | Semantic Score (SAS) | Description |
| --- | --- | --- | --- |
| **Ensemble (Final)** | **87.44%** | 0.9424 | **Best Overall.** |
| **RoBERTa (Base)** | 82.41% | **0.9528** | Strong baseline. |
| **DeBERTa (v3)** | 77.89% | 0.9522 | Good reasoning. |
| **Custom BERT** | 9.55% | 0.6586 | *Baseline Experiment*. |

> **Note**: *Semantic Answer Similarity (SAS)* is calculated using the `all-MiniLM-L6-v2` encoder to ensure the predicted answer carries the same meaning as the ground truth, even if the wording differs slightly.

---

## 📂 Project Structure

```bash
📦 Financial-Causality-Detection
 ┣ 📂 Data
 ┃ ┣ 📜 training_data_en.csv    # Training dataset
 ┃ ┗ 📜 evaluation_data_en.csv  # Blind test dataset
 ┣ 📂 Notebooks
 ┃ ┣ 📜 Ensemble.ipynb          # 🚀 MAIN: Ensemble logic & guardrails
 ┃ ┣ 📜 RoBERTa.ipynb           # Training: DAPT + QA Fine-tuning
 ┃ ┣ 📜 DeBERTa.ipynb           # Training: Strict filtering
 ┃ ┗ 📜 custom_BERT.ipynb       # Experimental baseline

```

---

## 🚀 Usage

### Prerequisites

* Python 3.8+
* GPU with CUDA support is highly recommended for training and inference.

### Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt

```

*(If `requirements.txt` is not present, install: `torch`, `transformers`, `pandas`, `numpy`, `sentence-transformers`, `datasets`, `tqdm`.)*

### Running the Code

1. **Prepare Data**: Ensure the `.csv` files are in the `Data/` directory.
2. **Generate Model Checkpoints**:
* Run `RoBERTa.ipynb` to save the `fin-qa-dapt` model directory.
* Run `DeBERTa.ipynb` to save the `deberta_sarang_plus` model directory.


3. **Run Inference**: Execute `Ensemble.ipynb`. This notebook loads the checkpoints created in step 2 to execute the final high-precision logic.

---

## 📝 License

This project is licensed under the MIT License.
