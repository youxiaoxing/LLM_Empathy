# LLM Empathy Evaluation Framework

This repository contains the implementation and evaluation scripts for the paper **"Position: LLMs Should Incorporate Explicit Mechanisms for Human Empathy"**.

The framework evaluates Large Language Models (LLMs) across three dimensions of empathy—**Cognitive, Cultural, and Relational**—and identifies specific failure mechanisms using novel metrics alongside standard benchmarks.

## 📂 Project Structure

```text
.
├── news_generation.py              # Cognitive Empathy Task (Newsroom)
├── literary_translation.py         # Cultural Empathy Task (WMT23)
├── pyschological_counseling.py     # Relational Empathy Task (Counseling)
│
└── metrics/                        # Evaluation Scripts
    ├── SA_score.py                 # Sentiment Attenuation (SA) Metric
    ├── EGM_score.py                # Empathic Granularity Mismatch (EGM) Metric
    ├── LD_score.py                 # Linguistic Distancing (LD) Metric
    ├── CA_score.py                 # Conflict Avoidance (CA) Metric
    ├── chrF_comet.py               # Standard Translation Metrics
    ├── run_empathy_epitome_hf.py   # Standard Counseling Metrics (EPITOME)
    ├── semantic_similarity.py      # Semantic preservation (Embedding Cosine Sim)
    └── ROUGE.py                    # Standard Summarization Metrics (ROUGE/CIDEr)

```

## 🛠️ Prerequisites

Please install the required Python libraries before running the scripts:

```bash
pip install openai tqdm spacy nltk numpy scipy pandas torch transformers sacrebleu unbabel-comet
python -m spacy download en_core_web_sm

```

> **Note:** Most scripts require an OpenAI API key (or compatible endpoint) to function, as GPT-4o/GPT-5.2 is used as a judge for several metrics. Please set your `api_key` in the respective Python files.

---

## 🚀 1. Task Generation

These scripts generate model outputs for the three empathy dimensions.

### **Cognitive Empathy: News Generation**

* **File:** `news_generation.py`
* **Description:** Generates full news articles based on high-impact summaries from the **Newsroom** dataset. It tests if the model preserves the informational salience and urgency of the original event.
* **Input:** `./dataset/newsroom.json`
* **Output:** `./result/article_generation.jsonl`

### **Cultural Empathy: Literary Translation**

* **File:** `literary_translation.py`
* **Description:** Performs discourse-level literary translation (Chinese to English) using the **WMT2023** dataset. It tests the model's ability to preserve narrative voice and cultural context.
* **Input:** `./dataset/wmt23.json`
* **Output:** `./result/wmt23_translation_[model]_results.jsonl`

### **Relational Empathy: Psychological Counseling**

* **File:** `pyschological_counseling.py`
* **Description:** Generates therapist-style responses to client narratives (e.g., workplace bullying, relationship betrayal). It tests the model's ability to engage with conflict and valid affect.
* **Input:** `./dataset/data.json`
* **Output:** `./result/[model].jsonl`

---

## 📊 2. Evaluation Metrics

The `metrics/` folder contains scripts to calculate both standard performance metrics and the specific empathy failure mechanisms.

### **A. Empathy Failure Mechanisms**

#### **1. Sentiment Attenuation (SA)**

* **File:** `metrics/SA_score.py`
* **Mechanism:** Measures if the model systematically downscales emotional intensity compared to the reference.
* **Method:** Uses LLM-as-a-judge to score emotional intensity on a scale of 0.0–1.0 for both reference and generation, then calculates the compression rate.

#### **2. Empathic Granularity Mismatch (EGM)**

* **File:** `metrics/EGM_score.py`
* **Mechanism:** Detects miscalibration in the level of disturbing or graphic detail (Cognitive Empathy).
* **Method:** Counts "aversive" elements (violence, injury, gore) per sentence and calculates the density difference between reference and output.

#### **3. Linguistic Distancing (LD)**

* **File:** `metrics/LD_score.py`
* **Mechanism:** Quantifies the use of abstract, depersonalized language (Cultural Empathy).
* **Method:** Uses `spaCy` and `WordNet` to analyze:
* Agentless passive voice density.
* Nominalization density.
* Impersonal subject ratios.



#### **4. Conflict Avoidance (CA)**

* **File:** `metrics/CA_score.py`
* **Mechanism:** Measures the tendency to prematurely resolve tension or offer solutions instead of exploring the conflict (Relational Empathy).
* **Method:** A binary classification metric using an LLM judge.

### **B. Standard Quality Metrics**

* **`metrics/chrF_comet.py`**: Calculates **chrF**, **COMET**, and **TER** scores. Used primarily for the Literary Translation task.
* **`metrics/run_empathy_epitome_hf.py`**: Uses Hugging Face models to calculate **EPITOME** scores (Emotional Reactions, Interpretations, Explorations). Used for the Counseling task.
* **`metrics/semantic_similarity.py`**: Computes Cosine Similarity using **Qwen3-Embedding** to ensure the model outputs remain semantically faithful to the reference (even if empathy fails).
* **`metrics/ROUGE.py`**: Computes **ROUGE** and **CIDEr** scores. Used for News Generation.

---
