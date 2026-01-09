# ClusterRAG (Anonymous Repository)

This repository contains the implementation of **ClusterRAG**, a collaborative and hybrid user profiling framework for personalized text generation. ClusterRAG enhances retrieval-augmented generation by clustering similar users and leveraging both individual and collaborative profiles during generation.

The code is evaluated on the **LaMP benchmark**, following the experimental setup described in the accompanying paper.

---

## Dataset: LaMP Benchmark

ClusterRAG is evaluated on the **LaMP (Language Model Personalization) dataset**.

**Official LaMP dataset repository:**  
https://github.com/LaMP-benchmark/LaMP

### Dataset Setup (Required)

1. Download the LaMP dataset from the official repository.
2. Paste the downloaded files into the corresponding subdirectories inside the `data/` folder of this repository.
3. Ensure the directory structure matches the expected LaMP format before running any scripts.

> **Important:** The code will not run correctly unless the LaMP dataset is downloaded and placed in the correct subdirectories under `data/`.

---

## Environment Setup

- Python ≥ 3.8  
- PyTorch  
- HuggingFace Transformers  
- ColBERT  
- HDBSCAN  
- NumPy, pandas, scikit-learn  

Dependencies can be installed based on the imports used in the scripts.

---

## Execution Steps

All commands should be executed from the repository root.

### Step 1: Generate User Sets
```bash
python user_embed/get_user_set.py --task LaMP_1 --use_date
