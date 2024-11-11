# LCE-Multi-Head-Attention

## Table of Contents
1. [Project Overview](#project-overview)
2. [Datasets](#datasets)
3. [Setup and Installation](#setup-and-installation)
4. [Python Files and Their Purpose](#python-files-and-their-purpose)
5. [Results and Evaluation](#results-and-evaluation)

## Project Overview
We used these models:
1. **Traditional Models**: TF-IDF with Cosine Similarity, Word2Vec embeddings with Cosine Similarity
3. **Transformer Models**: BERT, RoBERTa, and XLNet without attention optimization (using pre-trained models from Hugging Face)
4. **Proposed Model**: Fine-tuned model with multi-head attention, optimized with varying attention head configurations

Evaluation metrics include accuracy, precision, recall, F1-score, and AUC-ROC. We conducted experiments on the Quora Question Pairs (QQP) dataset as the primary dataset and tested on the StackExchange dataset for cross-domain validation.

## Datasets
### 1. Quora Question Pairs (QQP)
- **Description**: QQP consists of about 400,000 question pairs labeled as duplicate or non-duplicate.
- **Access**: We used the dataset directly from Hugging Face.
- **Hugging Face Path**: `AlekseyKorshuk/quora-question-pairs`

### 2. StackExchange Question Pairs
- **Description**: This dataset contains question pairs from StackExchange forums, reflecting a more domain-specific vocabulary. It's used for cross-domain validation.
- **Access**: It can be accessed from kaggle 

## Setup and Installation
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/manal1409/LCE-Multi-Head-Attention
    cd project_name
    ```

2. **Install Dependencies**:
    It requires Python 3.10+ and the libraries mentioned in the requirements:
   
    - scikit-learn
    - torch torchvision torchaudio
    - gensim
    - transformers
    - datasets

    ```bash
    pip install -r requirements.txt
    ```
If you got error on PyTorch then install this one as well:
pip install "accelerate>=0.26.0"

4. **Download Word2Vec Model** (if required):
    - Run `model_weights.py` to download and prepare the Google News Word2Vec embeddings.
    - This file will download the embeddings file and save it as `GoogleNews-vectors-negative300.bin`.

5. **Setup Hugging Face Datasets and Models**:
    - Ensure that the `transformers` and `datasets` libraries are installed.
    - It will automatically download pre-trained models from Hugging Face.

## Python Files and Their Purpose

| File             | Description |
|------------------|-------------|
| **main.py**      | Main file to run experiments on QQP and StackExchange datasets. Runs TF-IDF, Word2Vec, Transformer models and Proposed Model experiments. |
| **models.py**    | Contained model, TraditionalModels (TF-IDF, Word2Vec), TransformerModel (BERT, RoBERTa, XLNet), and ProposedModel with multi-head attention. Including fine-tuning and evaluation methods for each model. |
| **dataset.py**   | Handle dataset loading and preprocessing. Use Hugging Face `datasets` to load QQP and StackExchange data, creating PyTorch datasets. |
| **experiments.py** | Functions to run each experiment separately. Call models from `models.py` and prepares results. |
| **metrics.py**   | It contain custom evaluation metric functions to calculate accuracy, precision, recall, F1-score, and AUC-ROC. Used with Hugging Face’s `Trainer` to compute metrics after evaluation. |
| **model_weights.py** | To download the Google News Word2Vec embeddings and saving as `GoogleNews-vectors-negative300.bin`. |

## Results and Evaluation
| Model       | Dataset     | Accuracy | Precision | Recall | F1-Score |
|-------------|-------------|----------|-----------|--------|----------|
| TF-IDF      | QQP         | 0.00     | 0.00     | 0.00   | 0.00     |
| Word2Vec    | QQP         | 0.00     | 0.00      | 0.00   | 0.00     |
| BERT        | QQP         | 0.00     | 0.00      | 0.00   | 0.00     |
| RoBERTa     | QQP         | 0.00     | 0.00      | 0.00   | 0.00     |
| Proposed    | QQP         | 0.00     | 0.00      | 0.00   | 0.00   |

## Author 
Manal 
