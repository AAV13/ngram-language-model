# N-Gram Language Modeling and Evaluation

This project provides a Python implementation for building and evaluating various **N-gram language models** from scratch.  
It explores the impact of N-gram order, the necessity of smoothing, and compares several common techniques for handling data sparsity, including **Add-1 Smoothing**, **Linear Interpolation**, and **Stupid Backoff**.  

All models are trained and evaluated on the **Penn Treebank (PTB)** dataset.

Dataset Link: https://www.kaggle.com/datasets/aliakay8/penn-treebank-dataset

---

## 🚀 Features

* **N-gram Models**: Implements models for any N-gram order (Unigram, Bigram, Trigram, etc.).
* **Maximum Likelihood Estimation (MLE)**: Basic unsmoothed probability estimation.
* **Smoothing Techniques**:
  * Add-1 (Laplace) Smoothing
  * Linear Interpolation (with hyperparameter tuning on a validation set)
  * Stupid Backoff
* **Evaluation**: Calculates perplexity to measure model performance.
* **Text Generation**: Generates new text sequences using a trained model.

---

## 📁 Project Structure

The project is organized into a modular structure for clarity and maintainability.

```plaintext
ngram-language-model/
├── data/                     #Contains the Penn Treebank dataset files
├── src/                      #All source code for the project
│   ├── preprocess.py         #Data loading and tokenization logic
│   ├── language_model.py     #Core NgramLM class
│   ├── train.py              #Script to train and save models
│   ├── evaluate.py           #Script to evaluate models and get perplexity
│   └── generate.py           #Script for text generation
├── notebooks/
│   └── analysis.ipynb        #Exploratory Data Analysis (EDA) of the dataset
├── README.md                 #This file
└── Report.pdf                #The final project report and analysis
```

---

## ⚙️ Setup Instructions

Follow these steps to set up the project environment.

### 1. Clone the Repository

```bash
git clone <your-repository-link>
cd ngram-language-model
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

**MacOS/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install numpy nltk tqdm
```

---

## 🧠 How to Run & Reproduce Results

All scripts must be run as modules from the project’s **root directory**.  
The following commands will reproduce all the key results from `report.md`.

### 1. Training the Models

Train and save models for N = 1 through N = 4.

```bash
#Train Unigram (N=1)
python -m src.train --train-file data/ptb.train.txt --model-file unigram.pkl --n 1

#Train Bigram (N=2)
python -m src.train --train-file data/ptb.train.txt --model-file bigram.pkl --n 2

#Train Trigram (N=3)
python -m src.train --train-file data/ptb.train.txt --model-file trigram.pkl --n 3

#Train Four-gram (N=4)
python -m src.train --train-file data/ptb.train.txt --model-file fourgram.pkl --n 4
```

---

### 2. Evaluating Maximum Likelihood Estimation (MLE) Models

Evaluate each saved model using MLE to get perplexity scores for the first table in `report.md`.

```bash
#Evaluate Unigram MLE
python -m src.evaluate --model-file unigram.pkl --test-file data/ptb.test.txt --smoothing mle

#Evaluate Bigram MLE
python -m src.evaluate --model-file bigram.pkl --test-file data/ptb.test.txt --smoothing mle

#Evaluate Trigram MLE
python -m src.evaluate --model-file trigram.pkl --test-file data/ptb.test.txt --smoothing mle

#Evaluate Four-gram MLE
python -m src.evaluate --model-file fourgram.pkl --test-file data/ptb.test.txt --smoothing mle
```

---

### 3. Evaluating Smoothed Trigram Models

Use the `trigram.pkl` model to evaluate different smoothing and backoff strategies.

```bash
#Evaluate Add-1 (Laplace) Smoothing
python -m src.evaluate --model-file trigram.pkl --test-file data/ptb.test.txt --smoothing add1

#Evaluate Stupid Backoff (with alpha=0.4)
python -m src.evaluate --model-file trigram.pkl --test-file data/ptb.test.txt --smoothing backoff --alpha 0.4

#Evaluate Linear Interpolation with optimal lambdas
#Note: These lambdas were found by testing on data/ptb.valid.txt first.
python -m src.evaluate --model-file trigram.pkl --test-file data/ptb.test.txt --smoothing interpolation --lambdas "0.2,0.5,0.3"
```

---

### 4. Generating Text

Generate sample text using the best-performing model (Trigram with Stupid Backoff).

```bash
#Generate 5 sentences
python -m src.generate --model-file trigram.pkl --smoothing backoff --num-sentences 5
```

---

## 📊 Results

The final analysis and a detailed comparison of all models can be found in [`Report.pdf`](./Report.pdf).  
The best-performing model was the **Trigram model with Stupid Backoff**, achieving a **perplexity of 188.11** on the test set.
