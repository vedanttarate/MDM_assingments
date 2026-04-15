Lab Assignment 4 - NLP Preprocessing and Text Classification

Overview
This lab implements a complete NLP workflow on the 20 Newsgroups dataset. It
covers text preprocessing, feature engineering with TF-IDF and CountVectorizer,
classical machine learning baselines, and three deep learning architectures.
Results are evaluated with accuracy, precision, recall, F1-score, and
confusion matrices, with additional error analysis and interpretability.

Objectives

- Apply NLP preprocessing: cleaning, tokenization, stopword removal, stemming,
  and lemmatization.
- Compare vectorization methods (TF-IDF vs CountVectorizer).
- Train and evaluate multiple classical ML models.
- Build and evaluate deep learning models (MLP, CNN, BiLSTM).
- Perform error analysis and provide model interpretability.

Notebook

- NLP_Lab4.ipynb

Dataset

- 20 Newsgroups (subset of 6 categories to keep training manageable).
- The dataset is fetched via scikit-learn.
- Important: scikit-learn assigns numeric labels alphabetically by category
  name; the notebook uses target_names to preserve correct label mapping.

Environment and Dependencies
Python 3.9+ is recommended. The notebook installs these packages:

- nltk
- scikit-learn
- matplotlib
- seaborn
- wordcloud
- tensorflow
- keras

NLTK resources downloaded in the notebook:

- punkt
- stopwords
- wordnet
- averaged_perceptron_tagger
- punkt_tab

How to Run

1. Open NLP_Lab4.ipynb.
2. Run the install cell once to install dependencies and download NLTK data.
3. Run the remaining cells in order.

Execution Notes

- The deep learning section can be time-consuming on CPU. If needed, reduce
  epochs or use a GPU-enabled runtime.
- The dataset is downloaded automatically; ensure network access is available.

Notebook Structure (Section Guide)

1. Install and import dependencies
2. Dataset loading and EDA
3. NLP preprocessing pipeline
4. Text vectorization (TF-IDF and CountVectorizer)
5. Classical ML models
6. Deep learning models
7. Comprehensive model comparison
8. Error analysis and interpretability
9. Live prediction demo
10. Summary and discussion

Models Implemented
Classical ML

- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM
- SGD Classifier
- Random Forest

Deep Learning

- MLP (Embedding + GlobalMaxPooling)
- 1D CNN (multi-layer conv + pooling)
- Bidirectional LSTM

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score (weighted)
- Confusion matrices (raw and normalized)

Key Outputs

- EDA plots (class distribution, text length stats)
- Word clouds and top tokens per class
- ML model comparison table and charts
- Confusion matrices for best ML and all DL models
- Per-class F1 comparison
- Error analysis with misclassified examples

Reproducibility

- Random seeds are set where supported to improve repeatability.
- Results can vary slightly due to randomized training and data split.

Credits
Author: Rohit Thorat
Roll No: 202402070041
Date: 10 April 2026
