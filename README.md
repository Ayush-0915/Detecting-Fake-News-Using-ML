# 📰 Fake News Detection Using Machine Learning

A machine learning project that classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP) and supervised learning algorithms. The project compares the performance of **Logistic Regression**, **Multinomial Naive Bayes**, and **Random Forest** on a publicly available news dataset.

## 📌 Project Overview

The rapid spread of misinformation on digital platforms has created a growing need for automated fake news detection systems. This project uses text preprocessing and TF-IDF feature extraction to transform news articles into numerical representations and evaluates multiple machine learning models for binary classification.

## 🚀 Features

- Preprocessing of news article text
- TF-IDF feature extraction
- Comparison of three machine learning algorithms:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Random Forest
- Performance evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
- Prediction of unseen news articles

## 📂 Dataset

The project uses the **Fake and Real News Dataset** from Kaggle.

- `Fake.csv` – Fake news articles
- `True.csv` – Real news articles

Combined dataset statistics:
- Total Articles: **44,898**
- Fake Articles: **23,481**
- Real Articles: **21,417**

Dataset Link:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## 🔄 Workflow

1. Load Fake and Real news datasets
2. Assign binary labels
3. Merge and shuffle the data
4. Preprocess text (cleaning and normalization)
5. Combine title and article content
6. Convert text into TF-IDF features
7. Split data into training and testing sets
8. Train multiple machine learning models
9. Evaluate model performance
10. Compare results and make predictions

## 🤖 Models Evaluated

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest

## 📊 Performance Summary

| Model | Accuracy |
|---------|----------|
| Random Forest | **99.51%** |
| Logistic Regression | **98.54%** |
| Multinomial Naive Bayes | **94.06%** |

The Random Forest classifier achieved the best overall performance on the test dataset.


```

4. Run all cells in `Fake_News_Detection.ipynb`.

## 📈 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix


## 👨‍💻 Author

Ayushh 

## 📜 License

This project is intended for educational and research purposes.