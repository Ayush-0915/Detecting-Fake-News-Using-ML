# 📰 AI-Powered Fake News Detection using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-blueviolet?style=for-the-badge)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![ML](https://img.shields.io/badge/Machine%20Learning-Supervised-blue?style=for-the-badge)](https://en.wikipedia.org/wiki/Machine_learning)

An interactive, high-end AI SaaS application designed to combat digital misinformation. This project utilizes Natural Language Processing (NLP) and supervised Machine Learning algorithms to classify news articles as **Fake** or **Real** through a stunning, recruiter-ready dashboard.

---

## 📌 Overview

With the rapid proliferation of information on digital channels, verifying article authenticity is more critical than ever. This project implements a full NLP model engineering workflow—incorporating text cleaning, stop-word removal, and TF-IDF term frequency-inverse document frequency coordinate mappings—to feed high-accuracy classifiers. 

The dashboard provides a futuristic **Gen Z dark-themed SaaS aesthetic** with rotating vector SVG loaders, real-time typography metrics, multi-model evaluation visualizations, and confidence indicators.

---

## ⚡ Features

- 🧠 **Live Fake News Detector**: Input any custom article body or headline to run real-time inference.
- ⏱️ **Real-Time Word Metrics**: Automatic analysis of Character Count, Word Count, and Estimated Reading Time.
- 🔮 **Model Confidence Gauge**: Dynamic circular Plotly gauge illustrating classifier prediction weights.
- 📖 **AI Decision Summary**: Concise steps explaining the cleaning, TF-IDF vectorization, and Random Forest decision paths.
- 📊 **Dataset Analytics**: Interactive data exploration featuring category distributions, length histograms, and responsive tables.
- ⚔️ **Model Comparison**: Interactive toggle between grouped bar charts and radar grids comparing accuracy, recall, precision, and F1 metrics.
- 🌀 **Animated Splash Screen**: Startup experience displaying typewriter logo rotations and dot loaders.
- 🌌 **Premium UI/UX**: Fluid page transitions, hardware-accelerated background particles, card lifting animations, and Lucide SVG icons.
- 📱 **Fully Responsive Layout**: Stacks timelines vertically and wraps metric rows dynamically on tablet and mobile viewports.

---

## 📂 Dataset

The machine learning models are trained and evaluated using the **Fake and Real News Dataset** from Kaggle:
- **Total Articles**: 44,898 articles
- **Fake Articles**: 23,481 records (`Fake.csv`)
- **Real Articles**: 21,417 records (`True.csv`)
- **Source**: [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## 🛠️ Tech Stack

- **Core Logic**: Python 3.9+
- **Data Engineering**: Pandas, NumPy
- **Machine Learning & NLP**: Scikit-learn (TF-IDF Vectorizer, Random Forest, Logistic Regression, Multinomial Naive Bayes)
- **Interactive Visualizations**: Plotly, Matplotlib
- **Web Interface**: Streamlit (with custom HTML/CSS injections)

---

## 📊 Model Performance

Multiple machine learning algorithms were trained and benchmarked. The **Random Forest Classifier** achieved the highest accuracy:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| 🏆 **Random Forest** | **99.51%** | **99.42%** | **99.56%** | **99.49%** |
| 🥈 **Logistic Regression** | 98.54% | 98.15% | 98.81% | 98.48% |
| 🥉 **Multinomial Naive Bayes** | 94.06% | 93.91% | 93.63% | 93.77% |

*Note: Random Forest is selected as the primary prediction model inside the Live Prediction engine.*

---

## 🔄 Machine Learning Pipeline

```
  [ Raw Dataset ]
         │
         ▼
  [ Text Cleaning ] ───► Lowercasing, punctuation stripping, digits & URL removal
         │
         ▼
[ TF-IDF Vectorizer ] ──► Map content to 61,913 high-dimensional features
         │
         ▼
  [ Model Training ] ───► Random Forest Ensemble Fit (100 Decision Trees)
         │
         ▼
  [ Prediction ] ──────► Compute class outcomes and confidence intervals
         │
         ▼
  [ Evaluation ] ──────► Benchmarking (Confusion Matrix & classification metrics)
```

---

## 🖥️ Dashboard Interface

The application is structured into multiple functional modules accessible via the custom sidebar navigation menu:
1. **Home**: High-impact greeting banner, real-time KPI overview card strip, and animated pipeline flow steps.
2. **Dataset Analytics**: Interactive data grid, category breakdown charts, and length distribution histograms.
3. **Model Comparison**: Deep-dive performance leaderboard table, confusion matrices, and interactive Plotly metric plots.
4. **Live Prediction**: Input field with helper sample buttons, real-time typing analytics, and a confidence dial showing decision summaries.
5. **About**: Details regarding technical hyper-parameters, database metrics, and developer references.

---

## 🚀 Installation & Setup

Follow these commands to deploy and run the app locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ayush-0915/Detecting-Fake-News-Using-ML.git
   cd Detecting-Fake-News-Using-ML
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

---

## 🔮 Future Improvements

- **Transformer Integration**: Implement state-of-the-art encoder architectures like BERT or RoBERTa for context-aware predictions.
- **Explainable AI (XAI)**: Incorporate SHAP or LIME visualizations directly into the "AI Decision Summary" block.
- **Multilingual Support**: Enable translation and verification pipelines for non-English publications.
- **Cloud Deployment**: Host the application via Streamlit Community Cloud or Docker containers on AWS/GCP.

---

## 👨‍💻 Author

- **Ayush Singh** — AI & Machine Learning Developer
- **GitHub**: [Ayush-0915](https://github.com/Ayush-0915)

---

## 📜 License

This project is licensed under the **MIT License** - see the LICENSE details for permissions.