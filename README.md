# Roy Peddipedderu — Data Science Portfolio App

A production-ready, interactive **Data Science & AI portfolio** built with Streamlit — demonstrating the exact skills required for the Co-op Bank Data Scientist role (DS/IID/2026).

## 🌐 Live Demo
**[https://coop-bank-data-scientist.fly.dev](https://coop-bank-data-scientist.fly.dev)**

## 📋 What This App Demonstrates

| Module | Skills Demonstrated | Relevance to Co-op Bank |
|--------|-------------------|------------------------|
| 💳 Credit Scoring | Logistic Regression, RF, GBM, AUC-ROC, KS-Gini, Feature Engineering | Credit risk, lending decisions |
| 🚨 Fraud Detection | Ensemble ML, imbalance handling, PR curves, real-time scoring | Transaction monitoring, fraud prevention |
| 👥 Customer Analytics | Segmentation, churn prediction, CLV, clustering | Customer engagement, retention |
| 📈 Model Monitoring | PSI drift detection, performance dashboards, governance | MLOps, responsible AI, model risk |
| 🤖 LLM Demo | Banking chatbot, RAG-ready architecture | GenAI for customer experience |

## 🛠️ Tech Stack

- **Languages**: Python 3.11
- **ML Libraries**: Scikit-learn, NumPy, Pandas, SciPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Framework**: Streamlit
- **Containerization**: Docker
- **Deployment**: Fly.io (port 8502)
- **Version Control**: Git + GitHub

## 🚀 Quick Start

### Run Locally
```bash
git clone https://github.com/peddipedderu/coop-bank-data-scientist.git
cd coop-bank-data-scientist
pip install -r requirements.txt
streamlit run app.py --server.port 8502
```

### Docker
```bash
docker build -t roy-ds-portfolio .
docker run -p 8502:8502 roy-ds-portfolio
```

### Deploy to Fly.io
```bash
fly auth login
fly deploy
```

## 📊 Key Model Metrics (Demo)

| Model | AUC-ROC | Precision | Recall |
|-------|---------|-----------|--------|
| Credit Default (GBM) | 0.87 | 0.79 | 0.72 |
| Fraud Detection (RF) | 0.94 | 0.81 | 0.78 |
| Churn Prediction (LR) | 0.82 | 0.74 | 0.69 |

## 🔗 Connect

| Platform | Profile |
|----------|---------|
| 💼 LinkedIn | [linkedin.com/in/peddipedderu](https://www.linkedin.com/in/peddipedderu) |
| 🐙 GitHub | [github.com/peddipedderu](https://github.com/peddipedderu) |
| 🤗 HuggingFace | [huggingface.co/peddipedderu](https://huggingface.co/peddipedderu) |
| 📊 Kaggle | [kaggle.com/peddipedderu](https://www.kaggle.com/peddipedderu) |

---
*Portfolio built to support Co-op Bank Data Scientist application — Reference: DS/IID/2026*
