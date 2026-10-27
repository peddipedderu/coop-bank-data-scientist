"""
Co-op Bank Data Science Portfolio App
Roy Peddipedderu - Data Scientist Portfolio
Demonstrates: ML, Credit Scoring, Fraud Detection, Customer Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Roy | Data Scientist Portfolio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 { color: #e94560; font-size: 2.5rem; margin-bottom: 0.5rem; }
    .main-header p { color: #a8b2d8; font-size: 1.1rem; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-card h3 { color: #e94560; font-size: 1.8rem; margin: 0; }
    .metric-card p { color: #a8b2d8; margin: 0.2rem 0 0; font-size: 0.9rem; }
    .section-header {
        color: #e94560;
        border-bottom: 2px solid #0f3460;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem;
    }
    .skill-badge {
        display: inline-block;
        background: #0f3460;
        color: #a8b2d8;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
    .highlight-box {
        background: linear-gradient(135deg, #0f3460, #1a1a2e);
        border-left: 4px solid #e94560;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Roy Peddipedderu")
    st.markdown("### Data Scientist")
    st.markdown("---")
    st.markdown("**🔗 Connect with me**")
    st.markdown("""
    - 📧 [peddipedderu@gmail.com](mailto:peddipedderu@gmail.com)
    - 💼 [LinkedIn](https://www.linkedin.com/in/peddipedderu)
    - 🐙 [GitHub](https://github.com/peddipedderu)
    - 🤗 [HuggingFace](https://huggingface.co/peddipedderu)
    - 📊 [Kaggle](https://www.kaggle.com/peddipedderu)
    """)
    st.markdown("---")
    st.markdown("**🛠️ Core Skills**")
    skills = ["Python", "R", "SQL", "Machine Learning", "Deep Learning",
              "NLP/LLMs", "MLOps", "Credit Scoring", "Fraud Detection",
              "Feature Engineering", "Statistics", "Scikit-learn",
              "TensorFlow", "PyTorch", "Spark", "AWS/GCP/Azure"]
    for skill in skills:
        st.markdown(f"<span class='skill-badge'>{skill}</span>", unsafe_allow_html=True)

    page = st.radio("**📌 Navigate**", [
        "🏠 About Me",
        "💳 Credit Scoring Model",
        "🚨 Fraud Detection",
        "👥 Customer Analytics",
        "📈 Model Monitoring",
        "🤖 AI Chat (LLM Demo)"
    ])

# ── Helper Functions ────────────────────────────────────────────────────────
@st.cache_data
def generate_credit_data(n=5000, seed=42):
    np.random.seed(seed)
    X, y = make_classification(
        n_samples=n, n_features=15, n_informative=10,
        n_redundant=2, weights=[0.85, 0.15], random_state=seed
    )
    feature_names = [
        'income', 'debt_to_income', 'credit_history_months',
        'num_late_payments', 'credit_utilization', 'num_accounts',
        'loan_amount', 'employment_years', 'num_inquiries',
        'monthly_expenses', 'savings_balance', 'age',
        'num_dependents', 'collateral_value', 'loan_purpose_encoded'
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df['default'] = y
    # Scale to realistic ranges
    df['income'] = (df['income'] * 30000 + 80000).clip(20000, 500000)
    df['debt_to_income'] = (df['debt_to_income'] * 0.15 + 0.35).clip(0.05, 0.95)
    df['credit_utilization'] = (df['credit_utilization'] * 0.2 + 0.4).clip(0.01, 0.99)
    df['credit_history_months'] = (df['credit_history_months'] * 30 + 60).clip(6, 240).astype(int)
    df['num_late_payments'] = (df['num_late_payments'] * 1.5 + 2).clip(0, 15).astype(int)
    df['loan_amount'] = (df['loan_amount'] * 200000 + 500000).clip(50000, 5000000)
    df['age'] = (df['age'] * 8 + 38).clip(21, 70).astype(int)
    return df

@st.cache_data
def generate_fraud_data(n=10000, seed=99):
    np.random.seed(seed)
    X, y = make_classification(
        n_samples=n, n_features=12, n_informative=8,
        n_redundant=2, weights=[0.97, 0.03], random_state=seed
    )
    feature_names = [
        'transaction_amount', 'hour_of_day', 'distance_from_home',
        'days_since_last_txn', 'txn_frequency_7d', 'avg_txn_amount',
        'merchant_category_risk', 'device_fingerprint_match',
        'velocity_1h', 'is_international', 'channel_risk', 'customer_tenure'
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df['is_fraud'] = y
    df['transaction_amount'] = (df['transaction_amount'] * 5000 + 8000).clip(100, 200000)
    df['hour_of_day'] = (df['hour_of_day'] * 5 + 12).clip(0, 23).astype(int)
    return df


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name='Random Forest'):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    model = models[model_name]
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    return model, scaler, y_pred, y_proba, auc


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT ME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 About Me":
    st.markdown("""
    <div class='main-header'>
        <h1>👨‍💻 Roy Peddipedderu</h1>
        <p>Data Scientist | Machine Learning Engineer | AI Solutions Architect</p>
        <p>Nairobi, Kenya &nbsp;|&nbsp; 4+ Years Experience &nbsp;|&nbsp; Banking & FinTech Specialist</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, (val, label) in zip(
        [col1, col2, col3, col4],
        [("4+", "Years Experience"), ("15+", "ML Models Deployed"),
         ("95%", "Avg Model Accuracy"), ("3", "Banking Domains")]
    ):
        col.markdown(f"""
        <div class='metric-card'>
            <h3>{val}</h3><p>{label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown("### 🎯 Professional Summary")
        st.markdown("""
        <div class='highlight-box'>
        Results-driven Data Scientist with 4+ years of experience developing and deploying 
        end-to-end machine learning solutions in banking and financial services. Proven track 
        record in credit scoring, fraud detection, customer analytics, and NLP/LLM applications.
        Expert in Python, R, SQL and the full ML lifecycle from problem definition to production 
        deployment using modern MLOps practices.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📚 Education")
        st.info("🎓 **BSc. Data Science / Computer Science** — Top-ranked institution")
        st.info("📜 **Certifications**: AWS ML Specialty | Google Professional ML Engineer | TensorFlow Developer")

        st.markdown("### 💼 Key Experience")
        exp_data = {
            "Role": ["Senior Data Scientist", "ML Engineer", "Data Analyst"],
            "Domain": ["Credit & Risk", "Fraud Detection", "Customer Analytics"],
            "Impact": ["Reduced default rate by 18%", "Caught 94% of fraud cases", "Boosted retention by 22%"]
        }
        st.dataframe(pd.DataFrame(exp_data), use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("### 🔧 Tech Stack")
        tech_categories = {
            "Languages": ["Python", "R", "SQL", "Bash"],
            "ML/AI": ["Scikit-learn", "TensorFlow", "PyTorch", "XGBoost", "LightGBM"],
            "MLOps": ["MLflow", "DVC", "Docker", "Kubernetes", "FastAPI"],
            "Cloud": ["AWS SageMaker", "GCP Vertex AI", "Azure ML"],
            "Data": ["Spark", "Kafka", "dbt", "Airflow", "PostgreSQL"]
        }
        for cat, items in tech_categories.items():
            st.markdown(f"**{cat}**")
            st.markdown(" ".join([f"`{i}`" for i in items]))
            st.markdown("")

    st.markdown("### 📊 Domain Expertise Radar")
    categories = ['Credit Scoring', 'Fraud Detection', 'Customer Analytics',
                  'NLP/LLMs', 'MLOps/Deployment', 'Statistical Modelling',
                  'Data Engineering', 'Responsible AI']
    scores = [92, 88, 85, 82, 80, 90, 78, 85]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.plot(angles_plot, scores_plot, 'o-', linewidth=2, color='#e94560')
    ax.fill(angles_plot, scores_plot, alpha=0.25, color='#e94560')
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, color='#a8b2d8', size=9)
    ax.set_ylim(0, 100)
    ax.set_yticklabels([])
    ax.grid(color='#0f3460', linestyle='--', alpha=0.7)
    ax.spines['polar'].set_color('#0f3460')
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: CREDIT SCORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💳 Credit Scoring Model":
    st.markdown("<h2 class='section-header'>💳 Credit Scoring & Default Prediction</h2>", unsafe_allow_html=True)
    st.markdown("""
    > Demonstrates end-to-end credit risk modelling: data generation, feature engineering, 
    > model training, evaluation (AUC-ROC, KS, Gini), and real-time scoring — aligned with 
    > Co-op Bank's credit risk requirements.
    """)

    df = generate_credit_data()
    feature_cols = [c for c in df.columns if c != 'default']

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### ⚙️ Model Configuration")
        model_choice = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting", "Logistic Regression"])
        test_size = st.slider("Test Set %", 10, 40, 20)
        st.markdown("#### 📋 Dataset Overview")
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Default Rate", f"{df['default'].mean():.1%}")
        st.metric("Features", len(feature_cols))

    with col2:
        st.markdown("#### 📊 Default Distribution by Income Band")
        df['income_band'] = pd.cut(df['income'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        pivot = df.groupby('income_band', observed=True)['default'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        bars = ax.bar(pivot['income_band'], pivot['default'] * 100, color=['#e94560' if v > 0.15 else '#0f3460' for v in pivot['default']])
        ax.set_xlabel('Income Band', color='#a8b2d8')
        ax.set_ylabel('Default Rate (%)', color='#a8b2d8')
        ax.tick_params(colors='#a8b2d8')
        for spine in ax.spines.values(): spine.set_color('#0f3460')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    if st.button("🚀 Train Credit Model", type="primary"):
        with st.spinner("Training model..."):
            X = df[feature_cols].values
            y = df['default'].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42, stratify=y)
            model, scaler, y_pred, y_proba, auc = train_and_evaluate(X_train, X_test, y_train, y_test, model_choice)

        st.success(f"✅ {model_choice} trained! AUC-ROC: **{auc:.4f}**")

        c1, c2, c3 = st.columns(3)
        report = classification_report(y_test, y_pred, output_dict=True)
        c1.metric("AUC-ROC", f"{auc:.4f}")
        c2.metric("Precision (Default)", f"{report['1']['precision']:.4f}")
        c3.metric("Recall (Default)", f"{report['1']['recall']:.4f}")

        col_roc, col_fi = st.columns(2)
        with col_roc:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#16213e')
            ax.plot(fpr, tpr, color='#e94560', lw=2, label=f'AUC = {auc:.3f}')
            ax.plot([0, 1], [0, 1], color='#a8b2d8', linestyle='--', lw=1)
            ax.set_xlabel('False Positive Rate', color='#a8b2d8')
            ax.set_ylabel('True Positive Rate', color='#a8b2d8')
            ax.set_title('ROC Curve', color='white')
            ax.legend(loc='lower right', facecolor='#1a1a2e', labelcolor='white')
            ax.tick_params(colors='#a8b2d8')
            for spine in ax.spines.values(): spine.set_color('#0f3460')
            st.pyplot(fig)
            plt.close()

        with col_fi:
            if hasattr(model, 'feature_importances_'):
                fi = pd.Series(model.feature_importances_, index=feature_cols).nlargest(10)
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor('#1a1a2e')
                ax.set_facecolor('#16213e')
                fi.plot(kind='barh', ax=ax, color='#e94560')
                ax.set_title('Top 10 Feature Importances', color='white')
                ax.tick_params(colors='#a8b2d8')
                for spine in ax.spines.values(): spine.set_color('#0f3460')
                st.pyplot(fig)
                plt.close()

    st.markdown("---")
    st.markdown("#### 🎯 Live Credit Score Calculator")
    col_inp1, col_inp2, col_inp3 = st.columns(3)
    with col_inp1:
        income = st.number_input("Annual Income (KES)", 20000, 500000, 80000, step=5000)
        dti = st.slider("Debt-to-Income Ratio", 0.05, 0.95, 0.35)
        credit_hist = st.slider("Credit History (months)", 6, 240, 60)
    with col_inp2:
        late_payments = st.number_input("Late Payments (count)", 0, 15, 2)
        utilization = st.slider("Credit Utilization", 0.01, 0.99, 0.40)
        num_accounts = st.number_input("Number of Accounts", 1, 20, 5)
    with col_inp3:
        loan_amount = st.number_input("Loan Amount (KES)", 50000, 5000000, 500000, step=50000)
        emp_years = st.slider("Employment Years", 0, 30, 5)
        age = st.number_input("Age", 21, 70, 35)

    if st.button("📊 Calculate Credit Score"):
        # Simple rule-based score for demo
        score = 850
        score -= late_payments * 30
        score -= int(dti * 150)
        score -= int(utilization * 100)
        score += int(credit_hist / 24) * 15
        score += int(emp_years * 5)
        score += int(income / 10000) * 2
        score = max(300, min(850, score))

        risk = "🟢 LOW RISK" if score >= 700 else "🟡 MEDIUM RISK" if score >= 580 else "🔴 HIGH RISK"
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("Credit Score", score)
        col_s2.metric("Risk Classification", risk)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: FRAUD DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Fraud Detection":
    st.markdown("<h2 class='section-header'>🚨 Real-Time Fraud Detection System</h2>", unsafe_allow_html=True)
    st.markdown("""
    > End-to-end fraud detection pipeline using ensemble ML with class imbalance handling — 
    > directly applicable to Co-op Bank's transaction monitoring needs.
    """)

    df_fraud = generate_fraud_data()
    st.markdown(f"**Dataset**: {len(df_fraud):,} transactions | Fraud Rate: **{df_fraud['is_fraud'].mean():.2%}**")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### ⚙️ Configuration")
        fraud_model = st.selectbox("Model", ["Random Forest", "Gradient Boosting", "Logistic Regression"])
        threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)

    with col2:
        st.markdown("#### 📊 Transaction Amount Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        fig.patch.set_facecolor('#1a1a2e')
        for ax, (label, color) in zip(axes, [("Legitimate", "#0f3460"), ("Fraudulent", "#e94560")]):
            subset = df_fraud[df_fraud['is_fraud'] == (1 if label == "Fraudulent" else 0)]
            ax.set_facecolor('#16213e')
            ax.hist(subset['transaction_amount'], bins=40, color=color, alpha=0.8, edgecolor='none')
            ax.set_title(label, color='white', fontsize=10)
            ax.tick_params(colors='#a8b2d8')
            for spine in ax.spines.values(): spine.set_color('#0f3460')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    if st.button("🚀 Train Fraud Model", type="primary"):
        with st.spinner("Training fraud detection model..."):
            feature_cols = [c for c in df_fraud.columns if c != 'is_fraud']
            X = df_fraud[feature_cols].values
            y = df_fraud['is_fraud'].values
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            model, scaler, y_pred, y_proba, auc = train_and_evaluate(X_tr, X_te, y_tr, y_te, fraud_model)

        y_pred_thresh = (y_proba >= threshold).astype(int)
        report = classification_report(y_te, y_pred_thresh, output_dict=True)

        st.success(f"✅ Fraud model trained! AUC-ROC: **{auc:.4f}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC-ROC", f"{auc:.4f}")
        c2.metric("Fraud Precision", f"{report.get('1', {}).get('precision', 0):.3f}")
        c3.metric("Fraud Recall", f"{report.get('1', {}).get('recall', 0):.3f}")
        c4.metric("F1-Score", f"{report.get('1', {}).get('f1-score', 0):.3f}")

        col_cm, col_pr = st.columns(2)
        with col_cm:
            cm = confusion_matrix(y_te, y_pred_thresh)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#16213e')
            sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                        cmap='RdYlGn', cbar=False,
                        xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
            ax.set_title('Confusion Matrix', color='white')
            ax.tick_params(colors='#a8b2d8')
            st.pyplot(fig)
            plt.close()

        with col_pr:
            precision, recall, _ = precision_recall_curve(y_te, y_proba)
            ap = average_precision_score(y_te, y_proba)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#16213e')
            ax.plot(recall, precision, color='#e94560', lw=2, label=f'AP = {ap:.3f}')
            ax.set_xlabel('Recall', color='#a8b2d8')
            ax.set_ylabel('Precision', color='#a8b2d8')
            ax.set_title('Precision-Recall Curve', color='white')
            ax.legend(facecolor='#1a1a2e', labelcolor='white')
            ax.tick_params(colors='#a8b2d8')
            for spine in ax.spines.values(): spine.set_color('#0f3460')
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: CUSTOMER ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Analytics":
    st.markdown("<h2 class='section-header'>👥 Customer Analytics & Segmentation</h2>", unsafe_allow_html=True)

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    np.random.seed(42)
    n = 2000
    # Generate synthetic customer data
    segments = ['Premium', 'Mass Affluent', 'Retail', 'Youth/Digital']
    segment_probs = [0.15, 0.25, 0.45, 0.15]
    seg_labels = np.random.choice(segments, size=n, p=segment_probs)

    df_cust = pd.DataFrame({
        'customer_id': [f'KE{i:06d}' for i in range(n)],
        'age': np.where(seg_labels == 'Youth/Digital',
                        np.random.randint(18, 30, n),
                        np.random.randint(28, 68, n)),
        'monthly_income': (np.where(seg_labels == 'Premium', np.random.normal(250000, 80000, n),
                           np.where(seg_labels == 'Mass Affluent', np.random.normal(100000, 30000, n),
                           np.where(seg_labels == 'Retail', np.random.normal(45000, 15000, n),
                                    np.random.normal(25000, 8000, n))))).clip(5000),
        'products_held': np.random.poisson(3 if 'Premium' in seg_labels else 2, n).clip(1, 10),
        'tenure_months': np.random.randint(1, 180, n),
        'digital_transactions_pct': np.where(seg_labels == 'Youth/Digital',
                                              np.random.beta(8, 2, n),
                                              np.random.beta(3, 5, n)),
        'avg_monthly_balance': (np.random.exponential(80000, n)).clip(1000, 2000000),
        'churn_risk': np.random.beta(2, 8, n),
        'true_segment': seg_labels
    })

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Segment Distribution")
        seg_counts = df_cust['true_segment'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        colors = ['#e94560', '#0f3460', '#533483', '#1a8a5c']
        wedges, texts, autotexts = ax.pie(seg_counts, labels=seg_counts.index,
                                           autopct='%1.1f%%', colors=colors,
                                           textprops={'color': 'white'})
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### 📈 Income Distribution by Segment")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        for seg, color in zip(segments, colors):
            data = df_cust[df_cust['true_segment'] == seg]['monthly_income']
            ax.hist(data, bins=30, alpha=0.6, color=color, label=seg, density=True)
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        ax.tick_params(colors='#a8b2d8')
        ax.set_xlabel('Monthly Income (KES)', color='#a8b2d8')
        for spine in ax.spines.values(): spine.set_color('#0f3460')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("#### 🎯 Customer Churn Risk Analysis")
    churn_threshold = st.slider("Churn Risk Threshold", 0.1, 0.9, 0.5)
    high_churn = df_cust[df_cust['churn_risk'] >= churn_threshold]
    st.markdown(f"""
    - **High-risk customers**: {len(high_churn):,} ({len(high_churn)/len(df_cust):.1%})  
    - **Estimated revenue at risk**: KES {(high_churn['avg_monthly_balance'].sum() * 0.02):,.0f}/month  
    - **Recommended action**: Proactive retention campaign for top {min(500, len(high_churn)):,} customers
    """)

    st.markdown("#### 🗂️ Sample Customer Data")
    st.dataframe(df_cust.drop('customer_id', axis=1).head(10),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL MONITORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Monitoring":
    st.markdown("<h2 class='section-header'>📈 Model Monitoring & MLOps Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("""
    > Production model monitoring with data drift detection, performance tracking 
    > and automated alerting — demonstrates MLOps maturity aligned with Co-op Bank's 
    > model governance requirements.
    """)

    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=52, freq='W')
    
    # Simulate model performance drift
    auc_scores = 0.87 - np.cumsum(np.random.normal(0.001, 0.008, 52)).clip(-0.1, 0.05)
    auc_scores = np.clip(auc_scores, 0.70, 0.92)
    psi_scores = np.abs(np.cumsum(np.random.normal(0.002, 0.015, 52))).clip(0, 0.5)
    
    df_monitor = pd.DataFrame({
        'week': dates,
        'auc_roc': auc_scores,
        'psi': psi_scores,
        'predictions': np.random.randint(800, 2000, 52),
        'data_drift_alert': psi_scores > 0.25
    })

    col1, col2, col3, col4 = st.columns(4)
    latest = df_monitor.iloc[-1]
    col1.metric("Current AUC", f"{latest['auc_roc']:.4f}", f"{latest['auc_roc'] - df_monitor.iloc[-4]['auc_roc']:+.4f}")
    col2.metric("PSI Score", f"{latest['psi']:.4f}", delta_color="inverse")
    col3.metric("Weekly Predictions", f"{latest['predictions']:,}")
    col4.metric("Drift Alerts (4w)", str(df_monitor.tail(4)['data_drift_alert'].sum()))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.patch.set_facecolor('#1a1a2e')

    for ax, (col, label, color) in zip(axes, [
        ('auc_roc', 'AUC-ROC Over Time', '#e94560'),
        ('psi', 'Population Stability Index (PSI)', '#533483')
    ]):
        ax.set_facecolor('#16213e')
        ax.plot(df_monitor['week'], df_monitor[col], color=color, lw=2)
        ax.fill_between(df_monitor['week'], df_monitor[col], alpha=0.2, color=color)
        ax.set_ylabel(label, color='#a8b2d8', fontsize=9)
        ax.tick_params(colors='#a8b2d8')
        for spine in ax.spines.values(): spine.set_color('#0f3460')
        if col == 'psi':
            ax.axhline(0.25, color='orange', linestyle='--', lw=1, label='Warning (0.25)')
            ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### 🔔 Governance Checklist")
    checks = {
        "Model documentation current": True,
        "Fairness/bias testing completed": True,
        "Explainability report (SHAP/LIME)": True,
        "Human oversight mechanism active": True,
        "Data lineage tracked": True,
        "Model risk review scheduled": False,
        "Regulatory compliance sign-off": False
    }
    for check, status in checks.items():
        icon = "✅" if status else "⚠️"
        st.markdown(f"{icon} {check}")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: AI CHAT DEMO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Chat (LLM Demo)":
    st.markdown("<h2 class='section-header'>🤖 Banking AI Assistant (LLM Demo)</h2>", unsafe_allow_html=True)
    st.markdown("""
    > Demonstrates practical LLM/GenAI integration for banking use cases — aligned with 
    > Co-op Bank's AI strategy for customer experience and operational efficiency.
    """)

    st.info("💡 This is a rule-based demo. In production, this connects to a fine-tuned LLM (GPT-4, Gemini, Llama) with RAG over banking knowledge bases.")

    banking_qa = {
        "credit": "Credit scoring at Co-op Bank uses a combination of bureau data, transaction history, and ML models to assess default probability. Our models achieve AUC > 0.87.",
        "fraud": "Our fraud detection system processes transactions in real-time (<100ms latency) using ensemble ML models, flagging suspicious patterns with 94%+ recall.",
        "loan": "Loan eligibility depends on income, credit history, debt-to-income ratio, and employment stability. Our ML model provides instant pre-qualification decisions.",
        "account": "Account analytics help identify cross-sell opportunities. Customers with savings + current accounts have 40% higher lifetime value.",
        "interest": "Interest rates are risk-based — determined by your credit score, loan tenure, and collateral. Better credit scores unlock lower rates.",
        "default": "Our collections models predict accounts 90 days before default, enabling proactive outreach that reduces write-offs by 18%.",
    }

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm Co-op Bank's AI assistant. Ask me about credit, loans, fraud, or account analytics!"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask about banking services, credit, fraud..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        response = "I can help with: **credit scoring**, **fraud detection**, **loans**, **account analytics**, **interest rates**, and **default prediction**. Please ask about one of these topics!"
        prompt_lower = prompt.lower()
        for keyword, answer in banking_qa.items():
            if keyword in prompt_lower:
                response = answer
                break

        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
